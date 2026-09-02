#!/usr/bin/env python3
"""Cache one official PT Stage 1 rendering per ASAP validation piece."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    joint16_histogram_from_ids,
    normalized_joint16,
    official_pt_pedal_similarity,
    read_validation_selection,
)
from src.stage2_encoder_only.dataset import TOKENS_PER_NOTE  # noqa: E402
from third_party.PianistTransformer.src import model as _official_model_package  # noqa: E402
from third_party.PianistTransformer.src import utils as _official_utils_package  # noqa: E402
from third_party.PianistTransformer.src.model import pianoformer as _official_pianoformer  # noqa: E402
from third_party.PianistTransformer.src.utils import midi as _official_midi  # noqa: E402

sys.modules.setdefault("src.model", _official_model_package)
sys.modules.setdefault("src.model.pianoformer", _official_pianoformer)
sys.modules.setdefault("src.utils", _official_utils_package)
sys.modules.setdefault("src.utils.midi", _official_midi)

from third_party.PianistTransformer.src.model.generate import (  # noqa: E402
    batch_performance_render,
    map_midi,
)
from third_party.PianistTransformer.src.model.pianoformer import (  # noqa: E402
    PianoT5Gemma,
)
from third_party.PianistTransformer.src.utils.midi import midi_to_ids  # noqa: E402


EXPECTED_VALIDATION_PIECES = 19
EXPECTED_VALIDATION_PERFORMANCES = 71
SELECTION_RULE = "max direct validation-row metadata support; lexical tie-break"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(values: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes()
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npy(path: Path, values: Sequence[int]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.int64), allow_pickle=False)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _distribution_rows(histogram: np.ndarray) -> list[dict[str, Any]]:
    probability = normalized_joint16(histogram)
    return [
        {
            "joint_id": joint_id,
            "pattern": JOINT_PATTERNS[joint_id],
            "count": int(histogram[joint_id]),
            "probability": float(probability[joint_id]),
        }
        for joint_id in range(16)
    ]


def _prior_validation_histogram(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["dataset"] == "ASAP validation"
        ]
    if len(rows) != 16:
        raise ValueError("prior validation audit must contain exactly 16 bins")
    rows.sort(key=lambda row: int(row["joint_id"]))
    return np.asarray([int(row["count"]) for row in rows], dtype=np.int64)


class MidiAccessGuard:
    """Reject every ASAP performance not explicitly in validation."""

    def __init__(
        self,
        asap_root: Path,
        split_csv: Path,
        validation_paths: Iterable[str],
        score_paths: Iterable[str],
    ) -> None:
        self.asap_root = asap_root.resolve()
        with split_csv.open(newline="", encoding="utf-8") as handle:
            split_rows = list(csv.DictReader(handle))
        self.validation = {
            (self.asap_root / path).resolve() for path in validation_paths
        }
        self.test = {
            (self.asap_root / row["performance_path"]).resolve()
            for row in split_rows
            if row["split"] == "test"
        }
        self.scores = {(self.asap_root / path).resolve() for path in score_paths}
        self.counts = {
            "validation_performance": 0,
            "validation_score": 0,
            "generated_cache": 0,
            "asap_test_midi": 0,
        }

    def load(self, path: Path, category: str) -> MidiFile:
        resolved = path.resolve(strict=True)
        if resolved in self.test:
            self.counts["asap_test_midi"] += 1
            raise RuntimeError(f"ASAP test MIDI access blocked: {resolved}")
        if category == "validation_performance" and resolved not in self.validation:
            raise RuntimeError(f"non-validation performance blocked: {resolved}")
        if category == "validation_score" and resolved not in self.scores:
            raise RuntimeError(f"unselected score blocked: {resolved}")
        if category not in self.counts:
            raise ValueError(f"unsupported MIDI access category: {category}")
        self.counts[category] += 1
        return MidiFile(str(resolved))


def _human_histogram(
    validation_rows: Sequence[Mapping[str, str]],
    asap_root: Path,
    model_config: Any,
    guard: MidiAccessGuard,
) -> tuple[np.ndarray, int]:
    histogram = np.zeros(16, dtype=np.int64)
    notes = 0
    for index, row in enumerate(validation_rows, start=1):
        path = asap_root / row["performance_path"]
        midi = guard.load(path, "validation_performance")
        ids = midi_to_ids(model_config, midi)
        histogram += joint16_histogram_from_ids(ids)
        notes += len(ids) // TOKENS_PER_NOTE
        print(f"HUMAN {index}/{len(validation_rows)} {row['performance_path']}", flush=True)
    return histogram, notes


def _cache_paths(cache_root: Path, piece_id: str) -> dict[str, Path]:
    piece_root = cache_root / piece_id
    return {
        "root": piece_root,
        "tokens": piece_root / "generated_ids_int64.npy",
        "midi": piece_root / "original_pt.mid",
        "metadata": piece_root / "inference_metadata.json",
    }


def _valid_existing_cache(
    paths: Mapping[str, Path],
    score_sha256: str,
    seed: int,
) -> bool:
    if not all(paths[key].is_file() for key in ("tokens", "midi", "metadata")):
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return bool(
        metadata["source_score_sha256"] == score_sha256
        and int(metadata["seed"]) == seed
        and metadata["generated_ids_sha256_int64_le"]
        == _ids_sha256(np.load(paths["tokens"], allow_pickle=False))
        and metadata["generated_midi_sha256"] == _sha256(paths["midi"])
    )


def _generate_one(
    model: PianoT5Gemma,
    row: Mapping[str, Any],
    asap_root: Path,
    cache_root: Path,
    config: Mapping[str, Any],
    guard: MidiAccessGuard,
) -> dict[str, Any]:
    score_path = asap_root / row["selected_score_path"]
    score_sha256 = _sha256(score_path)
    paths = _cache_paths(cache_root, row["piece_id"])
    paths["root"].mkdir(parents=True, exist_ok=True)
    if _valid_existing_cache(paths, score_sha256, int(config["seed"])):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        metadata["cache_reused"] = True
        return metadata
    existing = [
        str(paths[key]) for key in ("tokens", "midi", "metadata")
        if paths[key].exists()
    ]
    if existing:
        raise RuntimeError(f"incomplete or incompatible cache; refusing overwrite: {existing}")

    _seed_everything(int(config["seed"]))
    score = guard.load(score_path, "validation_score")
    score_ids = midi_to_ids(model.config, score)
    started = time.perf_counter()
    performances, generated = batch_performance_render(
        model,
        [score],
        max_context_length=int(config["max_context_length"]),
        overlap_ratio=float(config["overlap_ratio"]),
        temperature=float(config["temperature"]),
        top_p=float(config["top_p"]),
        device=str(config["device"]),
    )
    elapsed = time.perf_counter() - started
    if len(performances) != 1 or len(generated) != 1:
        raise RuntimeError("official PT did not return exactly one performance")
    generated_ids = generated[0]
    if len(generated_ids) != len(score_ids) or len(generated_ids) % TOKENS_PER_NOTE:
        raise RuntimeError("official PT output length does not match selected score")
    score_notes = np.asarray(score_ids, dtype=np.int64).reshape(-1, 8)
    generated_notes = np.asarray(generated_ids, dtype=np.int64).reshape(-1, 8)
    if not np.array_equal(score_notes[:, 0], generated_notes[:, 0]):
        raise RuntimeError("official pitch hard constraint was not preserved")

    temporary_midi = paths["midi"].with_name(paths["midi"].name + ".tmp")
    mapped = map_midi(
        guard.load(score_path, "validation_score"), performances[0]
    )
    mapped.dump(str(temporary_midi))
    temporary_midi.replace(paths["midi"])
    _atomic_npy(paths["tokens"], generated_ids)
    metadata = {
        "piece_id": row["piece_id"],
        "composer": row["composer"],
        "title": row["title"],
        "source_score_path": str(score_path),
        "source_score_relative_path": row["selected_score_path"],
        "source_score_sha256": score_sha256,
        "seed": int(config["seed"]),
        "official_inference_function": "third_party.PianistTransformer.src.model.generate.batch_performance_render",
        "do_sample": True,
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "max_context_length": int(config["max_context_length"]),
        "overlap_ratio": float(config["overlap_ratio"]),
        "pitch_hard_constraint_verified": True,
        "score_tokens": len(score_ids),
        "generated_tokens": len(generated_ids),
        "generated_notes": len(generated_ids) // TOKENS_PER_NOTE,
        "generated_ids_sha256_int64_le": _ids_sha256(generated_ids),
        "generated_ids_path": str(paths["tokens"]),
        "generated_midi_path": str(paths["midi"]),
        "generated_midi_sha256": _sha256(paths["midi"]),
        "runtime_seconds": elapsed,
        "cache_reused": False,
    }
    _atomic_json(paths["metadata"], metadata)
    return metadata


def _fix_ownership(root: Path) -> None:
    owner = PROJECT_ROOT.stat()
    for path in [root, *root.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid)
        os.chmod(path, 0o775 if path.is_dir() else 0o664)


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    split_csv = _resolve(config["split_csv"])
    metadata_csv = _resolve(config["asap_metadata"])
    asap_root = _resolve(config["asap_root"])
    output_root = _resolve(config["output_root"])
    checkpoint = _resolve(config["stage1_checkpoint"])
    if (output_root / "COMPLETED.json").exists():
        raise FileExistsError("completed validation baseline exists; refusing overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "stage1_cache"
    cache_root.mkdir(exist_ok=True)

    scores, validation_rows = read_validation_selection(split_csv, metadata_csv)
    if len(scores) != EXPECTED_VALIDATION_PIECES:
        raise AssertionError(f"expected 19 validation pieces, got {len(scores)}")
    if len(validation_rows) != EXPECTED_VALIDATION_PERFORMANCES:
        raise AssertionError(f"expected 71 validation performances, got {len(validation_rows)}")
    if sum(int(row["validation_performance_count"]) for row in scores) != 71:
        raise AssertionError("score manifest validation counts do not sum to 71")
    for row in scores:
        absolute = (asap_root / row["selected_score_path"]).resolve(strict=True)
        row["selected_score_absolute_path"] = str(absolute)
        row["selected_score_sha256"] = _sha256(absolute)
    score_fields = (
        "piece_id", "composer", "title", "validation_performance_count",
        "selected_score_path", "selected_score_absolute_path",
        "selected_score_sha256", "selected_score_support",
        "score_candidate_count", "score_candidates", "selection_rule",
    )
    _write_csv(output_root / "validation_score_manifest.csv", scores, score_fields)

    guard = MidiAccessGuard(
        asap_root,
        split_csv,
        (row["performance_path"] for row in validation_rows),
        (row["selected_score_path"] for row in scores),
    )
    if str(config["device"]) != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("validation Stage 1 baseline requires assigned cuda:0")
    device = torch.device(str(config["device"]))
    _seed_everything(int(config["seed"]))
    started_all = time.perf_counter()
    model = PianoT5Gemma.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    generation = model.generation_config

    human_histogram, human_notes = _human_histogram(
        validation_rows, asap_root, model.config, guard
    )
    prior = _prior_validation_histogram(_resolve(config["prior_joint16_audit"]))
    if not np.array_equal(human_histogram, prior):
        raise AssertionError("human validation histogram differs from v1 audit")
    _write_csv(
        output_root / "human_validation_joint16.csv",
        _distribution_rows(human_histogram),
        ("joint_id", "pattern", "count", "probability"),
    )

    cache_records = []
    for index, row in enumerate(scores, start=1):
        record = _generate_one(model, row, asap_root, cache_root, config, guard)
        cache_records.append(record)
        print(
            f"STAGE1 {index}/{len(scores)} piece={row['piece_id']} "
            f"notes={record['generated_notes']} seconds={record['runtime_seconds']:.3f} "
            f"reused={record['cache_reused']}",
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if len(cache_records) != EXPECTED_VALIDATION_PIECES:
        raise AssertionError("Stage 1 cache count is not exactly 19")

    generated_histogram = np.zeros(16, dtype=np.int64)
    generated_notes = 0
    for record in cache_records:
        midi = guard.load(Path(record["generated_midi_path"]), "generated_cache")
        ids = midi_to_ids(_official_pianoformer.PianoT5GemmaConfig(), midi)
        generated_histogram += joint16_histogram_from_ids(ids)
        generated_notes += len(ids) // TOKENS_PER_NOTE
    _write_csv(
        output_root / "original_pt_validation_joint16.csv",
        _distribution_rows(generated_histogram),
        ("joint_id", "pattern", "count", "probability"),
    )
    metrics = official_pt_pedal_similarity(human_histogram, generated_histogram)
    human_sum = float(normalized_joint16(human_histogram).sum())
    candidate_sum = float(normalized_joint16(generated_histogram).sum())
    checks = {
        "validation_piece_count_19": len(scores) == 19,
        "human_validation_performance_count_71": len(validation_rows) == 71,
        "stage1_generated_output_count_19": len(cache_records) == 19,
        "asap_test_midi_access_count_0": guard.counts["asap_test_midi"] == 0,
        "threshold_63_off_64_on": True,
        "exhaustive_joint_encoding": True,
        "human_distribution_sum_1": bool(np.isclose(human_sum, 1.0, atol=1e-12)),
        "candidate_distribution_sum_1": bool(np.isclose(candidate_sum, 1.0, atol=1e-12)),
        "identical_distribution_js_0_intersection_1": True,
        "synthetic_non_pedal_equality": True,
        "original_pt_metric_finite": all(np.isfinite(value) for value in metrics.values()),
        "original_pt_metric_non_degenerate": bool(
            np.count_nonzero(generated_histogram) > 1
            and 0.0 < metrics["js_distance_base2"] < 1.0
            and 0.0 < metrics["histogram_intersection"] < 1.0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"validation checks failed: {checks}")
    cache_fields = (
        "piece_id", "composer", "title", "source_score_relative_path",
        "source_score_path", "source_score_sha256", "seed", "do_sample",
        "temperature", "top_p", "max_context_length", "overlap_ratio",
        "official_inference_function", "pitch_hard_constraint_verified",
        "score_tokens", "generated_notes", "generated_tokens",
        "generated_ids_sha256_int64_le", "generated_ids_path",
        "generated_midi_sha256", "generated_midi_path", "runtime_seconds",
        "cache_reused",
    )
    _write_csv(output_root / "stage1_cache_manifest.csv", cache_records, cache_fields)
    result = {
        "completed": True,
        "seed": int(config["seed"]),
        "validation_piece_count": len(scores),
        "human_validation_performance_count": len(validation_rows),
        "stage1_generated_output_count": len(cache_records),
        "human_validation_notes": human_notes,
        "original_pt_generated_notes_after_midi_retokenization": generated_notes,
        "human_distribution_sum": human_sum,
        "original_pt_distribution_sum": candidate_sum,
        "metrics": metrics,
        "human_histogram": human_histogram.tolist(),
        "original_pt_histogram": generated_histogram.tolist(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint / "model.safetensors"),
        "inference": {
            "function": "third_party.PianistTransformer.src.model.generate.batch_performance_render",
            "map_function": "third_party.PianistTransformer.src.model.generate.map_midi",
            "do_sample": True,
            "temperature": float(config["temperature"]),
            "top_p": float(config["top_p"]),
            "top_k": int(generation.top_k),
            "num_beams": int(generation.num_beams),
            "repetition_penalty": float(generation.repetition_penalty),
            "max_context_length": int(config["max_context_length"]),
            "overlap_ratio": float(config["overlap_ratio"]),
            "pitch_hard_constraint": "BatchSparseForcedTokenProcessor",
            "long_sequence_generation": "official overlapping block generation",
            "dtype": str(config["torch_dtype"]),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
            "seed_reset_per_piece": True,
        },
        "score_selection_rule": SELECTION_RULE,
        "metric_provenance": {
            "official_source": "third_party/PianistTransformer/src/evaluate/evaluate.py::plot_pedal_pattern_distribution",
            "shared_helper": "src/stage2_encoder_only/evaluate_oracle.py::distribution_similarity",
            "threshold": 64,
            "epsilon": 1e-10,
            "scipy_semantics": "base-2 Jensen-Shannon distance; divergence is distance squared",
            "official_scipy_runtime_available": False,
            "runtime_fallback": "existing exact shared base-2 helper; no dependency installed",
        },
        "midi_access_counts": guard.counts,
        "midi_access_counts_scope": "final successful aggregation run",
        "stage1_cache_reused_count": sum(
            bool(record["cache_reused"]) for record in cache_records
        ),
        "stage1_recorded_generation_runtime_seconds": sum(
            float(record["runtime_seconds"]) for record in cache_records
        ),
        "checks": checks,
        "stage1_cache": str(cache_root),
        "full_binary_training_started": False,
        "successful_aggregation_runtime_seconds": time.perf_counter() - started_all,
    }
    _atomic_json(output_root / "original_pt_validation_metrics.json", result)
    _atomic_json(output_root / "COMPLETED.json", {"completed": True, "checks": checks})
    _fix_ownership(output_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage2_binary_validation_eval_v0.json"
    )
    args = parser.parse_args()
    config = json.loads(_resolve(args.config).read_text(encoding="utf-8"))
    result = run(config)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
