#!/usr/bin/env python3
"""Build and permanently freeze the validation-only Original PT Stage-1 bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mido
import numpy as np
import torch
import transformers
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    cc64_schedule,
    sha256_file,
    signature_sha256,
)
from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    joint16_histogram_from_ids,
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
    PianoT5GemmaConfig,
)
from third_party.PianistTransformer.src.utils.midi import midi_to_ids  # noqa: E402


EXPERIMENT_ID = "stage2_binary_canonical_v1"
OUTPUT_ROOT = ROOT / "analysis/stage2_binary_canonical_v1"
BANK_ROOT = OUTPUT_ROOT / "canonical_validation_stage1"
PROTOCOL_PATH = OUTPUT_ROOT / "experiment_protocol.json"
STATUS_PATH = OUTPUT_ROOT / "run_status.json"
MANIFEST_PATH = OUTPUT_ROOT / "canonical_validation_stage1_manifest.csv"
BASELINE_PATH = OUTPUT_ROOT / "canonical_validation_baseline.json"
DISTRIBUTION_PATH = OUTPUT_ROOT / "canonical_validation_joint16_distribution.csv"
REPORT_PATH = OUTPUT_ROOT / "CANONICAL_VALIDATION_STAGE1_REPORT.md"
LOG_PATH = OUTPUT_ROOT / "canonical_validation.log"
SPLIT_PATH = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
SELECTION_SOURCE = ROOT / "analysis/stage2_binary_v0/validation_eval_v0/validation_score_manifest.csv"
TRAIN_MANIFEST = ROOT / "analysis/stage2_binary_v0/data_prep_v1/train_manifest.csv"
CACHE_ROOT = ROOT / "analysis/stage2_binary_v0/train_setup_v0/shared_cache"
CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
PT_ROOT = ROOT / "third_party/PianistTransformer"
OFFICIAL_GENERATOR = PT_ROOT / "src/model/generate.py"
OFFICIAL_TOKENIZER = PT_ROOT / "src/utils/midi.py"
OFFICIAL_EVALUATOR = PT_ROOT / "src/evaluate/evaluate.py"

PINNED_PT_REVISION = "747df2d12291e37f6638b39f1b71517e579ad48c"
EXPECTED_SPLIT_SHA256 = "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b"
EXPECTED_SELECTION_SHA256 = "4aac49a07a5bf0fdac24c9f9c3de7a268455f6a78f3ddb7bcf331ce7817e3711"
EXPECTED_CHECKPOINT_SHA256 = "8fb9111efc147adcc61a4885bd42a6e6b625626d9fb8576811916927aa8281d0"
EXPECTED_TRAIN_MANIFEST_SHA256 = "999b10e3d41bca7992b4e46ce85a70bbd2a0b313391247a53da510587efd4f20"
EXPECTED_PIECES = 19
EXPECTED_HUMANS = 71
EXPECTED_COUNTS = {"train": 892, "validation": 71, "test": 104}
SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_CONTEXT = 4096
OVERLAP = 0.5
EPSILON = 1e-10

MANIFEST_FIELDS = (
    "piece_id",
    "composer",
    "title",
    "validation_performance_count",
    "selected_score_path",
    "selected_score_absolute_path",
    "selected_score_sha256",
    "selected_score_support",
    "score_candidate_count",
    "score_candidates",
    "score_selection_rule",
    "score_selection_source_manifest",
    "score_selection_source_manifest_sha256",
    "canonical_midi_path",
    "canonical_midi_sha256",
    "canonical_non_cc64_signature_sha256",
    "generated_ids_path",
    "generated_token_sha256_int64_le",
    "note_count",
    "midi_duration_seconds",
    "cc64_event_count",
    "seed",
    "inference_configuration_identifier",
    "status",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_npy(path: Path, values: Sequence[int]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.int64), allow_pickle=False)
    temporary.replace(path)


def _ids_sha256(values: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _json_identifier(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_head(path: Path) -> str:
    git_dir = path / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_name = head.removeprefix("ref: ")
    loose = git_dir / ref_name
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and not line.startswith("^"):
                digest, name = line.split(" ", 1)
                if name == ref_name:
                    return digest
    raise RuntimeError(f"cannot resolve PT git revision: {ref_name}")


def _seed_everything(seed: int) -> None:
    """Match the validated historical CPU runner exactly."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _status(state: str, **extra: Any) -> None:
    prior: dict[str, Any] = {}
    if STATUS_PATH.is_file():
        prior = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    payload = {
        **prior,
        "experiment_id": EXPERIMENT_ID,
        "state": state,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "stage2_training_started": False,
        "asap_test_midi_access_count": 0,
        **extra,
    }
    _atomic_json(STATUS_PATH, payload)


def _assert_source_contracts() -> dict[str, str]:
    hashes = {
        "split_sha256": sha256_file(SPLIT_PATH),
        "selection_manifest_sha256": sha256_file(SELECTION_SOURCE),
        "checkpoint_model_sha256": sha256_file(CHECKPOINT / "model.safetensors"),
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "official_generator_sha256": sha256_file(OFFICIAL_GENERATOR),
        "official_tokenizer_sha256": sha256_file(OFFICIAL_TOKENIZER),
        "official_evaluator_sha256": sha256_file(OFFICIAL_EVALUATOR),
        "pt_source_revision": _git_head(PT_ROOT),
    }
    expected = {
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "selection_manifest_sha256": EXPECTED_SELECTION_SHA256,
        "checkpoint_model_sha256": EXPECTED_CHECKPOINT_SHA256,
        "train_manifest_sha256": EXPECTED_TRAIN_MANIFEST_SHA256,
        "pt_source_revision": PINNED_PT_REVISION,
    }
    changed = {name: (hashes[name], digest) for name, digest in expected.items() if hashes[name] != digest}
    if changed:
        raise RuntimeError(f"pinned source/provenance changed: {changed}")

    evaluator = OFFICIAL_EVALUATOR.read_text(encoding="utf-8")
    required = (
        'gt_midi = miditoolkit.MidiFile(item["gt"])',
        'pred_midi = miditoolkit.MidiFile(item["pred"])',
        "pedal_binarize_threshold: int = 64",
        "binary_value = 1 if pedal_value >= pedal_binarize_threshold else 0",
        "epsilon = 1e-10",
        "jensenshannon(gt_prob, pred_prob, base=2)",
        "np.sum(np.minimum(gt_prob, pred_prob))",
    )
    missing = [fragment for fragment in required if fragment not in evaluator]
    if missing:
        raise RuntimeError(f"official evaluator contract changed: {missing}")
    return hashes


def _split_inventory() -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_csv(SPLIT_PATH)
    by_split = {name: [row for row in rows if row["split"] == name] for name in EXPECTED_COUNTS}
    for name, expected in EXPECTED_COUNTS.items():
        if len(by_split[name]) != expected:
            raise RuntimeError(f"ASAP {name} count changed: {len(by_split[name])} != {expected}")
    piece_counts = {name: len({row["piece_id"] for row in group}) for name, group in by_split.items()}
    if piece_counts != {"train": 180, "validation": 19, "test": 23}:
        raise RuntimeError(f"ASAP piece inventory changed: {piece_counts}")
    validation = by_split["validation"]
    return validation, {
        "performance_counts": {name: len(group) for name, group in by_split.items()},
        "piece_counts": piece_counts,
        "split_sha256": sha256_file(SPLIT_PATH),
    }


def _selection_rows(validation_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    source = _read_csv(SELECTION_SOURCE)
    if len(source) != EXPECTED_PIECES or len({row["piece_id"] for row in source}) != EXPECTED_PIECES:
        raise RuntimeError("existing validation score manifest is not 19 unique pieces")
    split_piece_counts: dict[str, int] = {}
    for row in validation_rows:
        split_piece_counts[row["piece_id"]] = split_piece_counts.get(row["piece_id"], 0) + 1
    if set(split_piece_counts) != {row["piece_id"] for row in source}:
        raise RuntimeError("validation score manifest piece set differs from pinned split")
    output: list[dict[str, Any]] = []
    for row in source:
        score = Path(row["selected_score_absolute_path"])
        expected_score = ASAP_ROOT / row["selected_score_path"]
        if score.resolve(strict=True) != expected_score.resolve(strict=True):
            raise RuntimeError(f"score root/path mismatch: {row['piece_id']}")
        digest = sha256_file(score)
        if digest != row["selected_score_sha256"]:
            raise RuntimeError(f"selected score hash changed: {row['piece_id']}")
        if int(row["validation_performance_count"]) != split_piece_counts[row["piece_id"]]:
            raise RuntimeError(f"score support inventory changed: {row['piece_id']}")
        output.append(
            {
                **{key: value for key, value in row.items() if key != "selection_rule"},
                "score_selection_rule": row["selection_rule"],
                "score_selection_source_manifest": str(SELECTION_SOURCE),
                "score_selection_source_manifest_sha256": EXPECTED_SELECTION_SHA256,
                "canonical_midi_path": str(BANK_ROOT / row["piece_id"] / "original_pt.mid"),
                "canonical_midi_sha256": "",
                "canonical_non_cc64_signature_sha256": "",
                "generated_ids_path": str(BANK_ROOT / row["piece_id"] / "generated_ids_int64.npy"),
                "generated_token_sha256_int64_le": "",
                "note_count": "",
                "midi_duration_seconds": "",
                "cc64_event_count": "",
                "seed": SEED,
                "inference_configuration_identifier": "",
                "status": "pending",
            }
        )
    return output


def _runtime_inference(model: PianoT5Gemma) -> dict[str, Any]:
    generation = model.generation_config
    parameter = next(model.parameters())
    attention = getattr(model.config, "_attn_implementation", None)
    encoder_attention = getattr(model.config.encoder, "_attn_implementation", None)
    decoder_attention = getattr(model.config.decoder, "_attn_implementation", None)
    config = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_provenance": "yhj137/pianist-transformer-rendering local official checkpoint",
        "model_safetensors_sha256": EXPECTED_CHECKPOINT_SHA256,
        "pt_source_revision": PINNED_PT_REVISION,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "from_pretrained_dtype_argument": "torch.bfloat16",
        "attention_implementation_argument": "omitted (validated historical Transformers default/auto path)",
        "resolved_attention_implementation": attention,
        "resolved_encoder_attention_implementation": encoder_attention,
        "resolved_decoder_attention_implementation": decoder_attention,
        "seed": SEED,
        "seed_reset_per_piece": True,
        "seed_reset_semantics": "random.seed; numpy.random.seed; torch.manual_seed; torch.cuda.manual_seed and manual_seed_all if CUDA runtime is available",
        "function": "third_party.PianistTransformer.src.model.generate.batch_performance_render",
        "map_function": "third_party.PianistTransformer.src.model.generate.map_midi",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": int(generation.top_k),
        "num_beams": int(generation.num_beams),
        "repetition_penalty": float(generation.repetition_penalty),
        "max_context_length": MAX_CONTEXT,
        "overlap_ratio": OVERLAP,
        "pitch_hard_constraint": "official BatchSparseForcedTokenProcessor",
        "long_sequence_generation": "official overlapping block generation",
        "map_midi_semantics": "official map_midi on a freshly reloaded selected score",
        "other_generation_arguments": {
            "bos_token_id": int(model.config.bos_token_id),
            "eos_token_id": int(model.config.eos_token_id),
            "pad_token_id": int(model.config.pad_token_id),
            "separate_torch_generator": False,
        },
    }
    config["inference_configuration_identifier"] = _json_identifier(config)
    return config


def _piece_paths(piece_id: str) -> dict[str, Path]:
    root = BANK_ROOT / piece_id
    return {
        "root": root,
        "ids": root / "generated_ids_int64.npy",
        "midi": root / "original_pt.mid",
        "metadata": root / "inference_metadata.json",
    }


def _valid_frozen_piece(row: Mapping[str, Any], inference_id: str) -> dict[str, Any] | None:
    paths = _piece_paths(str(row["piece_id"]))
    if not all(paths[name].is_file() for name in ("ids", "midi", "metadata")):
        if any(paths[name].exists() for name in ("ids", "midi", "metadata")):
            raise RuntimeError(f"incomplete frozen artifact; refusing regeneration: {paths['root']}")
        return None
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    ids = np.load(paths["ids"], allow_pickle=False)
    checks = (
        metadata["source_score_sha256"] == row["selected_score_sha256"],
        int(metadata["seed"]) == SEED,
        metadata["inference_configuration_identifier"] == inference_id,
        metadata["generated_ids_sha256_int64_le"] == _ids_sha256(ids),
        metadata["canonical_midi_sha256"] == sha256_file(paths["midi"]),
        metadata["integrity_status"] == "PASS",
    )
    if not all(checks):
        raise RuntimeError(f"existing frozen artifact failed integrity; refusing regeneration: {paths['root']}")
    return metadata


def _generate_piece(
    model: PianoT5Gemma,
    row: Mapping[str, Any],
    inference: Mapping[str, Any],
    progress_callback: Callable[[float], None] | None,
) -> dict[str, Any]:
    frozen = _valid_frozen_piece(row, str(inference["inference_configuration_identifier"]))
    if frozen is not None:
        print(f"CANONICAL_REUSE piece={row['piece_id']}", flush=True)
        return {**frozen, "frozen_artifact_reused": True}

    paths = _piece_paths(str(row["piece_id"]))
    paths["root"].mkdir(parents=True, exist_ok=False)
    score_path = Path(str(row["selected_score_absolute_path"]))
    _seed_everything(SEED)
    score = MidiFile(str(score_path))
    score_ids = midi_to_ids(model.config, score)
    started = time.perf_counter()
    performances, generated = batch_performance_render(
        model,
        [score],
        max_context_length=MAX_CONTEXT,
        overlap_ratio=OVERLAP,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        device="cpu",
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - started
    if len(performances) != 1 or len(generated) != 1:
        raise RuntimeError("official PT did not return exactly one performance")
    generated_ids = np.asarray(generated[0], dtype=np.int64)
    if generated_ids.size != len(score_ids) or generated_ids.size % TOKENS_PER_NOTE:
        raise RuntimeError("official PT output length differs from selected score")
    score_notes = np.asarray(score_ids, dtype=np.int64).reshape(-1, TOKENS_PER_NOTE)
    generated_notes = generated_ids.reshape(-1, TOKENS_PER_NOTE)
    if not np.array_equal(score_notes[:, 0], generated_notes[:, 0]):
        raise RuntimeError("official pitch hard constraint was not preserved")

    temporary_midi = paths["midi"].with_name(paths["midi"].name + ".tmp")
    mapped = map_midi(MidiFile(str(score_path)), performances[0])
    mapped.dump(str(temporary_midi))
    MidiFile(str(temporary_midi))
    temporary_midi.replace(paths["midi"])
    _atomic_npy(paths["ids"], generated_ids)

    raw_midi = mido.MidiFile(str(paths["midi"]), clip=False)
    note_count = sum(
        1
        for track in raw_midi.tracks
        for message in track
        if not message.is_meta and message.type == "note_on" and message.velocity > 0
    )
    if note_count != len(generated_notes):
        raise RuntimeError(f"canonical MIDI note count mismatch: {note_count} != {len(generated_notes)}")
    metadata = {
        "piece_id": row["piece_id"],
        "composer": row["composer"],
        "title": row["title"],
        "source_score_path": str(score_path),
        "source_score_sha256": row["selected_score_sha256"],
        "score_selection_support": int(row["selected_score_support"]),
        "score_selection_provenance": str(SELECTION_SOURCE),
        "seed": SEED,
        "seed_reset_immediately_before_piece": True,
        "inference_configuration_identifier": inference["inference_configuration_identifier"],
        "inference": dict(inference),
        "generated_ids_path": str(paths["ids"]),
        "generated_ids_sha256_int64_le": _ids_sha256(generated_ids),
        "canonical_midi_path": str(paths["midi"]),
        "canonical_midi_sha256": sha256_file(paths["midi"]),
        "canonical_non_cc64_signature_sha256": signature_sha256(paths["midi"]),
        "score_tokens": len(score_ids),
        "generated_tokens": int(generated_ids.size),
        "note_count": note_count,
        "midi_duration_seconds": raw_midi.length,
        "cc64_event_count": len(cc64_schedule(paths["midi"])),
        "pitch_hard_constraint_verified": True,
        "file_dump_reload_verified": True,
        "integrity_status": "PASS",
        "runtime_seconds": elapsed,
        "frozen_artifact_reused": False,
    }
    _atomic_json(paths["metadata"], metadata)
    return metadata


def _update_manifest(rows: list[dict[str, Any]], piece_id: str, metadata: Mapping[str, Any]) -> None:
    row = next(item for item in rows if item["piece_id"] == piece_id)
    row.update(
        {
            "canonical_midi_sha256": metadata["canonical_midi_sha256"],
            "canonical_non_cc64_signature_sha256": metadata["canonical_non_cc64_signature_sha256"],
            "generated_token_sha256_int64_le": metadata["generated_ids_sha256_int64_le"],
            "note_count": metadata["note_count"],
            "midi_duration_seconds": metadata["midi_duration_seconds"],
            "cc64_event_count": metadata["cc64_event_count"],
            "inference_configuration_identifier": metadata["inference_configuration_identifier"],
            "status": "frozen_pass",
        }
    )
    _atomic_csv(MANIFEST_PATH, rows, MANIFEST_FIELDS)


def _strict_metric(human: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    def empirical(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (16,) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("expected non-empty 16-bin histogram")
        return values / values.sum()

    first = empirical(human) + EPSILON
    second = empirical(candidate) + EPSILON
    p = first / first.sum()
    q = second / second.sum()
    midpoint = (p + q) / 2.0
    terms_p = np.where(p > 0, p * np.log2(p / midpoint), 0.0)
    terms_q = np.where(q > 0, q * np.log2(q / midpoint), 0.0)
    divergence = 0.5 * float(terms_p.sum() + terms_q.sum())
    distance = math.sqrt(max(divergence, 0.0))
    return {
        "js_distance_base2": distance,
        "js_divergence_base2": distance * distance,
        "histogram_intersection_official": float(np.minimum(first, second).sum()),
    }


def _evaluate_baseline(
    validation_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, Any]],
    inference: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    tokenizer_config = PianoT5GemmaConfig()
    human_histogram = np.zeros(16, dtype=np.int64)
    human_loads = 0
    for row in validation_rows:
        path = (ASAP_ROOT / row["performance_path"]).resolve(strict=True)
        try:
            path.relative_to(ASAP_ROOT.resolve())
        except ValueError as exc:
            raise PermissionError(f"validation path escaped ASAP root: {path}") from exc
        ids = midi_to_ids(tokenizer_config, MidiFile(str(path)))
        human_histogram += joint16_histogram_from_ids(ids)
        human_loads += 1
    if human_loads != EXPECTED_HUMANS or int(human_histogram.sum()) != 283928:
        raise RuntimeError("human validation official-tokenizer inventory mismatch")

    candidate_histogram = np.zeros(16, dtype=np.int64)
    candidate_loads = 0
    for row in manifest_rows:
        if row["status"] != "frozen_pass":
            raise RuntimeError("baseline attempted before 19/19 frozen PASS")
        midi_path = Path(str(row["canonical_midi_path"]))
        if sha256_file(midi_path) != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical hash changed before metric: {row['piece_id']}")
        reloaded = MidiFile(str(midi_path))
        ids = midi_to_ids(tokenizer_config, reloaded)
        candidate_histogram += joint16_histogram_from_ids(ids)
        candidate_loads += 1
    if candidate_loads != EXPECTED_PIECES:
        raise RuntimeError("canonical validation MIDI load count is not 19")

    metric = _strict_metric(human_histogram, candidate_histogram)
    human_probability = human_histogram.astype(np.float64) / human_histogram.sum()
    candidate_probability = candidate_histogram.astype(np.float64) / candidate_histogram.sum()
    rows = []
    for joint_id, pattern in enumerate(JOINT_PATTERNS):
        rows.append(
            {
                "joint_id": joint_id,
                "pattern": pattern,
                "human_count": int(human_histogram[joint_id]),
                "human_probability": float(human_probability[joint_id]),
                "canonical_original_pt_count": int(candidate_histogram[joint_id]),
                "canonical_original_pt_probability": float(candidate_probability[joint_id]),
            }
        )
    _atomic_csv(DISTRIBUTION_PATH, rows, tuple(rows[0]))
    steady = float(candidate_probability[0] + candidate_probability[15])
    result = {
        "completed": True,
        "experiment_id": EXPERIMENT_ID,
        "validation_piece_count": EXPECTED_PIECES,
        "human_validation_performance_count": human_loads,
        "canonical_original_pt_midi_count": candidate_loads,
        "human_note_count_after_official_midi_to_ids": int(human_histogram.sum()),
        "canonical_original_pt_note_count_after_file_dump_reload_and_official_midi_to_ids": int(candidate_histogram.sum()),
        "human_histogram": human_histogram.tolist(),
        "canonical_original_pt_histogram": candidate_histogram.tolist(),
        "human_probability": human_probability.tolist(),
        "canonical_original_pt_probability": candidate_probability.tolist(),
        "metrics": metric,
        "diagnostics": {
            "canonical_original_pt_steady_mass_0000_plus_1111": steady,
            "canonical_original_pt_transition_containing_mass": 1.0 - steady,
            "human_steady_mass_0000_plus_1111": float(human_probability[0] + human_probability[15]),
            "human_transition_containing_mass": float(1.0 - human_probability[0] - human_probability[15]),
        },
        "metric_provenance": {
            "official_evaluator": str(OFFICIAL_EVALUATOR),
            "official_evaluator_sha256": provenance["official_evaluator_sha256"],
            "official_tokenizer": str(OFFICIAL_TOKENIZER),
            "official_tokenizer_sha256": provenance["official_tokenizer_sha256"],
            "threshold": 64,
            "pattern_order": "Pedal1 MSB through Pedal4 LSB; 0000..1111",
            "epsilon": EPSILON,
            "js": "SciPy-compatible base-2 Jensen-Shannon distance after its internal normalization",
            "intersection": "official sum(min(empirical_probability + epsilon)); no post-epsilon renormalization",
            "candidate_path": "canonical MIDI file dump/reload then official midi_to_ids",
        },
        "inference": dict(inference),
        "checks": {
            "canonical_generation_19_of_19_pass": True,
            "all_canonical_hashes_verified_before_metric": True,
            "human_midi_loads_validation_only_71": True,
            "canonical_midi_loads_validation_only_19": True,
            "asap_test_midi_access_count_0": True,
            "stage2_training_started_false": True,
        },
    }
    _atomic_json(BASELINE_PATH, result)
    return result


def _report(
    manifest: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    split_inventory: Mapping[str, Any],
    inference: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> str:
    metric = baseline["metrics"]
    diagnostics = baseline["diagnostics"]
    score_lines = [
        f"| {row['piece_id']} | {row['composer']} | {row['title']} | {row['selected_score_path']} | {row['selected_score_support']}/{row['validation_performance_count']} | `{row['selected_score_sha256']}` |"
        for row in manifest
    ]
    hash_lines = [
        f"| {row['piece_id']} | `{row['generated_token_sha256_int64_le']}` | `{row['canonical_midi_sha256']}` | {row['note_count']} | {float(row['midi_duration_seconds']):.6f} |"
        for row in manifest
    ]
    reuse = json.loads((CACHE_ROOT / "cache_statistics.json").read_text(encoding="utf-8"))
    lines = [
        "# Canonical Validation Stage 1 Report",
        "",
        "## 1. Scientific goal",
        "",
        "This new, validation-controlled lineage asks whether a binary Stage 2 that changes sustain CC64 only can improve the official Pianist Transformer pedal-distribution metric while Original PT expressive performance is permanently fixed as Stage 1.",
        "",
        "## 2. Canonical Stage 1 invariant",
        "",
        "All Stage 2 candidates must share these 19 exact Original PT validation performances. Stage 1 neural inference is never rerun after freeze. Stage 2 may transplant CC64 only, and strict non-CC64 equality is a hard pre-metric gate.",
        "",
        "## 3. Split and data provenance",
        "",
        f"- ASAP split: `{SPLIT_PATH}` SHA-256 `{provenance['split_sha256']}`.",
        f"- Pieces: train {split_inventory['piece_counts']['train']}, validation {split_inventory['piece_counts']['validation']}, test {split_inventory['piece_counts']['test']}.",
        f"- Performances: train {split_inventory['performance_counts']['train']}, validation {split_inventory['performance_counts']['validation']}, test {split_inventory['performance_counts']['test']}.",
        "- Future training pool: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances.",
        "",
        "## 4. Validation score selection (19 pieces)",
        "",
        f"Existing source-of-truth manifest: `{SELECTION_SOURCE}` SHA-256 `{provenance['selection_manifest_sha256']}`. Rule: max direct validation-row metadata support; lexical tie-break. No new selection rule was introduced.",
        "",
        "| Piece | Composer | Title | Selected score | Support | Score SHA-256 |",
        "|---|---|---|---|---:|---|",
        *score_lines,
        "",
        "## 5. Exact Original PT CPU inference provenance",
        "",
        f"- Checkpoint: `{inference['checkpoint']}` (`{inference['model_safetensors_sha256']}`).",
        f"- PT revision: `{inference['pt_source_revision']}`.",
        f"- Runtime: Python {inference['python_version']}, torch {inference['torch_version']}, transformers {inference['transformers_version']}.",
        f"- Device/dtype: `{inference['device']}` / `{inference['dtype']}`.",
        f"- Attention: argument `{inference['attention_implementation_argument']}`; resolved model/encoder/decoder `{inference['resolved_attention_implementation']}` / `{inference['resolved_encoder_attention_implementation']}` / `{inference['resolved_decoder_attention_implementation']}`.",
        f"- Sampling: seed {SEED}, reset per piece; do_sample=True; temperature={TEMPERATURE}; top-p={TOP_P}; top-k={inference['top_k']}; num_beams={inference['num_beams']}; repetition_penalty={inference['repetition_penalty']}.",
        f"- Context/overlap: {MAX_CONTEXT} tokens / {OVERLAP}. Official pitch constraint, overlapping generation, and map_midi are unchanged.",
        f"- Inference configuration identifier: `{inference['inference_configuration_identifier']}`.",
        "",
        "## 6. Canonical generation integrity",
        "",
        f"- **19/19 PASS**. Every piece has generated IDs, dumped/reloaded `original_pt.mid`, and inference metadata.",
        "- Score/output token length and pitch hard constraint were checked; every saved MIDI reloaded; no completed piece is regenerated.",
        "",
        "## 7. Canonical hashes",
        "",
        "| Piece | Generated IDs SHA-256 | Canonical MIDI SHA-256 | Notes | Duration (s) |",
        "|---|---|---|---:|---:|",
        *hash_lines,
        "",
        "## 8. Fixed validation Original PT baseline",
        "",
        f"- Official strict base-2 JS distance: **{metric['js_distance_base2']:.15f}** (lower is better).",
        f"- Official Intersection: **{metric['histogram_intersection_official']:.15f}** (higher is better).",
        "- Both human and candidate paths use MIDI load plus official midi_to_ids; candidate MIDI is the frozen file after dump/reload.",
        "",
        "## 9. Distribution diagnostics",
        "",
        f"- Canonical Original PT steady mass P(0000)+P(1111): `{diagnostics['canonical_original_pt_steady_mass_0000_plus_1111']:.15f}`.",
        f"- Canonical Original PT transition-containing mass: `{diagnostics['canonical_original_pt_transition_containing_mass']:.15f}`.",
        f"- Full human/candidate 16-pattern distribution: `{DISTRIBUTION_PATH}`.",
        "",
        "## 10. Existing code/cache reusable for later training",
        "",
        f"- Shared cache `{CACHE_ROOT}`: cache ID `{reuse['cache_id']}`, train 2,062 performances / {reuse['splits']['train']['notes']:,} notes / {reuse['splits']['train']['windows']:,} 512/256 windows.",
        "- Existing threshold-64 target generation, pedal masking, pretrained encoder initialization, independent 4x2 and joint-16 heads, AdamW LR configuration, CC64-only transplant, and strict non-CC64 assertion are path-compatible.",
        "- Strict future validation must use the pinned MIDI-roundtrip evaluator semantics, not the legacy direct-token headline metric. No training cache was rebuilt and no Stage 2 training was started.",
        "",
        "## 11. ASAP test isolation",
        "",
        "ASAP test human MIDI loads, canonical test-bank metric reads, prior test-result model choice, and test evaluation were not performed by this runner. Test remains deferred until validation-only model/checkpoint selection is complete.",
        "",
    ]
    return "\n".join(lines)


def _fix_ownership() -> None:
    owner = ROOT.stat()
    for path in [OUTPUT_ROOT, *OUTPUT_ROOT.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid)
        os.chmod(path, 0o775 if path.is_dir() else 0o664)


def run() -> dict[str, Any]:
    if BASELINE_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("completed canonical validation outputs exist; refusing overwrite")
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BANK_ROOT.mkdir(parents=True, exist_ok=True)
    _status("preflight", started_at_unix=time.time())
    provenance = _assert_source_contracts()
    validation_rows, split_inventory = _split_inventory()
    manifest = _selection_rows(validation_rows)
    _atomic_csv(MANIFEST_PATH, manifest, MANIFEST_FIELDS)
    _status("loading_original_pt_cpu", completed_pieces=0, target_pieces=EXPECTED_PIECES)

    _seed_everything(SEED)
    model = PianoT5Gemma.from_pretrained(
        str(CHECKPOINT),
        torch_dtype=torch.bfloat16,
    ).eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("historical Stage 1 model unexpectedly left CPU")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("historical Stage 1 model unexpectedly changed dtype")
    inference = _runtime_inference(model)
    first_signal_sent = False

    for index, row in enumerate(manifest, start=1):
        piece_id = str(row["piece_id"])
        _status(
            "canonical_generation_running",
            current_piece_index=index,
            current_piece_id=piece_id,
            completed_pieces=index - 1,
            target_pieces=EXPECTED_PIECES,
            inference=inference,
        )

        def progress_callback(progress: float) -> None:
            nonlocal first_signal_sent
            if not first_signal_sent:
                first_signal_sent = True
                _status(
                    "first_inference_active",
                    current_piece_index=index,
                    current_piece_id=piece_id,
                    completed_pieces=index - 1,
                    target_pieces=EXPECTED_PIECES,
                    first_generation_progress=float(progress),
                    inference=inference,
                )
                print(
                    f"FIRST_INFERENCE_ACTIVE piece={piece_id} progress={float(progress):.9f}",
                    flush=True,
                )

        metadata = _generate_piece(model, row, inference, progress_callback)
        _update_manifest(manifest, piece_id, metadata)
        print(
            f"CANONICAL_PASS {index}/{EXPECTED_PIECES} piece={piece_id} "
            f"notes={metadata['note_count']} seconds={float(metadata['runtime_seconds']):.3f} "
            f"reused={metadata['frozen_artifact_reused']}",
            flush=True,
        )
        _status(
            "canonical_generation_running",
            current_piece_index=index,
            current_piece_id=piece_id,
            completed_pieces=index,
            target_pieces=EXPECTED_PIECES,
            inference=inference,
        )

    if len(manifest) != EXPECTED_PIECES or any(row["status"] != "frozen_pass" for row in manifest):
        raise RuntimeError("canonical validation bank is not 19/19 PASS")
    _status("baseline_evaluation_running", completed_pieces=EXPECTED_PIECES, target_pieces=EXPECTED_PIECES)
    baseline = _evaluate_baseline(validation_rows, manifest, inference, provenance)
    REPORT_PATH.write_text(
        _report(manifest, baseline, split_inventory, inference, provenance),
        encoding="utf-8",
    )
    result = {
        "completed": True,
        "canonical_piece_count": EXPECTED_PIECES,
        "metrics": baseline["metrics"],
        "report": str(REPORT_PATH),
        "baseline": str(BASELINE_PATH),
        "distribution": str(DISTRIBUTION_PATH),
        "manifest": str(MANIFEST_PATH),
    }
    _status(
        "completed",
        completed=True,
        completed_pieces=EXPECTED_PIECES,
        target_pieces=EXPECTED_PIECES,
        result=result,
        inference=inference,
    )
    _fix_ownership()
    print("RUNNER_COMPLETED " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run()
    except Exception as exc:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        _status(
            "failed",
            completed=False,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(f"RUNNER_FAILED {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
