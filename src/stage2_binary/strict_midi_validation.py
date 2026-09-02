"""Official PT MIDI-roundtrip validation for binary Stage 2 models."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile

from scripts.audit_stage2_binary_metric_fidelity import (
    _assert_nonpedal_midi_equality,
    _strict_official_similarity,
    ids_to_midi,
    map_midi,
    midi_to_ids,
    PianoT5GemmaConfig,
)
from src.stage2_encoder_only.evaluate_oracle import distribution_similarity

from .validation_evaluator import (
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
)


STRICT_EVALUATOR_PROVENANCE = {
    "official_function": "third_party/PianistTransformer/src/evaluate/evaluate.py::plot_pedal_pattern_distribution",
    "official_evaluator_lines": "243-338",
    "official_evaluator_sha256": "42a6066496c570b974c3288eb3e8ac3b83e91c00add94da2a19b59c575a5e02a",
    "official_tokenizer": "third_party/PianistTransformer/src/utils/midi.py::{midi_to_ids,ids_to_midi}",
    "official_tokenizer_sha256": "2fb37eaca3d6e4f4a775eb8a59379e7e09651fda36c8146cf0067cde8ad78633",
    "official_mapping": "third_party/PianistTransformer/src/model/generate.py::map_midi",
    "official_mapping_sha256": "f4c0de409938cb0576a9076be61ed9bc9a27af140ff1dbb673157d36b4720472",
    "pinned_pt_commit": "747df2d12291e37f6638b39f1b71517e579ad48c",
    "epsilon": 1e-10,
    "threshold": 64,
    "js": "SciPy jensenshannon(..., base=2) numerical-equivalent helper",
    "intersection": "sum(min(p,q)) after epsilon without post-renormalization",
}


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19 or len({row["piece_id"] for row in rows}) != 19:
        raise RuntimeError("strict validation requires the fixed 19-piece Stage 1 cache")
    for row in rows:
        if "validation_eval_v0/stage1_cache" not in row["generated_ids_path"]:
            raise PermissionError("unexpected Stage 1 ID path")
        if "validation_eval_v0/stage1_cache" not in row["generated_midi_path"]:
            raise PermissionError("unexpected Original PT MIDI path")
        if "/workspace/public/ASAP/asap-dataset-v1.1/" not in row[
            "source_score_path"
        ]:
            raise PermissionError("unexpected validation score path")
    return rows


def evaluate_strict_midi_roundtrip(
    model: torch.nn.Module,
    *,
    architecture: str,
    stage1_manifest_csv: str | Path,
    human_histogram: Sequence[int],
    device: torch.device,
    temporary_root: str | Path,
) -> dict[str, Any]:
    """Evaluate all 19 cached pieces through official render/dump/re-tokenize."""

    rows = _read_manifest(stage1_manifest_csv)
    human = np.asarray(human_histogram, dtype=np.int64)
    if human.shape != (16,) or int(human.sum()) != 283928:
        raise RuntimeError("strict validation human histogram provenance mismatch")
    root = Path(temporary_root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_config = PianoT5GemmaConfig()
    direct_histogram = np.zeros(16, dtype=np.int64)
    strict_histogram = np.zeros(16, dtype=np.int64)
    windows = 0
    equality_rows: list[dict[str, Any]] = []
    score_midi_access_count = 0
    original_pt_midi_access_count = 0
    with tempfile.TemporaryDirectory(prefix="strict_validation_", dir=root) as temporary:
        midi_root = Path(temporary)
        for row in rows:
            piece_id = row["piece_id"]
            cached_ids = np.load(row["generated_ids_path"], allow_pickle=False)
            candidate_ids, details = infer_cached_binary_pedals(
                model,
                cached_ids,
                architecture=architecture,
                device=device,
                window_notes=512,
                stride_notes=256,
            )
            if not details["non_pedal_tokens_preserved"]:
                raise AssertionError("Stage 2 changed a pre-render non-pedal token")
            direct_histogram += joint16_histogram_from_ids(candidate_ids)
            windows += int(details["windows"])

            score_path = Path(row["source_score_path"])
            score_for_ids = MidiFile(str(score_path))
            score_midi_access_count += 1
            score_ids = midi_to_ids(tokenizer_config, score_for_ids)
            performance = ids_to_midi(
                tokenizer_config,
                candidate_ids,
                ref=score_ids,
            )
            score_for_mapping = MidiFile(str(score_path))
            score_midi_access_count += 1
            mapped = map_midi(score_for_mapping, performance)
            candidate_path = midi_root / f"{piece_id}.mid"
            mapped.dump(str(candidate_path))
            candidate_midi = MidiFile(str(candidate_path))

            original_path = Path(row["generated_midi_path"])
            original_midi = MidiFile(str(original_path))
            original_pt_midi_access_count += 1
            equality_rows.append(
                _assert_nonpedal_midi_equality(
                    original_midi,
                    candidate_midi,
                    metadata={
                        "piece_id": piece_id,
                        "architecture": architecture,
                    },
                )
            )
            strict_ids = midi_to_ids(tokenizer_config, candidate_midi)
            strict_histogram += joint16_histogram_from_ids(strict_ids)

    if windows != 182:
        raise RuntimeError(f"strict validation window count mismatch: {windows}")
    if int(direct_histogram.sum()) != 48894:
        raise RuntimeError("direct-token validation note count mismatch")
    if int(strict_histogram.sum()) != 48894:
        raise RuntimeError("strict roundtrip validation note count mismatch")
    if len(equality_rows) != 19 or not all(
        row["all_nonpedal_midi_exact"] for row in equality_rows
    ):
        raise AssertionError("strict validation non-pedal MIDI equality failed")
    strict = _strict_official_similarity(human, strict_histogram)
    direct = distribution_similarity(human, direct_histogram, epsilon=1e-10)
    return {
        "pieces": 19,
        "windows": windows,
        "direct_notes": int(direct_histogram.sum()),
        "strict_notes": int(strict_histogram.sum()),
        "direct_candidate_histogram": direct_histogram.tolist(),
        "strict_candidate_histogram": strict_histogram.tolist(),
        "direct_js_distance": float(direct["js_distance_base2"]),
        "direct_intersection": float(direct["histogram_intersection"]),
        "strict_js_distance": float(strict["js_distance_base2"]),
        "strict_js_divergence": float(strict["js_divergence_base2"]),
        "strict_intersection": float(strict["histogram_intersection"]),
        "nonpedal_midi_exact": True,
        "nonpedal_midi_equality_count": len(equality_rows),
        "score_midi_access_count": score_midi_access_count,
        "original_pt_midi_access_count": original_pt_midi_access_count,
        "stage1_inference_regeneration": 0,
        "asap_test_midi_access_count": 0,
    }


def strict_original_pt_reference(
    *, stage1_manifest_csv: str | Path, human_histogram: Sequence[int]
) -> dict[str, Any]:
    """Reproduce the cached Original PT strict validation baseline."""

    rows = _read_manifest(stage1_manifest_csv)
    tokenizer_config = PianoT5GemmaConfig()
    candidate = np.zeros(16, dtype=np.int64)
    for row in rows:
        ids = midi_to_ids(
            tokenizer_config,
            MidiFile(str(Path(row["generated_midi_path"]))),
        )
        candidate += joint16_histogram_from_ids(ids)
    human = np.asarray(human_histogram, dtype=np.int64)
    strict = _strict_official_similarity(human, candidate)
    return {
        "pieces": 19,
        "notes": int(candidate.sum()),
        "candidate_histogram": candidate.tolist(),
        "strict_js_distance": float(strict["js_distance_base2"]),
        "strict_intersection": float(strict["histogram_intersection"]),
        "stage1_inference_regeneration": 0,
        "asap_test_midi_access_count": 0,
    }
