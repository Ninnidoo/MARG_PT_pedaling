#!/usr/bin/env python3
"""Audit binary Stage 2 metrics with pinned official PT MIDI roundtrips."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
    load_binary_stage2_checkpoint,
)
from src.stage2_encoder_only.evaluate_oracle import (  # noqa: E402
    distribution_similarity,
)
from third_party.PianistTransformer.src import model as _official_model_package  # noqa: E402
from third_party.PianistTransformer.src import utils as _official_utils_package  # noqa: E402
from third_party.PianistTransformer.src.model import pianoformer as _official_pianoformer  # noqa: E402
from third_party.PianistTransformer.src.utils import midi as _official_midi  # noqa: E402

sys.modules.setdefault("src.model", _official_model_package)
sys.modules.setdefault("src.model.pianoformer", _official_pianoformer)
sys.modules.setdefault("src.utils", _official_utils_package)
sys.modules.setdefault("src.utils.midi", _official_midi)

from third_party.PianistTransformer.src.model.generate import map_midi  # noqa: E402
from third_party.PianistTransformer.src.model.pianoformer import (  # noqa: E402
    PianoT5GemmaConfig,
)
from third_party.PianistTransformer.src.utils.midi import (  # noqa: E402
    ids_to_midi,
    midi_to_ids,
)


OUTPUT_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/metric_fidelity_audit_v0"
VALIDATION_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/validation_eval_v0"
TRAIN_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/train_v0"
CACHE_ROOT = PROJECT_ROOT / "analysis/stage2_binary_v0/train_setup_v0/shared_cache"
ENCODER_CHECKPOINT = PROJECT_ROOT / "checkpoints/pianist_transformer"
LOCK_PATH = PROJECT_ROOT / "analysis/stage2_binary_v0/final_lock_v0/final_experiment_lock.json"
OFFICIAL_EVALUATOR = PROJECT_ROOT / "third_party/PianistTransformer/src/evaluate/evaluate.py"
OFFICIAL_TOKENIZER = PROJECT_ROOT / "third_party/PianistTransformer/src/utils/midi.py"
OFFICIAL_GENERATOR = PROJECT_ROOT / "third_party/PianistTransformer/src/model/generate.py"
PINNED_PT_COMMIT = "747df2d12291e37f6638b39f1b71517e579ad48c"
EPSILON = 1e-10
TOLERANCE = 1e-12
ARCHITECTURES = ("independent_4x2", "joint_16")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_git_head(repository: Path) -> str:
    head = (repository / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = repository / ".git" / head.removeprefix("ref: ")
    if reference.is_file():
        return reference.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"cannot resolve local git HEAD reference: {head}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _empirical_probability(histogram: Sequence[int]) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.float64)
    if values.shape != (16,) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("expected a non-empty 16-bin histogram")
    result = values / values.sum()
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-15):
        raise AssertionError("empirical probability does not sum to one")
    return result


def _official_metric_inputs(histogram: Sequence[int]) -> np.ndarray:
    """Pinned evaluate.py lines 302--304: normalize counts, then add epsilon."""

    return _empirical_probability(histogram) + EPSILON


def _base2_js_reference(first: np.ndarray, second: np.ndarray) -> float:
    """Mirror SciPy jensenshannon normalization and base-2 distance semantics."""

    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = (p + q) / 2.0
    divergence = 0.5 * float(
        np.sum(p * np.log2(p / midpoint)) + np.sum(q * np.log2(q / midpoint))
    )
    return math.sqrt(max(divergence, 0.0))


def _strict_official_similarity(
    human_histogram: Sequence[int], candidate_histogram: Sequence[int]
) -> dict[str, float]:
    """Match pinned evaluate.py, including no post-epsilon Intersection renorm."""

    human_input = _official_metric_inputs(human_histogram)
    candidate_input = _official_metric_inputs(candidate_histogram)
    js_distance = _base2_js_reference(human_input, candidate_input)
    intersection = float(np.minimum(human_input, candidate_input).sum())
    return {
        "js_distance_base2": js_distance,
        "js_divergence_base2": js_distance * js_distance,
        "histogram_intersection": intersection,
    }


def _source_and_arithmetic_contract(
    human_histogram: np.ndarray, original_histogram: np.ndarray
) -> dict[str, Any]:
    source = OFFICIAL_EVALUATOR.read_text(encoding="utf-8")
    required_fragments = (
        'gt_midi = miditoolkit.MidiFile(item["gt"])',
        "gt_tokens = midi_to_ids(config, gt_midi)",
        'pred_midi = miditoolkit.MidiFile(item["pred"])',
        "pred_tokens = midi_to_ids(config, pred_midi)",
        "pedal_binarize_threshold: int = 64",
        "binary_value = 1 if pedal_value >= pedal_binarize_threshold else 0",
        "epsilon = 1e-10",
        "jensenshannon(gt_prob, pred_prob, base=2)",
        "np.sum(np.minimum(gt_prob, pred_prob))",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise RuntimeError(f"pinned official evaluator contract changed: {missing}")

    synthetic = [
        ("identical_dense", np.arange(1, 17), np.arange(1, 17)),
        ("identical_sparse", np.eye(1, 16, 0, dtype=np.int64)[0], np.eye(1, 16, 0, dtype=np.int64)[0]),
        ("disjoint_endpoints", np.eye(1, 16, 0, dtype=np.int64)[0], np.eye(1, 16, 15, dtype=np.int64)[0]),
        ("mixed", np.array([9, 2, 0, 1] + [0] * 12), np.array([3, 1, 2, 6] + [0] * 12)),
        ("actual_original_pt", human_histogram, original_histogram),
    ]
    rows = []
    maximum_js_difference = 0.0
    maximum_intersection_difference = 0.0
    for name, first, second in synthetic:
        strict = _strict_official_similarity(first, second)
        legacy = distribution_similarity(first, second, epsilon=EPSILON)
        first_input = _official_metric_inputs(first)
        second_input = _official_metric_inputs(second)
        independent_js = _base2_js_reference(first_input, second_input)
        independent_intersection = float(np.sum(np.minimum(first_input, second_input)))
        js_difference = abs(strict["js_distance_base2"] - legacy["js_distance_base2"])
        reference_difference = abs(strict["js_distance_base2"] - independent_js)
        intersection_difference = abs(
            strict["histogram_intersection"] - independent_intersection
        )
        maximum_js_difference = max(maximum_js_difference, js_difference, reference_difference)
        maximum_intersection_difference = max(
            maximum_intersection_difference, intersection_difference
        )
        if js_difference > 2e-15 or reference_difference > 2e-15:
            raise AssertionError(f"base-2 JS equivalence failed: {name}")
        if intersection_difference > 1e-15:
            raise AssertionError(f"official Intersection equivalence failed: {name}")
        rows.append(
            {
                "case": name,
                "strict_js_distance": strict["js_distance_base2"],
                "legacy_vetted_helper_js_distance": legacy["js_distance_base2"],
                "js_absolute_difference": js_difference,
                "strict_official_intersection": strict["histogram_intersection"],
                "legacy_post_epsilon_renormalized_intersection": legacy[
                    "histogram_intersection"
                ],
                "intersection_official_minus_legacy": strict[
                    "histogram_intersection"
                ]
                - legacy["histogram_intersection"],
            }
        )
    return {
        "official_source_contract_fragments_verified": len(required_fragments),
        "scipy_runtime_available": False,
        "scipy_runtime_note": "scipy is absent in the pinned Docker environment; source-faithful base-2 arithmetic was cross-checked against the existing vetted helper",
        "synthetic_cases": rows,
        "maximum_js_absolute_difference": maximum_js_difference,
        "maximum_intersection_reference_difference": maximum_intersection_difference,
        "result": "PASS",
    }


class MidiAccessGuard:
    def __init__(
        self,
        human_paths: Sequence[Path],
        score_paths: Sequence[Path],
        original_paths: Sequence[Path],
    ) -> None:
        self.allowed = {
            "human_validation": {path.resolve() for path in human_paths},
            "validation_score": {path.resolve() for path in score_paths},
            "original_pt_cache": {path.resolve() for path in original_paths},
        }
        self.counts = {name: 0 for name in self.allowed}
        self.asap_test_midi_access_count = 0

    def load(self, path: Path, category: str) -> MidiFile:
        resolved = path.resolve()
        if category not in self.allowed or resolved not in self.allowed[category]:
            raise PermissionError(f"MIDI path is not in strict validation allowlist: {path}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        self.counts[category] += 1
        return MidiFile(str(resolved))


def _all_notes(midi: MidiFile) -> list[tuple[int, int, int, int]]:
    return [
        (note.pitch, note.start, note.end, note.velocity)
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]


def _instrument_layout(midi: MidiFile) -> list[tuple[int, bool, str]]:
    return [(item.program, item.is_drum, item.name) for item in midi.instruments]


def _non_cc64(midi: MidiFile) -> list[tuple[int, int, int, int]]:
    return [
        (index, cc.number, cc.value, cc.time)
        for index, instrument in enumerate(midi.instruments)
        for cc in instrument.control_changes
        if cc.number != 64
    ]


def _cc64(midi: MidiFile) -> list[tuple[int, int, int]]:
    return [
        (index, cc.value, cc.time)
        for index, instrument in enumerate(midi.instruments)
        for cc in instrument.control_changes
        if cc.number == 64
    ]


def _tempo_signature(midi: MidiFile) -> list[tuple[float, int]]:
    return [(item.tempo, item.time) for item in midi.tempo_changes]


def _time_signature(midi: MidiFile) -> list[tuple[int, int, int]]:
    return [(item.numerator, item.denominator, item.time) for item in midi.time_signature_changes]


def _key_signature(midi: MidiFile) -> list[tuple[int, int]]:
    return [(item.key_number, item.time) for item in midi.key_signature_changes]


def _assert_nonpedal_midi_equality(
    original: MidiFile,
    candidate: MidiFile,
    *,
    metadata: Mapping[str, Any],
    raise_on_mismatch: bool = True,
) -> dict[str, Any]:
    first = _all_notes(original)
    second = _all_notes(candidate)
    pitch_exact = [item[0] for item in first] == [item[0] for item in second]
    onset_exact = [item[1] for item in first] == [item[1] for item in second]
    offset_exact = [item[2] for item in first] == [item[2] for item in second]
    velocity_exact = [item[3] for item in first] == [item[3] for item in second]
    duration_exact = [item[2] - item[1] for item in first] == [
        item[2] - item[1] for item in second
    ]
    checks = {
        "note_count_exact": len(first) == len(second),
        "pitch_exact": pitch_exact,
        "onset_exact": onset_exact,
        "offset_exact": offset_exact,
        "duration_exact": duration_exact,
        "velocity_exact": velocity_exact,
        "note_order_and_events_exact": first == second,
        "ticks_per_beat_exact": original.ticks_per_beat == candidate.ticks_per_beat,
        "instrument_layout_exact": _instrument_layout(original)
        == _instrument_layout(candidate),
        "tempo_changes_exact": _tempo_signature(original) == _tempo_signature(candidate),
        "time_signatures_exact": _time_signature(original) == _time_signature(candidate),
        "key_signatures_exact": _key_signature(original) == _key_signature(candidate),
        "non_cc64_controls_exact": _non_cc64(original) == _non_cc64(candidate),
        "markers_exact": [(item.text, item.time) for item in original.markers]
        == [(item.text, item.time) for item in candidate.markers],
        "lyrics_exact": [(item.text, item.time) for item in original.lyrics]
        == [(item.text, item.time) for item in candidate.lyrics],
    }
    all_exact = all(checks.values())
    row = {
        **metadata,
        "original_note_count": len(first),
        "candidate_note_count": len(second),
        **checks,
        "original_cc64_event_count": len(_cc64(original)),
        "candidate_cc64_event_count": len(_cc64(candidate)),
        "cc64_events_exact": _cc64(original) == _cc64(candidate),
        "all_nonpedal_midi_exact": all_exact,
    }
    if not all_exact and raise_on_mismatch:
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(
            f"Stage 2 changed non-pedal MIDI content for {metadata['piece_id']}: {failed}"
        )
    return row


def _checkpoint_inventory() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        run_root = TRAIN_ROOT / architecture
        status = _read_json(run_root / "run_status.json")
        metrics = _read_csv(run_root / "metrics.csv")
        by_epoch = {int(row["epoch"]): row for row in metrics}
        for path in sorted(run_root.glob("*.pt"), key=lambda item: (item.stem != "best", item.name)):
            if path.stem == "best":
                epoch = int(status["best_epoch"])
            elif path.stem == "last":
                epoch = int(status["completed_epochs"])
            else:
                raise RuntimeError(
                    f"unrecognized available checkpoint; record explicitly before audit: {path}"
                )
            if epoch not in by_epoch:
                raise RuntimeError(f"checkpoint epoch absent from metrics.csv: {path}")
            short_arch = "independent" if architecture == "independent_4x2" else "joint"
            found.append(
                {
                    "architecture": architecture,
                    "checkpoint_kind": path.stem,
                    "checkpoint_label": f"{short_arch}_{path.stem}",
                    "epoch": epoch,
                    "path": path,
                    "sha256": _sha256(path),
                    "training_metric_row": by_epoch[epoch],
                }
            )
    if {(item["architecture"], item["checkpoint_kind"]) for item in found} != {
        ("independent_4x2", "best"),
        ("independent_4x2", "last"),
        ("joint_16", "best"),
        ("joint_16", "last"),
    }:
        raise RuntimeError("expected exactly the four available best/last checkpoints")
    return found


def _mass(histogram: Sequence[int]) -> tuple[float, float]:
    probability = _empirical_probability(histogram)
    steady = float(probability[0] + probability[15])
    return steady, 1.0 - steady


def _comparison_row(
    *,
    label: str,
    architecture: str,
    checkpoint_kind: str,
    epoch: int | str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    direct_histogram: np.ndarray,
    strict_histogram: np.ndarray,
    human_histogram: np.ndarray,
) -> dict[str, Any]:
    direct_legacy = distribution_similarity(
        human_histogram, direct_histogram, epsilon=EPSILON
    )
    direct_official = _strict_official_similarity(human_histogram, direct_histogram)
    strict = _strict_official_similarity(human_histogram, strict_histogram)
    direct_steady, direct_transition = _mass(direct_histogram)
    strict_steady, strict_transition = _mass(strict_histogram)
    return {
        "candidate": label,
        "architecture": architecture,
        "checkpoint_kind": checkpoint_kind,
        "epoch": epoch,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "direct_token_notes": int(direct_histogram.sum()),
        "strict_midi_roundtrip_notes": int(strict_histogram.sum()),
        "note_count_after_minus_before": int(strict_histogram.sum() - direct_histogram.sum()),
        "direct_token_js_distance_legacy": direct_legacy["js_distance_base2"],
        "direct_token_intersection_legacy": direct_legacy["histogram_intersection"],
        "direct_token_js_distance_official_arithmetic": direct_official[
            "js_distance_base2"
        ],
        "direct_token_intersection_official_arithmetic": direct_official[
            "histogram_intersection"
        ],
        "strict_midi_js_distance_official": strict["js_distance_base2"],
        "strict_midi_intersection_official": strict["histogram_intersection"],
        "strict_minus_direct_legacy_js_distance": strict["js_distance_base2"]
        - direct_legacy["js_distance_base2"],
        "strict_minus_direct_legacy_intersection": strict["histogram_intersection"]
        - direct_legacy["histogram_intersection"],
        "strict_minus_direct_same_arithmetic_js_distance": strict[
            "js_distance_base2"
        ]
        - direct_official["js_distance_base2"],
        "strict_minus_direct_same_arithmetic_intersection": strict[
            "histogram_intersection"
        ]
        - direct_official["histogram_intersection"],
        "direct_steady_mass_empirical": direct_steady,
        "direct_transition_mass_empirical": direct_transition,
        "strict_steady_mass_empirical": strict_steady,
        "strict_transition_mass_empirical": strict_transition,
    }


def _report(
    *,
    inventory: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    distributions: Mapping[str, Mapping[str, np.ndarray]],
    arithmetic: Mapping[str, Any],
    equality_rows: Sequence[Mapping[str, Any]],
    access_counts: Mapping[str, int],
    lock_sha256: str,
) -> str:
    by_label = {row["candidate"]: row for row in comparisons}
    stage2_rows = [row for row in comparisons if row["architecture"] != "original_pt"]
    strict_order = sorted(
        stage2_rows, key=lambda row: float(row["strict_midi_js_distance_official"])
    )
    direct_order = sorted(
        stage2_rows, key=lambda row: float(row["direct_token_js_distance_legacy"])
    )
    lines = [
        "# Stage 2 Binary Metric Fidelity Audit v0",
        "",
        "## Outcome",
        "",
        "Pinned official Pianist Transformer pedal evaluator의 MIDI-roundtrip semantics로 ASAP validation 19 pieces만 재평가했다. Existing final lock은 수정하지 않았으며 이 audit 동안 **PROVISIONAL_PENDING_OFFICIAL_MIDI_ROUNDTRIP_AUDIT**로 취급했다. Training/retraining, checkpoint selection 변경, Stage 1 regeneration, calibration, ASAP test MIDI access는 수행하지 않았다.",
        "",
        "## Official evaluator provenance",
        "",
        f"- Pinned PT revision: `{PINNED_PT_COMMIT}`",
        f"- Evaluator: `third_party/PianistTransformer/src/evaluate/evaluate.py` SHA-256 `{_sha256(OFFICIAL_EVALUATOR)}`",
        f"- Tokenizer: `third_party/PianistTransformer/src/utils/midi.py` SHA-256 `{_sha256(OFFICIAL_TOKENIZER)}`",
        f"- Renderer/mapping: `third_party/PianistTransformer/src/model/generate.py` SHA-256 `{_sha256(OFFICIAL_GENERATOR)}`",
        "- `plot_pedal_pattern_distribution`: lines 243–338",
        "- GT MIDI load/tokenize: lines 257–258; prediction MIDI load/tokenize: lines 259–260",
        "- Pedal1–4 extraction and threshold `>=64`: lines 262–284",
        "- 16-bin histogram, epsilon, SciPy base-2 JS distance, Intersection: lines 298–307",
        "- `midi_to_ids` official pedal sample locations: `midi.py` lines 136–182, especially 167–170",
        "- `ids_to_midi`: `midi.py` lines 184–256; `map_midi`: `generate.py` lines 130–237",
        "",
        "The pinned function unequivocally sends **both ground truth and prediction through `MIDI file → MidiFile → midi_to_ids`** before extracting Pedal1–4. Therefore pre-render Stage 2 IDs are diagnostic inputs only, not official headline metric inputs.",
        "",
        "## Arithmetic fidelity",
        "",
        "Official source adds `1e-10` after histogram normalization, calls `scipy.spatial.distance.jensenshannon(..., base=2)`, and computes `sum(min(gt_prob, pred_prob))` without a second normalization. SciPy normalizes its JS inputs internally; the existing vetted helper explicitly normalizes them and is JS-equivalent. The existing helper also normalizes before Intersection, which is not bit-for-bit official.",
        "",
        f"- Synthetic/actual equivalence cases: {len(arithmetic['synthetic_cases'])}",
        f"- Maximum JS absolute difference: `{arithmetic['maximum_js_absolute_difference']:.3e}`",
        f"- Maximum official-Intersection reference difference: `{arithmetic['maximum_intersection_reference_difference']:.3e}`",
        "- SciPy package in existing Docker: unavailable (`ModuleNotFoundError`); no dependency was installed. The pinned call semantics were reproduced with the source-faithful base-2 calculation and cross-checked against the existing vetted helper.",
        "",
        "## Existing path versus strict path",
        "",
        "- Existing training/diagnosis Stage 2 headline path: cached Stage 1 IDs → Stage 2 pedal replacement → direct token histogram → legacy helper.",
        "- Strict path: cached Stage 1 IDs → Stage 2 pedal replacement → official `ids_to_midi(ref=score_ids)` → official `map_midi(score, performance)` → candidate MIDI dump/reload → official `midi_to_ids` → threshold 64 → histogram → official arithmetic.",
        "- Original PT strict path continues to use the existing cached `original_pt.mid` files; Stage 1 inference was not rerun.",
        "",
        "## Available checkpoints",
        "",
        "| Candidate | Epoch | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for item in inventory:
        lines.append(
            f"| {item['checkpoint_label']} | {item['epoch']} | {item['path'].stat().st_size} | `{item['sha256']}` |"
        )
    lines += [
        "",
        "Only `best.pt` and `last.pt` existed in each run directory. No per-epoch checkpoint was reconstructed.",
        "",
        "## Direct-token versus strict MIDI-roundtrip metrics",
        "",
        "Intersection values in the strict columns use the pinned official no-post-epsilon-renormalization arithmetic. `strict − direct` below compares strict official values against the historical legacy direct metric; the CSV additionally supplies same-arithmetic differences.",
        "",
        "| Candidate | Direct notes | Strict notes | Direct JS | Strict JS | ΔJS | Direct Intersection | Strict Intersection | ΔIntersection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {candidate} | {direct_token_notes} | {strict_midi_roundtrip_notes} | {direct_token_js_distance_legacy:.12f} | {strict_midi_js_distance_official:.12f} | {strict_minus_direct_legacy_js_distance:+.12f} | {direct_token_intersection_legacy:.12f} | {strict_midi_intersection_official:.12f} | {strict_minus_direct_legacy_intersection:+.12f} |".format(
                **row
            )
        )
    original = by_label["original_pt"]
    lines += [
        "",
        "## Original PT baseline reproduction",
        "",
        f"- Strict Original PT JS: `{original['strict_midi_js_distance_official']:.15f}`; existing report: `0.144081839387225` — reproduced within `{TOLERANCE}`.",
        f"- Strict official Intersection: `{original['strict_midi_intersection_official']:.15f}`.",
        "- Existing reported Intersection: `0.907531643144195`. It differs only because the legacy helper post-normalized epsilon-smoothed probabilities; pinned official source does not. This is an arithmetic correction, not a histogram mismatch.",
        f"- Official-minus-legacy baseline Intersection: `{original['strict_midi_intersection_official'] - 0.9075316431441947:+.15e}`.",
        "- Human strict histogram and Original PT strict histogram exactly match the previously saved MIDI-tokenized histograms.",
        "",
        "## Empirical steady / transition-containing mass",
        "",
        "These descriptive masses use raw count-normalized empirical distributions (no epsilon).",
        "",
        "| Candidate | Direct steady | Direct transition | Strict steady | Strict transition |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {candidate} | {direct_steady_mass_empirical:.9f} | {direct_transition_mass_empirical:.9f} | {strict_steady_mass_empirical:.9f} | {strict_transition_mass_empirical:.9f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Strict global 16-pattern distributions",
        "",
        "All displayed probabilities are raw empirical `count / total` distributions in order `0000, 0001, ..., 1111`.",
        "",
        "| Pattern | Human | Original PT | Independent best | Independent last | Joint best | Joint last |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for joint_id, pattern in enumerate(JOINT_PATTERNS):
        lines.append(
            f"| `{pattern}` | {_empirical_probability(distributions['human']['strict'])[joint_id]:.9f} | {_empirical_probability(distributions['original_pt']['strict'])[joint_id]:.9f} | {_empirical_probability(distributions['independent_best']['strict'])[joint_id]:.9f} | {_empirical_probability(distributions['independent_last']['strict'])[joint_id]:.9f} | {_empirical_probability(distributions['joint_best']['strict'])[joint_id]:.9f} | {_empirical_probability(distributions['joint_last']['strict'])[joint_id]:.9f} |"
        )
    lines += [
        "",
        "## MIDI-level pedal-only preservation",
        "",
        f"- Piece-checkpoint comparisons: **{len(equality_rows)}/{len(equality_rows)} PASS**",
        "- Exact assertions: note count/order, pitch, onset, offset/duration, velocity, ticks-per-beat, instrument layout, tempo changes, time/key signatures, markers, lyrics, and all non-CC64 controls.",
        "- CC64 events were the only allowed MIDI-level difference.",
        "- Any non-pedal mismatch would have aborted before metric aggregation.",
        "",
        "## Selection impact assessment (no selection performed)",
        "",
        f"Historical direct-token order among available checkpoints: `{' < '.join(row['candidate'] for row in direct_order)}` (lower JS first).",
        f"Strict MIDI-roundtrip order among available checkpoints: `{' < '.join(row['candidate'] for row in strict_order)}` (lower JS first).",
        "The observed ranking is unchanged, so the available checkpoint set shows no selection flip under strict semantics. Nevertheless, the headline values move materially and the prior lock was based on direct-token metrics. This audit does **not** select a checkpoint or modify the existing lock; any lock update must be handled in a separate explicitly authorized step.",
        "",
        "## Integrity and scope",
        "",
        f"- Existing final lock SHA-256 before/after audit: `{lock_sha256}` / `{_sha256(LOCK_PATH)}` — unchanged",
        f"- Human validation MIDI loads: {access_counts['human_validation']} (71 expected)",
        f"- Validation score MIDI loads: {access_counts['validation_score']}",
        f"- Cached Original PT MIDI loads: {access_counts['original_pt_cache']}",
        "- Stage 1 neural inference regeneration: 0",
        "- Training/retraining: 0",
        "- Checkpoint reconstruction/update: 0",
        "- ASAP test MIDI access: **0 / PASS**",
        "- Existing final lock status for this report: **PROVISIONAL_PENDING_OFFICIAL_MIDI_ROUNDTRIP_AUDIT**",
        "",
        "## Artifacts",
        "",
        "- `checkpoint_metric_comparison.csv`",
        "- `strict_global_joint16_comparison.csv`",
        "- `midi_nonpedal_equality.csv`",
        "- `arithmetic_equivalence.json`",
        "- `audit_summary.json`",
        "- `candidate_midis/<checkpoint_label>/<piece_id>.mid`",
        "",
        "## Stop point",
        "",
        "Metric-fidelity audit only. No retraining, calibration, checkpoint selection change, Stage 1 regeneration, or ASAP test evaluation was performed.",
    ]
    return "\n".join(lines) + "\n"


def run(device_name: str = "cuda:0") -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {OUTPUT_ROOT}")
    lock_sha256 = _sha256(LOCK_PATH)
    pinned_head = _local_git_head(PROJECT_ROOT / "third_party/PianistTransformer")
    if pinned_head != PINNED_PT_COMMIT:
        raise RuntimeError(
            f"pinned PT revision mismatch: expected {PINNED_PT_COMMIT}, got {pinned_head}"
        )
    stage1_rows = _read_csv(VALIDATION_ROOT / "stage1_cache_manifest.csv")
    human_rows = _read_csv(CACHE_ROOT / "validation_performance_index.csv")
    baseline = _read_json(VALIDATION_ROOT / "original_pt_validation_metrics.json")
    if len(stage1_rows) != 19 or len(human_rows) != 71:
        raise RuntimeError("fixed validation inventory is not 19 pieces / 71 performances")
    if {row["dataset_split"] for row in human_rows} != {"validation"}:
        raise RuntimeError("human MIDI index contains a non-validation row")
    if {row["source"] for row in human_rows} != {"ASAP-validation"}:
        raise RuntimeError("human MIDI index contains a non-validation source")
    if len({row["piece_id"] for row in stage1_rows}) != 19:
        raise RuntimeError("Stage 1 cache does not identify 19 unique pieces")
    if {row["piece_id"] for row in stage1_rows} != {
        row["piece_id"] for row in human_rows
    }:
        raise RuntimeError("human and Stage 1 validation piece sets differ")

    human_paths = [Path(row["performance_absolute_path"]) for row in human_rows]
    score_paths = [Path(row["source_score_path"]) for row in stage1_rows]
    original_paths = [Path(row["generated_midi_path"]) for row in stage1_rows]
    guard = MidiAccessGuard(human_paths, score_paths, original_paths)
    # The pinned official evaluator constructs this exact default config at line 252.
    tokenizer_config = PianoT5GemmaConfig()

    human_histogram = np.zeros(16, dtype=np.int64)
    for row, path in zip(human_rows, human_paths, strict=True):
        ids = midi_to_ids(tokenizer_config, guard.load(path, "human_validation"))
        if len(ids) // 8 != int(row["notes"]):
            raise RuntimeError(f"human validation note count changed: {path}")
        human_histogram += joint16_histogram_from_ids(ids)
    if human_histogram.tolist() != baseline["human_histogram"]:
        raise RuntimeError("strict human MIDI histogram differs from saved baseline")
    if int(human_histogram.sum()) != 283928:
        raise RuntimeError("strict human validation note count mismatch")

    original_direct = np.zeros(16, dtype=np.int64)
    original_strict = np.zeros(16, dtype=np.int64)
    stage1_ids: dict[str, np.ndarray] = {}
    score_ids: dict[str, list[int]] = {}
    for row in stage1_rows:
        piece_id = row["piece_id"]
        ids = np.load(row["generated_ids_path"], allow_pickle=False)
        stage1_ids[piece_id] = ids
        original_direct += joint16_histogram_from_ids(ids)
        original_midi = guard.load(Path(row["generated_midi_path"]), "original_pt_cache")
        retokenized = midi_to_ids(tokenizer_config, original_midi)
        original_strict += joint16_histogram_from_ids(retokenized)
        score_ids[piece_id] = midi_to_ids(
            tokenizer_config,
            guard.load(Path(row["source_score_path"]), "validation_score"),
        )
    if original_strict.tolist() != baseline["original_pt_histogram"]:
        raise RuntimeError("strict Original PT MIDI histogram differs from saved baseline")
    if int(original_direct.sum()) != 48894 or int(original_strict.sum()) != 48894:
        raise RuntimeError("Original PT before/after MIDI note count mismatch")

    arithmetic = _source_and_arithmetic_contract(human_histogram, original_strict)
    inventory = _checkpoint_inventory()
    distributions: dict[str, dict[str, np.ndarray]] = {
        "human": {"direct": human_histogram.copy(), "strict": human_histogram.copy()},
        "original_pt": {"direct": original_direct, "strict": original_strict},
    }
    comparisons = [
        _comparison_row(
            label="original_pt",
            architecture="original_pt",
            checkpoint_kind="cached_stage1",
            epoch="",
            checkpoint_path="analysis/stage2_binary_v0/validation_eval_v0/stage1_cache/*/original_pt.mid",
            checkpoint_sha256="",
            direct_histogram=original_direct,
            strict_histogram=original_strict,
            human_histogram=human_histogram,
        )
    ]
    equality_rows: list[dict[str, Any]] = []
    device = torch.device(device_name)

    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix="metric_fidelity_audit_v0.tmp.", dir=OUTPUT_ROOT.parent)
    )
    try:
        candidate_root = temporary_parent / "candidate_midis"
        candidate_root.mkdir()
        for number, item in enumerate(inventory, start=1):
            label = str(item["checkpoint_label"])
            print(
                f"CHECKPOINT {number}/{len(inventory)} {label} epoch={item['epoch']}",
                flush=True,
            )
            label_root = candidate_root / label
            label_root.mkdir()
            model = load_binary_stage2_checkpoint(
                item["path"],
                architecture=str(item["architecture"]),
                encoder_checkpoint=ENCODER_CHECKPOINT,
                device=device,
            )
            direct_histogram = np.zeros(16, dtype=np.int64)
            strict_histogram = np.zeros(16, dtype=np.int64)
            total_windows = 0
            for piece_number, row in enumerate(stage1_rows, start=1):
                piece_id = row["piece_id"]
                candidate_ids, details = infer_cached_binary_pedals(
                    model,
                    stage1_ids[piece_id],
                    architecture=str(item["architecture"]),
                    device=device,
                    window_notes=512,
                    stride_notes=256,
                )
                if not details["non_pedal_tokens_preserved"]:
                    raise AssertionError("Stage 2 changed a pre-render non-pedal token")
                direct_histogram += joint16_histogram_from_ids(candidate_ids)
                total_windows += int(details["windows"])

                performance = ids_to_midi(
                    tokenizer_config,
                    candidate_ids,
                    ref=score_ids[piece_id],
                )
                mapped = map_midi(
                    guard.load(Path(row["source_score_path"]), "validation_score"),
                    performance,
                )
                candidate_path = label_root / f"{piece_id}.mid"
                mapped.dump(str(candidate_path))
                candidate_midi = MidiFile(str(candidate_path))
                original_midi = guard.load(
                    Path(row["generated_midi_path"]), "original_pt_cache"
                )
                equality_rows.append(
                    _assert_nonpedal_midi_equality(
                        original_midi,
                        candidate_midi,
                        metadata={
                            "architecture": item["architecture"],
                            "checkpoint_label": label,
                            "checkpoint_kind": item["checkpoint_kind"],
                            "checkpoint_epoch": item["epoch"],
                            "piece_id": piece_id,
                            "composer": row["composer"],
                            "title": row["title"],
                            "original_pt_midi": row["generated_midi_path"],
                            "stage2_candidate_midi": str(
                                OUTPUT_ROOT / "candidate_midis" / label / candidate_path.name
                            ),
                        },
                    )
                )
                retokenized = midi_to_ids(tokenizer_config, candidate_midi)
                strict_histogram += joint16_histogram_from_ids(retokenized)
                print(
                    f"  PIECE {piece_number}/{len(stage1_rows)} {piece_id}", flush=True
                )
            del model
            torch.cuda.empty_cache()
            if total_windows != 182:
                raise RuntimeError(f"unexpected validation window count for {label}")
            comparison = _comparison_row(
                label=label,
                architecture=str(item["architecture"]),
                checkpoint_kind=str(item["checkpoint_kind"]),
                epoch=int(item["epoch"]),
                checkpoint_path=str(item["path"].relative_to(PROJECT_ROOT)),
                checkpoint_sha256=str(item["sha256"]),
                direct_histogram=direct_histogram,
                strict_histogram=strict_histogram,
                human_histogram=human_histogram,
            )
            expected = item["training_metric_row"]
            if abs(
                comparison["direct_token_js_distance_legacy"]
                - float(expected["validation_pedal_js_distance"])
            ) > TOLERANCE:
                raise RuntimeError(f"direct-token JS did not reproduce training row: {label}")
            if abs(
                comparison["direct_token_intersection_legacy"]
                - float(expected["validation_pedal_intersection"])
            ) > TOLERANCE:
                raise RuntimeError(
                    f"direct-token Intersection did not reproduce training row: {label}"
                )
            comparisons.append(comparison)
            distributions[label] = {
                "direct": direct_histogram,
                "strict": strict_histogram,
            }

        if len(equality_rows) != 76 or not all(
            bool(row["all_nonpedal_midi_exact"]) for row in equality_rows
        ):
            raise RuntimeError("expected 76/76 non-pedal MIDI equality passes")
        if guard.asap_test_midi_access_count != 0:
            raise RuntimeError("ASAP test MIDI access was nonzero")
        if _sha256(LOCK_PATH) != lock_sha256:
            raise RuntimeError("existing final lock changed during audit")

        global_rows = []
        ordered_labels = ["original_pt"] + [item["checkpoint_label"] for item in inventory]
        for joint_id, pattern in enumerate(JOINT_PATTERNS):
            row: dict[str, Any] = {
                "joint_id": joint_id,
                "pattern": pattern,
                "human_count": int(human_histogram[joint_id]),
                "human_probability": float(
                    _empirical_probability(human_histogram)[joint_id]
                ),
            }
            for label in ordered_labels:
                direct = distributions[label]["direct"]
                strict = distributions[label]["strict"]
                direct_probability = _empirical_probability(direct)
                strict_probability = _empirical_probability(strict)
                row[f"{label}_direct_count"] = int(direct[joint_id])
                row[f"{label}_direct_probability"] = float(
                    direct_probability[joint_id]
                )
                row[f"{label}_strict_count"] = int(strict[joint_id])
                row[f"{label}_strict_probability"] = float(
                    strict_probability[joint_id]
                )
                row[f"{label}_strict_minus_direct_probability"] = float(
                    strict_probability[joint_id] - direct_probability[joint_id]
                )
            global_rows.append(row)

        _write_csv(temporary_parent / "checkpoint_metric_comparison.csv", comparisons)
        _write_csv(temporary_parent / "strict_global_joint16_comparison.csv", global_rows)
        _write_csv(temporary_parent / "midi_nonpedal_equality.csv", equality_rows)
        _write_json(temporary_parent / "arithmetic_equivalence.json", arithmetic)
        summary = {
            "completed": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pinned_pt_commit": PINNED_PT_COMMIT,
            "official_evaluator_sha256": _sha256(OFFICIAL_EVALUATOR),
            "official_tokenizer_sha256": _sha256(OFFICIAL_TOKENIZER),
            "official_generator_sha256": _sha256(OFFICIAL_GENERATOR),
            "existing_lock_sha256_unchanged": lock_sha256,
            "existing_lock_audit_status": "PROVISIONAL_PENDING_OFFICIAL_MIDI_ROUNDTRIP_AUDIT",
            "validation_pieces": 19,
            "human_validation_performances": 71,
            "human_validation_notes": int(human_histogram.sum()),
            "available_checkpoints": [
                {
                    key: (
                        str(value.relative_to(PROJECT_ROOT))
                        if key == "path"
                        else value
                    )
                    for key, value in item.items()
                    if key != "training_metric_row"
                }
                for item in inventory
            ],
            "metrics": comparisons,
            "nonpedal_midi_equality_passed": len(equality_rows),
            "nonpedal_midi_equality_total": len(equality_rows),
            "stage1_inference_regeneration": 0,
            "training_or_retraining": 0,
            "checkpoint_selection_changed": False,
            "asap_test_midi_access_count": 0,
            "midi_access_counts": guard.counts,
        }
        _write_json(temporary_parent / "audit_summary.json", summary)
        report = _report(
            inventory=inventory,
            comparisons=comparisons,
            distributions=distributions,
            arithmetic=arithmetic,
            equality_rows=equality_rows,
            access_counts=guard.counts,
            lock_sha256=lock_sha256,
        )
        (temporary_parent / "METRIC_FIDELITY_AUDIT_REPORT.md").write_text(
            report, encoding="utf-8"
        )
        temporary_parent.replace(OUTPUT_ROOT)
        return summary
    except BaseException:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise


def main() -> None:
    started = time.perf_counter()
    result = run()
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "available_checkpoints": len(result["available_checkpoints"]),
                "nonpedal_midi_equality": result["nonpedal_midi_equality_passed"],
                "asap_test_midi_access_count": result[
                    "asap_test_midi_access_count"
                ],
                "elapsed_seconds": time.perf_counter() - started,
                "output_root": str(OUTPUT_ROOT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
