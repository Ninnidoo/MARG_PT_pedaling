#!/usr/bin/env python3
"""CPU-only early-style pedal metrics for saved Original PT test MIDI.

This script performs no neural inference and loads no checkpoint.  It projects
both a saved Original PT performance and each ASAP human performance onto the
same canonical score-note grid with the exact Pianist Transformer alignment
pipeline (Nakamura MIDI-to-MIDI correspondence plus PT interpolation), then
reuses the existing raw-128 and endpoint-aware five-class metric functions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from miditoolkit import ControlChange, Instrument, MidiFile, Note

from src.stage2_encoder_only.evaluate_five_class import (
    CLASS_NAMES,
    classification_metrics,
    decoded_value_metrics,
    renderer_aware_metrics,
)
from src.stage2_encoder_only.evaluate_oracle import (
    aggregate_distribution,
    aggregate_intermediate,
    aggregate_standard,
    aggregate_transition,
    distribution_metrics,
    intermediate_metrics,
    macro_intermediate,
    macro_standard,
    macro_transition,
    standard_metrics,
    transition_metrics,
)
from src.stage2_encoder_only.five_class import (
    REPRESENTATIVES,
    classify_pedal_values,
)
from third_party.PianistTransformer.src.utils.midi import (
    interpolate,
    midi_to_ids,
    normalize_midi,
    read_corresp,
    segment_sequences,
)


TOKENS_PER_NOTE = 8
PEDAL_START = 5261
CLASS_BOUNDS = ((0, 0), (1, 63), (64, 95), (96, 126), (127, 127))
EXPECTED_TEST_PERFORMANCES = 104
EXPECTED_TEST_PIECES = 23
EXPECTED_ORIGINAL_PT_MIDIS = 23
ALIGNMENT_ZIP_SHA256 = "cf75af54435c6ad83a7b578b691724866f6df5701e8529d86ae2a3db1e85bddd"
ALIGNMENT_VERSION = "AlignmentTool_v240109"
ALIGNMENT_URL = "https://midialignment.github.io/AlignmentTool_v240109.zip"


class PinnedTokenizerConfig:
    pitch_start = 5
    velocity_start = 133
    timing_start = 261
    pedal_start = PEDAL_START
    valid_id_range = [
        (5, 133),
        (261, 5252),
        (133, 261),
        (261, 5261),
        *((PEDAL_START, PEDAL_START + 128),) * 4,
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def matrix_from_ids(ids: Sequence[int]) -> np.ndarray:
    values = np.asarray(ids, dtype=np.int64)
    if values.size == 0 or values.size % TOKENS_PER_NOTE:
        raise ValueError("token sequence is not a non-empty exact [N,8] sequence")
    return values.reshape(-1, TOKENS_PER_NOTE)


def max_consecutive(indices: Sequence[int]) -> int:
    best = current = 0
    previous: int | None = None
    for value in sorted(indices):
        current = current + 1 if previous is not None and value == previous + 1 else 1
        best = max(best, current)
        previous = value
    return best


def parse_corresp_counts(path: Path) -> dict[str, int]:
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    score_present = performance_present = both_present = 0
    for line in rows:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 10:
            raise ValueError(f"malformed correspondence line in {path}")
        score_ok = fields[0] != "*"
        performance_ok = fields[5] != "*"
        score_present += int(score_ok)
        performance_present += int(performance_ok)
        both_present += int(score_ok and performance_ok)
    return {
        "corresp_rows": len(rows),
        "score_present_rows": score_present,
        "performance_present_rows": performance_present,
        "both_present_rows": both_present,
        "extra_performance_note_rows": sum(
            1 for line in rows if line.split("\t")[0] == "*" and line.split("\t")[5] != "*"
        ),
    }


def cleanup_alignment_files(tool_dir: Path) -> None:
    for stem in ("eval_performance", "eval_score"):
        for path in tool_dir.glob(f"{stem}*"):
            if path.is_file():
                path.unlink()


def aligned_full_tokens(
    config: PinnedTokenizerConfig,
    score_path: Path,
    performance_path: Path,
    tool_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the full pre-segmentation PT-aligned label on the score grid."""

    cleanup_alignment_files(tool_dir)
    score = normalize_midi(MidiFile(str(score_path)))
    performance = normalize_midi(MidiFile(str(performance_path)))
    performance_file = tool_dir / "eval_performance.mid"
    score_file = tool_dir / "eval_score.mid"
    performance.dump(str(performance_file))
    score.dump(str(score_file))

    completed = subprocess.run(
        ["bash", "./MIDIToMIDIAlign.sh", "eval_performance", "eval_score"],
        cwd=tool_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    corresp_path = tool_dir / "eval_score_corresp.txt"
    if completed.returncode != 0 or not corresp_path.is_file():
        raise RuntimeError(
            f"alignment failed for {performance_path}: rc={completed.returncode}\n"
            f"{completed.stdout[-4000:]}"
        )
    corresp_hash = sha256(corresp_path)
    raw_counts = parse_corresp_counts(corresp_path)
    correspondence = read_corresp(str(corresp_path))

    score_notes = score.instruments[0].notes
    performance_notes = performance.instruments[0].notes
    if len(correspondence) != len(score_notes):
        raise RuntimeError(
            f"correspondence does not cover score: {len(correspondence)} != {len(score_notes)}"
        )
    if [int(item[0]) for item in correspondence] != list(range(len(score_notes))):
        raise RuntimeError("correspondence score indices are not exactly 0..N-1")

    score_starts: list[int] = []
    velocities: list[float] = []
    starts: list[float] = []
    durations: list[float] = []
    unknown: list[int] = []
    for index, (score_index, performance_index) in enumerate(correspondence):
        if performance_index != -1:
            note = performance_notes[performance_index]
            velocities.append(note.velocity)
            starts.append(note.start)
            durations.append(note.end - note.start)
        else:
            velocities.append(np.nan)
            durations.append(np.nan)
            unknown.append(index)
        score_starts.append(score_notes[score_index].start)

    # This intentionally follows PT's align_score_and_performance implementation.
    starts.sort()
    ordered_starts: list[float] = []
    cursor = 0
    unknown_set = set(unknown)
    for index in range(len(correspondence)):
        if index in unknown_set:
            ordered_starts.append(np.nan)
        else:
            ordered_starts.append(starts[cursor])
            cursor += 1
    aligned_starts = interpolate(score_starts, ordered_starts)
    aligned_velocities = interpolate(aligned_starts, velocities)
    aligned_durations = interpolate(aligned_starts, durations)

    output_notes: list[Note] = []
    ends: list[int] = []
    for index, (score_index, _) in enumerate(correspondence):
        end = aligned_starts[index] + aligned_durations[index]
        ends.append(end)
        output_notes.append(
            Note(
                aligned_velocities[index],
                score_notes[score_index].pitch,
                aligned_starts[index],
                end,
            )
        )
    max_tick = max(ends) + 4999
    output_ccs: list[ControlChange] = []
    for cc in performance.instruments[0].control_changes:
        if cc.time <= max_tick:
            output_ccs.append(cc)
        else:
            break
    aligned = MidiFile(ticks_per_beat=500)
    aligned.instruments.append(
        Instrument(
            program=0,
            is_drum=False,
            name="Piano",
            notes=output_notes,
            control_changes=output_ccs,
        )
    )
    score_tokens = matrix_from_ids(midi_to_ids(config, score))
    label_tokens = matrix_from_ids(midi_to_ids(config, aligned, normalize=False))
    if score_tokens.shape != label_tokens.shape:
        raise RuntimeError("aligned score/label shapes differ")
    if not np.array_equal(score_tokens[:, 0], label_tokens[:, 0]):
        raise RuntimeError("aligned label pitches differ from canonical score pitches")

    segments, _ = segment_sequences(
        score_tokens.reshape(-1).tolist(),
        label_tokens.reshape(-1).tolist(),
        unknown,
        len(score_notes),
        5,
        64,
    )
    audit = {
        "score_note_count": len(score_notes),
        "performance_normalized_note_count": len(performance_notes),
        "matched_score_note_count": len(score_notes) - len(unknown),
        "interpolated_score_note_count": len(unknown),
        "max_consecutive_interpolated_score_notes": max_consecutive(unknown),
        "pt_sft_segment_count_if_applied": len(segments),
        "pt_sft_segment_note_counts_if_applied": [len(item) // TOKENS_PER_NOTE for item in segments],
        "full_pre_segmentation_label_used": True,
        "correspondence_sha256": corresp_hash,
        **raw_counts,
    }
    cleanup_alignment_files(tool_dir)
    return label_tokens, audit


def strip_private(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("_") and key != "loss"
    }


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asap-root", type=Path, required=True)
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv",
    )
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "analysis/stage2_binary_canonical_v1/test_eval_v0/canonical_test_manifest.csv",
    )
    parser.add_argument(
        "--alignment-tool-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis/original_pt_early_metrics_test_v0",
    )
    args = parser.parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PinnedTokenizerConfig()

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty or -1 for this CPU-only evaluation")
    for program in (
        "midi2pianoroll",
        "SprToFmt3x",
        "Fmt3xToHmm",
        "ScorePerfmMatcher",
        "ErrorDetection",
        "RealignmentMOHMM",
        "MatchToCorresp",
    ):
        if not (args.alignment_tool_dir / "Programs" / program).is_file():
            raise FileNotFoundError(f"missing alignment program: {program}")

    split = pd.read_csv(args.split_csv)
    test = split.loc[split["split"] == "test"].copy()
    if len(test) != EXPECTED_TEST_PERFORMANCES:
        raise RuntimeError(f"expected 104 test performances, got {len(test)}")
    if test["piece_id"].nunique() != EXPECTED_TEST_PIECES:
        raise RuntimeError("test split does not contain exactly 23 pieces")
    if test["performance_path"].nunique() != len(test):
        raise RuntimeError("duplicate test performance_path detected")

    metadata = pd.read_csv(args.asap_root / "metadata.csv")
    test = test.merge(
        metadata[["midi_performance", "midi_score", "performance_annotations"]],
        left_on="performance_path",
        right_on="midi_performance",
        how="left",
        validate="one_to_one",
    )
    if test["midi_score"].isna().any():
        raise RuntimeError("split performance missing from ASAP metadata")

    manifest = pd.read_csv(args.canonical_manifest)
    if len(manifest) != EXPECTED_ORIGINAL_PT_MIDIS or set(manifest["num"]) != set(range(23)):
        raise RuntimeError("canonical manifest is not nums 0..22 exactly once")
    if manifest["piece_id"].nunique() != 23:
        raise RuntimeError("canonical manifest piece IDs are not unique")
    manifest_by_piece = {row.piece_id: row for _, row in manifest.iterrows()}
    if set(test["piece_id"]) != set(manifest_by_piece):
        raise RuntimeError("test split piece set differs from canonical Original PT manifest")
    for row in manifest_by_piece.values():
        midi_path = Path(row.canonical_original_pt_midi_path)
        if not midi_path.is_file() or sha256(midi_path) != row.canonical_midi_sha256:
            raise RuntimeError(f"canonical Original PT MIDI missing/hash mismatch: {midi_path}")

    print("PREFLIGHT PASS: 23 pieces, 104 unique human performances, 23 saved Original PT MIDI", flush=True)

    pt_by_piece: dict[str, np.ndarray] = {}
    pt_alignment_audit: list[dict[str, Any]] = []
    for piece_index, (_, row) in enumerate(manifest.sort_values("num").iterrows(), start=1):
        tokens, audit = aligned_full_tokens(
            config,
            Path(row.score_path),
            Path(row.canonical_original_pt_midi_path),
            args.alignment_tool_dir,
        )
        pt_by_piece[row.piece_id] = tokens[:, 4:8] - PEDAL_START
        pt_alignment_audit.append(
            {
                "num": int(row.num),
                "piece_id": row.piece_id,
                "composer": row.composer,
                "title": row.title,
                "canonical_score_path": row.score_path,
                "original_pt_midi_path": row.canonical_original_pt_midi_path,
                **audit,
            }
        )
        print(f"PT ALIGN {piece_index:02d}/23 PASS num={row.num} notes={len(tokens)}", flush=True)

    standard_rows: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    class_prediction_records: list[np.ndarray] = []
    class_target_records: list[np.ndarray] = []
    raw_target_records: list[np.ndarray] = []
    per_performance_rows: list[dict[str, Any]] = []

    test = test.sort_values(["metadata_index", "performance_path"]).reset_index(drop=True)
    for index, row in test.iterrows():
        canonical = manifest_by_piece[row.piece_id]
        human_tokens, audit = aligned_full_tokens(
            config,
            Path(canonical.score_path),
            args.asap_root / row.performance_path,
            args.alignment_tool_dir,
        )
        target = human_tokens[:, 4:8] - PEDAL_START
        predicted = pt_by_piece[row.piece_id]
        if predicted.shape != target.shape:
            raise RuntimeError(
                f"post-alignment shape mismatch for {row.performance_path}: "
                f"{predicted.shape} != {target.shape}"
            )
        if np.any((predicted < 0) | (predicted > 127)) or np.any((target < 0) | (target > 127)):
            raise RuntimeError("pedal value outside [0,127]")

        standard = standard_metrics(predicted, target)
        intermediate = intermediate_metrics(predicted, target)
        distribution = distribution_metrics(predicted, target)
        transition = transition_metrics(predicted, target)
        standard_rows.append(standard)
        intermediate_rows.append(intermediate)
        distribution_rows.append(distribution)
        transition_rows.append(transition)

        predicted_classes = np.asarray(classify_pedal_values(predicted), dtype=np.int64)
        target_classes = np.asarray(classify_pedal_values(target), dtype=np.int64)
        class_prediction_records.append(predicted_classes)
        class_target_records.append(target_classes)
        raw_target_records.append(target)
        per_class = classification_metrics(predicted_classes, target_classes)
        per_renderer = renderer_aware_metrics([(predicted_classes, target_classes)])
        per_decoded = decoded_value_metrics([(predicted_classes, target)])

        row_score_abs = (args.asap_root / row.midi_score).resolve()
        canonical_score_abs = Path(canonical.score_path).resolve()
        per_performance_rows.append(
            {
                "metadata_index": int(row.metadata_index),
                "num": int(canonical.num),
                "piece_id": row.piece_id,
                "composer": row.composer,
                "title": row.title,
                "performance_path": row.performance_path,
                "asap_row_score_path": row.midi_score,
                "asap_row_score_is_canonical_score": row_score_abs == canonical_score_abs,
                "canonical_score_note_count": len(target),
                "human_normalized_note_count": audit["performance_normalized_note_count"],
                "matched_score_note_count": audit["matched_score_note_count"],
                "interpolated_score_note_count": audit["interpolated_score_note_count"],
                "max_consecutive_interpolated_score_notes": audit[
                    "max_consecutive_interpolated_score_notes"
                ],
                "extra_human_note_rows": audit["extra_performance_note_rows"],
                "correspondence_sha256": audit["correspondence_sha256"],
                "raw_token_accuracy": standard["pedal_token_accuracy"],
                "raw_exact_note_accuracy": standard["exact_note_accuracy"],
                "raw_mae": standard["pedal_value_mae"],
                "raw_transition_accuracy": transition[
                    "transition_position_exact_accuracy"
                ],
                "raw_steady_accuracy": transition[
                    "steady_position_exact_accuracy"
                ],
                "raw_transition_precision": transition[
                    "transition_detection_precision"
                ],
                "raw_transition_recall": transition["transition_detection_recall"],
                "raw_transition_f1": transition["transition_detection_f1"],
                "five_class_token_accuracy": per_class["token_accuracy"],
                "five_class_exact_note_accuracy": per_class["exact_note_accuracy"],
                "five_class_macro_f1": per_class["macro_f1"],
                "off_on_state_accuracy": per_renderer["state"]["accuracy"],
                "binary_transition_precision": per_renderer[
                    "combined_binary_transition"
                ]["precision"],
                "binary_transition_recall": per_renderer[
                    "combined_binary_transition"
                ]["recall"],
                "binary_transition_f1": per_renderer[
                    "combined_binary_transition"
                ]["f1"],
                "five_class_decoded_mae": per_decoded["overall_micro_mae"],
            }
        )
        print(
            f"HUMAN ALIGN/EVAL {index + 1:03d}/104 PASS "
            f"piece={row.piece_id} notes={len(target)} missing={audit['interpolated_score_note_count']}",
            flush=True,
        )

    raw_global = {
        "standard": strip_private(aggregate_standard(standard_rows)),
        "standard_macro_over_performances": strip_private(macro_standard(standard_rows)),
        "intermediate": strip_private(aggregate_intermediate(intermediate_rows)),
        "intermediate_macro_over_performances": strip_private(
            macro_intermediate(intermediate_rows)
        ),
        "distribution": strip_private(aggregate_distribution(distribution_rows)),
        "transition": strip_private(aggregate_transition(transition_rows)),
        "transition_macro_over_performances": strip_private(
            macro_transition(transition_rows)
        ),
        "rmse": None,
        "tolerance_accuracy": None,
        "rmse_and_tolerance_note": (
            "Not calculated: the initial raw-128 evaluate_oracle.py does not define RMSE "
            "or +/-5/10/20 accuracy; no new raw metric definition was introduced."
        ),
    }

    all_predicted_classes = np.concatenate(class_prediction_records)
    all_target_classes = np.concatenate(class_target_records)
    five_classification = classification_metrics(
        all_predicted_classes, all_target_classes
    )
    five_renderer = renderer_aware_metrics(
        list(zip(class_prediction_records, class_target_records))
    )
    five_decoded = decoded_value_metrics(
        list(zip(class_prediction_records, raw_target_records))
    )

    total_interpolated = sum(
        int(row["interpolated_score_note_count"]) for row in per_performance_rows
    )
    total_matched = sum(int(row["matched_score_note_count"]) for row in per_performance_rows)
    alternate_rows = sum(
        not bool(row["asap_row_score_is_canonical_score"])
        for row in per_performance_rows
    )
    alignment_summary = {
        "canonical_score_grid_note_samples": sum(
            int(row["canonical_score_note_count"]) for row in per_performance_rows
        ),
        "matched_score_notes": total_matched,
        "interpolated_score_notes": total_interpolated,
        "matched_ratio": total_matched / (total_matched + total_interpolated),
        "maximum_consecutive_interpolated_score_notes": max(
            int(row["max_consecutive_interpolated_score_notes"])
            for row in per_performance_rows
        ),
        "alternate_asap_score_variant_performances": alternate_rows,
        "performance_rows_evaluated": len(per_performance_rows),
        "unique_performance_paths": len(
            {row["performance_path"] for row in per_performance_rows}
        ),
        "duplicate_evaluations": len(per_performance_rows)
        - len({row["performance_path"] for row in per_performance_rows}),
        "silent_skips": 0,
        "truncation": 0,
        "padding": 0,
        "correspondence_policy": (
            "Pianist Transformer align_score_and_performance pre-segmentation full score grid: "
            "Nakamura MIDI-to-MIDI correspondence; PT interpolate() for score notes without a match"
        ),
    }
    if alignment_summary["duplicate_evaluations"] != 0:
        raise RuntimeError("duplicate evaluation detected")

    oracle_reference = {
        "split": "ASAP test, 104 human performances",
        "condition": "Oracle Stage 2: human non-pedal Pitch/IOI/Velocity/Duration input",
        "pedal_token_accuracy": 0.568735,
        "exact_note_accuracy": 0.506394,
        "pedal_value_mae": 31.802348,
        "transition_position_exact_accuracy": 0.110027,
        "steady_position_exact_accuracy": 0.653181,
        "transition_detection_f1": 0.124416,
    }
    five_validation_reference = {
        "split": "ASAP validation, 71 human performances (not test)",
        "model": "endpoint-aware encoder-only five-class, best epoch 3",
        "five_class_token_accuracy": 0.5800229283480319,
        "five_class_exact_note_accuracy": 0.5053957341297793,
        "five_class_macro_f1": 0.34163966028264364,
        "five_class_micro_f1": 0.5800229283480319,
        "five_class_weighted_f1": 0.5201120408360368,
        "off_on_state_accuracy": 0.7908210884449579,
        "binary_transition_f1": 0.1271023631327715,
        "five_class_decoded_mae": 28.764686822011214,
    }

    metric_payload = {
        "status": "completed",
        "raw_128_style": raw_global,
        "five_class_style": {
            "classification": five_classification,
            "renderer_aware": five_renderer,
            "canonical_decoded_vs_raw_human": five_decoded,
            "raw_original_pt_mae": raw_global["standard"]["pedal_value_mae"],
            "five_class_decoded_mae": five_decoded["overall_micro_mae"],
        },
        "alignment_summary": alignment_summary,
        "original_pt_alignment_by_piece": pt_alignment_audit,
        "references": {
            "oracle_128_test": oracle_reference,
            "five_class_validation": five_validation_reference,
        },
    }
    write_json(output_dir / "metrics.json", metric_payload)

    per_fields = list(per_performance_rows[0])
    write_csv(output_dir / "per_performance_metrics.csv", per_performance_rows, per_fields)
    matrix = np.asarray(five_classification["confusion_matrix"], dtype=np.int64)
    confusion_rows = []
    for true_id, true_name in enumerate(CLASS_NAMES):
        confusion_rows.append(
            {
                "true_class_id": true_id,
                "true_class_name": true_name,
                **{
                    f"pred_{CLASS_NAMES[pred_id].lower()}": int(matrix[true_id, pred_id])
                    for pred_id in range(len(CLASS_NAMES))
                },
                "support": int(matrix[true_id].sum()),
            }
        )
    write_csv(
        output_dir / "five_class_confusion_matrix.csv",
        confusion_rows,
        list(confusion_rows[0]),
    )

    source_paths = {
        "progress": PROJECT_ROOT / "Progress.md",
        "oracle_report": PROJECT_ROOT
        / "analysis/stage2_encoder_only_v0/oracle_test_v0/ORACLE_TEST_EVALUATION.md",
        "raw_evaluator": PROJECT_ROOT / "src/stage2_encoder_only/evaluate_oracle.py",
        "five_class_report": PROJECT_ROOT
        / "analysis/stage2_encoder_only_5class_v0/FIVE_CLASS_TRAINING_REPORT.md",
        "five_class_evaluator": PROJECT_ROOT
        / "src/stage2_encoder_only/evaluate_five_class.py",
        "five_class_mapping": PROJECT_ROOT / "src/stage2_encoder_only/five_class.py",
        "pt_midi_implementation": PROJECT_ROOT
        / "third_party/PianistTransformer/src/utils/midi.py",
        "split_csv": args.split_csv,
        "canonical_manifest": args.canonical_manifest,
    }
    configuration = {
        "status": "completed",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pt_neural_inference_count": 0,
        "stage2_inference_count": 0,
        "checkpoint_load_count": 0,
        "training_count": 0,
        "official_js_intersection_metric_count": 0,
        "asap_root": str(args.asap_root),
        "output_dir": str(output_dir),
        "test_performance_count": len(test),
        "test_piece_count": test["piece_id"].nunique(),
        "saved_original_pt_midi_count": len(manifest),
        "tokenizer": "pinned official PT midi_to_ids after MIDI load",
        "alignment": {
            "version": ALIGNMENT_VERSION,
            "url": ALIGNMENT_URL,
            "zip_sha256": ALIGNMENT_ZIP_SHA256,
            "tool_dir": str(args.alignment_tool_dir),
            "program_sha256": {
                program.name: sha256(program)
                for program in sorted((args.alignment_tool_dir / "Programs").iterdir())
                if program.is_file()
            },
            "pt_source_function": (
                "third_party.PianistTransformer.src.utils.midi.align_score_and_performance"
            ),
            "full_pre_segmentation_grid": True,
            "interpolation_function": (
                "third_party.PianistTransformer.src.utils.midi.interpolate"
            ),
        },
        "raw_metric_source": "src/stage2_encoder_only/evaluate_oracle.py",
        "five_class_metric_source": "src/stage2_encoder_only/evaluate_five_class.py",
        "five_class_mapping": {
            name: {"id": index, "bounds": list(CLASS_BOUNDS[index]), "representative": int(REPRESENTATIVES[index])}
            for index, name in enumerate(CLASS_NAMES)
        },
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "runtime_seconds": time.time() - started,
    }
    write_json(output_dir / "config.json", configuration)

    raw = raw_global
    five = five_classification
    renderer = five_renderer
    decoded = five_decoded
    raw_table = markdown_table(
        ("Metric", "Original PT baseline"),
        (
            ("Pedal-token / exact-value accuracy", raw["standard"]["pedal_token_accuracy"]),
            ("Exact-note accuracy", raw["standard"]["exact_note_accuracy"]),
            ("Pedal1 accuracy", raw["standard"]["pedal1_accuracy"]),
            ("Pedal2 accuracy", raw["standard"]["pedal2_accuracy"]),
            ("Pedal3 accuracy", raw["standard"]["pedal3_accuracy"]),
            ("Pedal4 accuracy", raw["standard"]["pedal4_accuracy"]),
            ("Raw CC64 MAE", raw["standard"]["pedal_value_mae"]),
            ("Transition accuracy", raw["transition"]["transition_position_exact_accuracy"]),
            ("Steady-position accuracy", raw["transition"]["steady_position_exact_accuracy"]),
            ("Transition precision", raw["transition"]["transition_detection_precision"]),
            ("Transition recall", raw["transition"]["transition_detection_recall"]),
            ("Transition F1", raw["transition"]["transition_detection_f1"]),
            ("Intermediate exact accuracy", raw["intermediate"]["intermediate_exact_accuracy"]),
            ("Intermediate MAE", raw["intermediate"]["intermediate_mae"]),
            ("Intermediate endpoint collapse", raw["intermediate"]["endpoint_collapse_ratio"]),
            ("Predicted ZERO ratio", raw["distribution"]["predicted_zero_ratio"]),
            ("Predicted INTERMEDIATE ratio", raw["distribution"]["predicted_intermediate_ratio"]),
            ("Predicted FULL ratio", raw["distribution"]["predicted_full_ratio"]),
        ),
    )
    five_table = markdown_table(
        ("Metric", "Original PT baseline"),
        (
            ("5-class token accuracy", five["token_accuracy"]),
            ("5-class exact-note accuracy", five["exact_note_accuracy"]),
            ("Macro F1", five["macro_f1"]),
            ("Micro F1", five["micro_f1"]),
            ("Weighted F1", five["weighted_f1"]),
            ("OFF/ON state accuracy", renderer["state"]["accuracy"]),
            ("Binary transition precision", renderer["combined_binary_transition"]["precision"]),
            ("Binary transition recall", renderer["combined_binary_transition"]["recall"]),
            ("Binary transition F1", renderer["combined_binary_transition"]["f1"]),
            ("5-class canonical decoded MAE", decoded["overall_micro_mae"]),
            ("Decoded tolerance ±5", decoded["tolerance_accuracy"]["plus_minus_5"]),
            ("Decoded tolerance ±10", decoded["tolerance_accuracy"]["plus_minus_10"]),
            ("Decoded tolerance ±20", decoded["tolerance_accuracy"]["plus_minus_20"]),
        ),
    )
    per_class_table = markdown_table(
        ("Class", "Precision", "Recall", "F1", "Predicted ratio"),
        [
            (
                name,
                five["precision"][index],
                five["recall"][index],
                five["f1"][index],
                five["predicted_distribution"][index],
            )
            for index, name in enumerate(CLASS_NAMES)
        ],
    )
    reference_table = markdown_table(
        ("Metric", "Original PT baseline", "Oracle Stage 2 test reference"),
        (
            ("Token accuracy", raw["standard"]["pedal_token_accuracy"], oracle_reference["pedal_token_accuracy"]),
            ("Exact-note accuracy", raw["standard"]["exact_note_accuracy"], oracle_reference["exact_note_accuracy"]),
            ("MAE", raw["standard"]["pedal_value_mae"], oracle_reference["pedal_value_mae"]),
            ("Transition accuracy", raw["transition"]["transition_position_exact_accuracy"], oracle_reference["transition_position_exact_accuracy"]),
            ("Steady accuracy", raw["transition"]["steady_position_exact_accuracy"], oracle_reference["steady_position_exact_accuracy"]),
            ("Transition F1", raw["transition"]["transition_detection_f1"], oracle_reference["transition_detection_f1"]),
        ),
    )
    report = f"""# Original Pianist Transformer Early-Style Test Metrics v0

## Outcome

저장된 Original PT MIDI 23개만을 읽어 ASAP piece-wise test split의 23 pieces / 104 unique human performances에 대해 early raw-128-style 및 endpoint-aware 5-class-style metric을 CPU-only로 계산했다. PT inference, Stage 2 inference, checkpoint load, training, GPU/CUDA, official JS/Intersection 계산은 모두 0회다.

## Data and correspondence audit

- Original PT MIDI: 23 files, canonical nums 0..22 exactly once; manifest hash 검증 PASS.
- ASAP test: 104 unique performance paths, 23 piece IDs; duplicate evaluation 0.
- Human raw tokenizer note total은 331,576이지만 canonical score grid로 반복 집계한 note samples는 {alignment_summary['canonical_score_grid_note_samples']:,}이다. Note 수가 다르므로 direct index, truncate, pad, nearest-neighbour matching은 사용하지 않았다.
- Correspondence: PT repository의 `align_score_and_performance`와 동일한 Nakamura MIDI-to-MIDI correspondence, 동일 `normalize_midi`, `read_corresp`, `interpolate`, `midi_to_ids` 경로를 사용했다. Metric에는 PT SFT segmentation 전 full canonical score grid를 사용하여 silent skip은 0이다.
- Matched score notes: {total_matched:,}; PT interpolation이 적용된 unmatched score notes: {total_interpolated:,}; match ratio: {alignment_summary['matched_ratio']:.6%}; 최대 consecutive interpolation: {alignment_summary['maximum_consecutive_interpolated_score_notes']} notes.
- 3 human performances는 ASAP metadata상 alternate repeat score variant를 사용한다. Fixed Original PT가 생성된 canonical score에 같은 공식 aligner로 직접 대응시켰으며 per-performance CSV에서 명시적으로 flag했다.
- Saved PT 자체도 canonical score와 공식 aligner로 대응시킨 뒤, actual saved MIDI load와 official `midi_to_ids` semantics로 Pedal1–4를 추출했다.
- PT repository는 alignment tool download page만 지정하고 version/hash를 pin하지 않는다. 이번 실행은 공식 `{ALIGNMENT_VERSION}` ZIP ({ALIGNMENT_ZIP_SHA256})을 사용했으며 `config.json`에 static program hashes까지 기록했다.

이 결과는 alignment-dependent baseline이다. 특히 {total_interpolated:,} score notes와 최대 {alignment_summary['maximum_consecutive_interpolated_score_notes']:,}-note 연속 구간이 PT의 interpolation에 의존하므로, raw note-wise 숫자를 manual ground-truth note alignment 또는 Oracle Stage 2와 동일 조건인 것처럼 해석하면 안 된다.

이 baseline은 score-generated Original PT를 human performance pedal target에 score-note correspondence로 비교한다. Human non-pedal token을 그대로 입력으로 쓴 Oracle Stage 2와 입력 조건이 다르며, 동일 조건 controlled comparison이 아니다.

## Raw 128-class-style metrics

{raw_table}

RMSE 및 raw ±5/±10/±20 accuracy는 초기 `evaluate_oracle.py`에 정의되어 있지 않아 새 정의를 추가하지 않았다. Transition/steady는 performance별 note-major `P1→P2→P3→P4` flatten 후 집계하며 performance boundary를 넘지 않는다.

## Five-class-style metrics

Classes: ZERO=0, LOW=1–63, MID=64–95, HIGH=96–126, FULL=127. Representatives: `[0, 32, 80, 111, 127]`.

{five_table}

Raw Original PT CC64 MAE는 **{raw['standard']['pedal_value_mae']:.6f}**, Original PT raw value를 5-class로 quantize한 뒤 representative로 decode한 MAE는 **{decoded['overall_micro_mae']:.6f}**다. 정의는 서로 다르지만 이번 saved PT는 ZERO/FULL endpoint만 포함하므로 수치가 같아졌다.

### Per-class results

{per_class_table}

Confusion matrix는 `five_class_confusion_matrix.csv`, performance별 결과와 alignment audit는 `per_performance_metrics.csv`에 저장했다.

## Earlier Stage 2 references (reference only)

{reference_table}

위 Oracle Stage 2 값은 같은 104-performance test split이지만 각 human performance의 non-pedal input을 사용했다. Original PT baseline은 score에서 생성된 saved output이므로 직접적인 architecture-only 또는 controlled comparison으로 해석하면 안 된다.

5-class 기존 reference는 **validation 71 performances**의 endpoint-aware encoder-only model 결과다: token accuracy {five_validation_reference['five_class_token_accuracy']:.6f}, exact-note {five_validation_reference['five_class_exact_note_accuracy']:.6f}, macro-F1 {five_validation_reference['five_class_macro_f1']:.6f}, weighted-F1 {five_validation_reference['five_class_weighted_f1']:.6f}, decoded MAE {five_validation_reference['five_class_decoded_mae']:.6f}. Split이 다르므로 Original PT test baseline과 같은-split model comparison으로 사용하지 않는다.

## Integrity and scope

- All 104 split rows evaluated exactly once; duplicate 0.
- Silent skip / truncation / padding: 0 / 0 / 0.
- No Stage 2 result was recomputed and no model selection was performed.
- No PT or Stage 2 neural inference, checkpoint loading, training, audio, JS, or Intersection evaluation was performed.
- Full provenance, source hashes, alignment tool hash, and CPU-only controls are in `config.json`.
"""
    (output_dir / "ORIGINAL_PT_EARLY_METRICS_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(f"COMPLETED: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
