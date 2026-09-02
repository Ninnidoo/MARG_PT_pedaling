#!/usr/bin/env python3
"""CPU-only Bass Connectivity v0 evaluation on frozen ASAP-validation MIDI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bass_connectivity_v0 import (
    BETA_OCTAVE,
    GROUP_WINDOW_SECONDS,
    LOCAL_IOI_RADIUS,
    NEXT_LOW_MAX_PITCH,
    R_HIGH,
    R_LOW,
    STRUCTURAL_BASS_THRESHOLD,
    NoteAttack,
    connectivity_score,
    detect_structural_bass,
    earliest_anchor_groups,
    parse_midi,
    structural_groups,
)
from src.stage2_binary.canonical_stage1 import sha256_file, signature_sha256


OUTPUT = ROOT / "analysis/bass_connectivity_v0_validation"
STAGE1_MANIFEST = ROOT / "analysis/stage2_binary_canonical_v1/canonical_validation_stage1_manifest.csv"
SPLIT_MANIFEST = ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
DECODING_ROOT = ROOT / "analysis/stage2_encoder_only_4class_decoding_phase4_v0"
WEIGHTED_ROOT = ROOT / "analysis/stage2_encoder_only_4class_loss_phase3_v0"
CONTROL_MANIFEST = ROOT / "analysis/harmonic_metric_pure_hall_negative_v0/input_midis.csv"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")

SYSTEMS = (
    "HUMAN",
    "ORIGINAL_PT",
    "STANDARD_CE_ARGMAX",
    "STANDARD_CE_POSTERIOR_MEDIAN",
    "WEIGHTED_CE_ARGMAX",
    "NO_PEDAL",
    "ALWAYS_ON",
)
MODEL_PATH_TEMPLATE = {
    "STANDARD_CE_ARGMAX": DECODING_ROOT / "predictions/argmax/{piece_id}/candidate.mid",
    "STANDARD_CE_POSTERIOR_MEDIAN": DECODING_ROOT / "predictions/median/{piece_id}/candidate.mid",
    "WEIGHTED_CE_ARGMAX": WEIGHTED_ROOT / "weighted_ce/validation/predictions/weighted_ce/{piece_id}/candidate.mid",
}
SYSTEM_LABEL = {
    "HUMAN": "Human",
    "ORIGINAL_PT": "Original PT",
    "STANDARD_CE_ARGMAX": "Standard CE — argmax (4-class)",
    "STANDARD_CE_POSTERIOR_MEDIAN": "Standard CE — posterior median (4-class)",
    "WEIGHTED_CE_ARGMAX": "Weighted CE — argmax (4-class)",
    "NO_PEDAL": "No pedal / Always off",
    "ALWAYS_ON": "Always on",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def present(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> None:
    keys = set().union(*(row.keys() for row in rows)) if rows else set(preferred)
    fields = list(dict.fromkeys([*preferred, *sorted(keys - set(preferred))]))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: present(row.get(key)) for key in fields})
    temporary.replace(path)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def mean(values: Iterable[float]) -> float | None:
    selected = [float(value) for value in values]
    return statistics.fmean(selected) if selected else None


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    selected = [float(value) for value in values]
    if not selected:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": statistics.fmean(selected),
        "median": statistics.median(selected),
        "std": statistics.pstdev(selected),
    }


def build_input_manifest() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    stage_rows = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_MANIFEST) if row["split"] == "validation"]
    if len(stage_rows) != 19 or len({row["piece_id"] for row in stage_rows}) != 19:
        raise RuntimeError("canonical Stage-1 manifest is not 19 unique validation pieces")
    if len(validation) != 71 or len({row["piece_id"] for row in validation}) != 19:
        raise RuntimeError("validation split is not 71 performances / 19 pieces")
    stage_by_piece = {row["piece_id"]: row for row in stage_rows}
    if set(stage_by_piece) != {row["piece_id"] for row in validation}:
        raise RuntimeError("canonical and human validation piece universes differ")
    harmonic_rows = {
        (row["piece_id"], row["system"]): row
        for row in read_csv(CONTROL_MANIFEST)
    }
    controls = {
        key: row for key, row in harmonic_rows.items()
        if row["system"] in {"NO_PEDAL", "ALWAYS_ON"}
    }

    rows: list[dict[str, Any]] = []
    for item in validation:
        path = ASAP_ROOT / item["performance_path"]
        rows.append({
            "system": "HUMAN",
            "system_label": SYSTEM_LABEL["HUMAN"],
            "piece_id": item["piece_id"],
            "composer": item["composer"],
            "title": item["title"],
            "performance_id": item["performance_path"],
            "midi_path": str(path),
            "source_manifest": str(SPLIT_MANIFEST),
            "expected": True,
            "status": "available" if path.is_file() else "missing",
            "missing_reason": "" if path.is_file() else "validation human MIDI absent",
            "invariant_group": "HUMAN_SEPARATE",
        })

    for piece_id, stage in sorted(stage_by_piece.items()):
        base = {
            "piece_id": piece_id,
            "composer": stage["composer"],
            "title": stage["title"],
            "performance_id": piece_id,
            "expected": True,
            "invariant_group": f"CANONICAL_NON_PEDAL::{piece_id}",
        }
        original = Path(stage["canonical_midi_path"])
        rows.append({
            **base,
            "system": "ORIGINAL_PT",
            "system_label": SYSTEM_LABEL["ORIGINAL_PT"],
            "midi_path": str(original),
            "source_manifest": str(STAGE1_MANIFEST),
            "status": "available" if original.is_file() else "missing",
            "missing_reason": "" if original.is_file() else "canonical Original PT MIDI absent",
            "expected_sha256": stage["canonical_midi_sha256"],
        })
        for system, template in MODEL_PATH_TEMPLATE.items():
            path = Path(str(template).format(piece_id=piece_id))
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            expected_hash = metadata.get("candidate_midi_sha256", "")
            harmonic = harmonic_rows.get((piece_id, system))
            if harmonic and expected_hash and harmonic["sha256"] != expected_hash:
                raise RuntimeError(f"4-class hash provenance disagreement: {system} {piece_id}")
            rows.append({
                **base,
                "system": system,
                "system_label": SYSTEM_LABEL[system],
                "midi_path": str(path),
                "source_manifest": str(metadata_path),
                "status": "available" if path.is_file() else "missing",
                "missing_reason": "" if path.is_file() else f"existing {system} candidate MIDI absent",
                "expected_sha256": expected_hash or (harmonic["sha256"] if harmonic else ""),
            })
        for system in ("NO_PEDAL", "ALWAYS_ON"):
            control = controls.get((piece_id, system))
            path = Path(control["midi_path"]) if control else None
            available = bool(path and path.is_file())
            rows.append({
                **base,
                "system": system,
                "system_label": SYSTEM_LABEL[system],
                "midi_path": str(path) if path else "",
                "source_manifest": str(CONTROL_MANIFEST),
                "status": "available" if available else "missing",
                "missing_reason": "" if available else "no pre-existing Harmonic Muddiness control MIDI",
                "expected_sha256": control["sha256"] if control else "",
            })

    system_order = {system: i for i, system in enumerate(SYSTEMS)}
    rows.sort(key=lambda row: (system_order[row["system"]], row["piece_id"], row["performance_id"]))
    for row in rows:
        if row["status"] == "available":
            path = Path(row["midi_path"])
            row["sha256"] = sha256_file(path)
            expected = row.get("expected_sha256", "")
            row["hash_verified"] = not expected or expected == row["sha256"]
            if not row["hash_verified"]:
                raise RuntimeError(f"frozen input hash changed: {path}")
            row["used"] = True
        else:
            row["sha256"] = ""
            row["hash_verified"] = ""
            row["used"] = False
    return rows, stage_by_piece


def required_logic_tests() -> list[dict[str, Any]]:
    exp_half = math.exp(-0.5)
    base = dict(
        keyoff_seconds=1.0,
        next_onset_seconds=2.0,
        current_pitch=40,
        next_pitch=43,
        pedal_on_at_keyoff=True,
        first_pedal_off_seconds=2.0,
    )

    def result(**updates: Any):
        args = dict(base)
        args.update(updates)
        return connectivity_score(**args)

    synthetic = [
        NoteAttack(i, 0, 60 + i, 64, i, time, i)
        for i, time in enumerate((0.000, 0.015, 0.030), 1)
    ]
    grouping = [[round(1000 * note.onset_seconds) for note in group] for group in earliest_anchor_groups(synthetic)]
    checks = [
        ("1_keyoff_at_pedal_off", "0", result(pedal_on_at_keyoff=False).score, result(pedal_on_at_keyoff=False).score == 0.0),
        ("2_different_off_at_next", "1", result().score, result().score == 1.0),
        ("3_different_left_half_gap", f"{exp_half:.8f}", result(first_pedal_off_seconds=1.5).score, math.isclose(result(first_pedal_off_seconds=1.5).score or -1, exp_half, abs_tol=1e-12)),
        ("4_different_right_quarter_gap", f"{exp_half:.8f}", result(first_pedal_off_seconds=2.25).score, math.isclose(result(first_pedal_off_seconds=2.25).score or -1, exp_half, abs_tol=1e-12)),
        ("5_same_connected_through_next", "1", result(next_pitch=40, first_pedal_off_seconds=2.5).score, result(next_pitch=40, first_pedal_off_seconds=2.5).score == 1.0),
        ("6_no_subsequent_off", "different=0;same=1", f"different={result(first_pedal_off_seconds=None).score:g};same={result(next_pitch=40, first_pedal_off_seconds=None).score:g}", result(first_pedal_off_seconds=None).score == 0.0 and result(next_pitch=40, first_pedal_off_seconds=None).score == 1.0),
        ("7_keyoff_at_next_excluded", "excluded", result(keyoff_seconds=2.0).status, result(keyoff_seconds=2.0).score is None),
        ("8_earliest_anchor_grouping", "[[0,15],[30]]", json.dumps(grouping, separators=(",", ":")), grouping == [[0, 15], [30]]),
    ]
    rows = [{"test": name, "expected": expected, "actual": actual, "passed": passed} for name, expected, actual, passed in checks]
    if not all(row["passed"] for row in rows):
        raise AssertionError("required Bass Connectivity synthetic logic test failed")
    return rows


def evaluate_performance(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    parsed = parse_midi(Path(row["midi_path"]))
    groups = detect_structural_bass(parsed)
    basses = structural_groups(groups)
    structural_rows: list[dict[str, Any]] = []
    duplicate_lowest = 0
    for bass_index, group in enumerate(basses):
        low_count = sum(note.pitch == group.lowest_pitch for note in group.attacks)
        duplicate_lowest += int(low_count > 1)
        note = group.lowest_attack
        structural_rows.append({
            "system": row["system"], "piece_id": row["piece_id"],
            "performance_id": row["performance_id"], "midi_path": row["midi_path"],
            "structural_bass_index": bass_index, "onset_group_index": group.index,
            "onset_time_sec": group.anchor_seconds, "anchor_tick": group.anchor_tick,
            "lowest_pitch": group.lowest_pitch, "group_pitches": " ".join(map(str, group.pitches)),
            "lowest_pitch_attack_count": low_count, "lowest_attack_id": note.identifier,
            "lowest_attack_time_sec": note.onset_seconds, "lowest_attack_channel": note.channel,
            "physical_keyoff_time_sec": note.keyoff_seconds, "physical_keyoff_tick": note.keyoff_tick,
            "keyoff_source": note.keyoff_source, "pedal_on_at_keyoff": note.pedal_on_at_keyoff,
            "first_pedal_off_time_sec": note.first_pedal_off_seconds,
            "local_median_ioi_sec": group.local_median_ioi,
            "local_positive_ioi_count": group.local_positive_ioi_count,
            "next_low_group_index": group.next_low_group_index,
            "next_low_gap_sec": group.next_low_gap_seconds, "R_B": group.r_b,
            "L_n": group.lowness, "S_n": group.structural_timescale,
            "O_oct": group.octave_bonus, "B_n": group.salience,
        })

    transitions: list[dict[str, Any]] = []
    scores: list[float] = []
    counters: Counter[str] = Counter()
    opportunity_signature = []
    for pair_index, (current, following) in enumerate(zip(basses, basses[1:])):
        note = current.lowest_attack
        result = connectivity_score(
            keyoff_seconds=note.keyoff_seconds,
            next_onset_seconds=following.anchor_seconds,
            current_pitch=current.lowest_pitch,
            next_pitch=following.lowest_pitch,
            pedal_on_at_keyoff=note.pedal_on_at_keyoff,
            first_pedal_off_seconds=note.first_pedal_off_seconds,
        )
        valid = result.score is not None
        if valid:
            scores.append(float(result.score))
            counters["valid_transition_count"] += 1
            counters["same_bass_transition_count" if result.same_bass else "different_bass_transition_count"] += 1
            counters["keyoff_at_pedal_off_count"] += int(result.keyoff_at_pedal_off)
            counters["no_subsequent_pedal_off_count"] += int(result.no_subsequent_pedal_off)
        elif result.status == "excluded_keyoff_at_or_after_next_bass":
            counters["excluded_keyoff_at_or_after_next_count"] += 1
        elif result.status == "excluded_missing_keyoff":
            counters["excluded_missing_keyoff_count"] += 1
        opportunity_signature.append((
            current.index, following.index,
            "valid" if valid else result.status,
            current.lowest_pitch, following.lowest_pitch,
            note.keyoff_tick, following.anchor_tick,
        ))
        transitions.append({
            "system": row["system"], "piece_id": row["piece_id"],
            "performance_id": row["performance_id"], "midi_path": row["midi_path"],
            "pair_index": pair_index, "current_onset_group_index": current.index,
            "next_onset_group_index": following.index,
            "current_bass_pitch": current.lowest_pitch, "next_bass_pitch": following.lowest_pitch,
            "same_bass": result.same_bass, "current_bass_onset_sec": current.anchor_seconds,
            "physical_keyoff_sec": note.keyoff_seconds, "next_bass_onset_sec": following.anchor_seconds,
            "gap_sec": result.gap_seconds, "pedal_on_at_keyoff": note.pedal_on_at_keyoff,
            "first_pedal_off_sec": result.pedal_off_seconds, "status": result.status,
            "is_valid_opportunity": valid, "keyoff_at_pedal_off": result.keyoff_at_pedal_off,
            "no_subsequent_pedal_off": result.no_subsequent_pedal_off, "C_r": result.score,
        })

    structural_signature = [
        (group.index, group.anchor_tick, group.lowest_pitch, group.lowest_attack.keyoff_tick)
        for group in basses
    ]
    performance = {
        "system": row["system"], "system_label": row["system_label"],
        "piece_id": row["piece_id"], "composer": row["composer"], "title": row["title"],
        "performance_id": row["performance_id"], "midi_path": row["midi_path"],
        "structural_bass_count": len(basses),
        "consecutive_bass_pair_count": max(0, len(basses) - 1),
        "last_structural_bass_excluded_count": int(bool(basses)),
        "valid_transition_count": counters["valid_transition_count"],
        "excluded_keyoff_at_or_after_next_count": counters["excluded_keyoff_at_or_after_next_count"],
        "excluded_missing_keyoff_count": counters["excluded_missing_keyoff_count"],
        "same_bass_transition_count": counters["same_bass_transition_count"],
        "different_bass_transition_count": counters["different_bass_transition_count"],
        "keyoff_at_pedal_off_count": counters["keyoff_at_pedal_off_count"],
        "no_subsequent_pedal_off_count": counters["no_subsequent_pedal_off_count"],
        "no_opportunity": not scores, "C_Bass_performance": mean(scores),
        "duplicate_lowest_structural_group_count": duplicate_lowest,
        "all_attack_count": len(parsed.attacks), "all_onset_group_count": len(groups),
        **{f"parser_{key}": value for key, value in parsed.diagnostics.items()},
    }
    signatures = {
        "non_cc64_signature_sha256": signature_sha256(Path(row["midi_path"])),
        "structural_bass_signature_sha256": stable_hash(structural_signature),
        "opportunity_signature_sha256": stable_hash(opportunity_signature),
    }
    return performance, transitions, structural_rows, signatures


def aggregate(performance_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]], input_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    piece_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in performance_rows:
        piece_groups[(row["system"], row["piece_id"])].append(row)
    piece_rows: list[dict[str, Any]] = []
    for (system, piece_id), rows in sorted(piece_groups.items(), key=lambda item: (SYSTEMS.index(item[0][0]), item[0][1])):
        values = [row["C_Bass_performance"] for row in rows if row["C_Bass_performance"] is not None]
        piece_rows.append({
            "system": system, "system_label": SYSTEM_LABEL[system], "piece_id": piece_id,
            "composer": rows[0]["composer"], "title": rows[0]["title"],
            "performance_count": len(rows), "scored_performance_count": len(values),
            "no_opportunity_performance_count": sum(row["no_opportunity"] for row in rows),
            "structural_bass_count": sum(row["structural_bass_count"] for row in rows),
            "valid_transition_count": sum(row["valid_transition_count"] for row in rows),
            "C_Bass_piece": mean(values),
        })

    system_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        perfs = [row for row in performance_rows if row["system"] == system]
        pieces = [row for row in piece_rows if row["system"] == system]
        valid = [row for row in transition_rows if row["system"] == system and row["is_valid_opportunity"]]
        scores = [float(row["C_r"]) for row in valid]
        same = [float(row["C_r"]) for row in valid if row["same_bass"]]
        different = [float(row["C_r"]) for row in valid if not row["same_bass"]]
        stats, same_stats, different_stats = distribution(scores), distribution(same), distribution(different)
        macro_values = [row["C_Bass_piece"] for row in pieces if row["C_Bass_piece"] is not None]
        expected = [row for row in input_rows if row["system"] == system]
        system_rows.append({
            "system": system, "system_label": SYSTEM_LABEL[system],
            "number_of_performances": len(perfs), "number_of_pieces": len({row["piece_id"] for row in perfs}),
            "expected_input_count": len(expected), "missing_input_count": sum(row["status"] == "missing" for row in expected),
            "detected_structural_bass_count": sum(row["structural_bass_count"] for row in perfs),
            "valid_bass_transition_count": len(valid),
            "excluded_keyoff_at_or_after_next_count": sum(row["excluded_keyoff_at_or_after_next_count"] for row in perfs),
            "excluded_missing_keyoff_count": sum(row["excluded_missing_keyoff_count"] for row in perfs),
            "no_opportunity_performance_count": sum(row["no_opportunity"] for row in perfs),
            "same_bass_transition_count": len(same), "different_bass_transition_count": len(different),
            "keyoff_at_pedal_off_count": sum(row["keyoff_at_pedal_off_count"] for row in perfs),
            "no_subsequent_pedal_off_count": sum(row["no_subsequent_pedal_off_count"] for row in perfs),
            "C_r_mean": stats["mean"], "C_r_median": stats["median"], "C_r_std": stats["std"],
            "same_bass_C_r_mean": same_stats["mean"], "same_bass_C_r_median": same_stats["median"], "same_bass_C_r_std": same_stats["std"],
            "different_bass_C_r_mean": different_stats["mean"], "different_bass_C_r_median": different_stats["median"], "different_bass_C_r_std": different_stats["std"],
            "micro_transition_weighted_mean": stats["mean"],
            "scored_piece_count": len(macro_values), "piece_balanced_C_Bass": mean(macro_values),
        })
    return piece_rows, system_rows


def audit_invariants(input_rows: list[dict[str, Any]], signatures: Mapping[tuple[str, str], Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in input_rows:
        if item["status"] != "available" or item["system"] == "HUMAN":
            continue
        reference = signatures[("ORIGINAL_PT", item["piece_id"])]
        candidate = signatures[(item["system"], item["piece_id"])]
        non_cc = candidate["non_cc64_signature_sha256"] == reference["non_cc64_signature_sha256"]
        structural = candidate["structural_bass_signature_sha256"] == reference["structural_bass_signature_sha256"]
        opportunity = candidate["opportunity_signature_sha256"] == reference["opportunity_signature_sha256"]
        rows.append({
            "piece_id": item["piece_id"], "system": item["system"],
            "reference_system": "ORIGINAL_PT", "non_cc64_events_exact": non_cc,
            "structural_bass_set_exact": structural, "valid_transition_set_exact": opportunity,
            "passed": non_cc and structural and opportunity,
            **candidate,
        })
    if not all(row["passed"] for row in rows):
        failures = [(row["piece_id"], row["system"]) for row in rows if not row["passed"]]
        raise RuntimeError(f"cross-system invariant failure; comparison aborted: {failures}")
    return rows


def f6(value: Any) -> str:
    return "NA" if value is None or value == "" else f"{float(value):.6f}"


def build_report(input_rows: list[dict[str, Any]], tests: list[dict[str, Any]], performance_rows: list[dict[str, Any]], piece_rows: list[dict[str, Any]], system_rows: list[dict[str, Any]], transitions: list[dict[str, Any]], invariants: list[dict[str, Any]]) -> str:
    by_system = {row["system"]: row for row in system_rows}
    missing = [row for row in input_rows if row["status"] == "missing"]
    anomalies = {
        "missing_keyoff_pairs": sum(row["excluded_missing_keyoff_count"] for row in performance_rows),
        "duplicate_lowest_groups": sum(row["duplicate_lowest_structural_group_count"] for row in performance_rows),
        "cc120_events": sum(row.get("parser_cc120_events", 0) for row in performance_rows),
        "cc123_events": sum(row.get("parser_cc123_events", 0) for row in performance_rows),
        "unmatched_note_off": sum(row.get("parser_unmatched_note_off", 0) for row in performance_rows),
    }
    comparable = ["HUMAN", "ORIGINAL_PT", "STANDARD_CE_ARGMAX", "STANDARD_CE_POSTERIOR_MEDIAN", "WEIGHTED_CE_ARGMAX"]
    ordering = sorted(comparable, key=lambda system: by_system[system]["piece_balanced_C_Bass"], reverse=True)
    ordering_text = " > ".join(f"{SYSTEM_LABEL[system]} ({f6(by_system[system]['piece_balanced_C_Bass'])})" for system in ordering)
    lines = [
        "# Bass Connectivity v0 — ASAP Validation Evaluation",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Scope and provenance",
        "",
        "This is a measurement-only sanity evaluation over pre-existing ASAP **validation** MIDI. Neural inference, training, checkpoint loading/execution, audio rendering, parameter search, and GPU/CUDA use were all zero. ASAP test MIDI access was zero. No MIDI was generated or modified.",
        "",
        f"- Human: 71 existing validation performances across 19 pieces from `{SPLIT_MANIFEST}`.",
        f"- Original PT: 19 frozen canonical Stage-1 MIDI files from `{STAGE1_MANIFEST}`.",
        f"- Canonical 4-class systems recovered from the Harmonic Muddiness system inventory and its source reports: STANDARD_CE_ARGMAX, STANDARD_CE_POSTERIOR_MEDIAN, and WEIGHTED_CE_ARGMAX (19 existing pieces each). Source roots: `{DECODING_ROOT}` and `{WEIGHTED_ROOT}`.",
        f"- Existing Harmonic Muddiness extremes from `{CONTROL_MANIFEST}`: NO_PEDAL and ALWAYS_ON, 13 pieces each. The six unavailable pieces per control remain `missing`; no controls were synthesized.",
        "- Excluded deliberately: HYBRID_REGRESSION_ONLY (regression) and CUSTOM_EVENT_V0 (noncanonical custom-event system). The Harmonic Muddiness report explicitly excludes CUSTOM_EVENT_V0 from its canonical Stage2 median, and the request forbids mixing regression systems.",
        "",
        "## Exact fixed metric",
        "",
        f"Non-drum positive-velocity note attacks are sorted in merged MIDI event order and grouped by an inclusive earliest-anchor window of {1000*GROUP_WINDOW_SECONDS:.0f} ms (not single linkage). Each group uses its earliest time and lowest newly attacked pitch `p_n`. Held/sustained notes do not enter detection.",
        "",
        "`L_n=1` for `p_n<=52`, `(57-p_n)/5` for `52<p_n<57`, and `0` for `p_n>=57`. The next-low attack is the first later group with `p<=57`. The local timescale is the median positive grouped IOI in the existing ±8-group convention. `R_B=Delta_B/local_IOI`, `S_n=clip((R_B-1.5)/1.5,0,1)`, and `O_oct=1` only when `p_n+12` is in the same group. `B_n=clip(L_n*S_n*(1+0.2*O_oct),0,1)`; Structural Bass iff `B_n>=0.5`. Terminal next-low and insufficient-timescale values remain missing and are never replaced by sentinels or detected.",
        "",
        "Only consecutive Structural Bass pairs are considered. The final bass is excluded. A pair is a valid pedal opportunity only when physical `keyoff < next onset`; otherwise it is counted separately. At key-off, channel-specific `CC64>=64` is ON. Same-timestamp messages follow the Harmonic Muddiness merged sequential order and same-pitch notes use FIFO instance pairing. If pedal is OFF at key-off, `C_r=0`. Otherwise the first later OFF event is used. For different pitches the Gaussian has `sigma_L=d/2`, `sigma_R=d/4`; no later OFF gives 0. For exactly equal pitches, release at/after the next onset (including no later OFF) gives 1; early release uses the left Gaussian. No octave-equivalent exception and no decay/velocity weighting are used.",
        "",
        "Performance scores average valid `C_r`; no-opportunity performances are NA. Human performances are averaged within piece, then all scored pieces receive equal weight in the primary system macro. The pooled transition-weighted mean is diagnostic only. Reported standard deviations are population standard deviations.",
        "",
        "## Required synthetic logic tests",
        "",
        "| Test | Expected | Actual | Result |",
        "|---|---:|---:|---|",
    ]
    lines.extend(f"| {row['test']} | {row['expected']} | {row['actual']} | {'PASS' if row['passed'] else 'FAIL'} |" for row in tests)
    lines += [
        "",
        "All 8 required tests passed.",
        "",
        "## Main system comparison",
        "",
        "Primary score is the piece-balanced macro mean.",
        "",
        "| System | Perf. | Pieces | Structural bass | Valid | Excluded key-held | No opp. perf. | C mean | C median | C std | Micro | Piece macro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = by_system[system]
        lines.append(f"| {row['system_label']} | {row['number_of_performances']} | {row['number_of_pieces']} | {row['detected_structural_bass_count']} | {row['valid_bass_transition_count']} | {row['excluded_keyoff_at_or_after_next_count']} | {row['no_opportunity_performance_count']} | {f6(row['C_r_mean'])} | {f6(row['C_r_median'])} | {f6(row['C_r_std'])} | {f6(row['micro_transition_weighted_mean'])} | **{f6(row['piece_balanced_C_Bass'])}** |")
    lines += [
        "",
        "## Same-bass versus different-bass",
        "",
        "| System | Same n | Same mean | Same median | Different n | Different mean | Different median | Pedal OFF at key-off | No later pedal OFF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = by_system[system]
        lines.append(f"| {row['system_label']} | {row['same_bass_transition_count']} | {f6(row['same_bass_C_r_mean'])} | {f6(row['same_bass_C_r_median'])} | {row['different_bass_transition_count']} | {f6(row['different_bass_C_r_mean'])} | {f6(row['different_bass_C_r_median'])} | {row['keyoff_at_pedal_off_count']} | {row['no_subsequent_pedal_off_count']} |")

    lines += ["", "## Per-piece comparison", ""]
    piece_lookup = {(row["piece_id"], row["system"]): row for row in piece_rows}
    pieces = sorted({row["piece_id"] for row in piece_rows})
    lines.append("| Piece | " + " | ".join(SYSTEM_LABEL[s] for s in SYSTEMS) + " |")
    lines.append("|---|" + "---:|" * len(SYSTEMS))
    for piece in pieces:
        meta = next(row for row in piece_rows if row["piece_id"] == piece)
        values = [f6(piece_lookup.get((piece, system), {}).get("C_Bass_piece")) for system in SYSTEMS]
        lines.append(f"| {meta['composer']} — {meta['title']} (`{piece}`) | " + " | ".join(values) + " |")

    invariant_fail = [row for row in invariants if not row["passed"]]
    lines += [
        "",
        "## Cross-system invariants",
        "",
        f"All `{len(invariants)}` available canonical-variant comparisons passed strict non-CC64 event equality, exact Structural Bass set equality, and exact valid-opportunity set equality. Failures: `{len(invariant_fail)}`. HUMAN is intentionally excluded because it is a distinct performance.",
        "",
        "## Missing systems and anomalies",
        "",
        f"There are `{len(missing)}` expected-but-missing inputs: six NO_PEDAL and six ALWAYS_ON piece variants. No primary HUMAN/ORIGINAL_PT/4-class MIDI is missing.",
        "",
    ]
    for system in ("NO_PEDAL", "ALWAYS_ON"):
        ids = [row["piece_id"] for row in missing if row["system"] == system]
        lines.append(f"- {system}: {', '.join(f'`{item}`' for item in ids)}")
    lines += [
        f"- Structural-pair missing-keyoff exclusions: `{anomalies['missing_keyoff_pairs']}`.",
        f"- Detected groups with duplicate lowest-pitch attacks (deterministic earliest merged-event representative used): `{anomalies['duplicate_lowest_groups']}`.",
        f"- Parser diagnostics: CC120=`{anomalies['cc120_events']}`, CC123=`{anomalies['cc123_events']}`, unmatched note-off=`{anomalies['unmatched_note_off']}`. These are exposed in `performance_scores.csv` rather than hidden.",
        "",
        "## Measurement-only sanity interpretation",
        "",
    ]
    off = by_system["NO_PEDAL"]
    on = by_system["ALWAYS_ON"]
    lines += [
        f"- NO_PEDAL piece macro is `{f6(off['piece_balanced_C_Bass'])}` (micro `{f6(off['micro_transition_weighted_mean'])}`); its key-off-at-pedal-OFF count is `{off['keyoff_at_pedal_off_count']}` of `{off['valid_bass_transition_count']}` valid opportunities.",
        f"- ALWAYS_ON same-bass mean is `{f6(on['same_bass_C_r_mean'])}` and different-bass mean is `{f6(on['different_bass_C_r_mean'])}`, directly testing the fixed no-OFF exception/asymmetry.",
        f"- On the common 19-piece universe, the measured primary ordering is: {ordering_text}.",
        "- ALWAYS_ON is based on only the 13 pieces with existing control MIDI, so its macro is not presented as a common-19-piece rank against the main systems.",
        "- No coefficient, threshold, event handling, or post-processing was changed after seeing this ordering.",
        "",
        "## Output files and reproducibility",
        "",
        "- `input_manifest.csv`: every expected input, including missing controls and frozen hashes.",
        "- `structural_bass_events.csv`: all detected fixed-v0 Structural Bass events.",
        "- `transition_diagnostics.csv`: every consecutive pair, including exclusions and score reasons.",
        "- `performance_scores.csv`, `piece_scores.csv`, `system_summary.csv`: the three aggregation levels.",
        "- `cross_system_invariants.csv`, `sanity_tests.csv`: identity and required logic audits.",
        "",
        "Tested command (inside the project container, with CUDA hidden):",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES='' python scripts/evaluate_bass_connectivity_v0_validation.py",
        "CUDA_VISIBLE_DEVICES='' python -c \"import runpy; d=runpy.run_path('tests/test_bass_connectivity_v0.py'); [d[n]() for n in sorted(d) if n.startswith('test_')]\"",
        "```",
        "",
        f"Run totals: existing MIDI opened=`{sum(row['status']=='available' for row in input_rows)}`, ASAP validation human MIDI opened=`{sum(row['system']=='HUMAN' and row['used'] for row in input_rows)}`, ASAP test MIDI opened=`0`, inference=`0`, training=`0`, GPU/CUDA=`0`, generated MIDI=`0`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    tests = required_logic_tests()
    inputs, _ = build_input_manifest()
    available = [row for row in inputs if row["status"] == "available"]
    performance_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    signatures: dict[tuple[str, str], dict[str, str]] = {}
    for index, row in enumerate(available, 1):
        print(f"EVAL {index}/{len(available)} {row['system']} {row['performance_id']}", flush=True)
        performance, transitions, structural, signature = evaluate_performance(row)
        performance_rows.append(performance)
        transition_rows.extend(transitions)
        structural_rows.extend(structural)
        if row["system"] != "HUMAN":
            signatures[(row["system"], row["piece_id"])] = signature

    invariants = audit_invariants(inputs, signatures)
    piece_rows, system_rows = aggregate(performance_rows, transition_rows, inputs)
    write_csv(output / "input_manifest.csv", inputs, ("system", "piece_id", "performance_id", "status", "used", "midi_path"))
    write_csv(output / "sanity_tests.csv", tests, ("test", "expected", "actual", "passed"))
    write_csv(output / "structural_bass_events.csv", structural_rows, ("system", "piece_id", "performance_id", "structural_bass_index", "onset_group_index", "onset_time_sec", "lowest_pitch", "R_B", "B_n"))
    write_csv(output / "transition_diagnostics.csv", transition_rows, ("system", "piece_id", "performance_id", "pair_index", "current_bass_pitch", "next_bass_pitch", "status", "is_valid_opportunity", "C_r"))
    write_csv(output / "performance_scores.csv", performance_rows, ("system", "piece_id", "performance_id", "structural_bass_count", "valid_transition_count", "C_Bass_performance"))
    write_csv(output / "piece_scores.csv", piece_rows, ("system", "piece_id", "performance_count", "C_Bass_piece"))
    write_csv(output / "system_summary.csv", system_rows, ("system", "number_of_performances", "number_of_pieces", "detected_structural_bass_count", "valid_bass_transition_count", "piece_balanced_C_Bass"))
    write_csv(output / "cross_system_invariants.csv", invariants, ("piece_id", "system", "reference_system", "non_cc64_events_exact", "structural_bass_set_exact", "valid_transition_set_exact", "passed"))
    report = build_report(inputs, tests, performance_rows, piece_rows, system_rows, transition_rows, invariants)
    (output / "BASS_CONNECTIVITY_V0_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "complete": True, "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_only": True, "asap_test_midi_access_count": 0,
        "neural_inference_count": 0, "training_count": 0, "checkpoint_load_count": 0,
        "gpu_cuda_use_count": 0, "generated_midi_count": 0, "parameter_search_count": 0,
        "existing_midi_used": len(available), "missing_expected_inputs": len(inputs) - len(available),
        "required_logic_tests_passed": all(row["passed"] for row in tests),
        "cross_system_invariants_passed": all(row["passed"] for row in invariants),
        "fixed_parameters": {
            "group_window_seconds": GROUP_WINDOW_SECONDS, "local_ioi_radius": LOCAL_IOI_RADIUS,
            "next_low_max_pitch": NEXT_LOW_MAX_PITCH, "R_low": R_LOW, "R_high": R_HIGH,
            "beta_octave": BETA_OCTAVE, "structural_bass_threshold": STRUCTURAL_BASS_THRESHOLD,
        },
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"COMPLETE output={output} performances={len(performance_rows)} transitions={len(transition_rows)}", flush=True)


if __name__ == "__main__":
    main()
