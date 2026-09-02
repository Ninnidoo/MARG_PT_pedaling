#!/usr/bin/env python3
"""Finalize Stage 2 binary manifests and audit official PT pedal targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from miditoolkit import MidiFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from third_party.PianistTransformer.src.model.pianoformer import PianoT5GemmaConfig
from third_party.PianistTransformer.src.utils.midi import midi_to_ids


DATASET_ORDER = ("MAESTRO-clean", "ASAP train", "ASAP validation")
SLOT_NAMES = ("Pedal1", "Pedal2", "Pedal3", "Pedal4")
PEDAL_OFFSET = 5261
BINARY_THRESHOLD = 64
JOINT_WEIGHTS = np.asarray([8, 4, 2, 1], dtype=np.int64)
EXPECTED_SPLIT_SHA256 = "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b"


@dataclass
class AuditAccumulator:
    expected_performances: int
    success_count: int = 0
    total_notes: int = 0
    raw_class_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    binary_counts: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int64))
    slot_binary_counts: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.int64))
    joint_counts: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.int64))


def read_csv(path: Path, encoding: str = "utf-8") -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        stat = path.stat()
        record = f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8", errors="surrogateescape"))
        count += 1
    return count, digest.hexdigest()


def binary_and_joint(raw_pedals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw_pedals.ndim != 2 or raw_pedals.shape[1] != 4:
        raise AssertionError(f"pedal array must be [notes,4], got {raw_pedals.shape}")
    if raw_pedals.size and (raw_pedals.min() < 0 or raw_pedals.max() > 127):
        raise AssertionError("raw pedal value outside [0,127]")
    binary = (raw_pedals >= BINARY_THRESHOLD).astype(np.int64)
    if not np.all((binary == 0) | (binary == 1)):
        raise AssertionError("binary target outside {0,1}")
    joint = binary @ JOINT_WEIGHTS
    if joint.size and (joint.min() < 0 or joint.max() > 15):
        raise AssertionError("joint target outside [0,15]")
    decoded = np.column_stack(tuple((joint >> shift) & 1 for shift in (3, 2, 1, 0)))
    if not np.array_equal(binary, decoded):
        raise AssertionError("binary -> joint -> binary round trip failed")
    return binary, joint


def boundary_unit_test() -> None:
    boundary = np.asarray([[63, 64, 63, 64]], dtype=np.int64)
    binary, joint = binary_and_joint(boundary)
    expected = np.asarray([[0, 1, 0, 1]], dtype=np.int64)
    if not np.array_equal(binary, expected) or int(joint[0]) != 5:
        raise AssertionError("boundary test failed: 63 must be OFF and 64 must be ON")
    all_patterns = np.asarray(
        [[(joint_id >> shift) & 1 for shift in (3, 2, 1, 0)] for joint_id in range(16)],
        dtype=np.int64,
    )
    reconstructed = all_patterns @ JOINT_WEIGHTS
    if not np.array_equal(reconstructed, np.arange(16, dtype=np.int64)):
        raise AssertionError("exhaustive 16-pattern round trip failed")


def js_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    p = first.astype(np.float64) / int(first.sum())
    q = second.astype(np.float64) / int(second.sum())
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    divergence = max(0.0, 0.5 * (kl(p) + kl(q)))
    return {
        "js_divergence_base2": divergence,
        "js_distance_base2": math.sqrt(divergence),
        "intersection_area": float(np.minimum(p, q).sum()),
    }


def dataset_entries(
    maestro_rows: list[dict[str, str]],
    asap_train_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    asap_root: Path,
) -> dict[str, list[dict[str, str]]]:
    maestro_entries = [{
        "source": "MAESTRO-clean",
        "performance_path": row["midi_filename"],
        "absolute_path": row["midi_path"],
        "expected_notes": "",
    } for row in maestro_rows]
    train_entries = [{
        "source": "ASAP-train",
        "performance_path": row["performance_path"],
        "absolute_path": row["performance_absolute_path"],
        "expected_notes": row["num_normalized_notes"],
    } for row in asap_train_rows]
    validation_entries = [{
        "source": "ASAP-validation",
        "performance_path": row["performance_path"],
        "absolute_path": str(asap_root / row["performance_path"]),
        "expected_notes": row["num_normalized_notes"],
    } for row in split_rows if row["split"] == "validation"]
    return {
        "MAESTRO-clean": maestro_entries,
        "ASAP train": train_entries,
        "ASAP validation": validation_entries,
    }


def tokenize_audit(
    groups: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, AuditAccumulator], list[dict[str, object]], list[dict[str, object]]]:
    config = PianoT5GemmaConfig()
    if config.pedal_start != PEDAL_OFFSET:
        raise RuntimeError(f"official pedal_start changed: {config.pedal_start}")
    if list(config.valid_id_range[4:]) != [(5261, 5389)] * 4:
        raise RuntimeError("official Pedal1-4 vocabulary ranges changed")
    failures: list[dict[str, object]] = []
    sparse: list[dict[str, object]] = []
    accumulators = {
        name: AuditAccumulator(expected_performances=len(entries))
        for name, entries in groups.items()
    }
    started = time.monotonic()
    completed = 0
    total_expected = sum(len(entries) for entries in groups.values())
    for dataset in DATASET_ORDER:
        entries = groups[dataset]
        accumulator = accumulators[dataset]
        print(f"dataset_start={dataset} performances={len(entries)}", flush=True)
        for index, entry in enumerate(entries, start=1):
            path = Path(entry["absolute_path"])
            raw_cc64_events: int | None = None
            try:
                midi = MidiFile(str(path))
                raw_cc64_events = sum(
                    1
                    for instrument in midi.instruments
                    if not instrument.is_drum
                    for event in instrument.control_changes
                    if event.number == 64
                )
                ids = midi_to_ids(config, midi)
            except Exception as error:  # dataset failures are audited, not hidden
                failures.append({
                    "dataset": dataset,
                    "source": entry["source"],
                    "performance_path": entry["performance_path"],
                    "performance_absolute_path": str(path),
                    "reason": "midi_load_or_official_tokenization_failure",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "raw_cc64_events_if_available": "" if raw_cc64_events is None else raw_cc64_events,
                })
                completed += 1
                continue
            tokens = np.asarray(ids, dtype=np.int64)
            if tokens.size == 0:
                failures.append({
                    "dataset": dataset,
                    "source": entry["source"],
                    "performance_path": entry["performance_path"],
                    "performance_absolute_path": str(path),
                    "reason": "zero_tokenized_notes",
                    "error_type": "EmptyTokenSequence",
                    "error_message": "official midi_to_ids returned no notes",
                    "raw_cc64_events_if_available": raw_cc64_events,
                })
                completed += 1
                continue
            if tokens.size % 8:
                raise RuntimeError(f"official token length not divisible by 8: {path}")
            notes = tokens.reshape(-1, 8)
            expected_notes = entry["expected_notes"]
            if expected_notes and len(notes) != int(expected_notes):
                raise RuntimeError(
                    f"stored ASAP note count differs after official tokenization: {path}: "
                    f"{len(notes)} != {expected_notes}"
                )
            raw_pedals = notes[:, 4:8] - config.pedal_start
            binary, joint = binary_and_joint(raw_pedals)
            targets = int(raw_pedals.size)
            accumulator.success_count += 1
            accumulator.total_notes += len(notes)
            accumulator.raw_class_counts += np.asarray([
                np.count_nonzero(raw_pedals == 0),
                np.count_nonzero((raw_pedals >= 1) & (raw_pedals <= 126)),
                np.count_nonzero(raw_pedals == 127),
            ], dtype=np.int64)
            accumulator.binary_counts += np.bincount(binary.reshape(-1), minlength=2)
            for slot in range(4):
                accumulator.slot_binary_counts[slot] += np.bincount(binary[:, slot], minlength=2)
            accumulator.joint_counts += np.bincount(joint, minlength=16)
            if int(accumulator.raw_class_counts.sum()) != accumulator.total_notes * 4:
                raise AssertionError("raw target accounting mismatch")
            if int(accumulator.binary_counts.sum()) != accumulator.total_notes * 4:
                raise AssertionError("binary target accounting mismatch")
            if int(accumulator.joint_counts.sum()) != accumulator.total_notes:
                raise AssertionError("joint target accounting mismatch")
            sparse_reason = ""
            if raw_cc64_events == 0:
                sparse_reason = "no_raw_cc64_events"
            elif raw_cc64_events <= 4:
                sparse_reason = "sparse_raw_cc64_events_1_to_4"
            if sparse_reason:
                sparse.append({
                    "dataset": dataset,
                    "source": entry["source"],
                    "performance_path": entry["performance_path"],
                    "performance_absolute_path": str(path),
                    "reason": sparse_reason,
                    "raw_cc64_event_count": raw_cc64_events,
                    "tokenized_note_count": len(notes),
                    "total_pedal_targets": targets,
                    "tokenized_off_count": int(np.count_nonzero(binary == 0)),
                    "tokenized_on_count": int(np.count_nonzero(binary == 1)),
                    "unique_sampled_pedal_values": len(np.unique(raw_pedals)),
                    "excluded_from_manifest": "False",
                })
            completed += 1
            if index % 25 == 0 or index == len(entries):
                print(
                    f"progress dataset={dataset} {index}/{len(entries)} "
                    f"global={completed}/{total_expected} elapsed_s={time.monotonic()-started:.1f}",
                    flush=True,
                )
    return accumulators, failures, sparse


def distribution_rows(
    accumulators: dict[str, AuditAccumulator],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    binary_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    raw_labels = ("zero_0", "intermediate_1_126", "full_127")
    for dataset in DATASET_ORDER:
        accumulator = accumulators[dataset]
        target_total = accumulator.total_notes * 4
        for label, count in zip(raw_labels, accumulator.raw_class_counts):
            binary_rows.append({
                "dataset": dataset,
                "distribution": "raw_pedal_value_class",
                "slot": "All",
                "label": label,
                "count": int(count),
                "ratio": float(count / target_total),
                "denominator": target_total,
            })
        for slot, counts in [("All", accumulator.binary_counts), *zip(SLOT_NAMES, accumulator.slot_binary_counts)]:
            denominator = int(counts.sum())
            for label, count in zip(("OFF", "ON"), counts):
                binary_rows.append({
                    "dataset": dataset,
                    "distribution": "binary_threshold_64",
                    "slot": slot,
                    "label": label,
                    "count": int(count),
                    "ratio": float(count / denominator),
                    "denominator": denominator,
                })
        for joint_id, count in enumerate(accumulator.joint_counts):
            joint_rows.append({
                "dataset": dataset,
                "joint_id": joint_id,
                "pattern": f"{joint_id:04b}",
                "count": int(count),
                "ratio": float(count / accumulator.total_notes),
                "denominator_notes": accumulator.total_notes,
            })
    return binary_rows, joint_rows


def pct(count: int, total: int) -> str:
    return f"{count / total:.6%}" if total else "n/a"


def render_report(
    *,
    args: argparse.Namespace,
    hashes: dict[str, str],
    ambiguous_filenames: list[str],
    confirmed_filenames: set[str],
    final_maestro_rows: list[dict[str, str]],
    asap_train_rows: list[dict[str, str]],
    final_train_rows: list[dict[str, str]],
    accumulators: dict[str, AuditAccumulator],
    failures: list[dict[str, object]],
    sparse: list[dict[str, object]],
    similarity: dict[str, float],
    integrity: dict[str, bool],
    inventories: dict[str, tuple[int, str]],
) -> str:
    ambiguous_lines = "\n".join(f"  - `{name}`" for name in ambiguous_filenames)
    summary_lines = []
    binary_lines = []
    sparse_counts = Counter(row["dataset"] for row in sparse)
    failure_counts = Counter(row["dataset"] for row in failures)
    for dataset in DATASET_ORDER:
        accumulator = accumulators[dataset]
        total_targets = accumulator.total_notes * 4
        summary_lines.append(
            f"| {dataset} | {accumulator.expected_performances:,} | {accumulator.success_count:,} | "
            f"{failure_counts[dataset]:,} | {accumulator.total_notes:,} | {total_targets:,} |"
        )
        binary_lines.append(
            f"| {dataset} | {int(accumulator.binary_counts[0]):,} ({pct(int(accumulator.binary_counts[0]), total_targets)}) | "
            f"{int(accumulator.binary_counts[1]):,} ({pct(int(accumulator.binary_counts[1]), total_targets)}) |"
        )
    slot_lines = []
    for dataset in DATASET_ORDER:
        accumulator = accumulators[dataset]
        for slot_index, slot in enumerate(SLOT_NAMES):
            counts = accumulator.slot_binary_counts[slot_index]
            total = int(counts.sum())
            slot_lines.append(
                f"| {dataset} | {slot} | {int(counts[0]):,} ({pct(int(counts[0]), total)}) | "
                f"{int(counts[1]):,} ({pct(int(counts[1]), total)}) |"
            )
    pattern_lines = []
    for joint_id in range(16):
        cells = [f"| {joint_id} | `{joint_id:04b}`"]
        for dataset in DATASET_ORDER:
            accumulator = accumulators[dataset]
            count = int(accumulator.joint_counts[joint_id])
            cells.append(f"{count:,} ({pct(count, accumulator.total_notes)})")
        pattern_lines.append(" | ".join(cells) + " |")
    missing_lines = []
    for dataset in DATASET_ORDER:
        rows = [row for row in sparse if row["dataset"] == dataset]
        no_cc = sum(row["reason"] == "no_raw_cc64_events" for row in rows)
        low_cc = sum(row["reason"] == "sparse_raw_cc64_events_1_to_4" for row in rows)
        missing_lines.append(f"| {dataset} | {no_cc} | {low_cc} | {len(rows)} |")
    failure_detail = "\n".join(
        f"- `{row['dataset']}` / `{row['performance_path']}`: {row['error_type']}: {row['error_message']}"
        for row in failures
    ) or "- 0건. 모든 selected performance가 성공적으로 tokenization되었다."
    duplicate_rows = read_csv(args.v0_dir / "asap_train_maestro_duplicates.csv")[1]
    duplicate_sources = {row["direct_maestro_midi_filename"] for row in duplicate_rows}
    excerpt_count = sum(row["asap_is_source_excerpt"] == "True" for row in duplicate_rows)
    return f"""# Stage 2 binary v1: final dataset and target-distribution audit

## 범위

- 이 단계는 final manifest 확정과 target 통계 audit만 수행했다. 모델/head/loss/weighting/balancing/window/cache/training/Stage 1 inference는 수행하지 않았다.
- tokenization 대상: final MAESTRO-clean, ASAP train, ASAP validation.
- **ASAP test MIDI는 열거나 tokenization하지 않았고 distribution/metric 계산에도 사용하지 않았다.** 저장 split은 contamination membership 확인에만 읽었다.

## Manifest 확정

- v0 source: `{args.v0_dir}`
- v0 MAESTRO-clean: 1,174 performances.
- v0 ambiguous audit 6 relationships의 distinct MAESTRO MIDI 4개를 conservative leakage prevention으로 모두 추가 제외했다:
{ambiguous_lines}
- confirmed overlap {len(confirmed_filenames)}개와 위 ambiguous 4개는 final MAESTRO manifest에 0개 남았다.
- final MAESTRO-clean: **{len(final_maestro_rows):,} performances**
- ASAP train: **{len(asap_train_rows):,} performances**
- final training pool (naive concatenation): **{len(final_train_rows):,} performances**
- ASAP-train ↔ MAESTRO direct-source duplicates는 정책대로 제거하지 않았다: v0 audit 기준 **{len(duplicate_rows):,} ASAP rows / {len(duplicate_sources):,} distinct MAESTRO source filenames**, source excerpt {excerpt_count:,} rows.

## Official tokenizer provenance

- Pianist Transformer commit: `747df2d12291e37f6638b39f1b71517e579ad48c`
- implementation: `third_party/PianistTransformer/src/utils/midi.py::normalize_midi` 및 `midi_to_ids`
- config: `third_party/PianistTransformer/src/model/pianoformer.py::PianoT5GemmaConfig`
- tokenizer SHA-256: `{hashes['tokenizer']}`
- 새 pedal sampling logic은 구현하지 않았다. official `midi_to_ids`가 생성한 8-token note `[Pitch, IOI, Velocity, Duration, Pedal1, Pedal2, Pedal3, Pedal4]`에서 Pedal token offset 5,261만 제거했다.
- official sampling: Pedal1=note onset, Pedal2–4=다음-note IOI의 1/4, 2/4, 3/4 위치에서 직전 CC64 value를 샘플한다.

## Target 정의와 unit assertions

- binary threshold: **64** (official PT evaluator와 동일)
- `raw < 64 -> OFF=0`; `raw >= 64 -> ON=1`
- independent target: `[P1,P2,P3,P4]`, 각 bit는 `{{0,1}}`
- joint target: `joint_id = 8*P1 + 4*P2 + 2*P3 + P4`, 범위 `[0,15]`
- 모든 note에서 binary→joint→4-bit decode exact equality를 assertion했다.
- exhaustive 16-pattern round-trip과 boundary `63→OFF`, `64→ON` unit assertion을 통과했다.

## Tokenization accounting

| dataset | manifest performances | success | failed/skipped | total notes | Pedal1–4 targets |
| --- | ---: | ---: | ---: | ---: | ---: |
{os.linesep.join(summary_lines)}

Failure detail:
{failure_detail}

lightweight token cache는 생성하지 않았다. 전체 note/token array가 필요 없는 descriptive audit이므로 performance 하나씩 official tokenizer에 통과시킨 뒤 integer histogram만 누적하는 streaming 방식이 더 작고 명확하다.

## Raw pedal-value distribution

| dataset | raw=0 | intermediate 1–126 | raw=127 |
| --- | ---: | ---: | ---: |
""" + "\n".join(
        f"| {dataset} | " + " | ".join(
            f"{int(count):,} ({pct(int(count), accumulators[dataset].total_notes*4)})"
            for count in accumulators[dataset].raw_class_counts
        ) + " |"
        for dataset in DATASET_ORDER
    ) + f"""

## Binary OFF/ON distribution

| dataset | OFF | ON |
| --- | ---: | ---: |
{os.linesep.join(binary_lines)}

### Slot별 OFF/ON

| dataset | slot | OFF | ON |
| --- | --- | ---: | ---: |
{os.linesep.join(slot_lines)}

## Joint 16-pattern distribution

Pattern 순서는 `0000, 0001, 0010, ..., 1111`이며 P1이 most-significant bit이다.

| joint_id | pattern | MAESTRO-clean | ASAP train | ASAP validation |
| ---: | --- | ---: | ---: | ---: |
{os.linesep.join(pattern_lines)}

## MAESTRO-clean vs ASAP train descriptive similarity

- official-PT-style base-2 JS metric (SciPy `jensenshannon`과 같은 **distance**): **{similarity['js_distance_base2']:.12f}**
- base-2 JS divergence (distance², 명칭 모호성 방지용 병기): **{similarity['js_divergence_base2']:.12f}**
- Intersection Area `sum(min(p_i,q_i))`: **{similarity['intersection_area']:.12f}**
- descriptive audit 전용이다. 현재 naive concatenation 결정, dataset/class weighting, balancing, oversampling에는 사용하지 않았다.

## Pedal missing/sparse audit

`missing = raw non-drum CC64 event 0`, `sparse = 1–4 events`로 사전 정의해 audit했다. 임의 제외에는 사용하지 않았다.

| dataset | missing (0) | sparse (1–4) | total flagged |
| --- | ---: | ---: | ---: |
{os.linesep.join(missing_lines)}

상세 목록과 tokenized OFF/ON 수는 `pedal_missing_or_sparse.csv`에 있다.

## 필수 무결성 검증

| 조건 | 결과 | 근거 |
| --- | --- | --- |
| A. final train에 ASAP validation/test performance 0 | {'PASS' if integrity['A'] else 'FAIL'} | ASAP source는 저장 split train 892행과 exact equality; train/held-out path 교집합 0 |
| B. confirmed overlap + ambiguous 4가 final MAESTRO에 0 | {'PASS' if integrity['B'] else 'FAIL'} | confirmed {len(confirmed_filenames)} + ambiguous {len(ambiguous_filenames)} filename 교집합 0 |
| C. 모든 binary label이 `{{0,1}}` | {'PASS' if integrity['C'] else 'FAIL'} | tokenized note마다 assertion |
| D. 모든 joint label이 `[0,15]` | {'PASS' if integrity['D'] else 'FAIL'} | tokenized note마다 assertion |
| E. binary→joint→binary round-trip | {'PASS' if integrity['E'] else 'FAIL'} | 전 note 및 exhaustive 16-pattern assertion |
| F. boundary 63 OFF / 64 ON | {'PASS' if integrity['F'] else 'FAIL'} | explicit unit assertion, `[63,64,63,64] -> 0101 -> 5` |
| G. 원본 MIDI/metadata 미수정 | {'PASS' if integrity['G'] else 'FAIL'} | input hash와 dataset `(relative path,size,mtime_ns)` inventory 전/후 동일 |

추가 contamination 확인: ASAP train/validation/test piece_id는 split 간 교집합 0; ASAP held-out performance path와 final ASAP-train path 교집합 0; ASAP test tokenization count 0.

Dataset inventory after audit: ASAP {inventories['ASAP'][0]} files / `{inventories['ASAP'][1]}`; MAESTRO {inventories['MAESTRO'][0]} files / `{inventories['MAESTRO'][1]}`.

## Input hashes

- v0 maestro_clean_manifest: `{hashes['v0_maestro']}`
- v0 asap_train_manifest: `{hashes['v0_asap_train']}`
- v0 train_manifest: `{hashes['v0_train']}`
- v0 ambiguous_matches: `{hashes['v0_ambiguous']}`
- ASAP split: `{hashes['split']}`
- ASAP metadata: `{hashes['asap_metadata']}`
- MAESTRO metadata: `{hashes['maestro_metadata']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-dir", type=Path, default=Path("/workspace/project/analysis/stage2_binary_v0/data_prep_v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/project/analysis/stage2_binary_v0/data_prep_v1"))
    parser.add_argument("--split-csv", type=Path, default=Path("/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv"))
    parser.add_argument("--asap-root", type=Path, default=Path("/workspace/public/ASAP/asap-dataset-v1.1"))
    parser.add_argument("--maestro-root", type=Path, default=Path("/workspace/public/MAESTRO/maestro-v3.0.0"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing v1 output: {args.output_dir}")
    boundary_unit_test()

    paths = {
        "v0_maestro": args.v0_dir / "maestro_clean_manifest.csv",
        "v0_asap_train": args.v0_dir / "asap_train_manifest.csv",
        "v0_train": args.v0_dir / "train_manifest.csv",
        "v0_ambiguous": args.v0_dir / "ambiguous_matches.csv",
        "v0_overlap": args.v0_dir / "maestro_overlap_audit.csv",
        "split": args.split_csv,
        "asap_metadata": args.asap_root / "metadata.csv",
        "maestro_metadata": args.maestro_root / "maestro-v3.0.0.csv",
        "tokenizer": Path("/workspace/project/third_party/PianistTransformer/src/utils/midi.py"),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError([str(path) for path in paths.values() if not path.is_file()])
    hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    if hashes_before["split"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("ASAP split source changed")
    inventories_before = {"ASAP": inventory(args.asap_root), "MAESTRO": inventory(args.maestro_root)}

    maestro_fields, v0_maestro_rows = read_csv(paths["v0_maestro"])
    asap_train_fields, asap_train_rows = read_csv(paths["v0_asap_train"])
    train_fields, v0_train_rows = read_csv(paths["v0_train"])
    _, ambiguous_rows = read_csv(paths["v0_ambiguous"])
    _, overlap_rows = read_csv(paths["v0_overlap"])
    _, split_rows = read_csv(paths["split"])
    ambiguous_filenames = sorted({row["candidate_maestro_midi_filename"] for row in ambiguous_rows})
    if len(ambiguous_filenames) != 4:
        raise RuntimeError(f"expected 4 distinct ambiguous filenames, found {len(ambiguous_filenames)}")
    confirmed_filenames = {row["excluded_maestro_midi_filename"] for row in overlap_rows}
    final_maestro_rows = [row for row in v0_maestro_rows if row["midi_filename"] not in ambiguous_filenames]
    if len(v0_maestro_rows) != 1174 or len(final_maestro_rows) != 1170:
        raise RuntimeError("final MAESTRO accounting mismatch")
    final_maestro_names = {row["midi_filename"] for row in final_maestro_rows}
    final_train_rows = [
        row for row in v0_train_rows
        if not (row["source"] == "MAESTRO-clean" and row["performance_path"] in ambiguous_filenames)
    ]
    if len(asap_train_rows) != 892 or len(final_train_rows) != 2062:
        raise RuntimeError("final train accounting mismatch")
    if Counter(row["source"] for row in final_train_rows) != Counter({"MAESTRO-clean": 1170, "ASAP-train": 892}):
        raise RuntimeError("final train source accounting mismatch")
    if {row["performance_path"] for row in final_train_rows if row["source"] == "MAESTRO-clean"} != final_maestro_names:
        raise RuntimeError("final train MAESTRO rows differ from final MAESTRO manifest")

    splits_by_piece: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        splits_by_piece[row["piece_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in splits_by_piece.values()):
        raise RuntimeError("ASAP piece crosses split boundary")
    split_counts = Counter(row["split"] for row in split_rows)
    if split_counts != Counter({"train": 892, "validation": 71, "test": 104}):
        raise RuntimeError("ASAP performance split counts changed")
    asap_train_paths = {row["performance_path"] for row in asap_train_rows}
    stored_train_paths = {row["performance_path"] for row in split_rows if row["split"] == "train"}
    heldout_paths = {row["performance_path"] for row in split_rows if row["split"] != "train"}
    integrity_a = asap_train_paths == stored_train_paths and not (asap_train_paths & heldout_paths)
    integrity_b = not (final_maestro_names & (confirmed_filenames | set(ambiguous_filenames)))
    if not (integrity_a and integrity_b):
        raise RuntimeError(f"manifest integrity A/B failed: {integrity_a}/{integrity_b}")

    groups = dataset_entries(final_maestro_rows, asap_train_rows, split_rows, args.asap_root)
    if tuple(groups) != DATASET_ORDER or len(groups["ASAP validation"]) != 71:
        raise RuntimeError("audit group construction mismatch")
    accumulators, failures, sparse = tokenize_audit(groups)
    for name in DATASET_ORDER:
        accumulator = accumulators[name]
        if accumulator.success_count + sum(row["dataset"] == name for row in failures) != accumulator.expected_performances:
            raise AssertionError(f"success/failure accounting mismatch for {name}")
    similarity = js_metrics(
        accumulators["MAESTRO-clean"].joint_counts,
        accumulators["ASAP train"].joint_counts,
    )
    binary_rows, joint_rows = distribution_rows(accumulators)

    inventories_after = {"ASAP": inventory(args.asap_root), "MAESTRO": inventory(args.maestro_root)}
    hashes_after = {name: sha256_file(path) for name, path in paths.items()}
    integrity_g = inventories_before == inventories_after and hashes_before == hashes_after
    integrity = {"A": integrity_a, "B": integrity_b, "C": True, "D": True, "E": True, "F": True, "G": integrity_g}
    if not all(integrity.values()):
        raise RuntimeError(f"integrity failure: {integrity}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="data_prep_v1.", dir=args.output_dir.parent) as temporary:
        temp = Path(temporary)
        write_csv(temp / "maestro_clean_manifest.csv", maestro_fields, final_maestro_rows)
        write_csv(temp / "asap_train_manifest.csv", asap_train_fields, asap_train_rows)
        write_csv(temp / "train_manifest.csv", train_fields, final_train_rows)
        write_csv(temp / "binary_distribution_by_dataset.csv", [
            "dataset", "distribution", "slot", "label", "count", "ratio", "denominator",
        ], binary_rows)
        write_csv(temp / "joint16_distribution_by_dataset.csv", [
            "dataset", "joint_id", "pattern", "count", "ratio", "denominator_notes",
        ], joint_rows)
        write_csv(temp / "tokenization_failures.csv", [
            "dataset", "source", "performance_path", "performance_absolute_path", "reason",
            "error_type", "error_message", "raw_cc64_events_if_available",
        ], failures)
        write_csv(temp / "pedal_missing_or_sparse.csv", [
            "dataset", "source", "performance_path", "performance_absolute_path", "reason",
            "raw_cc64_event_count", "tokenized_note_count", "total_pedal_targets",
            "tokenized_off_count", "tokenized_on_count", "unique_sampled_pedal_values",
            "excluded_from_manifest",
        ], sparse)
        report = render_report(
            args=args, hashes=hashes_before, ambiguous_filenames=ambiguous_filenames,
            confirmed_filenames=confirmed_filenames, final_maestro_rows=final_maestro_rows,
            asap_train_rows=asap_train_rows, final_train_rows=final_train_rows,
            accumulators=accumulators, failures=failures, sparse=sparse,
            similarity=similarity, integrity=integrity, inventories=inventories_after,
        )
        (temp / "BINARY_DATASET_AUDIT.md").write_text(report, encoding="utf-8")
        for path in temp.iterdir():
            path.chmod(0o664)
        os.rename(temp, args.output_dir)
    print(f"wrote={args.output_dir}", flush=True)
    print(
        "summary " + " ".join(
            f"{name.replace(' ', '_')}={accumulators[name].success_count}/{accumulators[name].expected_performances}"
            for name in DATASET_ORDER
        ),
        flush=True,
    )
    print(
        f"notes MAESTRO={accumulators['MAESTRO-clean'].total_notes} "
        f"ASAP_train={accumulators['ASAP train'].total_notes} "
        f"ASAP_validation={accumulators['ASAP validation'].total_notes}",
        flush=True,
    )
    print(f"js_distance={similarity['js_distance_base2']:.12f} intersection={similarity['intersection_area']:.12f}", flush=True)
    print(f"failures={len(failures)} sparse={len(sparse)} integrity=A-G:PASS", flush=True)


if __name__ == "__main__":
    main()
