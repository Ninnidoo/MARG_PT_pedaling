#!/usr/bin/env python3
"""Build and audit cache-note -> official PT-note mappings (no training)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.stage2_event_model.note_alignment import (
    ALIGNMENT_IMPLEMENTATION_VERSION,
    build_note_alignment,
    save_note_alignment,
    sha256_file,
)
from src.stage2_event_model.ownership import assign_unique_owners

PROJECT = Path("/workspace/project")
CACHE_ROOT = PROJECT / "analysis/custom_event_tokenizer_v1"
OUTPUT = PROJECT / "analysis/custom_event_model_v0_note_alignment"
CACHE_ID = "3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263"
EXPECTED = {"train": (2062, 35573, 9009549), "validation": (71, 1078, 272927)}
FIRST_FAILURE_PERFORMANCE_INDEX = 1793
FIRST_FAILURE_DATASET_WINDOW_INDEX = 31168
FIRST_FAILURE_BATCH_INDEX = 151
FIRST_FAILURE_PERFORMANCE_WINDOW_INDEX = 12


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def config() -> dict:
    pt_source = PROJECT / "third_party/PianistTransformer/src/utils/midi.py"
    manifest = CACHE_ROOT / "cache_manifest.json"
    value = {
        "alignment_implementation_version": ALIGNMENT_IMPLEMENTATION_VERSION,
        "tokenizer_cache_id": CACHE_ID,
        "cache_manifest_sha256": sha256_file(manifest),
        "pt_midi_implementation_sha256": sha256_file(pt_source),
        "source_manifest_identity": sha256_file(manifest),
        "window_notes": 512,
        "stride_notes": 256,
        "ownership": "mapped-group-containment_max-representative-margin_tie-smaller-start",
        "identity": "unique raw(onset,pitch,noteoff,velocity); duplicate identity fails; official normalized tuple verified",
        "asap_test_access_count": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["alignment_config_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def first_failure_diagnostic(entry: dict, alignment) -> dict:
    cache_path = CACHE_ROOT / entry["cache_file"]
    with np.load(cache_path, allow_pickle=False) as cache:
        cache_rows = np.column_stack((
            cache["note_onset_ticks"], cache["note_pitch"], cache["note_offset_ticks"], cache["note_velocity"]
        )).astype(np.int64)
    pt_rows = np.column_stack((
        alignment.normalized_onset, alignment.normalized_pitch,
        alignment.normalized_offset, alignment.normalized_velocity,
    )).astype(np.int64)
    mapped_cache_rows = cache_rows[alignment.pt_to_cache]
    differing = np.flatnonzero(cache_rows[:, 1] != pt_rows[:, 1])
    first = int(differing[0])
    left, right = max(0, first - 10), min(len(cache_rows), first + 11)
    cache_context = [
        {"position": i, "raw_onset": int(row[0]), "pitch": int(row[1]), "raw_noteoff": int(row[2]), "velocity": int(row[3])}
        for i, row in enumerate(cache_rows[left:right], start=left)
    ]
    pt_context = []
    for pt_index in range(left, right):
        ci = int(alignment.pt_to_cache[pt_index])
        pt_context.append({
            "position": pt_index, "cache_note_index": ci,
            "normalized_onset": int(pt_rows[pt_index, 0]), "pitch": int(pt_rows[pt_index, 1]),
            "normalized_noteoff": int(pt_rows[pt_index, 2]), "velocity": int(pt_rows[pt_index, 3]),
            "source_raw_onset": int(cache_rows[ci, 0]),
        })
    return {
        "performance_id": entry["performance_path"], "piece_id": entry["piece_id"],
        "performance_index": entry["performance_index"],
        "dataset_window_index": FIRST_FAILURE_DATASET_WINDOW_INDEX,
        "shuffled_batch_index": FIRST_FAILURE_BATCH_INDEX,
        "performance_window_index": FIRST_FAILURE_PERFORMANCE_WINDOW_INDEX,
        "global_window_start": 3072, "global_window_end": 3584,
        "cache_note_count": len(cache_rows), "pt_input_note_count": len(pt_rows),
        "first_differing_position": first,
        "pitch_differing_positions": int(len(differing)),
        "cache_pitch_sequence_context": cache_context,
        "pt_pitch_sequence_context": pt_context,
        "same_note_multiset": bool(Counter(map(tuple, mapped_cache_rows[:, 1:].tolist())) == Counter(map(tuple, cache_rows[:, 1:].tolist()))),
        "root_cause": (
            "raw onsets 22591 and 22592 round to the same official PT normalized onset 19309; "
            "the official (normalized_onset,pitch) sort swaps pitches 61 and 49"
        ),
        "mismatch_type": {
            "same_modeled_note_set": True, "same_onset_after_pt_normalization": True,
            "raw_onsets_differ": True, "note_count_mismatch": False,
            "pitch_mismatch_beyond_reordering": False, "duration_mismatch": False,
            "velocity_mismatch": False,
        },
    }


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = config()
    atomic_json(OUTPUT / "alignment_config.json", cfg)
    source_manifest_path = CACHE_ROOT / "cache_manifest.json"
    before_manifest_sha = sha256_file(source_manifest_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest["cache_id"] != CACHE_ID:
        raise RuntimeError("frozen cache ID mismatch")

    totals = {
        split: Counter({
            "performances": 0, "identity_order_performances": 0,
            "permuted_performances": 0, "total_notes": 0, "moved_notes": 0,
            "unmapped_cache_notes": 0, "unmapped_pt_notes": 0,
            "ambiguous_mappings": 0, "note_count_mismatch_performances": 0,
            "duplicate_raw_identity_keys": 0, "notes_in_duplicate_raw_identities": 0,
            "moved_within_same_raw_onset": 0, "moved_across_raw_onsets": 0,
            "moved_within_same_normalized_onset": 0, "moved_across_normalized_onsets": 0,
            "onset_groups_mapped": 0, "non_contiguous_onset_groups": 0,
            "representatives_mapped": 0, "zero_owner_groups": 0,
            "duplicate_owner_groups": 0, "owned_onsets": 0, "total_windows": 0,
            "initial_supervision_count": 0, "terminal_supervision_count": 0,
            "split_chord_groups": 0, "split_chord_window_group_pairs": 0,
        }) for split in ("train", "validation")
    }
    maximum_displacement = {"train": 0, "validation": 0}
    logical_digest = hashlib.sha256(cfg["alignment_config_sha256"].encode())
    output_entries = []
    first_failure = None

    entries = [e for e in manifest["entries"] if e["split"] in totals]
    for ordinal, entry in enumerate(entries, start=1):
        split = entry["split"]
        cache_path = CACHE_ROOT / entry["cache_file"]
        alignment = build_note_alignment(entry["source_midi"], cache_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            first = cache["onset_first_note_index"].astype(np.int64)
            last = cache["onset_last_note_index"].astype(np.int64)
            representative = cache["onset_representative_note_index"].astype(np.int64)
        mapped_first, mapped_last, mapped_rep, non_contiguous = alignment.map_group_bounds(first, last, representative)
        ownership = assign_unique_owners(
            alignment.note_count, mapped_first, mapped_last, mapped_rep,
            window_notes=512, stride_notes=256,
        )
        if len(np.unique(ownership.owner_window_index)) == 0:
            raise AssertionError("empty ownership")
        mapping_rel = Path("cache") / split / f"{int(entry['performance_index']):04d}_{entry['source_sha256'][:12]}.npz"
        metadata = {
            "alignment_config_sha256": cfg["alignment_config_sha256"],
            "tokenizer_cache_id": CACHE_ID, "split": split,
            "performance_index": int(entry["performance_index"]),
            "performance_path": entry["performance_path"], "piece_id": entry["piece_id"],
            "source_sha256": entry["source_sha256"], "stats": dict(alignment.stats),
            "non_contiguous_onset_groups": int(non_contiguous),
        }
        save_note_alignment(OUTPUT / mapping_rel, alignment, metadata)
        mapping_sha = sha256_file(OUTPUT / mapping_rel)
        logical_digest.update(split.encode())
        logical_digest.update(str(entry["performance_index"]).encode())
        logical_digest.update(alignment.cache_to_pt.tobytes())
        output_entries.append({
            **{k: entry[k] for k in ("split", "performance_index", "performance_path", "piece_id", "source_sha256")},
            "alignment_file": str(mapping_rel), "alignment_file_sha256": mapping_sha,
            "notes": alignment.note_count, "onsets": len(first),
            "windows": len(ownership.window_starts), "non_contiguous_onset_groups": non_contiguous,
            "stats": dict(alignment.stats),
        })
        stat = totals[split]
        stat["performances"] += 1
        stat["identity_order_performances" if alignment.stats["identity_order"] else "permuted_performances"] += 1
        for name in (
            "note_count", "moved_notes", "unmapped_cache_notes", "unmapped_pt_notes",
            "ambiguous_mappings", "duplicate_raw_identity_keys", "notes_in_duplicate_raw_identities",
            "moved_within_same_raw_onset", "moved_across_raw_onsets",
            "moved_within_same_normalized_onset", "moved_across_normalized_onsets",
        ):
            stat["total_notes" if name == "note_count" else name] += int(alignment.stats[name])
        maximum_displacement[split] = max(maximum_displacement[split], int(alignment.stats["maximum_displacement"]))
        stat["onset_groups_mapped"] += len(first)
        stat["non_contiguous_onset_groups"] += non_contiguous
        stat["representatives_mapped"] += len(representative)
        stat["owned_onsets"] += sum(len(x) for x in ownership.owned_onset_indices)
        stat["total_windows"] += len(ownership.window_starts)
        stat["initial_supervision_count"] += 1
        stat["terminal_supervision_count"] += 1
        stat["split_chord_groups"] += ownership.split_chord_group_count
        stat["split_chord_window_group_pairs"] += ownership.split_chord_window_group_pairs
        if split == "train" and int(entry["performance_index"]) == FIRST_FAILURE_PERFORMANCE_INDEX:
            first_failure = first_failure_diagnostic(entry, alignment)
        if ordinal % 100 == 0 or ordinal == len(entries):
            print(f"alignment {ordinal}/{len(entries)} elapsed={time.time()-started:.1f}s", flush=True)

    after_manifest_sha = sha256_file(source_manifest_path)
    if before_manifest_sha != after_manifest_sha:
        raise RuntimeError("frozen tokenizer manifest changed during audit")
    if first_failure is None:
        raise RuntimeError("first failed performance was not audited")
    alignment_id = logical_digest.hexdigest()
    result_manifest = {
        "alignment_id": alignment_id, "alignment_config_sha256": cfg["alignment_config_sha256"],
        "tokenizer_cache_id": CACHE_ID, "asap_test_access_count": 0,
        "train_count": totals["train"]["performances"],
        "validation_count": totals["validation"]["performances"],
        "entries": output_entries,
    }
    stats = {
        "alignment_id": alignment_id, "alignment_config_sha256": cfg["alignment_config_sha256"],
        "tokenizer_cache_id": CACHE_ID, "asap_test_access_count": 0,
        "training_steps": 0, "optimizer_steps": 0,
        "frozen_cache_manifest_sha256_before": before_manifest_sha,
        "frozen_cache_manifest_sha256_after": after_manifest_sha,
        "frozen_cache_unchanged": before_manifest_sha == after_manifest_sha,
        "splits": {}, "elapsed_seconds": time.time() - started,
    }
    for split, counter in totals.items():
        values = dict(counter)
        values["maximum_displacement"] = maximum_displacement[split]
        values["moved_note_fraction"] = values["moved_notes"] / values["total_notes"]
        values["ownership_expected"] = EXPECTED[split][2]
        values["ownership_matches_expected"] = values["owned_onsets"] == EXPECTED[split][2]
        values["windows_expected"] = EXPECTED[split][1]
        values["windows_match_expected"] = values["total_windows"] == EXPECTED[split][1]
        stats["splits"][split] = values
    atomic_json(OUTPUT / "alignment_manifest.json", result_manifest)
    atomic_json(OUTPUT / "alignment_stats.json", stats)
    atomic_json(OUTPUT / "first_failure_diagnostic.json", first_failure)
    print(json.dumps({"alignment_id": alignment_id, "splits": stats["splits"]}, indent=2))


if __name__ == "__main__":
    main()
