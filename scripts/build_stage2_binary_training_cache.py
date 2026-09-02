#!/usr/bin/env python3
"""Build one persistent int16 cache shared by both binary architectures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_encoder_only.dataset import (  # noqa: E402
    TOKENS_PER_NOTE,
    _PinnedTokenizerConfig,
    generate_window_starts,
)
from third_party.PianistTransformer.src.utils.midi import midi_to_ids  # noqa: E402


INDEX_FIELDS = (
    "performance_index",
    "source",
    "dataset_split",
    "composer",
    "title",
    "piece_id",
    "performance_path",
    "performance_absolute_path",
    "token_offset",
    "notes",
    "windows",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _validation_rows(split_csv: Path, asap_root: Path) -> list[dict[str, str]]:
    result = []
    for row in _read_csv(split_csv):
        if row["split"] != "validation":
            continue
        result.append(
            {
                "source": "ASAP-validation",
                "dataset_split": "validation",
                "composer": row["composer"],
                "title": row["title"],
                "piece_id": row["piece_id"],
                "performance_path": row["performance_path"],
                "performance_absolute_path": str(asap_root / row["performance_path"]),
            }
        )
    return result


def _write_split_cache(
    *,
    output_root: Path,
    split: str,
    rows: list[dict[str, str]],
    expected_notes: int,
    window_notes: int,
    stride_notes: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    token_filename = f"{split}_tokens_int16.npy"
    index_filename = f"{split}_performance_index.csv"
    windows_filename = f"{split}_windows_int32.npy"
    token_path = output_root / token_filename
    token_map = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.int16, shape=(expected_notes, TOKENS_PER_NOTE)
    )
    tokenizer = _PinnedTokenizerConfig()
    offset = 0
    windows: list[tuple[int, int, int]] = []
    records: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"performances": 0, "notes": 0, "windows": 0}
    )
    token_digest = hashlib.sha256()
    for performance_index, row in enumerate(rows):
        midi_path = Path(row["performance_absolute_path"])
        if not midi_path.is_file():
            raise FileNotFoundError(midi_path)
        ids = midi_to_ids(tokenizer, MidiFile(str(midi_path)))
        array = np.asarray(ids, dtype=np.int64)
        if array.size == 0 or array.size % TOKENS_PER_NOTE:
            raise ValueError(f"invalid official token count: {midi_path}")
        array = array.reshape(-1, TOKENS_PER_NOTE)
        if array.min() < np.iinfo(np.int16).min or array.max() > np.iinfo(np.int16).max:
            raise OverflowError(f"official token ID does not fit int16: {midi_path}")
        tokens = array.astype(np.int16)
        note_count = len(tokens)
        if offset + note_count > expected_notes:
            raise RuntimeError(f"{split} note count exceeds audited expectation")
        starts = generate_window_starts(note_count, window_notes, stride_notes)
        for start in starts:
            windows.append((performance_index, start, min(start + window_notes, note_count)))
        token_map[offset : offset + note_count] = tokens
        token_digest.update(tokens.astype("<i2", copy=False).tobytes())
        record = {
            "performance_index": performance_index,
            **{field: row.get(field, "") for field in INDEX_FIELDS[1:8]},
            "token_offset": offset,
            "notes": note_count,
            "windows": len(starts),
        }
        records.append(record)
        stats = source_stats[row["source"]]
        stats["performances"] += 1
        stats["notes"] += note_count
        stats["windows"] += len(starts)
        offset += note_count
        if (performance_index + 1) % 100 == 0:
            print(
                f"cache {split} {performance_index + 1}/{len(rows)} "
                f"notes={offset} windows={len(windows)}",
                flush=True,
            )
    token_map.flush()
    del token_map
    if offset != expected_notes:
        raise RuntimeError(
            f"{split} audited notes mismatch: expected {expected_notes}, got {offset}"
        )
    with (output_root / index_filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    window_array = np.asarray(windows, dtype=np.int32)
    if window_array.ndim != 2 or window_array.shape[1] != 3:
        raise AssertionError("window cache must have shape [W,3]")
    np.save(output_root / windows_filename, window_array, allow_pickle=False)
    return {
        "performances": len(rows),
        "notes": offset,
        "windows": len(windows),
        "token_dtype": "int16",
        "window_dtype": "int32",
        "tokens_file": token_filename,
        "index_file": index_filename,
        "windows_file": windows_filename,
        "token_payload_bytes": offset * TOKENS_PER_NOTE * np.dtype(np.int16).itemsize,
        "tokens_sha256_int16_le": token_digest.hexdigest(),
        "source_statistics": dict(source_stats),
        "creation_seconds": time.perf_counter() - started,
    }


def _fix_ownership(path: Path) -> None:
    owner = PROJECT_ROOT.stat()
    for item in [path, *path.rglob("*")]:
        try:
            os.chown(item, owner.st_uid, owner.st_gid)
        except PermissionError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_binary_full_training_v0.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["cache_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite shared cache: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    manifest = Path(config["train_manifest"])
    split_csv = Path(config["asap_split_artifact"])
    asap_root = Path(config["asap_root"])
    for name, path in (("manifest", manifest), ("split", split_csv), ("ASAP root", asap_root)):
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if name == "ASAP root":
            asap_root = path
        elif name == "manifest":
            manifest = path
        else:
            split_csv = path
    train_rows = _read_csv(manifest)
    split_rows = _read_csv(split_csv)
    validation_rows = _validation_rows(split_csv, asap_root)
    counts = defaultdict(int)
    for row in train_rows:
        counts[row["source"]] += 1
    if dict(counts) != config["expected_train_source_performances"]:
        raise RuntimeError(f"training source counts differ: {dict(counts)}")
    if len(train_rows) != int(config["expected_train_performances"]):
        raise RuntimeError("training performance count differs from audit")
    if len(validation_rows) != int(config["expected_validation_performances"]):
        raise RuntimeError("validation performance count differs from split")
    split_by_path = {row["performance_path"]: row["split"] for row in split_rows}
    asap_train_paths = {
        row["performance_path"] for row in train_rows if row["source"] == "ASAP-train"
    }
    if any(split_by_path.get(path) != "train" for path in asap_train_paths):
        raise RuntimeError("train manifest contains non-train ASAP performance")
    heldout_paths = {row["performance_path"] for row in split_rows if row["split"] != "train"}
    if asap_train_paths & heldout_paths:
        raise RuntimeError("train manifest intersects ASAP validation/test paths")
    train_stats = _write_split_cache(
        output_root=output_root,
        split="train",
        rows=train_rows,
        expected_notes=int(config["expected_train_notes"]),
        window_notes=int(config["window_notes"]),
        stride_notes=int(config["stride_notes"]),
    )
    validation_stats = _write_split_cache(
        output_root=output_root,
        split="validation",
        rows=validation_rows,
        expected_notes=int(config["expected_validation_notes"]),
        window_notes=int(config["window_notes"]),
        stride_notes=int(config["stride_notes"]),
    )
    cache_id_payload = {
        "manifest_sha256": _sha256_file(manifest),
        "split_sha256": _sha256_file(split_csv),
        "train_tokens": train_stats["tokens_sha256_int16_le"],
        "validation_tokens": validation_stats["tokens_sha256_int16_le"],
        "window_notes": int(config["window_notes"]),
        "stride_notes": int(config["stride_notes"]),
    }
    cache_id = hashlib.sha256(
        json.dumps(cache_id_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    statistics = {
        "completed": True,
        "cache_id": cache_id,
        "cache_format": "single concatenated NumPy .npy int16 token array per split; mmap read-only",
        "official_tokenizer": "third_party.PianistTransformer.src.utils.midi.midi_to_ids",
        "train_manifest": str(manifest),
        "train_manifest_sha256": cache_id_payload["manifest_sha256"],
        "asap_split_artifact": str(split_csv),
        "asap_split_sha256": cache_id_payload["split_sha256"],
        "window_notes": int(config["window_notes"]),
        "stride_notes": int(config["stride_notes"]),
        "splits": {"train": train_stats, "validation": validation_stats},
        "asap_test_midi_access_count": 0,
        "original_midi_modified": False,
        "creation_seconds_total": time.perf_counter() - started,
    }
    files = [path for path in output_root.iterdir() if path.is_file()]
    statistics["cache_bytes_before_statistics_json"] = sum(path.stat().st_size for path in files)
    stats_path = output_root / "cache_statistics.json"
    stats_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for _ in range(3):
        statistics["cache_bytes_total"] = sum(
            path.stat().st_size for path in output_root.iterdir() if path.is_file()
        )
        stats_path.write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _fix_ownership(output_root)
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
