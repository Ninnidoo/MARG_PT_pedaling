#!/usr/bin/env python3
"""Build leakage-audited Stage 2 binary training manifests without MIDI writes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


EXPECTED_PIECES = {"train": 180, "validation": 19, "test": 23}
EXPECTED_PERFORMANCES = {"train": 892, "validation": 71, "test": 104}
EXPECTED_SPLIT_SHA256 = "d1fe379eb123ff7abca93773296f735f40a215dcd6e6363b55756383708c965b"


# These are exact metadata values, not fuzzy-match patterns. Each rule was
# manually reviewed against explicit catalogue/work/movement text in MAESTRO.
CONFIRMED_EXTRA_PAIRS: dict[tuple[str, str], set[tuple[str, str]]] = {
    ("Bach", "Italian_concerto"): {
        ("Johann Sebastian Bach", "Italian Concerto in F Major, BWV 971 (Complete)"),
        ("Johann Sebastian Bach", "Italian Concerto, BWV 971"),
        ("Johann Sebastian Bach", "Italian Concerto, BWV 971 (Complete)"),
    },
    ("Beethoven", "Piano_Sonatas_4-1"): {
        ("Ludwig van Beethoven", "Sonata No. 4 in E-flat Major, Op. 7, 1st mov."),
    },
    ("Beethoven", "Piano_Sonatas_29-2"): {
        ("Ludwig van Beethoven", "Sonata No. 29 in B-flat Major, Op. 106"),
    },
    ("Chopin", "Etudes_op_25_4"): {
        ("Frédéric Chopin", "Etudes Op. 25 Nos. 1-5"),
    },
    ("Chopin", "Sonata_2_2nd"): {
        ("Frédéric Chopin", "Sonata No. 2 in B-flat Minor, Op. 35"),
    },
    ("Chopin", "Sonata_3_4th"): {
        ("Frédéric Chopin", "Sonata 3 Op. 58"),
        ("Frédéric Chopin", "Sonata No. 3"),
        ("Frédéric Chopin", "Sonata in B Minor, Op. 58"),
    },
    ("Haydn", "Keyboard_Sonatas_46-1"): {
        ("Joseph Haydn", "Sonata in A-flat Major, No. 31 Hob. XVI:46, 1st mov."),
    },
    ("Haydn", "Keyboard_Sonatas_49-1"): {
        ("Joseph Haydn", "Sonata E-Flat Major, Hob. XVI:49, First Movement"),
        ("Joseph Haydn", "Sonata in E flat Major, Hob. XVI: 49, I. Allegro"),
    },
    ("Liszt", "Concert_Etude_S145_2"): {
        ("Franz Liszt", 'Concert Etude No. 2 "Gnomenreigen"'),
    },
    ("Liszt", "Mephisto_Waltz"): {
        ("Franz Liszt", "Mephisto-Waltz No. 1"),
    },
    ("Liszt", "Transcendental_Etudes_10"): {
        ("Franz Liszt", "Transcendental Etude No. 10 in F Minor"),
    },
    ("Mozart", "Piano_Sonatas_12-2"): {
        ("Wolfgang Amadeus Mozart", "Sonata in F K332"),
        ("Wolfgang Amadeus Mozart", "Sonata in F Major, K 332 (Complete)"),
    },
    ("Schumann", "Kreisleriana_1"): {
        ("Robert Schumann", "Kreisleriana, Op. 16"),
    },
    ("Schumann", "Kreisleriana_6"): {
        ("Robert Schumann", "Kreisleriana"),
        ("Robert Schumann", "Kreisleriana, Op. 16"),
        ("Robert Schumann", "Kreisleriana, Op. 16 (1st edition) (Complete)"),
        ("Robert Schumann", "Kreisleriana, Op. 16 (Complete)"),
    },
    ("Scriabin", "Etudes_op_8_11"): {
        ("Alexander Scriabin", "Etude Op. 2 No.1;  Etudes Op. 8, Nos. 5, 11 and 12"),
    },
}


# A canonical title shared by two files is not enough here: the 1,409-second
# file is consistent with the complete Op. 25 set, while the 327-second file is
# not. The latter is explicitly audited below and deliberately retained.
CHOPIN_OP25_COMPLETE_FILENAME = (
    "2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--"
    "AUDIO_05_R1_2004_02_Track02_wav.midi"
)
CONFIRMED_EXTRA_FILENAMES: dict[tuple[str, str], set[str]] = {
    ("Chopin", "Etudes_op_25_4"): {CHOPIN_OP25_COMPLETE_FILENAME},
    ("Chopin", "Etudes_op_25_8"): {CHOPIN_OP25_COMPLETE_FILENAME},
    ("Chopin", "Etudes_op_25_12"): {CHOPIN_OP25_COMPLETE_FILENAME},
}


AMBIGUOUS_SELECTORS: dict[tuple[str, str], list[dict[str, str]]] = {
    ("Chopin", "Etudes_op_25_4"): [{
        "filename": "2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi",
        "reason": "canonical title says 12 Etudes, Op. 25 but 327 s duration is inconsistent with a complete set; included etude numbers are unspecified",
    }],
    ("Chopin", "Etudes_op_25_8"): [{
        "filename": "2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi",
        "reason": "canonical title says 12 Etudes, Op. 25 but 327 s duration is inconsistent with a complete set; included etude numbers are unspecified",
    }],
    ("Chopin", "Etudes_op_25_12"): [{
        "filename": "2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav.midi",
        "reason": "canonical title says 12 Etudes, Op. 25 but 327 s duration is inconsistent with a complete set; included etude numbers are unspecified",
    }],
    ("Haydn", "Keyboard_Sonatas_32-1"): [{
        "pair_composer": "Joseph Haydn",
        "pair_title": "Son. Hob. XVI:32",
        "reason": "work catalogue is exact but the MAESTRO title does not identify whether this file contains the held-out first movement",
    }],
    ("Haydn", "Keyboard_Sonatas_39-2"): [{
        "pair_composer": "Joseph Haydn",
        "pair_title": "Son. Hob. XVI:39",
        "reason": "work catalogue is exact but the MAESTRO title does not identify whether this file contains the held-out second movement",
    }],
    ("Mozart", "Piano_Sonatas_12-2"): [{
        "pair_composer": "Wolfgang Amadeus Mozart",
        "pair_title": "Sonata in F",
        "reason": "key-only MAESTRO title omits K. 332 and movement identity",
    }],
}


def read_csv(path: Path, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open(newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[int, str]:
    """Hash relative names, sizes, and mtimes without reading dataset payloads."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        record = f"{path.relative_to(root)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8", errors="surrogateescape"))
        count += 1
    return count, digest.hexdigest()


def stable_piece_id(prefix: str, composer: str, title: str) -> str:
    digest = hashlib.sha256((composer + "\0" + title).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asap-root", type=Path, default=Path("/workspace/public/ASAP/asap-dataset-v1.1"))
    parser.add_argument("--maestro-root", type=Path, default=Path("/workspace/public/MAESTRO/maestro-v3.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/project/analysis/stage2_binary_v0/data_prep_v0"))
    args = parser.parse_args()

    asap_metadata_path = args.asap_root / "metadata.csv"
    maestro_metadata_path = args.maestro_root / "maestro-v3.0.0.csv"
    for path in (args.asap_root, args.maestro_root, args.split_csv, asap_metadata_path, maestro_metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    input_hashes_before = {
        "split": sha256_file(args.split_csv),
        "asap_metadata": sha256_file(asap_metadata_path),
        "maestro_metadata": sha256_file(maestro_metadata_path),
    }
    if input_hashes_before["split"] != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"stored split SHA-256 changed: {input_hashes_before['split']}")
    inventories_before = {
        "ASAP": inventory(args.asap_root),
        "MAESTRO": inventory(args.maestro_root),
    }

    split_rows = read_csv(args.split_csv, "utf-8")
    asap_rows = read_csv(asap_metadata_path)
    maestro_rows = read_csv(maestro_metadata_path)
    if len(split_rows) != len(asap_rows):
        raise RuntimeError("split and ASAP metadata row counts differ")
    for index, (split_row, asap_row) in enumerate(zip(split_rows, asap_rows)):
        expected = (str(index), asap_row["composer"], asap_row["title"], asap_row["midi_performance"])
        actual = (split_row["metadata_index"], split_row["composer"], split_row["title"], split_row["performance_path"])
        if actual != expected:
            raise RuntimeError(f"split/ASAP metadata mismatch at row {index}: {actual} != {expected}")

    piece_splits: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        piece_splits[row["piece_id"]].add(row["split"])
    if any(len(values) != 1 for values in piece_splits.values()):
        raise RuntimeError("one or more ASAP pieces cross split boundaries")
    piece_counts = Counter(next(iter(values)) for values in piece_splits.values())
    performance_counts = Counter(row["split"] for row in split_rows)
    if dict(piece_counts) != EXPECTED_PIECES:
        raise RuntimeError(f"piece counts differ: {dict(piece_counts)} != {EXPECTED_PIECES}")
    if dict(performance_counts) != EXPECTED_PERFORMANCES:
        raise RuntimeError(f"performance counts differ: {dict(performance_counts)} != {EXPECTED_PERFORMANCES}")

    maestro_by_filename = {row["midi_filename"]: row for row in maestro_rows}
    if len(maestro_by_filename) != len(maestro_rows):
        raise RuntimeError("MAESTRO metadata contains duplicate MIDI filenames")
    missing_maestro_midi = [row["midi_filename"] for row in maestro_rows if not (args.maestro_root / row["midi_filename"]).is_file()]
    if missing_maestro_midi:
        raise RuntimeError(f"MAESTRO MIDI files missing: {missing_maestro_midi[:5]}")
    missing_asap_midi = [row["midi_performance"] for row in asap_rows if not (args.asap_root / row["midi_performance"]).is_file()]
    if missing_asap_midi:
        raise RuntimeError(f"ASAP performance MIDI files missing: {missing_asap_midi[:5]}")

    joined = list(zip(split_rows, asap_rows))
    heldout_groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for split_row, asap_row in joined:
        if split_row["split"] in {"validation", "test"}:
            key = (split_row["split"], split_row["piece_id"], split_row["composer"], split_row["title"])
            heldout_groups[key].append(asap_row)
    if len(heldout_groups) != EXPECTED_PIECES["validation"] + EXPECTED_PIECES["test"]:
        raise RuntimeError("held-out ASAP piece count differs")

    heldout_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    ambiguous_rows: list[dict[str, object]] = []
    excluded_filenames: set[str] = set()
    heldout_direct_filenames: set[str] = set()
    confirmed_overlap_piece_ids: set[str] = set()

    for (split, piece_id, composer, title), group in sorted(heldout_groups.items()):
        direct_filenames = {
            row["maestro_midi_performance"].removeprefix("{maestro}/")
            for row in group if row["maestro_midi_performance"]
        }
        missing_direct = sorted(direct_filenames - maestro_by_filename.keys())
        if missing_direct:
            raise RuntimeError(f"held-out direct MAESTRO links missing: {missing_direct}")
        heldout_direct_filenames.update(direct_filenames)
        direct_pairs = {
            (maestro_by_filename[name]["canonical_composer"], maestro_by_filename[name]["canonical_title"])
            for name in direct_filenames
        }
        asap_key = (composer, title)
        extra_pairs = CONFIRMED_EXTRA_PAIRS.get(asap_key, set())
        extra_filenames = CONFIRMED_EXTRA_FILENAMES.get(asap_key, set())
        matched: dict[str, tuple[str, str]] = {}
        for row in maestro_rows:
            filename = row["midi_filename"]
            pair = (row["canonical_composer"], row["canonical_title"])
            if filename in direct_filenames:
                matched[filename] = ("direct_filename_link", "exact")
            elif pair in direct_pairs:
                matched[filename] = ("direct_seed_canonical_exact", "high")
            elif pair in extra_pairs:
                matched[filename] = ("explicit_catalog_identity", "high")
            elif filename in extra_filenames:
                matched[filename] = ("explicit_complete_collection_filename", "high")
        if matched:
            confirmed_overlap_piece_ids.add(piece_id)
        excluded_filenames.update(matched)
        heldout_rows.append({
            "split": split,
            "composer": composer,
            "piece/title": title,
            "piece_id": piece_id,
            "asap_performance_count": len(group),
            "direct_maestro_link_count": len(direct_filenames),
            "confirmed_maestro_overlap_performance_count": len(matched),
        })
        for filename, (method, confidence) in sorted(matched.items()):
            maestro = maestro_by_filename[filename]
            overlap_rows.append({
                "asap_split": split,
                "asap_piece_id": piece_id,
                "asap_composer": composer,
                "asap_composition": title,
                "maestro_canonical_composer": maestro["canonical_composer"],
                "maestro_composition": maestro["canonical_title"],
                "match_method": method,
                "match_confidence": confidence,
                "is_direct_source_filename": str(filename in direct_filenames),
                "excluded_maestro_midi_filename": filename,
            })
        for selector in AMBIGUOUS_SELECTORS.get(asap_key, []):
            candidates: list[dict[str, str]]
            if "filename" in selector:
                candidate = maestro_by_filename.get(selector["filename"])
                if candidate is None:
                    raise RuntimeError(f"ambiguous filename missing: {selector['filename']}")
                candidates = [candidate]
            else:
                candidates = [row for row in maestro_rows if (
                    row["canonical_composer"] == selector["pair_composer"]
                    and row["canonical_title"] == selector["pair_title"]
                )]
                if not candidates:
                    raise RuntimeError(f"ambiguous canonical pair missing: {selector}")
            for candidate in candidates:
                if candidate["midi_filename"] in excluded_filenames:
                    raise RuntimeError("an ambiguous candidate was also marked confirmed")
                ambiguous_rows.append({
                    "asap_split": split,
                    "asap_piece_id": piece_id,
                    "asap_composer": composer,
                    "asap_composition": title,
                    "candidate_maestro_canonical_composer": candidate["canonical_composer"],
                    "candidate_maestro_composition": candidate["canonical_title"],
                    "candidate_maestro_midi_filename": candidate["midi_filename"],
                    "reason": selector["reason"],
                    "action": "retained_in_maestro_clean_pending_manual_resolution",
                })

    maestro_clean_rows: list[dict[str, object]] = []
    for row in maestro_rows:
        if row["midi_filename"] in excluded_filenames:
            continue
        maestro_clean_rows.append({
            "source": "MAESTRO-clean",
            "canonical_composer": row["canonical_composer"],
            "canonical_title": row["canonical_title"],
            "piece_id": stable_piece_id("maestro_piece_", row["canonical_composer"], row["canonical_title"]),
            "original_maestro_split": row["split"],
            "year": row["year"],
            "midi_filename": row["midi_filename"],
            "midi_path": str(args.maestro_root / row["midi_filename"]),
            "audio_filename": row["audio_filename"],
            "audio_path": str(args.maestro_root / row["audio_filename"]),
            "duration": row["duration"],
        })

    asap_train_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    for split_row, asap_row in joined:
        if split_row["split"] != "train":
            continue
        direct_raw = asap_row["maestro_midi_performance"]
        direct_filename = direct_raw.removeprefix("{maestro}/") if direct_raw else ""
        maestro = maestro_by_filename.get(direct_filename) if direct_filename else None
        if direct_filename and maestro is None:
            raise RuntimeError(f"train direct MAESTRO link missing: {direct_filename}")
        asap_train_rows.append({
            "source": "ASAP-train",
            "split": "train",
            "metadata_index": split_row["metadata_index"],
            "composer": split_row["composer"],
            "title": split_row["title"],
            "piece_id": split_row["piece_id"],
            "performance_path": split_row["performance_path"],
            "performance_absolute_path": str(args.asap_root / split_row["performance_path"]),
            "folder": asap_row["folder"],
            "midi_score": asap_row["midi_score"],
            "maestro_midi_performance": direct_raw,
            "maestro_source_start_seconds": asap_row["start"],
            "maestro_source_end_seconds": asap_row["end"],
            "has_raw_cc64": split_row["has_raw_cc64"],
            "has_intermediate_pedal": split_row["has_intermediate_pedal"],
            "num_raw_cc64_events": split_row["num_raw_cc64_events"],
            "num_pedal_intermediate": split_row["num_pedal_intermediate"],
            "num_normalized_notes": split_row["num_normalized_notes"],
        })
        if maestro is not None:
            duplicate_rows.append({
                "asap_split": "train",
                "asap_piece_id": split_row["piece_id"],
                "asap_composer": split_row["composer"],
                "asap_composition": split_row["title"],
                "asap_performance_path": split_row["performance_path"],
                "asap_metadata_index": split_row["metadata_index"],
                "direct_maestro_midi_filename": direct_filename,
                "maestro_canonical_composer": maestro["canonical_composer"],
                "maestro_canonical_title": maestro["canonical_title"],
                "maestro_source_start_seconds": asap_row["start"],
                "maestro_source_end_seconds": asap_row["end"],
                "asap_is_source_excerpt": str(bool(asap_row["start"] or asap_row["end"])),
                "match_method": "direct_filename_link",
                "match_confidence": "exact",
                "maestro_in_clean_manifest": str(direct_filename not in excluded_filenames),
            })

    train_rows: list[dict[str, object]] = []
    for row in maestro_clean_rows:
        train_rows.append({
            "source": "MAESTRO-clean",
            "dataset_split": row["original_maestro_split"],
            "composer": row["canonical_composer"],
            "title": row["canonical_title"],
            "piece_id": row["piece_id"],
            "performance_path": row["midi_filename"],
            "performance_absolute_path": row["midi_path"],
            "metadata_index": "",
            "year": row["year"],
            "duration": row["duration"],
        })
    for row in asap_train_rows:
        train_rows.append({
            "source": "ASAP-train",
            "dataset_split": "train",
            "composer": row["composer"],
            "title": row["title"],
            "piece_id": row["piece_id"],
            "performance_path": row["performance_path"],
            "performance_absolute_path": row["performance_absolute_path"],
            "metadata_index": row["metadata_index"],
            "year": "",
            "duration": "",
        })

    # Integrity A/B/C before any output is committed.
    clean_filenames = {row["midi_filename"] for row in maestro_clean_rows}
    integrity_a = not (heldout_direct_filenames & clean_filenames)
    integrity_b = not (excluded_filenames & clean_filenames)
    heldout_asap_paths = {
        split_row["performance_path"] for split_row in split_rows
        if split_row["split"] in {"validation", "test"}
    }
    train_asap_paths = {
        row["performance_path"] for row in train_rows if row["source"] == "ASAP-train"
    }
    integrity_c = not (heldout_asap_paths & train_asap_paths) and all(
        row["dataset_split"] == "train" for row in train_rows if row["source"] == "ASAP-train"
    )
    if not (integrity_a and integrity_b and integrity_c):
        raise RuntimeError(f"integrity A/B/C failed: {integrity_a}/{integrity_b}/{integrity_c}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "heldout_asap_pieces.csv", [
        "split", "composer", "piece/title", "piece_id", "asap_performance_count",
        "direct_maestro_link_count", "confirmed_maestro_overlap_performance_count",
    ], heldout_rows)
    write_csv(args.output_dir / "maestro_overlap_audit.csv", [
        "asap_split", "asap_piece_id", "asap_composer", "asap_composition",
        "maestro_canonical_composer", "maestro_composition", "match_method",
        "match_confidence", "is_direct_source_filename", "excluded_maestro_midi_filename",
    ], overlap_rows)
    write_csv(args.output_dir / "ambiguous_matches.csv", [
        "asap_split", "asap_piece_id", "asap_composer", "asap_composition",
        "candidate_maestro_canonical_composer", "candidate_maestro_composition",
        "candidate_maestro_midi_filename", "reason", "action",
    ], ambiguous_rows)
    write_csv(args.output_dir / "maestro_clean_manifest.csv", [
        "source", "canonical_composer", "canonical_title", "piece_id",
        "original_maestro_split", "year", "midi_filename", "midi_path",
        "audio_filename", "audio_path", "duration",
    ], maestro_clean_rows)
    write_csv(args.output_dir / "asap_train_manifest.csv", [
        "source", "split", "metadata_index", "composer", "title", "piece_id",
        "performance_path", "performance_absolute_path", "folder", "midi_score",
        "maestro_midi_performance", "maestro_source_start_seconds",
        "maestro_source_end_seconds", "has_raw_cc64", "has_intermediate_pedal",
        "num_raw_cc64_events", "num_pedal_intermediate", "num_normalized_notes",
    ], asap_train_rows)
    write_csv(args.output_dir / "train_manifest.csv", [
        "source", "dataset_split", "composer", "title", "piece_id",
        "performance_path", "performance_absolute_path", "metadata_index", "year", "duration",
    ], train_rows)
    write_csv(args.output_dir / "asap_train_maestro_duplicates.csv", [
        "asap_split", "asap_piece_id", "asap_composer", "asap_composition",
        "asap_performance_path", "asap_metadata_index", "direct_maestro_midi_filename",
        "maestro_canonical_composer", "maestro_canonical_title",
        "maestro_source_start_seconds", "maestro_source_end_seconds",
        "asap_is_source_excerpt", "match_method", "match_confidence",
        "maestro_in_clean_manifest",
    ], duplicate_rows)

    input_hashes_after = {
        "split": sha256_file(args.split_csv),
        "asap_metadata": sha256_file(asap_metadata_path),
        "maestro_metadata": sha256_file(maestro_metadata_path),
    }
    inventories_after = {
        "ASAP": inventory(args.asap_root),
        "MAESTRO": inventory(args.maestro_root),
    }
    integrity_d = input_hashes_before["split"] == input_hashes_after["split"] == EXPECTED_SPLIT_SHA256
    integrity_e = input_hashes_before == input_hashes_after and inventories_before == inventories_after
    if not (integrity_d and integrity_e):
        raise RuntimeError(f"integrity D/E failed: {integrity_d}/{integrity_e}")

    direct_duplicate_sources = {row["direct_maestro_midi_filename"] for row in duplicate_rows}
    duplicate_excerpt_count = sum(row["asap_is_source_excerpt"] == "True" for row in duplicate_rows)
    direct_overlap_rows = sum(row["match_method"] == "direct_filename_link" for row in overlap_rows)
    raw_canonical_pairs = {
        (row["maestro_canonical_composer"], row["maestro_composition"])
        for row in overlap_rows
    }
    ambiguous_table = "\n".join(
        "| " + " | ".join(md_escape(row[key]) for key in (
            "asap_split", "asap_composer", "asap_composition",
            "candidate_maestro_composition", "candidate_maestro_midi_filename", "reason",
        )) + " |"
        for row in ambiguous_rows
    ) or "| - | - | - | - | - | 0건 |"

    report = f"""# Stage 2 binary v0 dataset preparation report

## 범위와 입력

- 수행 범위: CSV manifest와 leakage/duplicate audit만 생성. tokenization, CC64 threshold, histogram, balancing, 모델 구현, 학습, 평가는 수행하지 않았다.
- 실제 bind 확인: host `/public/intern_2026_summer_public_dataset/ilkyun_data` → 기존 container `ilkyun-marg-pedaling-dev:/workspace/public`.
- ASAP root: `{args.asap_root}`
- ASAP metadata: `{asap_metadata_path}` (SHA-256 `{input_hashes_before['asap_metadata']}`)
- MAESTRO root: `{args.maestro_root}`
- MAESTRO metadata: `{maestro_metadata_path}` (SHA-256 `{input_hashes_before['maestro_metadata']}`)
- 발견했지만 사용하지 않은 보조 metadata: `{args.maestro_root / 'maestro-v3.0.0.json'}`
- 기존 ASAP split source of truth: `{args.split_csv}` (SHA-256 `{input_hashes_before['split']}`; Git commit `4c1d5daa527d76e815fcbda0239bcdec6af9458a`).
- full split 생성 스크립트는 현재 repository에서 발견되지 않았다. `tests/test_stage2_dataset.py`는 original PT seed-42 test-score membership만 재구성해 검증하며, 이번 작업은 seed를 실행하거나 split을 재생성하지 않고 저장 CSV를 그대로 사용했다.

## 기존 split 검증

| split | piece 수 | performance 수 |
| --- | ---: | ---: |
| train | {piece_counts['train']} | {performance_counts['train']} |
| validation | {piece_counts['validation']} | {performance_counts['validation']} |
| test | {piece_counts['test']} | {performance_counts['test']} |

- split CSV 1,067행은 ASAP metadata 1,067행과 `metadata_index`, composer, title, performance path 기준으로 전 행 exact alignment를 확인했다.
- 한 `piece_id`가 둘 이상의 split에 속하는 경우는 0건이다.
- 기대값 180/19/23 piece와 정확히 일치했다.

## Composition overlap 방법

1. ASAP validation/test의 `maestro_midi_performance`를 `{{maestro}}/` 뒤 exact filename으로 해석했다. 직접 링크 {len(heldout_direct_filenames)}개는 모두 MAESTRO CSV row와 일치했다.
2. 직접 row의 exact canonical `(composer, title)` pair가 같은 MAESTRO row를 동일 metadata composition으로 확장했다.
3. 추가 표기 변형은 BWV/Op./Hob./K./S. 번호와 movement 또는 Complete/명시적 collection 범위가 확인되는 exact allowlist만 사용했다. fuzzy score/threshold는 사용하지 않았다.
4. 확신할 수 없는 후보는 clean 제외 집합에 넣지 않고 `ambiguous_matches.csv`에 기록했다.

## 결과 수량

| 항목 | 수 |
| --- | ---: |
| ASAP held-out composition (validation 19 + test 23) | {len(heldout_groups)} |
| confirmed MAESTRO overlap이 있는 held-out ASAP composition | {len(confirmed_overlap_piece_ids)} |
| overlap audit의 distinct MAESTRO canonical pair | {len(raw_canonical_pairs)} |
| 제거되는 distinct MAESTRO performance | {len(excluded_filenames)} |
| 그중 held-out direct filename | {len(heldout_direct_filenames)} |
| overlap audit row (ASAP composition × MAESTRO performance) | {len(overlap_rows)} |
| 최종 MAESTRO-clean performance | {len(maestro_clean_rows)} |
| 최종 ASAP-train performance | {len(asap_train_rows)} |
| 최종 train_manifest performance | {len(train_rows)} |
| ambiguous match row | {len(ambiguous_rows)} |

MAESTRO 원본은 {len(maestro_rows)} performances이며, `MAESTRO-clean = {len(maestro_rows)} - {len(excluded_filenames)} = {len(maestro_clean_rows)}`이다. `train_manifest = {len(maestro_clean_rows)} + {len(asap_train_rows)} = {len(train_rows)}`이다.

## Ambiguous cases

아래 후보는 임의 제외하지 않았으며 현재 `maestro_clean_manifest.csv`에 남아 있다. 각 행의 후속 수동 확인 전에는 clean 확정으로 과해석하지 않아야 한다.

| split | ASAP composer | ASAP composition | candidate MAESTRO title | MIDI filename | reason |
| --- | --- | --- | --- | --- | --- |
{ambiguous_table}

## ASAP-train ↔ MAESTRO direct source duplicate audit

- direct/exact ASAP-train performance row: **{len(duplicate_rows)}**
- distinct MAESTRO source MIDI filename: **{len(direct_duplicate_sources)}**
- source `start` 또는 `end`가 있어 excerpt로 표시된 ASAP row: **{duplicate_excerpt_count}**
- 목록: `asap_train_maestro_duplicates.csv`
- 이것은 ASAP train leakage가 아니라 concatenate 시 source performance 이중 집계 가능성 audit이다. 이번 단계에서는 제거 정책을 적용하지 않았다.
- held-out composition 제거 때문에 duplicate source 중 일부가 MAESTRO-clean에서 빠질 수 있으며, 각 행의 `maestro_in_clean_manifest`에 상태를 기록했다.

## 무결성 검증

| 조건 | 결과 | 근거 |
| --- | --- | --- |
| A. validation/test 직접 연결 MAESTRO performance가 clean에 0개 | {'PASS' if integrity_a else 'FAIL'} | direct {len(heldout_direct_filenames)} filename; clean 교집합 0 |
| B. confirmed held-out composition MAESTRO performance가 clean에 0개 | {'PASS' if integrity_b else 'FAIL'} | confirmed exclusion {len(excluded_filenames)} filename; clean 교집합 0 |
| C. ASAP validation/test performance가 train manifest에 0개 | {'PASS' if integrity_c else 'FAIL'} | ASAP source 행은 저장 split의 train {len(asap_train_rows)}행만 사용 |
| D. 기존 ASAP split 미수정 | {'PASS' if integrity_d else 'FAIL'} | 전/후 SHA-256 `{input_hashes_after['split']}` 동일 |
| E. 원본 dataset 미수정 | {'PASS' if integrity_e else 'FAIL'} | metadata hash와 전체 file `(relative path,size,mtime_ns)` inventory digest 전/후 동일 |

Dataset inventory: ASAP {inventories_after['ASAP'][0]} files / `{inventories_after['ASAP'][1]}`; MAESTRO {inventories_after['MAESTRO'][0]} files / `{inventories_after['MAESTRO'][1]}`.

## 산출물

- `heldout_asap_pieces.csv`: 42개 held-out ASAP piece composition
- `maestro_overlap_audit.csv`: confirmed match와 제외 filename ({direct_overlap_rows} direct-link audit rows 포함)
- `ambiguous_matches.csv`: 미확정 후보; clean 제외에 미반영
- `maestro_clean_manifest.csv`: confirmed held-out overlap 제거 후 MAESTRO
- `asap_train_manifest.csv`: 저장 split artifact의 ASAP train performance
- `train_manifest.csv`: `source` 포함 MAESTRO-clean + ASAP-train
- `asap_train_maestro_duplicates.csv`: train direct-source 중복 audit

## 주의

- ambiguous 후보는 요청에 따라 추측으로 제거하지 않았다. 따라서 이 manifest의 “clean”은 confirmed identity 기준이며, ambiguous 후보의 수동 판정 전까지 provisional이다.
- 원본 MIDI/audio/metadata는 삭제, 이동, 수정, 복사하지 않았다. 모든 manifest는 원본 경로를 참조한다.
"""
    (args.output_dir / "DATASET_PREP_REPORT.md").write_text(report, encoding="utf-8")

    for path in args.output_dir.iterdir():
        if path.is_file():
            path.chmod(0o664)
    print(f"wrote {args.output_dir}")
    print(f"heldout_pieces={len(heldout_groups)} excluded_maestro={len(excluded_filenames)} maestro_clean={len(maestro_clean_rows)} asap_train={len(asap_train_rows)} train_total={len(train_rows)} ambiguous={len(ambiguous_rows)} duplicates={len(duplicate_rows)}")
    print("integrity=A:PASS B:PASS C:PASS D:PASS E:PASS")


if __name__ == "__main__":
    main()
