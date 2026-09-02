#!/usr/bin/env python3
"""Build the deterministic, blind Structural Bass annotation package v0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIT = ROOT / "analysis/structural_bass_event_detector_v0_regroup20ms"
SOURCE_CANDIDATES = SOURCE_AUDIT / "manual_annotation_candidates.csv"
SOURCE_FEATURES = SOURCE_AUDIT / "onset_features_regroup20ms.csv"
WEB_SOURCE = ROOT / "src/structural_bass_annotation"
DEFAULT_OUTPUT = ROOT / "analysis/structural_bass_annotation_v0"
SEED = 42
SETS = ("A", "B")
LABELS = ("STRUCTURAL_BASS", "NOT_STRUCTURAL_BASS", "AMBIGUOUS")
R_STRATA = ("very_small", "small_moderate", "intermediate", "moderately_large", "very_large")
PITCH_BANDS = ("p_le_52", "p_53_55", "p_56_57")
FORBIDDEN_BLIND_KEY_FRAGMENTS = (
    "r_b", "n_b", "next_low", "local_median", "stratum", "pitch_band",
    "onset_group_index", "candidate_identifier", "piece_id", "performance_path",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_sha256(row: Mapping[str, str], fields: Sequence[str]) -> str:
    payload = "\x1f".join(str(row[field]) for field in fields).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in (SEED, *parts)).encode("utf-8")).hexdigest()


def original_id(row: Mapping[str, str]) -> str:
    return f"{row['piece_id']}:{int(row['onset_group_index'])}"


def validate_sources(candidates: Sequence[Mapping[str, str]], features: Sequence[Mapping[str, str]]) -> None:
    required_candidate = {
        "piece", "piece_id", "performance", "onset_group_index", "lowest_pitch",
        "lowest_note_name", "R_B", "N_B", "next_low_gap_sec", "local_median_ioi",
        "R_stratum", "pitch_band", "annotation", "comment",
    }
    required_feature = {
        "composer", "title", "piece_id", "onset_group_index", "onset_time_sec",
        "lowest_pitch", "lowest_note_name", "group_pitches", "group_note_names",
    }
    if len(candidates) != 100:
        raise AssertionError(f"expected exactly 100 source candidates, got {len(candidates)}")
    if not candidates or not required_candidate.issubset(candidates[0]):
        raise AssertionError("candidate CSV schema is incomplete")
    if not features or not required_feature.issubset(features[0]):
        raise AssertionError("feature CSV schema is incomplete")
    identifiers = [original_id(row) for row in candidates]
    if len(set(identifiers)) != 100:
        raise AssertionError("source candidates are not 100 unique piece/onset pairs")
    if any(row["annotation"] or row["comment"] for row in candidates):
        raise AssertionError("source annotations/comments must be blank")

    by_piece: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in candidates:
        by_piece[row["piece_id"]].append(row)
    if len(by_piece) != 5 or any(len(rows) != 20 for rows in by_piece.values()):
        raise AssertionError("source candidates must be five pieces with 20 rows each")
    for piece_id, rows in by_piece.items():
        strata = Counter(row["R_stratum"] for row in rows)
        bands = Counter(row["pitch_band"] for row in rows)
        if strata != Counter({stratum: 4 for stratum in R_STRATA}):
            raise AssertionError(f"unexpected R-stratum balance for {piece_id}: {strata}")
        if bands != Counter({"p_le_52": 7, "p_53_55": 7, "p_56_57": 6}):
            raise AssertionError(f"unexpected pitch-band balance for {piece_id}: {bands}")


def choose_set_a(piece_id: str, rows: Sequence[Mapping[str, str]]) -> set[str]:
    """Choose 2/4 per R stratum, minimizing pitch-band half-split error."""

    by_stratum = {
        stratum: sorted(
            (row for row in rows if row["R_stratum"] == stratum),
            key=original_id,
        )
        for stratum in R_STRATA
    }
    options = [list(itertools.combinations(by_stratum[stratum], 2)) for stratum in R_STRATA]
    totals = Counter(row["pitch_band"] for row in rows)
    best_score: int | None = None
    best: list[tuple[Mapping[str, str], ...]] = []
    for parts in itertools.product(*options):
        selected = tuple(row for pair in parts for row in pair)
        counts = Counter(row["pitch_band"] for row in selected)
        # Twice the absolute deviation avoids floats. The theoretical optimum is 2
        # for source totals 7/7/6 (3-or-4, 3-or-4, and 3 in Set A).
        score = sum(abs(2 * counts[band] - totals[band]) for band in PITCH_BANDS)
        if best_score is None or score < best_score:
            best_score, best = score, [selected]
        elif score == best_score:
            best.append(selected)
    if not best:
        raise AssertionError(f"no split candidates for {piece_id}")
    chosen = min(
        best,
        key=lambda selected: stable_seed_key(
            "split", piece_id, *sorted(original_id(row) for row in selected)
        ),
    )
    return {original_id(row) for row in chosen}


def split_candidates(candidates: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_piece: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in candidates:
        by_piece[row["piece_id"]].append(row)

    split: dict[str, list[dict[str, str]]] = {"A": [], "B": []}
    for piece_id in sorted(by_piece):
        rows = by_piece[piece_id]
        set_a = choose_set_a(piece_id, rows)
        for source in rows:
            split["A" if original_id(source) in set_a else "B"].append(dict(source))

    for set_name in SETS:
        split[set_name].sort(key=lambda row: stable_seed_key("order", set_name, original_id(row)))
        for position, row in enumerate(split[set_name], start=1):
            row["review_id"] = f"{set_name}-{position:03d}"
            row["set"] = set_name
    return split


def parse_pitches(value: str) -> list[int]:
    return [int(token) for token in value.split()]


def build_context(
    candidate: Mapping[str, str],
    feature_index: Mapping[tuple[str, int], Mapping[str, str]],
) -> tuple[str, str, list[dict[str, Any]]]:
    piece_id = candidate["piece_id"]
    center = int(candidate["onset_group_index"])
    source = feature_index[(piece_id, center)]
    center_time = float(source["onset_time_sec"])
    context: list[dict[str, Any]] = []
    for relative_group in range(-4, 9):
        key = (piece_id, center + relative_group)
        if key not in feature_index:
            raise AssertionError(f"candidate lacks full -4/+8 context: {original_id(candidate)}")
        row = feature_index[key]
        pitches = parse_pitches(row["group_pitches"])
        if not pitches:
            raise AssertionError(f"empty pitch group in context: {key}")
        context.append({
            "relative_group": relative_group,
            "relative_time_sec": round(float(row["onset_time_sec"]) - center_time, 6),
            "pitches": pitches,
        })
    if int(source["lowest_pitch"]) != int(candidate["lowest_pitch"]):
        raise AssertionError(f"candidate/feature lowest-pitch mismatch: {original_id(candidate)}")
    return source["composer"], source["title"], context


def assert_blind_payload(payload: Mapping[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).lower()
                if any(fragment in normalized for fragment in FORBIDDEN_BLIND_KEY_FRAGMENTS):
                    raise AssertionError(f"forbidden key in blind payload: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)


def build_payloads(
    split: Mapping[str, Sequence[Mapping[str, str]]],
    features: Sequence[Mapping[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_index = {
        (row["piece_id"], int(row["onset_group_index"])): row
        for row in features
    }
    payloads: dict[str, dict[str, Any]] = {}
    masters: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    source_fields = tuple(split["A"][0].keys())
    for set_name in SETS:
        blind_rows: list[dict[str, Any]] = []
        for position, row in enumerate(split[set_name], start=1):
            composer, title, context = build_context(row, feature_index)
            blind_rows.append({
                "review_id": row["review_id"],
                "set": set_name,
                "composer": composer,
                "piece_title": title,
                "lowest_pitch": int(row["lowest_pitch"]),
                "lowest_note_name": row["lowest_note_name"],
                "context": context,
            })
            master = {
                "review_id": row["review_id"],
                "set": set_name,
                "original_candidate_identifier": original_id(row),
                "piece": row["piece"],
                "piece_id": row["piece_id"],
                "performance": row["performance"],
                "onset_group_index": int(row["onset_group_index"]),
                "lowest_pitch": int(row["lowest_pitch"]),
                "lowest_note_name": row["lowest_note_name"],
                "R_B": float(row["R_B"]),
                "N_B": int(row["N_B"]),
                "next_low_gap_sec": float(row["next_low_gap_sec"]),
                "local_median_ioi": float(row["local_median_ioi"]),
                "original_R_stratum": row["R_stratum"],
                "pitch_band": row["pitch_band"],
                "source_candidate_row_sha256": row_sha256(row, source_fields),
            }
            masters.append(master)
            split_rows.append({
                "review_id": row["review_id"],
                "set": set_name,
                "review_position": position,
                "original_candidate_identifier": original_id(row),
                "piece": row["piece"],
                "piece_id": row["piece_id"],
                "original_R_stratum": row["R_stratum"],
                "pitch_band": row["pitch_band"],
                "split_seed": SEED,
                "order_rule": "SHA-256 seed-42 deterministic shuffle",
            })
        payload = {"schema_version": 1, "set": set_name, "candidates": blind_rows}
        assert_blind_payload(payload)
        payloads[set_name] = payload
    return payloads, masters, split_rows


def validate_final(
    payloads: Mapping[str, Mapping[str, Any]],
    masters: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ids = {set_name: {row["review_id"] for row in payloads[set_name]["candidates"]} for set_name in SETS}
    if any(len(payloads[name]["candidates"]) != 50 for name in SETS):
        raise AssertionError("Set A and Set B must each contain 50 candidates")
    if ids["A"] & ids["B"]:
        raise AssertionError("blind review IDs overlap")
    original_by_set = {
        name: {row["original_candidate_identifier"] for row in masters if row["set"] == name}
        for name in SETS
    }
    if original_by_set["A"] & original_by_set["B"] or len(original_by_set["A"] | original_by_set["B"]) != 100:
        raise AssertionError("candidate overlap or coverage invariant failed")
    summary: dict[str, Any] = {}
    for set_name in SETS:
        rows = [row for row in split_rows if row["set"] == set_name]
        piece_counts = Counter(row["piece_id"] for row in rows)
        stratum_counts = Counter((row["piece_id"], row["original_R_stratum"]) for row in rows)
        if set(piece_counts.values()) != {10} or len(piece_counts) != 5:
            raise AssertionError(f"{set_name} is not exactly 10 candidates per piece")
        if set(stratum_counts.values()) != {2}:
            raise AssertionError(f"{set_name} is not exactly 2 per piece/R stratum")
        summary[set_name] = {
            "candidate_count": len(rows),
            "piece_counts": dict(sorted(piece_counts.items())),
            "piece_pitch_band_counts": {
                piece_id: dict(Counter(
                    row["pitch_band"] for row in rows if row["piece_id"] == piece_id
                ))
                for piece_id in sorted(piece_counts)
            },
        }
    return summary


def render_readme(output: Path, summary: Mapping[str, Any]) -> str:
    return f"""# Structural Bass Blind Annotation v0

This local page presents exactly the existing 100 candidates as two blind, deterministic seed-{SEED} sets. It uses MIDI note-on pitch/onset structure only. Set A should be annotated first; Set B is reserved for later holdout review.

## Run

From the server host, start the existing development container server (CPU-only):

```bash
docker exec -it ilkyun-marg-pedaling-dev python /workspace/project/scripts/serve_structural_bass_annotation.py --host 0.0.0.0 --port 8765
```

Forward port `8765` in VS Code's **Ports** view, then open:

```text
http://localhost:8765
```

No package installation, GPU/CUDA, model inference, audio, or MIDI-file access is used.

## Annotate Set A

Choose **Annotate Set A** on the start screen. One candidate is shown at a time. The piano roll contains four preceding groups, the candidate, and eight following groups. Candidate attacks are orange; its lowest pitch is ringed and filled in red.

Keyboard shortcuts:

- `1`: Structural Bass
- `2`: Not Structural Bass
- `3`: Ambiguous
- `←`: previous candidate
- `→`: next candidate

Buttons and keyboard labels save immediately and advance. Comments autosave; previous/next navigation allows review and label correction. Set A and B progress are independent.

## Persistence and export

The server atomically updates this persistent file after every label/comment change:

```text
{output / 'annotation_results.csv'}
```

Browser refresh or closure does not clear saved work. **Export annotations** downloads the current set as CSV with `review_id,set,annotation,comment`.

## Blind/hidden separation

- Browser-served files live only under `public/`.
- `annotation_master_hidden.csv` preserves the metrics and original candidate mapping but is outside `public/` and is not exposed by the server.
- `annotation_split_manifest.csv` records the deterministic split and balance audit and is likewise not exposed.
- Never open hidden master files during blind annotation.

## Verified split

- Set A: {summary['A']['candidate_count']} candidates, exactly 10 per piece and 2 per piece/R-stratum
- Set B: {summary['B']['candidate_count']} candidates, exactly 10 per piece and 2 per piece/R-stratum
- A/B overlap: 0
- Initial labels/comments: blank

Run the smoke test inside the container:

```bash
python -m unittest tests.test_structural_bass_annotation_v0 -v
```
"""


def run(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    for required in (SOURCE_CANDIDATES, SOURCE_FEATURES, WEB_SOURCE / "index.html", WEB_SOURCE / "app.js", WEB_SOURCE / "styles.css"):
        if not required.is_file():
            raise FileNotFoundError(required)

    candidates = read_csv(SOURCE_CANDIDATES)
    features = read_csv(SOURCE_FEATURES)
    validate_sources(candidates, features)
    split = split_candidates(candidates)
    payloads, masters, split_rows = build_payloads(split, features)
    summary = validate_final(payloads, masters, split_rows)

    public = output / "public"
    public.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "styles.css"):
        shutil.copyfile(WEB_SOURCE / name, public / name)
    blind_fields = (
        "review_id", "set", "composer", "piece_title", "lowest_pitch",
        "lowest_note_name", "context_json",
    )
    for set_name in SETS:
        json_path = public / f"set_{set_name}_blind.json"
        json_path.write_text(json.dumps(payloads[set_name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        csv_rows = [
            {
                **{field: row[field] for field in blind_fields if field != "context_json"},
                "context_json": json.dumps(row["context"], separators=(",", ":")),
            }
            for row in payloads[set_name]["candidates"]
        ]
        write_csv(public / f"set_{set_name}_blind.csv", csv_rows, blind_fields)

    master_fields = tuple(masters[0].keys())
    split_fields = tuple(split_rows[0].keys())
    write_csv(output / "annotation_master_hidden.csv", masters, master_fields)
    write_csv(output / "annotation_split_manifest.csv", split_rows, split_fields)
    result_rows = [
        {"review_id": row["review_id"], "set": row["set"], "annotation": "", "comment": ""}
        for row in split_rows
    ]
    result_fields = ("review_id", "set", "annotation", "comment")
    write_csv(output / "annotation_results_template.csv", result_rows, result_fields)
    write_csv(output / "annotation_results.csv", result_rows, result_fields)

    metadata = {
        "schema_version": 1,
        "seed": SEED,
        "source_candidates": str(SOURCE_CANDIDATES),
        "source_candidates_sha256": sha256(SOURCE_CANDIDATES),
        "source_features": str(SOURCE_FEATURES),
        "source_features_sha256": sha256(SOURCE_FEATURES),
        "source_unique_candidates": 100,
        "split_summary": summary,
        "blind_forbidden_key_fragments": list(FORBIDDEN_BLIND_KEY_FRAGMENTS),
        "allowed_annotations": list(LABELS),
        "asap_test_midi_access_count": 0,
        "asap_midi_files_opened": 0,
        "cc64_features_used": 0,
        "key_off_features_used": 0,
        "duration_features_used": 0,
        "audio_used": False,
        "gpu_cuda_used": False,
        "split_rule": "2 of 4 candidates per piece/R-stratum, minimum pitch-band half-split error, SHA-256 seed-42 tie-break",
        "order_rule": "SHA-256 seed-42 deterministic shuffle independently within Set A and Set B",
    }
    (output / "build_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(render_readme(output, summary), encoding="utf-8")
    print(f"wrote blind annotation package: {output}")
    print("Set A: 50; Set B: 50; each set has exactly 10 per piece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
