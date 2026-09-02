"""Strict canonical Frozen-PT piece to ASAP human-reference mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FrozenHumanReference:
    piece_id: str
    metadata_index: str
    performance_path: str
    source_midi: str
    dataset_index: int


def _required(row: Mapping[str, Any], key: str, scope: str) -> str:
    if key not in row or row[key] is None or str(row[key]) == "":
        raise ValueError(f"{scope} missing required identifier: {key}")
    return str(row[key])


def build_frozen_human_reference_map(
    stage1_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    dataset_entries: Sequence[Mapping[str, Any]],
    *,
    expected_pieces: int | None = 19,
    expected_humans: int | None = 71,
    verify_paths: bool = False,
) -> tuple[FrozenHumanReference, ...]:
    """Join exactly as the canonical 4-class evaluator: by performance_path.

    ``metadata_index`` belongs to the canonical split row, not the cached
    Stage-2 dataset entry.  Piece identity is checked on both sides of the join.
    """

    stage1_piece_ids = [_required(row, "piece_id", "stage1 row") for row in stage1_rows]
    if len(stage1_piece_ids) != len(set(stage1_piece_ids)):
        raise ValueError("ambiguous frozen Stage-1 piece_id")
    if expected_pieces is not None and len(stage1_piece_ids) != expected_pieces:
        raise ValueError(f"expected {expected_pieces} frozen pieces, found {len(stage1_piece_ids)}")
    stage1_universe = set(stage1_piece_ids)

    split_by_path: dict[str, Mapping[str, Any]] = {}
    metadata_seen: set[str] = set()
    for row in validation_rows:
        path = _required(row, "performance_path", "validation split row")
        piece = _required(row, "piece_id", "validation split row")
        metadata = _required(row, "metadata_index", "validation split row")
        if path in split_by_path:
            raise ValueError(f"duplicate/ambiguous validation performance_path: {path}")
        if metadata in metadata_seen:
            raise ValueError(f"duplicate validation metadata_index: {metadata}")
        if piece not in stage1_universe:
            raise ValueError(f"validation row maps to unknown frozen piece: {piece}")
        split_by_path[path] = row
        metadata_seen.add(metadata)
    if expected_humans is not None and len(split_by_path) != expected_humans:
        raise ValueError(f"expected {expected_humans} validation references, found {len(split_by_path)}")

    dataset_by_path: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for dataset_index, entry in enumerate(dataset_entries):
        path = _required(entry, "performance_path", "validation dataset entry")
        _required(entry, "piece_id", "validation dataset entry")
        if path in dataset_by_path:
            raise ValueError(f"duplicate human dataset assignment: {path}")
        dataset_by_path[path] = (dataset_index, entry)
    missing_dataset = sorted(set(split_by_path) - set(dataset_by_path))
    unexpected_dataset = sorted(set(dataset_by_path) - set(split_by_path))
    if missing_dataset or unexpected_dataset:
        raise ValueError(
            f"human path universe mismatch: missing={missing_dataset}, unexpected={unexpected_dataset}"
        )

    result = []
    for path, row in split_by_path.items():
        dataset_index, entry = dataset_by_path[path]
        split_piece = _required(row, "piece_id", "validation split row")
        dataset_piece = _required(entry, "piece_id", "validation dataset entry")
        if split_piece != dataset_piece:
            raise ValueError(
                f"wrong-piece mapping for {path}: split={split_piece}, dataset={dataset_piece}"
            )
        source_midi = _required(entry, "source_midi", "validation dataset entry")
        if verify_paths and not Path(source_midi).is_file():
            raise FileNotFoundError(source_midi)
        result.append(FrozenHumanReference(
            piece_id=split_piece,
            metadata_index=_required(row, "metadata_index", "validation split row"),
            performance_path=path,
            source_midi=source_midi,
            dataset_index=dataset_index,
        ))
    if len(result) != len({item.performance_path for item in result}):
        raise AssertionError("a human performance was assigned more than once")
    counts = {piece: 0 for piece in stage1_piece_ids}
    for item in result:
        counts[item.piece_id] += 1
    for row in stage1_rows:
        expected = row.get("validation_performance_count")
        if expected is not None and str(expected) != "" and counts[str(row["piece_id"])] != int(expected):
            raise ValueError(f"piece reference count mismatch: {row['piece_id']}")
    return tuple(sorted(result, key=lambda item: int(item.metadata_index)))


def references_by_piece(
    references: Sequence[FrozenHumanReference],
) -> dict[str, tuple[FrozenHumanReference, ...]]:
    grouped: dict[str, list[FrozenHumanReference]] = {}
    for reference in references:
        grouped.setdefault(reference.piece_id, []).append(reference)
    return {piece: tuple(values) for piece, values in grouped.items()}
