#!/usr/bin/env python3
"""Focused regressions for canonical Frozen-PT mapping and resume safety."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary_2slot.checkpointing import (
    atomic_save_resumable_checkpoint,
    build_resumable_epoch_payload,
    restore_resumable_epoch_payload,
)
from src.stage2_binary_2slot.dataset import mask_human_pedal_features
from src.stage2_binary_2slot.frozen_validation_mapping import build_frozen_human_reference_map


def synthetic_universe():
    counts = [4] * 14 + [3] * 5
    stage1 = [
        {"piece_id": f"piece_{piece}", "validation_performance_count": str(count)}
        for piece, count in enumerate(counts)
    ]
    validation = []
    entries = []
    identifier = 100
    for piece, count in enumerate(counts):
        for performance in range(count):
            path = f"Composer{piece}/Work/performance_{performance}.mid"
            validation.append({
                "metadata_index": str(identifier), "piece_id": f"piece_{piece}",
                "performance_path": path, "split": "validation",
            })
            entries.append({
                "piece_id": f"piece_{piece}", "performance_path": path,
                "source_midi": __file__,
            })
            identifier += 1
    return stage1, validation, entries


def test_path_join_succeeds_without_dataset_metadata_index():
    stage1, validation, entries = synthetic_universe()
    assert all("metadata_index" not in entry for entry in entries)
    result = build_frozen_human_reference_map(stage1, validation, entries, verify_paths=True)
    assert len(result) == 71
    assert len({item.metadata_index for item in result}) == 71
    assert len({item.performance_path for item in result}) == 71
    assert len({item.piece_id for item in result}) == 19


def test_missing_identifier_fails_fast():
    stage1, validation, entries = synthetic_universe()
    validation[0] = {key: value for key, value in validation[0].items() if key != "metadata_index"}
    try:
        build_frozen_human_reference_map(stage1, validation, entries)
    except ValueError as error:
        assert "metadata_index" in str(error)
    else:
        raise AssertionError("missing identifier was accepted")


def test_ambiguous_piece_mapping_fails_fast():
    stage1, validation, entries = synthetic_universe()
    stage1[1] = dict(stage1[1], piece_id=stage1[0]["piece_id"])
    try:
        build_frozen_human_reference_map(stage1, validation, entries)
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("ambiguous frozen piece was accepted")


def test_duplicate_human_assignment_fails_fast():
    stage1, validation, entries = synthetic_universe()
    entries[1] = dict(entries[1], performance_path=entries[0]["performance_path"])
    try:
        build_frozen_human_reference_map(stage1, validation, entries)
    except ValueError as error:
        assert "duplicate human dataset assignment" in str(error)
    else:
        raise AssertionError("duplicate human assignment was accepted")


def test_wrong_piece_mapping_is_detected():
    stage1, validation, entries = synthetic_universe()
    entries[0] = dict(entries[0], piece_id="piece_1")
    try:
        build_frozen_human_reference_map(stage1, validation, entries)
    except ValueError as error:
        assert "wrong-piece" in str(error)
    else:
        raise AssertionError("wrong-piece mapping was accepted")


def test_frozen_pt_pedal_columns_are_masked_without_changing_nonpedal():
    tokens = torch.tensor([[5, 261, 133, 262, 400, 401, 402, 403]], dtype=torch.long)
    masked = mask_human_pedal_features(tokens)
    assert torch.equal(masked[:, :4], tokens[:, :4])
    assert masked[:, 4:].tolist() == [[1, 1, 1, 1]]


def test_resumable_checkpoint_round_trip():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    payload = build_resumable_epoch_payload(
        model=model, optimizer=optimizer, scaler=scaler,
        completed_training_epoch=1, global_step=516,
        run_configuration={"seed": 42},
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "resume.pt"
        atomic_save_resumable_checkpoint(path, payload)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        epoch, step = restore_resumable_epoch_payload(
            loaded, model=model, optimizer=optimizer, scaler=scaler, restore_rng=False
        )
    assert (epoch, step) == (1, 516)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in expected.items())


def main():
    tests = [(name, value) for name, value in globals().items() if name.startswith("test_")]
    results = {}
    for name, test in sorted(tests):
        test(); results[name] = "passed"
    print(results)


if __name__ == "__main__":
    main()
