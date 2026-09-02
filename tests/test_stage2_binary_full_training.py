"""Focused tests for the shared binary full-training path (no dataset MIDI)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.stage2_binary.full_training import (
    PedalMetricEarlyStopping,
    SharedBinaryWindowDataset,
    binary_batch_metrics,
    order_sha256,
    training_window_order,
)
from src.stage2_encoder_only.dataset import PEDAL_TOKEN_OFFSET


def _synthetic_cache(root: Path) -> None:
    tokens = np.array(
        [
            [1, 129, 257, 385, *[PEDAL_TOKEN_OFFSET + value for value in (63, 64, 0, 127)]],
            [2, 130, 258, 386, *[PEDAL_TOKEN_OFFSET + value for value in (127, 0, 64, 63)]],
        ],
        dtype=np.int16,
    )
    np.save(root / "train_tokens_int16.npy", tokens, allow_pickle=False)
    np.save(root / "train_windows_int32.npy", np.array([[0, 0, 2]], dtype=np.int32), allow_pickle=False)
    with (root / "train_performance_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "performance_index", "source", "dataset_split", "composer", "title",
                "piece_id", "performance_path", "performance_absolute_path", "token_offset",
                "notes", "windows",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "performance_index": 0,
                "source": "ASAP-train",
                "dataset_split": "train",
                "performance_path": "fixture.mid",
                "token_offset": 0,
                "notes": 2,
                "windows": 1,
            }
        )
    metadata = {
        "completed": True,
        "cache_id": "fixture-cache",
        "splits": {
            "train": {
                "tokens_file": "train_tokens_int16.npy",
                "windows_file": "train_windows_int32.npy",
                "index_file": "train_performance_index.csv",
                "notes": 2,
                "windows": 1,
                "performances": 1,
            }
        },
    }
    (root / "cache_statistics.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_shared_cache_masks_only_pedals_and_derives_raw_targets(tmp_path: Path) -> None:
    _synthetic_cache(tmp_path)
    dataset = SharedBinaryWindowDataset(tmp_path, "train")
    sample = dataset[0]
    input_notes = sample["input_ids"].reshape(-1, 8)
    assert input_notes[:, :4].tolist() == [[1, 129, 257, 385], [2, 130, 258, 386]]
    assert sample["pedal_targets"].tolist() == [[63, 64, 0, 127], [127, 0, 64, 63]]
    assert dataset.cache_id == "fixture-cache"


def test_architecture_independent_epoch_order_is_reproducible() -> None:
    first = training_window_order(101, 42, 1)
    second = training_window_order(101, 42, 1)
    assert first == second
    assert order_sha256(first) == order_sha256(second)
    assert first != training_window_order(101, 42, 2)


def test_binary_metrics_decode_joint_and_independent_equivalently() -> None:
    bits = torch.tensor([[[0, 1, 1, 0], [1, 0, 0, 1]]])
    independent = torch.full((1, 2, 4, 2), -2.0)
    independent.scatter_(-1, bits.unsqueeze(-1), 2.0)
    joint_ids = 8 * bits[..., 0] + 4 * bits[..., 1] + 2 * bits[..., 2] + bits[..., 3]
    joint = torch.full((1, 2, 16), -2.0)
    joint.scatter_(-1, joint_ids.unsqueeze(-1), 2.0)
    for logits, architecture in (
        (independent, "independent_4x2"),
        (joint, "joint_16"),
    ):
        metrics = binary_batch_metrics(logits, bits, 0.5, architecture)
        assert metrics["binary_accuracy"] == 1.0
        assert metrics["exact_pattern_accuracy"] == 1.0


def test_js_primary_and_intersection_tiebreak_early_stopping() -> None:
    stopping = PedalMetricEarlyStopping(patience=3, tie_tolerance=1e-12)
    assert stopping.update(0.2, 0.8, 1) == (True, False)
    assert stopping.update(0.2, 0.81, 2) == (True, False)
    assert stopping.best_epoch == 2
    assert stopping.update(0.21, 0.99, 3) == (False, False)
    assert stopping.update(0.22, 0.99, 4) == (False, False)
    assert stopping.update(0.23, 0.99, 5) == (False, True)
