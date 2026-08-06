from __future__ import annotations

import csv
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.stage2_encoder_only import dataset as dataset_module
from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PAD_ID,
    PEDAL_TOKEN_OFFSET,
    TOKENS_PER_NOTE,
    Stage2PedalDataset,
    generate_window_starts,
    stage2_pedal_collate_fn,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
SPLIT_CSV = (
    PROJECT_ROOT / "analysis" / "stage2_encoder_only_v0" / "asap_split.csv"
)


class SplitIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with SPLIT_CSV.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_split_integrity(self) -> None:
        self.assertEqual(len(self.rows), 1067)
        self.assertEqual(
            {row["split"] for row in self.rows},
            {"train", "validation", "test"},
        )
        self.assertEqual(
            len({row["performance_path"] for row in self.rows}),
            len(self.rows),
        )
        splits_by_piece: dict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            splits_by_piece[row["piece_id"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_piece.values()))

    def test_all_original_pt_test_pieces_are_test(self) -> None:
        metadata = pd.read_csv(ASAP_ROOT / "metadata.csv")
        scores = sorted(set(metadata.midi_score))
        random.seed(42)
        random.shuffle(scores)
        selected_scores = set(scores[: int(0.1 * len(scores))])

        test_pieces = set()
        for score in selected_scores:
            matches = metadata[metadata.midi_score == score]
            identities = set(zip(matches.composer, matches.title))
            self.assertEqual(len(identities), 1)
            test_pieces.update(identities)

        assigned = defaultdict(set)
        for row in self.rows:
            assigned[(row["composer"], row["title"])].add(row["split"])
        self.assertEqual(len(test_pieces), 23)
        for piece in test_pieces:
            self.assertEqual(assigned[piece], {"test"})

    def test_no_cc64_performance_is_retained(self) -> None:
        no_cc64 = [row for row in self.rows if row["has_raw_cc64"] == "False"]
        self.assertEqual(len(no_cc64), 13)
        self.assertTrue(all(int(row["num_raw_cc64_events"]) == 0 for row in no_cc64))


class WindowTests(unittest.TestCase):
    def test_short_performance_has_one_window(self) -> None:
        self.assertEqual(generate_window_starts(414), [0])

    def test_long_performance_is_deterministic_and_tail_covered(self) -> None:
        starts = generate_window_starts(1000, 512, 256)
        self.assertEqual(starts, [0, 256, 488])
        self.assertEqual(starts[-1] + 512, 1000)
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(starts, sorted(starts))


class RealDatasetTests(unittest.TestCase):
    def make_dataset(self, split: str) -> Stage2PedalDataset:
        return Stage2PedalDataset(ASAP_ROOT, SPLIT_CSV, split)

    def test_exact_masking_targets_range_and_eight_token_grammar(self) -> None:
        dataset = self.make_dataset("train")
        row_index, start, end = dataset._windows[0]
        original = dataset._load_performance_tokens(
            dataset.performances[row_index]
        )[start:end]
        sample = dataset[0]
        restored = sample["input_ids"].reshape(-1, TOKENS_PER_NOTE)

        self.assertEqual(sample["input_ids"].numel() % TOKENS_PER_NOTE, 0)
        self.assertEqual(tuple(restored.shape), (end - start, TOKENS_PER_NOTE))
        self.assertTrue(
            torch.equal(
                restored[:, :NON_PEDAL_FEATURES],
                torch.from_numpy(original[:, :NON_PEDAL_FEATURES].copy()),
            )
        )
        self.assertTrue(torch.all(restored[:, NON_PEDAL_FEATURES:] == MASK_ID))
        expected = torch.from_numpy(
            original[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET
        )
        self.assertTrue(torch.equal(sample["pedal_targets"], expected))
        self.assertGreaterEqual(int(sample["pedal_targets"].min()), 0)
        self.assertLessEqual(int(sample["pedal_targets"].max()), 127)

    def test_small_smoke_sample_from_each_split(self) -> None:
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                dataset = self.make_dataset(split)
                sample = dataset[0]
                notes = sample["note_mask"].numel()
                self.assertGreater(dataset.performance_count, 0)
                self.assertGreater(dataset.window_count, 0)
                self.assertEqual(tuple(sample["input_ids"].shape), (notes * 8,))
                self.assertEqual(tuple(sample["pedal_targets"].shape), (notes, 4))
                self.assertTrue(torch.all(sample["note_mask"]))
                self.assertEqual(sample["metadata"]["split"], split)


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        with SPLIT_CSV.open(newline="", encoding="utf-8") as handle:
            row = next(row for row in csv.DictReader(handle) if row["split"] == "train")
        self.split_csv = Path(self.temp_dir.name) / "one_performance.csv"
        with self.split_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, row.keys())
            writer.writeheader()
            writer.writerow(row)

    def make_dataset(self, cache_mode: str) -> Stage2PedalDataset:
        return Stage2PedalDataset(
            ASAP_ROOT,
            self.split_csv,
            "train",
            cache_mode=cache_mode,
        )

    def assert_samples_equal(
        self, left: dict[str, object], right: dict[str, object]
    ) -> None:
        for key in ("input_ids", "pedal_targets", "note_mask"):
            self.assertTrue(torch.equal(left[key], right[key]))
        self.assertEqual(left["metadata"], right["metadata"])

    def test_cache_equivalence(self) -> None:
        uncached = self.make_dataset("none")
        preloaded = self.make_dataset("preload")
        self.assert_samples_equal(uncached[0], preloaded[0])

    def test_preload_tokenizes_once_and_getitem_does_not_tokenize(self) -> None:
        with mock.patch.object(
            dataset_module,
            "midi_to_ids",
            wraps=dataset_module.midi_to_ids,
        ) as tokenizer:
            dataset = self.make_dataset("preload")
            self.assertEqual(tokenizer.call_count, 1)
            self.assertGreaterEqual(dataset.window_count, 2)
            dataset[0]
            dataset[1]
            dataset[0]
            self.assertEqual(tokenizer.call_count, 1)

    def test_cache_is_immutable_int16_and_accounted_from_arrays(self) -> None:
        dataset = self.make_dataset("preload")
        cached = next(iter(dataset._token_cache.values()))
        original_pedals = cached[:, NON_PEDAL_FEATURES:].copy()

        self.assertEqual(cached.dtype, np.int16)
        self.assertEqual(cached.ndim, 2)
        self.assertEqual(cached.shape[1], TOKENS_PER_NOTE)
        self.assertFalse(cached.flags.writeable)
        dataset[0]
        self.assertTrue(
            np.array_equal(
                cached[:, NON_PEDAL_FEATURES:],
                original_pedals,
            )
        )
        self.assertEqual(dataset.cached_performance_count, 1)
        self.assertEqual(
            dataset.cached_token_count,
            sum(array.size for array in dataset._token_cache.values()),
        )
        self.assertEqual(
            dataset.cached_nbytes,
            sum(array.nbytes for array in dataset._token_cache.values()),
        )
        self.assertEqual(dataset.cached_nbytes, dataset.cached_token_count * 2)

    def test_unsupported_cache_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cache_mode"):
            self.make_dataset("disk")
class CollationTests(unittest.TestCase):
    @staticmethod
    def sample(notes: int, label: str) -> dict[str, object]:
        return {
            "input_ids": torch.arange(notes * 8, dtype=torch.long) + 2,
            "pedal_targets": torch.zeros((notes, 4), dtype=torch.long),
            "note_mask": torch.ones(notes, dtype=torch.bool),
            "metadata": {"label": label},
        }

    def test_note_granularity_padding_and_masks(self) -> None:
        batch = stage2_pedal_collate_fn(
            [self.sample(2, "long"), self.sample(1, "short")]
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 16))
        self.assertEqual(tuple(batch["pedal_targets"].shape), (2, 2, 4))
        self.assertEqual(tuple(batch["note_mask"].shape), (2, 2))
        self.assertEqual(tuple(batch["token_attention_mask"].shape), (2, 16))
        self.assertTrue(torch.all(batch["input_ids"][1, 8:] == PAD_ID))
        self.assertTrue(torch.all(batch["pedal_targets"][1, 1] == -100))
        self.assertFalse(bool(batch["note_mask"][1, 1]))
        self.assertTrue(torch.all(batch["token_attention_mask"][0] == 1))
        self.assertTrue(torch.all(batch["token_attention_mask"][1, :8] == 1))
        self.assertTrue(torch.all(batch["token_attention_mask"][1, 8:] == 0))
        self.assertEqual(
            batch["metadata"], [{"label": "long"}, {"label": "short"}]
        )


if __name__ == "__main__":
    unittest.main()
