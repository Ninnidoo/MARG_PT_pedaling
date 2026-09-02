"""Canonical data adapter for Binary 2-Slot v0.

This adapter reuses the frozen cache provenance, official PT tokenization,
cache-to-PT note alignment, and maximum-margin ownership.  It deliberately
does not expose any Custom Event v0 target.  Human-side binary interval
primitives are reconstructed from the canonical raw MIDI and are compressed
online by the rollout code.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PAD_ID,
    TOKENS_PER_NOTE,
)
from src.stage2_event_model.dataset import (
    DEFAULT_ALIGNMENT_ROOT,
    CustomEventWindowDataset,
)
from src.stage2_event_tokenizer.binary_2slot import (
    DOWN,
    OFF,
    ON,
    UP,
    BinaryTransition,
    state_after,
)
from src.stage2_event_tokenizer.tokenizer import parse_raw_midi
from src.stage2_event_tokenizer.tokenizer_v1 import _tick_second_converters


WINDOW_NOTES = 512
STRIDE_NOTES = 256
BOUNDARY_CONTEXT_NOTES = 511


@dataclass(frozen=True)
class HumanIntervalPrimitive:
    """Uncompressed human target information for one represented interval."""

    region: str
    global_index: int
    main_onset_index: int | None
    left_seconds: float
    right_seconds: float
    human_start_state: int
    human_transitions: tuple[BinaryTransition, ...]

    def __post_init__(self) -> None:
        if self.region not in {"PRE", "MAIN", "POST"}:
            raise ValueError(f"invalid interval region: {self.region}")
        if self.human_start_state not in {OFF, ON}:
            raise ValueError("human start state must be OFF or ON")
        if self.right_seconds <= self.left_seconds:
            raise ValueError("represented interval must have positive duration")
        state_after(self.human_start_state, self.human_transitions)

    @property
    def duration_seconds(self) -> float:
        return self.right_seconds - self.left_seconds

    @property
    def raw_transition_count(self) -> int:
        return len(self.human_transitions)

    @property
    def human_end_state(self) -> int:
        return state_after(self.human_start_state, self.human_transitions)


@dataclass(frozen=True)
class PerformanceTimeline:
    performance_id: str
    source_midi: str
    pre: HumanIntervalPrimitive
    main: tuple[HumanIntervalPrimitive, ...]
    post: HumanIntervalPrimitive

    @property
    def represented_intervals(self) -> tuple[HumanIntervalPrimitive, ...]:
        return (self.pre,) + self.main + (self.post,)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_binary_crossings(raw_events: Sequence[Any]) -> tuple[tuple[int, str], ...]:
    """Frozen same-tick-last projection followed by the CC64=64 threshold."""

    grouped: dict[int, list[Any]] = {}
    for event in raw_events:
        grouped.setdefault(int(event.tick), []).append(event)
    state = OFF
    result: list[tuple[int, str]] = []
    for tick in sorted(grouped):
        events = sorted(grouped[tick], key=lambda event: (event.track, event.message_index))
        next_state = ON if int(events[-1].value) >= 64 else OFF
        if next_state != state:
            result.append((tick, DOWN if state == OFF else UP))
            state = next_state
    return tuple(result)


def _interval_transitions(
    crossings: Sequence[tuple[int, str, float]],
    left: float,
    right: float,
    *,
    include_right: bool,
) -> tuple[BinaryTransition, ...]:
    denominator = right - left
    if denominator <= 0:
        raise ValueError("interval duration must be positive")
    result = []
    for tick, direction, seconds in crossings:
        inside = left <= seconds <= right if include_right else left <= seconds < right
        if inside:
            tau = (seconds - left) / denominator
            if abs(tau) <= 1e-12:
                tau = 0.0
            if abs(tau - 1.0) <= 1e-12:
                tau = 1.0
            result.append(BinaryTransition(direction, float(tau), source_tick=tick))
    return tuple(result)


def _state_before(crossings: Sequence[tuple[int, str, float]], boundary: float) -> int:
    state = OFF
    for _, _, seconds in crossings:
        if seconds >= boundary:
            break
        state = ON if state == OFF else OFF
    return state


def build_performance_timeline(
    source_midi: str | Path,
    onset_ticks: Sequence[int] | np.ndarray,
    latest_note_off_tick: int,
    *,
    performance_id: str,
) -> PerformanceTimeline:
    """Build PRE/MAIN/POST raw binary primitives in canonical musical time."""

    raw = parse_raw_midi(source_midi)
    cache_onsets = np.asarray(onset_ticks, dtype=np.int64)
    complete_onsets = np.asarray(
        sorted({note[3] for note in raw.note_signature}), dtype=np.int64
    )
    if not np.array_equal(cache_onsets, complete_onsets):
        raise RuntimeError("canonical cache/raw distinct-onset identity mismatch")
    if int(latest_note_off_tick) != raw.latest_note_off:
        raise RuntimeError("canonical cache/raw latest-note-off identity mismatch")
    tick_to_seconds, _ = _tick_second_converters(source_midi)
    crossings = tuple(
        (tick, direction, float(tick_to_seconds(tick)))
        for tick, direction in _effective_binary_crossings(raw.raw_cc64)
    )
    onset_seconds = tuple(float(tick_to_seconds(int(tick))) for tick in cache_onsets)
    note_end_seconds = float(tick_to_seconds(int(latest_note_off_tick)))
    if note_end_seconds <= onset_seconds[-1]:
        raise ValueError("final MAIN interval is not positive")

    first = onset_seconds[0]
    pre_left = first - 1.0
    pre = HumanIntervalPrimitive(
        region="PRE",
        global_index=0,
        main_onset_index=None,
        left_seconds=pre_left,
        right_seconds=first,
        human_start_state=_state_before(crossings, pre_left),
        human_transitions=_interval_transitions(
            crossings, pre_left, first, include_right=False
        ),
    )

    main: list[HumanIntervalPrimitive] = []
    for index, left in enumerate(onset_seconds):
        right = note_end_seconds if index + 1 == len(onset_seconds) else onset_seconds[index + 1]
        main.append(
            HumanIntervalPrimitive(
                region="MAIN",
                global_index=index + 1,
                main_onset_index=index,
                left_seconds=left,
                right_seconds=right,
                human_start_state=_state_before(crossings, left),
                human_transitions=_interval_transitions(
                    crossings, left, right, include_right=False
                ),
            )
        )
    post_right = note_end_seconds + 1.0
    post = HumanIntervalPrimitive(
        region="POST",
        global_index=len(main) + 1,
        main_onset_index=None,
        left_seconds=note_end_seconds,
        right_seconds=post_right,
        human_start_state=_state_before(crossings, note_end_seconds),
        human_transitions=_interval_transitions(
            crossings, note_end_seconds, post_right, include_right=True
        ),
    )

    # The represented intervals must reproduce the raw human trajectory across
    # every shared boundary. Events exactly on a boundary belong to the right.
    represented = (pre,) + tuple(main) + (post,)
    for left_interval, right_interval in zip(represented, represented[1:]):
        if left_interval.human_end_state != right_interval.human_start_state:
            raise AssertionError("human interval state chain is discontinuous")
    return PerformanceTimeline(
        performance_id=performance_id,
        source_midi=str(source_midi),
        pre=pre,
        main=tuple(main),
        post=post,
    )


def assert_masked_real_tokens(tokens: np.ndarray | torch.Tensor) -> None:
    values = torch.as_tensor(tokens)
    if values.ndim != 2 or values.shape[1] != TOKENS_PER_NOTE:
        raise ValueError("real-note tokens must have shape [N,8]")
    if not bool(torch.all(values[:, NON_PEDAL_FEATURES:] == MASK_ID)):
        raise AssertionError("human Pedal1-4 leakage: all pedal features must be MASK")


def mask_human_pedal_features(tokens: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Replace all four official PT human-pedal features with MASK=1."""

    values = (
        tokens.detach().clone().to(dtype=torch.long)
        if isinstance(tokens, torch.Tensor)
        else torch.from_numpy(np.array(tokens, dtype=np.int64, copy=True)).long()
    )
    if values.ndim != 2 or values.shape[1] != TOKENS_PER_NOTE:
        raise ValueError("PT note tokens must have shape [N,8]")
    values[:, NON_PEDAL_FEATURES:] = MASK_ID
    assert_masked_real_tokens(values)
    return values


def build_boundary_input(
    masked_real_tokens: np.ndarray | torch.Tensor,
    region: str,
) -> dict[str, torch.Tensor | int | str]:
    """Build an attended all-MASK pseudo-note in a separate PT pass."""

    real = torch.from_numpy(np.array(masked_real_tokens, dtype=np.int64, copy=True)).long()
    assert_masked_real_tokens(real)
    if region == "PRE":
        context = real[:BOUNDARY_CONTEXT_NOTES]
        query_position = 0
        notes = torch.cat((torch.full((1, 8), MASK_ID), context), dim=0)
    elif region == "POST":
        context = real[-BOUNDARY_CONTEXT_NOTES:]
        query_position = len(context)
        notes = torch.cat((context, torch.full((1, 8), MASK_ID)), dim=0)
    else:
        raise ValueError("boundary region must be PRE or POST")
    if len(notes) > WINDOW_NOTES:
        raise AssertionError("boundary pass exceeded 512 notes")
    token_mask = torch.ones(notes.numel(), dtype=torch.long)
    return {
        "region": region,
        "input_ids": notes.reshape(-1).clone(),
        "token_attention_mask": token_mask,
        "note_mask": torch.ones(len(notes), dtype=torch.bool),
        "query_position": int(query_position),
        "real_context_notes": int(len(context)),
    }


class Binary2SlotPerformanceDataset(CustomEventWindowDataset):
    """Window adapter with raw human primitives and no precomputed targets."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        window_notes: int = WINDOW_NOTES,
        stride_notes: int = STRIDE_NOTES,
        cache_input_tokens: bool = False,
        entry_limit: int | None = None,
        performance_indices: Sequence[int] | None = None,
        alignment_root: str | Path = DEFAULT_ALIGNMENT_ROOT,
    ) -> None:
        if window_notes != WINDOW_NOTES or stride_notes != STRIDE_NOTES:
            raise ValueError("Binary 2-Slot v0 freezes 512-note/stride-256 windows")
        if entry_limit is not None and performance_indices is not None:
            raise ValueError("entry_limit and performance_indices are mutually exclusive")
        selected = None if performance_indices is None else tuple(sorted(set(int(v) for v in performance_indices)))
        if selected is not None and (not selected or selected[0] < 0):
            raise ValueError("performance_indices must be non-empty non-negative train/validation indices")
        super().__init__(
            cache_root,
            split,
            window_notes=window_notes,
            stride_notes=stride_notes,
            cache_input_tokens=cache_input_tokens,
            entry_limit=(selected[-1] + 1 if selected is not None else entry_limit),
            alignment_root=alignment_root,
        )
        if selected is not None:
            wanted = set(selected)
            self.performances = [
                performance for performance in self.performances
                if int(performance.entry["performance_index"]) in wanted
            ]
            found = {int(performance.entry["performance_index"]) for performance in self.performances}
            if found != wanted:
                raise ValueError("requested performance index is not in the selected split")
            self.windows = [
                (performance_index, window_index)
                for performance_index, performance in enumerate(self.performances)
                for window_index in range(len(performance.ownership.window_starts))
            ]
        self._timeline_cache: dict[int, PerformanceTimeline] = {}
        self._masked_token_cache: dict[int, np.ndarray] = {}
        self._validate_ownership_order()

    def _validate_ownership_order(self) -> None:
        for performance in self.performances:
            ownership = performance.ownership
            membership = np.zeros(performance.num_onsets, dtype=np.int64)
            flattened: list[int] = []
            for window_index, owned in enumerate(ownership.owned_onset_indices):
                membership[owned] += 1
                flattened.extend(int(value) for value in owned)
                positions = ownership.owned_local_representative_indices[window_index]
                width = ownership.window_ends[window_index] - ownership.window_starts[window_index]
                if np.any((positions < 0) | (positions >= width)):
                    raise AssertionError("owned representative lies outside owner window")
            if not np.all(membership == 1):
                raise AssertionError("every onset must have exactly one owner")
            if np.any(np.diff(ownership.owner_window_index) < 0):
                raise AssertionError("owner window index is not non-decreasing")
            if flattened != list(range(performance.num_onsets)):
                raise AssertionError("owned global onset indices are not strictly chronological")

    def _load_input_tokens(self, performance_index: int) -> np.ndarray:
        tokens = super()._load_input_tokens(performance_index)
        performance = self.performances[performance_index]
        ioi = tokens[:, 1].astype(np.int64) - self._tokenizer_config.timing_start
        expected = np.diff(
            np.concatenate((np.asarray([0], dtype=np.int64), performance.alignment.normalized_onset))
        )
        lower, upper = self._tokenizer_config.valid_id_range[1]
        expected = np.clip(expected, lower - self._tokenizer_config.timing_start, upper - 1 - self._tokenizer_config.timing_start)
        if not np.array_equal(ioi, expected):
            raise ValueError("PT IOI sequence differs from audited normalized onset order")
        return tokens

    def timeline(self, performance_index: int) -> PerformanceTimeline:
        if performance_index not in self._timeline_cache:
            performance = self.performances[performance_index]
            if _sha256_file(performance.entry["source_midi"]) != performance.entry["source_sha256"]:
                raise RuntimeError("canonical source MIDI SHA changed")
            with np.load(performance.cache_path, allow_pickle=False) as cache:
                timeline = build_performance_timeline(
                    performance.entry["source_midi"],
                    cache["onset_ticks"].astype(np.int64, copy=True),
                    int(cache["latest_note_off_tick"]),
                    performance_id=str(performance.entry["performance_path"]),
                )
            if len(timeline.main) != performance.num_onsets:
                raise AssertionError("timeline/ownership onset count mismatch")
            self._timeline_cache[performance_index] = timeline
        return self._timeline_cache[performance_index]

    def masked_performance_tokens(self, performance_index: int) -> np.ndarray:
        if performance_index not in self._masked_token_cache:
            tokens = mask_human_pedal_features(
                self._load_input_tokens(performance_index)
            ).numpy().astype(np.int16, copy=False)
            tokens.setflags(write=False)
            if self.cache_input_tokens:
                self._masked_token_cache[performance_index] = tokens
            return tokens
        return self._masked_token_cache[performance_index]

    def boundary_input(self, performance_index: int, region: str) -> dict[str, Any]:
        result = build_boundary_input(self.masked_performance_tokens(performance_index), region)
        result["performance_index"] = performance_index
        result["performance_id"] = self.timeline(performance_index).performance_id
        result["interval"] = self.timeline(performance_index).pre if region == "PRE" else self.timeline(performance_index).post
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        performance_index, window_index = self.windows[index]
        performance = self.performances[performance_index]
        ownership = performance.ownership
        start = ownership.window_starts[window_index]
        end = ownership.window_ends[window_index]
        tokens = self.masked_performance_tokens(performance_index)[start:end]
        owned = ownership.owned_onset_indices[window_index]
        local = ownership.owned_local_representative_indices[window_index]
        if np.any((local < 0) | (local >= len(tokens))):
            raise AssertionError("owned representative lies outside returned window")
        timeline = self.timeline(performance_index)
        return {
            "input_ids": torch.from_numpy(tokens.reshape(-1).copy()).long(),
            "note_mask": torch.ones(len(tokens), dtype=torch.bool),
            "owned_global_onset_indices": torch.from_numpy(owned.copy()).long(),
            "owned_representative_positions": torch.from_numpy(local.copy()).long(),
            "human_intervals": tuple(timeline.main[int(i)] for i in owned),
            "metadata": {
                "split": self.split,
                "performance_index": performance_index,
                "canonical_performance_index": int(performance.entry["performance_index"]),
                "performance_id": timeline.performance_id,
                "window_index": window_index,
                "window_start_note": start,
                "window_end_note": end,
                "total_windows": len(ownership.window_starts),
                "total_notes": performance.num_notes,
            },
        }


class PerformanceMajorSampler(Sampler[int]):
    """Shuffle performances while preserving chronological windows within each."""

    def __init__(self, dataset: Binary2SlotPerformanceDataset, *, shuffle: bool, seed: int = 42) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        by_performance: list[list[int]] = [[] for _ in dataset.performances]
        for dataset_index, (performance_index, _) in enumerate(dataset.windows):
            by_performance[performance_index].append(dataset_index)
        self.indices_by_performance = tuple(tuple(values) for values in by_performance)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self.indices_by_performance)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for performance_index in order:
            yield from self.indices_by_performance[performance_index]


class PerformanceMajorBatchSampler(Sampler[list[int]]):
    """Chronological micro-batches that never cross a performance boundary.

    A trainer can detect the first/last batch from window metadata, execute PRE
    before the first batch and POST after the last, and keep one state across
    every micro-batch/gradient-accumulation/optimizer-step boundary.
    """

    def __init__(
        self,
        dataset: Binary2SlotPerformanceDataset,
        *,
        batch_size: int,
        shuffle_performances: bool,
        seed: int = 42,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.performance_sampler = PerformanceMajorSampler(
            dataset, shuffle=shuffle_performances, seed=seed
        )
        self.batch_size = int(batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.performance_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.performance_sampler.indices_by_performance
        )

    def __iter__(self) -> Iterator[list[int]]:
        index_to_performance = {
            index: performance_index
            for performance_index, indices in enumerate(
                self.performance_sampler.indices_by_performance
            )
            for index in indices
        }
        batch: list[int] = []
        active_performance: int | None = None
        for index in self.performance_sampler:
            performance = index_to_performance[index]
            if batch and (performance != active_performance or len(batch) == self.batch_size):
                yield batch
                batch = []
            active_performance = performance
            batch.append(index)
        if batch:
            yield batch


def binary_2slot_collate_fn(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad owner windows without materializing any online N/tau target."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    note_counts = [sample["input_ids"].numel() // TOKENS_PER_NOTE for sample in samples]
    owned_counts = [len(sample["owned_global_onset_indices"]) for sample in samples]
    max_notes = max(note_counts)
    max_owned = max(owned_counts, default=0)
    input_ids = torch.full((batch_size, max_notes * TOKENS_PER_NOTE), PAD_ID, dtype=torch.long)
    token_attention_mask = torch.zeros_like(input_ids)
    note_mask = torch.zeros((batch_size, max_notes), dtype=torch.bool)
    owned_mask = torch.zeros((batch_size, max_owned), dtype=torch.bool)
    owned_global = torch.full((batch_size, max_owned), -1, dtype=torch.long)
    owned_positions = torch.full((batch_size, max_owned), -1, dtype=torch.long)
    for batch_index, (sample, notes, owned_count) in enumerate(zip(samples, note_counts, owned_counts)):
        length = notes * TOKENS_PER_NOTE
        input_ids[batch_index, :length] = sample["input_ids"]
        token_attention_mask[batch_index, :length] = 1
        note_mask[batch_index, :notes] = True
        if owned_count:
            owned_mask[batch_index, :owned_count] = True
            owned_global[batch_index, :owned_count] = sample["owned_global_onset_indices"]
            owned_positions[batch_index, :owned_count] = sample["owned_representative_positions"]
    reshaped = input_ids.reshape(batch_size, max_notes, TOKENS_PER_NOTE)
    valid_notes = note_mask.unsqueeze(-1).expand_as(reshaped)
    pedal_valid = valid_notes[:, :, NON_PEDAL_FEATURES:]
    if not bool(torch.all(reshaped[:, :, NON_PEDAL_FEATURES:][pedal_valid] == MASK_ID)):
        raise AssertionError("collated real note exposes human Pedal1-4")
    return {
        "input_ids": input_ids,
        "token_attention_mask": token_attention_mask,
        "note_mask": note_mask,
        "owned_onset_mask": owned_mask,
        "owned_global_onset_indices": owned_global,
        "owned_representative_positions": owned_positions,
        "human_intervals": [sample["human_intervals"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }
