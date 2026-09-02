"""Decoders over one frozen four-class posterior.

All functions consume the same final overlap-averaged logits.  They do not
perform window inference, model updates, calibration, or temporal smoothing.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .representation import NUM_CLASSES, REPRESENTATIVES


CLASS_NAMES = ("ZERO", "LOW", "HALF", "FULL")


def softmax_final_logits(logits: np.ndarray) -> np.ndarray:
    """Return stable float64 posteriors for final logits shaped ``[..., 4]``."""

    values = np.asarray(logits)
    if values.ndim < 2 or values.shape[-1] != NUM_CLASSES:
        raise ValueError("final logits must have shape [...,4]")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("final logits contain NaN/Inf")
    shifted = values.astype(np.float64) - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    posterior = exponent / exponent.sum(axis=-1, keepdims=True)
    if not np.all(np.isfinite(posterior)) or not np.allclose(
        posterior.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise FloatingPointError("posterior normalization failed")
    return posterior


def decode_argmax(posterior: np.ndarray) -> np.ndarray:
    """Deterministic argmax; NumPy resolves exact ties to the lower index."""

    probabilities = _posterior(posterior)
    return probabilities.argmax(axis=-1).astype(np.int64)


def decode_ordered_median(posterior: np.ndarray) -> np.ndarray:
    """Smallest ordered class whose cumulative posterior is at least 0.5."""

    probabilities = _posterior(posterior)
    cumulative = probabilities.cumsum(axis=-1)
    return (cumulative >= 0.5).argmax(axis=-1).astype(np.int64)


def decode_expectation(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous representative expectation and canonical half-up MIDI integer."""

    probabilities = _posterior(posterior)
    representatives = np.asarray(REPRESENTATIVES, dtype=np.float64)
    continuous = np.clip(probabilities @ representatives, 0.0, 127.0)
    # Exact Raw-Huber/Hybrid convention: clip then nearest integer, half up.
    midi_integer = np.floor(continuous + 0.5).astype(np.int64)
    return continuous, midi_integer


def decode_auxiliary_median(posterior: np.ndarray) -> np.ndarray:
    """Hybrid auxiliary-only diagnostic, deliberately independent of regression."""

    return decode_ordered_median(posterior)


def decode_side_constrained(
    regression_cc64_integer: np.ndarray, auxiliary_posterior: np.ndarray
) -> np.ndarray:
    """Use regression for OFF/ON and auxiliary argmax only within that side."""

    raw = np.asarray(regression_cc64_integer, dtype=np.int64)
    probabilities = _posterior(auxiliary_posterior)
    if raw.shape != probabilities.shape[:-1]:
        raise ValueError("regression CC64 and auxiliary posterior shapes differ")
    if not np.all((raw >= 0) & (raw <= 127)):
        raise ValueError("regression CC64 integers must lie in [0,127]")
    off_choice = probabilities[..., :2].argmax(axis=-1)
    on_choice = probabilities[..., 2:].argmax(axis=-1) + 2
    result = np.where(raw < 64, off_choice, on_choice).astype(np.int64)
    if not np.array_equal(result >= 2, raw >= 64):
        raise AssertionError("side-constrained fusion changed an OFF/ON bit")
    return result


def stable_top2_indices(posterior: np.ndarray) -> np.ndarray:
    """Top-2 indices, descending probability and ascending class on ties."""

    probabilities = _posterior(posterior)
    return np.argsort(-probabilities, axis=-1, kind="stable")[..., :2].astype(
        np.int64
    )


def stable_sample_uniforms(
    piece_id: str, sample_shape: tuple[int, ...], *, seed: int = 42
) -> np.ndarray:
    """Stable per-piece/sample uniforms independent of batch/window traversal."""

    if seed != 42:
        raise ValueError("Phase 4 primary Top-2 seed is fixed at 42")
    digest = hashlib.sha256(f"phase4-top2:{seed}:{piece_id}".encode()).digest()
    piece_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.Generator(np.random.PCG64(piece_seed)).random(sample_shape)


def decode_top2(
    posterior: np.ndarray, *, piece_id: str, seed: int = 42
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample from renormalized Top-2 at T=1 using stable sample identities."""

    probabilities = _posterior(posterior)
    top2 = stable_top2_indices(probabilities)
    selected = np.take_along_axis(probabilities, top2, axis=-1)
    normalized = selected / selected.sum(axis=-1, keepdims=True)
    draws = stable_sample_uniforms(piece_id, probabilities.shape[:-1], seed=seed)
    classes = np.where(draws < normalized[..., 0], top2[..., 0], top2[..., 1])
    support_ok = np.all((classes == top2[..., 0]) | (classes == top2[..., 1]))
    if not support_ok:
        raise AssertionError("Top-2 sampling escaped its support")
    return classes.astype(np.int64), {
        "seed": seed,
        "temperature": 1.0,
        "rng": "PCG64 seeded by SHA256(phase4-top2:42:piece_id)",
        "top2_indices": top2,
        "top2_probabilities": normalized,
    }


def cc64_integer_to_classes(values: np.ndarray) -> np.ndarray:
    """Frozen canonical 0-25/26-63/64-103/104-127 binning."""

    raw = np.asarray(values, dtype=np.int64)
    if not np.all((raw >= 0) & (raw <= 127)):
        raise ValueError("MIDI CC64 integers must lie in [0,127]")
    return np.select(
        (raw <= 25, raw <= 63, raw <= 103), (0, 1, 2), default=3
    ).astype(np.int64)


def _posterior(value: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.ndim < 2 or probabilities.shape[-1] != NUM_CLASSES:
        raise ValueError("posterior must have shape [...,4]")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise FloatingPointError("posterior is invalid")
    if not np.allclose(
        probabilities.sum(axis=-1), 1.0, atol=1e-10, rtol=0.0
    ):
        raise ValueError("posterior rows must sum to one")
    return probabilities
