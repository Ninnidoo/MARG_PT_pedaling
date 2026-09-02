"""Canonical four-class inference and pooled validation metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.stage2_binary.validation_evaluator import replace_pedal_tokens
from src.stage2_decoder_only.model import FourClassPedalDecoderOnlyModel
from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PEDAL_TOKEN_OFFSET,
    TOKENS_PER_NOTE,
    generate_window_starts,
)
from src.stage2_four_class.model import (
    FourClassPedalEncoderDecoderModel,
    FourClassPedalEncoderModel,
)
from src.stage2_four_class.representation import NUM_CLASSES, REPRESENTATIVES


MODEL_NAMES = ("original_pt", "encoder_only", "decoder_only", "encoder_decoder")
MODEL_LABELS = {
    "original_pt": "Original PT",
    "encoder_only": "Encoder-only",
    "decoder_only": "Decoder-only FR",
    "encoder_decoder": "Encoder-Decoder FR",
}
PATTERN_COUNT = NUM_CLASSES ** 4
TRANSITION_TOLERANCE = 1
DISTRIBUTION_EPSILON = 1e-10


def as_note_tokens(values: Sequence[int] | np.ndarray) -> np.ndarray:
    tokens = np.asarray(values, dtype=np.int64)
    if tokens.ndim == 1:
        if not tokens.size or tokens.size % TOKENS_PER_NOTE:
            raise ValueError("token sequence must contain complete eight-token notes")
        tokens = tokens.reshape(-1, TOKENS_PER_NOTE)
    if tokens.ndim != 2 or tokens.shape[1] != TOKENS_PER_NOTE or not len(tokens):
        raise ValueError("tokens must have shape [N,8]")
    return tokens


def canonical_classes_from_tokens(values: Sequence[int] | np.ndarray) -> np.ndarray:
    from src.stage2_four_class.representation import classify_cc64

    tokens = as_note_tokens(values)
    raw = tokens[:, NON_PEDAL_FEATURES:] - PEDAL_TOKEN_OFFSET
    classes = np.asarray(classify_cc64(raw), dtype=np.int64)
    if classes.shape != (len(tokens), 4) or not np.all((classes >= 0) & (classes < 4)):
        raise AssertionError("canonical class conversion escaped [0,3]")
    return classes


def classes_to_candidate_tokens(
    source_tokens: Sequence[int] | np.ndarray, classes: np.ndarray
) -> np.ndarray:
    source = as_note_tokens(source_tokens)
    predicted = np.asarray(classes, dtype=np.int64)
    if predicted.shape != (len(source), 4) or not np.all((predicted >= 0) & (predicted < 4)):
        raise ValueError("predicted classes must have shape [N,4] in [0,3]")
    raw = np.asarray(REPRESENTATIVES, dtype=np.int64)[predicted]
    return replace_pedal_tokens(source, raw)


def load_four_class_model(
    architecture: str,
    checkpoint_path: str | Path,
    *,
    pretrained_checkpoint: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise TypeError("canonical checkpoint must contain model_state")
    configuration = dict(payload.get("configuration", {}))
    common = {
        "torch_dtype": torch.float32,
        "attn_implementation": "eager",
    }
    if architecture == "encoder_only":
        model = FourClassPedalEncoderModel.from_pretrained(
            pretrained_checkpoint, **common
        )
    elif architecture == "decoder_only":
        model = FourClassPedalDecoderOnlyModel.from_pretrained_performance_embeddings(
            pretrained_checkpoint,
            num_layers=int(configuration.get("decoder_only_num_layers", 2)),
            init_seed=int(configuration.get("decoder_only_init_seed", 42)),
            **common,
        )
    elif architecture == "encoder_decoder":
        model = FourClassPedalEncoderDecoderModel.from_pretrained_encoder(
            pretrained_checkpoint,
            decoder_init_seed=int(configuration.get("decoder_init_seed", 42)),
            freeze_encoder=False,
            **common,
        )
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("checkpoint contains a non-finite model parameter")
    metadata = {
        "epoch": int(payload["epoch"]),
        "validation_ce": float(payload["validation_ce"]),
        "configuration": configuration,
    }
    return model.to(device).eval(), metadata


def infer_four_class_pedals(
    model: torch.nn.Module,
    source_tokens: Sequence[int] | np.ndarray,
    *,
    architecture: str,
    device: torch.device,
    window_notes: int = 512,
    stride_notes: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Average canonical window logits; AR windows are independently free-running."""

    tokens = as_note_tokens(source_tokens)
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    starts = generate_window_starts(len(tokens), window_notes, stride_notes)
    accumulated = torch.zeros((len(tokens), 4, 4), dtype=torch.float32, device=device)
    contributions = torch.zeros(len(tokens), dtype=torch.float32, device=device)
    ground_truth_history_access_count = 0
    with torch.inference_mode():
        for start in starts:
            end = min(start + window_notes, len(tokens))
            input_ids = torch.from_numpy(masked[start:end].reshape(1, -1)).long().to(device)
            token_attention_mask = torch.ones_like(input_ids)
            note_mask = torch.ones((1, end - start), dtype=torch.bool, device=device)
            use_amp = device.type == "cuda"
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if use_amp else torch.bfloat16,
                enabled=use_amp,
            ):
                if architecture == "encoder_only":
                    output = model(
                        input_ids=input_ids,
                        token_attention_mask=token_attention_mask,
                        note_mask=note_mask,
                    )
                    logits = output.logits[0]
                elif architecture == "decoder_only":
                    window_classes, window_logits = model.generate_with_logits(
                        input_ids, token_attention_mask, note_mask
                    )
                    logits = window_logits[0]
                    if not torch.equal(window_classes[0], logits.argmax(dim=-1)):
                        raise AssertionError("Decoder-only greedy classes/logits disagree")
                elif architecture == "encoder_decoder":
                    output = model(
                        input_ids=input_ids,
                        token_attention_mask=token_attention_mask,
                        note_mask=note_mask,
                        decode_mode="greedy",
                    )
                    logits = output.logits[0]
                else:
                    raise ValueError(f"unsupported architecture: {architecture}")
            logits = logits[: end - start].float()
            if tuple(logits.shape) != (end - start, 4, 4):
                raise RuntimeError("four-class window logits have the wrong shape")
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("non-finite inference logits")
            accumulated[start:end] += logits
            contributions[start:end] += 1
    if bool(torch.any(contributions == 0)):
        raise AssertionError("one or more notes received no window prediction")
    mean_logits = accumulated / contributions[:, None, None]
    if not bool(torch.isfinite(mean_logits).all()):
        raise FloatingPointError("non-finite overlap-averaged logits")
    classes = mean_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    candidate = classes_to_candidate_tokens(tokens, classes)
    if not np.array_equal(
        candidate.reshape(-1, TOKENS_PER_NOTE)[:, :NON_PEDAL_FEATURES],
        tokens[:, :NON_PEDAL_FEATURES],
    ):
        raise AssertionError("Stage 2 changed non-pedal tokens")
    return classes, candidate, {
        "notes": len(tokens),
        "windows": len(starts),
        "window_notes": window_notes,
        "stride_notes": stride_notes,
        "overlap_merge": "free-running window step-logit mean then argmax",
        "deterministic_argmax": True,
        "ground_truth_pedal_history_access_count": ground_truth_history_access_count,
        "non_pedal_tokens_preserved": True,
        "finite_logits": True,
    }


def confusion_from_pairs(candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(candidate, dtype=np.int64)
    reference = np.asarray(target, dtype=np.int64)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("aligned classes must share shape [N,4]")
    if not np.all((predicted >= 0) & (predicted < 4)) or not np.all(
        (reference >= 0) & (reference < 4)
    ):
        raise ValueError("aligned classes must lie in [0,3]")
    return np.bincount(
        reference.reshape(-1) * 4 + predicted.reshape(-1), minlength=16
    ).reshape(4, 4)


def classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (4, 4) or np.any(matrix < 0) or matrix.sum() <= 0:
        raise ValueError("confusion matrix must be non-empty 4x4")
    classes = []
    for index in range(4):
        tp = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        support = int(matrix[index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes.append(
            {"class_id": index, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    return {
        "token_accuracy": float(np.trace(matrix) / matrix.sum()),
        "macro_f1": float(np.mean([item["f1"] for item in classes])),
        "class_metrics": classes,
        "confusion_matrix": matrix.tolist(),
        "prediction_class_distribution": (matrix.sum(axis=0) / matrix.sum()).tolist(),
        "target_class_distribution": (matrix.sum(axis=1) / matrix.sum()).tolist(),
        "pedal_samples": int(matrix.sum()),
        "zero_support_convention": "precision/recall/F1=0",
    }


def pattern_ids(classes: np.ndarray) -> np.ndarray:
    values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all((values >= 0) & (values < 4)):
        raise ValueError("patterns require [N,4] canonical classes")
    return values @ np.asarray([64, 16, 4, 1], dtype=np.int64)


def pattern_metrics(candidate_histogram: np.ndarray, target_histogram: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(candidate_histogram, dtype=np.float64)
    target = np.asarray(target_histogram, dtype=np.float64)
    if predicted.shape != (PATTERN_COUNT,) or target.shape != (PATTERN_COUNT,):
        raise ValueError("joint pattern histograms must have exactly 256 bins")
    if predicted.sum() <= 0 or target.sum() <= 0:
        raise ValueError("joint pattern histograms must be non-empty")
    p = predicted / predicted.sum() + DISTRIBUTION_EPSILON
    q = target / target.sum() + DISTRIBUTION_EPSILON
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * float(
        np.sum(p * np.log2(p / midpoint)) + np.sum(q * np.log2(q / midpoint))
    )
    divergence = max(divergence, 0.0)
    if not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0.0) or not np.isclose(
        q.sum(), 1.0, atol=1e-12, rtol=0.0
    ):
        raise AssertionError("normalized distribution does not sum to one")
    return {
        "js_divergence_base2": divergence,
        "js_distance_base2": math.sqrt(divergence),
        "intersection": float(np.minimum(p, q).sum()),
        "candidate_distribution": p.tolist(),
        "target_distribution": q.tolist(),
        "bins": PATTERN_COUNT,
        "epsilon": DISTRIBUTION_EPSILON,
    }


def pooled_transition_counts(
    candidate: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    from scripts.audit_pedal_event_metric_tolerance_mini import match_pair

    matched = match_pair(dict(candidate), dict(target), TRANSITION_TOLERANCE)
    result: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for direction in ("UP", "DOWN"):
        candidate_count = sum(event["direction"] == direction for event in candidate["transitions"])
        target_count = sum(event["direction"] == direction for event in target["transitions"])
        tp = sum(first["direction"] == direction for first, _ in matched)
        fp = candidate_count - tp
        fn = target_count - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[direction.lower()] = {
            "candidate": candidate_count,
            "reference": target_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["pooled"] = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate": total_tp + total_fp,
        "reference": total_tp + total_fn,
        "tolerance_distinct_onsets": TRANSITION_TOLERANCE,
    }
    return result
