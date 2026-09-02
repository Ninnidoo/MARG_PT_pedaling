#!/usr/bin/env python3
"""Standalone Run B State-Anchored Transition v1 full-training driver."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage2_binary_2slot_full_v0 as common
from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import load_frozen_human
from scripts.run_stage2_4class_validation_eval_v0 import (
    SPLIT_CSV,
    STAGE1_MANIFEST,
    finalize_transition,
    sum_transition,
)
from src.stage2_binary.canonical_stage1 import sha256_file, signature_sha256
from src.stage2_binary_2slot.checkpointing import (
    atomic_save_resumable_checkpoint,
    build_resumable_epoch_payload,
    restore_resumable_epoch_payload,
)
from src.stage2_binary_2slot.frozen_pt_validation import load_frozen_pt_piece
from src.stage2_binary_2slot.frozen_validation_mapping import (
    build_frozen_human_reference_map,
    references_by_piece,
)
from src.stage2_event_model.dataset import CANONICAL_ALIGNMENT_ID, CANONICAL_CACHE_ID
from src.stage2_four_class.validation_evaluator import pooled_transition_counts
from src.stage2_state_anchored.config import (
    MODE_CLASS_WEIGHTS,
    SEED,
    StateAnchoredModelConfig,
)
from src.stage2_state_anchored.dataset import (
    MODE_TO_ID,
    StateAnchoredWindowDataset,
    diagnose_ownership,
    state_anchored_collate_fn,
)
from src.stage2_state_anchored.decoding import (
    CHANGE_ID,
    RETURN_ID,
    state_mode_conflicts,
)
from src.stage2_state_anchored.inference import (
    assemble_performance_owner_logits,
    collect_frozen_pt_piece_logits,
    decode_performance,
)
from src.stage2_state_anchored.model import StateAnchoredTransitionModelV1
from src.stage2_state_anchored.serialization import (
    reconstruct_modeled_events,
    render_state_anchored_candidate,
)
from src.stage2_state_anchored.targets import MODE_NAMES


RUN_ROOT = ROOT / "analysis/stage2_state_anchored_transition_v1_full_v0"
SMOKE_ROOT = ROOT / "analysis/stage2_state_anchored_transition_v1_smoke"
RUN_A_ROOT = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v1"
RUN_A_COMMON = ROOT / "analysis/runA_to_runB_transition/runA_common_horizon_metrics.json"
CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
ALIGNMENT_ROOT = ROOT / "analysis/custom_event_model_v0_note_alignment"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
EXPECTED_STAGE1_MANIFEST_SHA = "f32e91da5d2196edee825236daa577f18709cf7d49e289d8f18ae8ea637b0037"
MAX_EPOCHS = 10
WINDOW_MICRO_BATCH = 4
PERFORMANCES_PER_STEP = 4
AMP_INIT_SCALE = 1024.0
MIN_FREE_BYTES = 7 * 2**30
RUN_A_P = 0.589162
RUN_A_R = 0.576268
RUN_A_F1 = 0.582644
RUN_A_STATE = 0.488841

common.RUN_ROOT = RUN_ROOT


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "training_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for row in read_csv(path):
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                converted[key] = None
                continue
            try:
                converted[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                converted[key] = value
        result.append(converted)
    return result


def replace_epoch(table: list[dict[str, Any]], row: dict[str, Any]) -> None:
    epoch = int(row["epoch"])
    table[:] = [item for item in table if int(item["epoch"]) != epoch]
    table.append(row)
    table.sort(key=lambda item: int(item["epoch"]))


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def chunks(values: Sequence[int], size: int = WINDOW_MICRO_BATCH) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def forward_batch(model, batch: Mapping[str, Any]):
    return model(
        input_ids=batch["input_ids"],
        token_attention_mask=batch["token_attention_mask"],
        note_mask=batch["note_mask"],
        owned_representative_positions=batch["owned_representative_positions"],
        owned_onset_mask=batch["owned_onset_mask"],
    )


def build_model(device: torch.device) -> StateAnchoredTransitionModelV1:
    model = StateAnchoredTransitionModelV1.from_pretrained(
        PRETRAINED,
        head_init_seed=SEED,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    if (
        model.hidden_size != 768
        or int(model.encoder.config.num_hidden_layers) != 10
        or model.encoder_parameter_count != 103_271_424
        or model.prediction_head_parameter_count != 5_383
    ):
        raise RuntimeError("Run B pretrained architecture identity changed")
    if model.head_init_seed != SEED or hasattr(model, "recurrent_state") or hasattr(model, "state_embedding"):
        raise AssertionError("Run B fresh-head/non-recurrent invariant failed")
    return model


def build_optimizer(model) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": common.ENCODER_LR, "group_name": "encoder"},
            {"params": list(model.prediction_head_parameters()), "lr": common.HEAD_LR, "group_name": "run_b_heads"},
        ],
        weight_decay=common.WEIGHT_DECAY,
    )


def raw_loss_sums(output, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor | int | float]:
    state_valid = batch["state_valid_mask"].bool()
    mode_valid = batch["mode_valid_mask"].bool()
    timing_mask = batch["timing_mask"].bool()
    state_targets = batch["state_targets"]
    mode_targets = batch["mode_targets"]
    timing_targets = batch["timing_targets"]
    weights = torch.tensor(MODE_CLASS_WEIGHTS, dtype=torch.float32, device=output.mode_logits.device)
    state_sum = F.cross_entropy(output.state_logits[state_valid].float(), state_targets[state_valid], reduction="sum")
    selected_modes = mode_targets[mode_valid]
    mode_sum = F.cross_entropy(
        output.mode_logits[mode_valid].float(), selected_modes, weight=weights, reduction="sum",
    )
    timing_sum = (
        F.smooth_l1_loss(
            output.timing_predictions[timing_mask].float(), timing_targets[timing_mask].float(),
            beta=0.1, reduction="sum",
        )
        if bool(timing_mask.any()) else output.timing_predictions.sum() * 0.0
    )
    return {
        "state_sum": state_sum,
        "mode_sum": mode_sum,
        "timing_sum": timing_sum,
        "state_den": int(state_valid.sum().item()),
        "mode_den": float(weights[selected_modes].sum().item()),
        "mode_count": int(mode_valid.sum().item()),
        "timing_den": int(timing_mask.sum().item()),
    }


def performance_denominators(dataset, performance_indices: Sequence[int]) -> dict[str, float]:
    state = mode = timing = 0.0
    for index in performance_indices:
        targets = dataset.performance_targets(index)
        state += len(targets.onset_states)
        for interval in targets.intervals:
            mode += MODE_CLASS_WEIGHTS[MODE_TO_ID[interval.mode]]
            timing += sum(interval.timing_mask)
    if state <= 0 or mode <= 0 or timing <= 0:
        raise ValueError("empty Run B performance-group denominator")
    return {"state": state, "mode": mode, "timing": timing}


def empty_loss_total() -> dict[str, float]:
    return {"state_sum": 0.0, "state_den": 0.0, "mode_sum": 0.0, "mode_den": 0.0,
            "timing_sum": 0.0, "timing_den": 0.0}


def add_loss_total(total: dict[str, float], terms: Mapping[str, Any]) -> None:
    for key in ("state_sum", "state_den", "mode_sum", "mode_den", "timing_sum", "timing_den"):
        value = terms[key]
        total[key] += float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)


def finish_loss(total: Mapping[str, float]) -> dict[str, float]:
    result = {
        "state_loss": total["state_sum"] / total["state_den"],
        "mode_loss": total["mode_sum"] / total["mode_den"],
        "timing_loss": total["timing_sum"] / total["timing_den"],
    }
    result["total_loss"] = sum(result.values())
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Run B aggregate loss is non-finite")
    return result


def replay_performance_group(
    model, dataset, performance_indices: Sequence[int], device: torch.device,
    *, amp: bool, backward=None,
) -> dict[str, float]:
    denominators = performance_denominators(dataset, performance_indices)
    total = empty_loss_total()
    for performance_index in performance_indices:
        window_indices = common.window_dataset_indices(dataset, performance_index)
        for selected in chunks(window_indices):
            batch = move_batch(state_anchored_collate_fn([dataset[index] for index in selected]), device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                output = forward_batch(model, batch)
                terms = raw_loss_sums(output, batch)
                loss = (
                    terms["state_sum"] / denominators["state"]
                    + terms["mode_sum"] / denominators["mode"]
                    + terms["timing_sum"] / denominators["timing"]
                )
            if backward is not None:
                backward(loss)
            add_loss_total(total, terms)
    return {**finish_loss(total), **{f"denominator_{key}": value for key, value in denominators.items()}}


def performance_inference(model, dataset, performance_index: int, device: torch.device, *, amp: bool):
    chunks_value = []
    loss_total = empty_loss_total()
    for selected in chunks(common.window_dataset_indices(dataset, performance_index)):
        batch = move_batch(state_anchored_collate_fn([dataset[index] for index in selected]), device)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            output = forward_batch(model, batch)
            terms = raw_loss_sums(output, batch)
        add_loss_total(loss_total, terms)
        for batch_index in range(len(selected)):
            count = int(batch["owned_onset_mask"][batch_index].sum().item())
            chunks_value.append((
                batch["owned_global_onset_indices"][batch_index, :count].detach().cpu(),
                output.state_logits[batch_index, :count].detach(),
                output.mode_logits[batch_index, :count].detach(),
                output.timing_predictions[batch_index, :count].detach(),
            ))
    num_onsets = int(dataset.performances[performance_index].num_onsets)
    return assemble_performance_owner_logits(num_onsets, chunks_value), finish_loss(loss_total)


def confusion_add(matrix: np.ndarray, target: Sequence[int] | torch.Tensor, predicted: Sequence[int] | torch.Tensor) -> None:
    truth = torch.as_tensor(target, dtype=torch.long).cpu().reshape(-1).tolist()
    guess = torch.as_tensor(predicted, dtype=torch.long).cpu().reshape(-1).tolist()
    if len(truth) != len(guess):
        raise ValueError("metric target/prediction length mismatch")
    for left, right in zip(truth, guess, strict=True):
        matrix[int(left), int(right)] += 1


def confusion_metrics(matrix: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    if matrix.sum() <= 0:
        return {"accuracy": None, "macro_f1": None, "class_metrics": [], "confusion": matrix.tolist()}
    rows = []
    for index, name in enumerate(names):
        tp = int(matrix[index, index]); support = int(matrix[index].sum()); predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1,
                     "support": support, "predicted": predicted})
    return {"accuracy": float(np.trace(matrix) / matrix.sum()),
            "macro_f1": float(np.mean([row["f1"] for row in rows])),
            "class_metrics": rows, "confusion": matrix.tolist()}


def transition_accumulator() -> dict[str, Any]:
    return {scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")}
            for scope in ("pooled", "up", "down")}


def interval_event_seconds(interval, event) -> float:
    """Map target tau through Run B canonical onset boundaries."""

    duration = float(interval.right_seconds) - float(interval.left_seconds)
    if not duration > 0.0:
        raise ValueError("Run B interval has non-positive onset duration")
    tau = float(event.tau)
    if not 0.0 <= tau < 1.0:
        raise ValueError("Run B target tau escaped the right-open interval")
    return float(interval.left_seconds) + tau * duration


def direct_same_performance_trajectory(
    events: Sequence[Mapping[str, Any]], onset_seconds: Sequence[float],
) -> dict[str, Any]:
    """Map events to the same performance canonical distinct-onset indices."""

    onsets = np.asarray(onset_seconds, dtype=np.float64)
    if onsets.ndim != 1 or len(onsets) < 2 or np.any(np.diff(onsets) <= 0):
        raise ValueError("direct Human mapping requires chronological distinct onsets")
    transitions = []
    for event in events:
        seconds = float(event["seconds"])
        if not math.isfinite(seconds):
            raise ValueError("Human metric event has non-finite time")
        right = int(np.searchsorted(onsets, seconds, side="left"))
        if right == 0:
            index = 0
        elif right == len(onsets):
            index = len(onsets) - 1
        else:
            index = right - 1 if seconds - onsets[right - 1] <= onsets[right] - seconds else right
        transitions.append({
            "direction": str(event["direction"]), "seconds": seconds,
            "onset_index": index, "score_position": index,
        })
    return {"transitions": transitions}


def timing_diagnostics(logits, targets) -> dict[str, Any]:
    predicted = logits.timing_predictions.detach().float().clamp(0.0, 1.0).cpu()
    values = torch.tensor([item.timing_targets for item in targets.intervals], dtype=torch.float32)
    mask = torch.tensor([item.timing_mask for item in targets.intervals], dtype=torch.bool)
    modes = torch.tensor([MODE_TO_ID[item.mode] for item in targets.intervals], dtype=torch.long)
    errors = (predicted - values).abs()
    change = modes == CHANGE_ID
    returns = modes == RETURN_ID
    return {
        "active": errors[mask].tolist(),
        "change": errors[change, 0].tolist(),
        "return_tau1": errors[returns, 0].tolist(),
        "return_tau2": errors[returns, 1].tolist(),
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def human_validation(model, dataset, device: torch.device, epoch: int) -> dict[str, Any]:
    model.eval()
    transition_total = transition_accumulator()
    matrices = {key: np.zeros(shape, dtype=np.int64) for key, shape in (
        ("raw_state", (2, 2)), ("vit_state", (2, 2)), ("raw_mode", (3, 3)), ("vit_mode", (3, 3)),
    )}
    loss_total = empty_loss_total()
    raw_conflicts = vit_conflicts = intervals = 0
    distributions = {key: Counter() for key in ("raw_state", "vit_state", "raw_mode", "vit_mode")}
    timing = {key: [] for key in ("active", "change", "return_tau1", "return_tau2")}
    failures = []
    for performance_index, performance in enumerate(dataset.performances):
        logits, losses = performance_inference(model, dataset, performance_index, device, amp=True)
        targets = dataset.performance_targets(performance_index)
        decoded = decode_performance(logits)
        state_target = torch.tensor(targets.onset_states, dtype=torch.long)
        mode_target = torch.tensor([MODE_TO_ID[item.mode] for item in targets.intervals], dtype=torch.long)
        confusion_add(matrices["raw_state"], state_target, decoded.raw.state_path)
        confusion_add(matrices["vit_state"], state_target, decoded.viterbi.state_path)
        confusion_add(matrices["raw_mode"], mode_target, decoded.raw.mode_path)
        confusion_add(matrices["vit_mode"], mode_target, decoded.viterbi.mode_path)
        raw_conflicts += decoded.raw.conflict_count
        conflict = state_mode_conflicts(decoded.viterbi.state_path, decoded.viterbi.mode_path)
        vit_conflicts += int(conflict["count"])
        intervals += len(mode_target)
        for key, path in (("raw_state", decoded.raw.state_path), ("vit_state", decoded.viterbi.state_path),
                          ("raw_mode", decoded.raw.mode_path), ("vit_mode", decoded.viterbi.mode_path)):
            distributions[key].update(path.cpu().tolist())
        current_timing = timing_diagnostics(logits, targets)
        for key in timing:
            timing[key].extend(current_timing[key])
        add_loss_total(loss_total, {
            "state_sum": losses["state_loss"] * len(targets.onset_states), "state_den": len(targets.onset_states),
            "mode_sum": losses["mode_loss"] * sum(MODE_CLASS_WEIGHTS[MODE_TO_ID[item.mode]] for item in targets.intervals),
            "mode_den": sum(MODE_CLASS_WEIGHTS[MODE_TO_ID[item.mode]] for item in targets.intervals),
            "timing_sum": losses["timing_loss"] * sum(sum(item.timing_mask) for item in targets.intervals),
            "timing_den": sum(sum(item.timing_mask) for item in targets.intervals),
        })
        try:
            onset_seconds = tuple(item.left_seconds for item in targets.intervals) + (
                targets.intervals[-1].right_seconds,
            )
            if len(onset_seconds) != len(targets.onset_states):
                raise AssertionError("Human direct State/onset cardinality mismatch")
            modeled = reconstruct_modeled_events(
                onset_seconds,
                decoded.viterbi.state_path, decoded.viterbi.mode_path, logits.timing_predictions,
            )
            modeled = tuple(
                event for event in modeled
                if onset_seconds[0] <= float(event["seconds"]) < onset_seconds[-1]
            )
            reference = [
                {"direction": event.direction,
                 "seconds": interval_event_seconds(interval, event),
                 "source_tick": event.source_tick}
                for interval in targets.intervals for event in interval.raw_transitions
            ]
            candidate_metric = direct_same_performance_trajectory(modeled, onset_seconds)
            reference_metric = direct_same_performance_trajectory(reference, onset_seconds)
            sum_transition(transition_total, pooled_transition_counts(candidate_metric, reference_metric))
        except Exception as error:
            failures.append({"performance_index": performance_index, "error": f"{type(error).__name__}: {error}"})
        if (performance_index + 1) % 10 == 0 or performance_index + 1 == len(dataset.performances):
            log(f"HUMAN_VALIDATION epoch={epoch} performances={performance_index + 1}/{len(dataset.performances)}")
    if failures:
        raise RuntimeError(f"human-input common-horizon mapping failures: {failures}")
    if vit_conflicts:
        raise AssertionError("human-input constrained decode produced conflicts")
    direction = finalize_transition(transition_total)
    pooled = direction["pooled"]
    return {
        "epoch": epoch,
        "raw_state": confusion_metrics(matrices["raw_state"], ("OFF", "ON")),
        "viterbi_state": confusion_metrics(matrices["vit_state"], ("OFF", "ON")),
        "raw_mode": confusion_metrics(matrices["raw_mode"], MODE_NAMES),
        "viterbi_mode": confusion_metrics(matrices["vit_mode"], MODE_NAMES),
        "raw_conflict_count": raw_conflicts,
        "raw_conflict_rate": raw_conflicts / intervals if intervals else 0.0,
        "viterbi_conflict_count": vit_conflicts, "viterbi_conflict_rate": 0.0,
        "transition_precision": pooled["precision"], "transition_recall": pooled["recall"],
        "transition_f1": pooled["f1"], "predicted_transitions": pooled["candidate"],
        "reference_transitions": pooled["reference"],
        "predicted_reference_ratio": pooled["candidate"] / pooled["reference"] if pooled["reference"] else 0.0,
        "timing_active_mae": mean_or_none(timing["active"]),
        "timing_change_mae": mean_or_none(timing["change"]),
        "timing_return_tau1_mae": mean_or_none(timing["return_tau1"]),
        "timing_return_tau2_mae": mean_or_none(timing["return_tau2"]),
        "raw_state_distribution": dict(distributions["raw_state"]),
        "viterbi_state_distribution": dict(distributions["vit_state"]),
        "raw_mode_distribution": dict(distributions["raw_mode"]),
        "viterbi_mode_distribution": dict(distributions["vit_mode"]),
        "losses": finish_loss(loss_total), "alignment_failures": failures,
        "metric_performances": len(dataset.performances) - len(failures),
        "mapping_strategy": "direct same-performance canonical distinct-onset indices",
        "external_alignment_calls": 0,
        "modeled_horizon": "[t1,tM)", "asap_test_access": 0,
    }


def audit_stage1() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(STAGE1_MANIFEST)
    validation = [row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"]
    manifest_sha = sha256_file(STAGE1_MANIFEST)
    if manifest_sha != EXPECTED_STAGE1_MANIFEST_SHA:
        raise RuntimeError("canonical Stage 1 manifest SHA changed")
    if len(rows) != 19 or len(validation) != 71 or {row["piece_id"] for row in rows} != {row["piece_id"] for row in validation}:
        raise RuntimeError("frozen Stage 1 / validation universe changed")
    if {row["seed"] for row in rows} != {"42"} or {row["status"] for row in rows} != {"frozen_pass"}:
        raise RuntimeError("frozen Stage 1 seed/status changed")
    mapping = []
    for row in rows:
        path = Path(row["canonical_midi_path"])
        if not path.is_file() or sha256_file(path) != row["canonical_midi_sha256"]:
            raise RuntimeError(f"frozen Stage 1 MIDI missing/hash mismatch: {row['piece_id']}")
        if signature_sha256(path) != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"frozen Stage 1 non-CC64 signature mismatch: {row['piece_id']}")
        mapped = [item["performance_path"] for item in validation if item["piece_id"] == row["piece_id"]]
        if len(mapped) != int(row["validation_performance_count"]):
            raise RuntimeError("frozen Stage 1 piece/reference mapping changed")
        mapping.append({"piece_id": row["piece_id"], "canonical_midi_path": row["canonical_midi_path"],
                        "canonical_midi_sha256": row["canonical_midi_sha256"], "human_performances": mapped})
    provenance = {
        "status": "PASS", "canonical_manifest": str(STAGE1_MANIFEST),
        "canonical_manifest_sha256": manifest_sha, "pieces": 19, "human_performances": 71,
        "seed": 42, "stage1_inference_regeneration": 0,
        "stage2_input_choice": "official midi_to_ids of saved canonical Original-PT MIDI; pedal columns masked",
        "mapping": mapping, "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "stage1_validation_provenance.json", provenance)
    return rows, validation, provenance


def run_a_score_positions(row: Mapping[str, str]) -> tuple[list[int], list[int]]:
    from scripts.audit_pedal_event_metric_tolerance_mini import raw_events
    from miditoolkit import MidiFile

    cache = RUN_A_ROOT / "frozen_pt_score_positions" / f"{row['piece_id']}.json"
    value = json.loads(cache.read_text(encoding="utf-8"))
    onsets, _ = raw_events(MidiFile(row["canonical_midi_path"]))
    if value["source_sha256"] != row["canonical_midi_sha256"] or value["raw_onsets"] != onsets:
        raise RuntimeError(f"Run A score-position cache provenance changed: {row['piece_id']}")
    return onsets, [int(item) for item in value["score_positions"]]


def binary_states_from_aligned_tokens(tokens: np.ndarray) -> list[int]:
    from src.stage2_encoder_only.dataset import NON_PEDAL_FEATURES, PEDAL_TOKEN_OFFSET
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim == 1:
        values = values.reshape(-1, 8)
    iois = values[:, 1] - 261
    absolute = np.cumsum(iois)
    first = np.flatnonzero(np.r_[True, np.diff(absolute) != 0])
    return [int(values[index, NON_PEDAL_FEATURES] - PEDAL_TOKEN_OFFSET >= 64) for index in first]


def mapped_human_modes(
    transitions: Sequence[Mapping[str, Any]], human_states: Sequence[int], score_positions: Sequence[int],
) -> tuple[list[int], list[int]]:
    indices = []
    modes = []
    for index, (left, right) in enumerate(zip(score_positions, score_positions[1:])):
        if left < 0 or right <= left or right >= len(human_states):
            continue
        count = sum(left <= int(event["score_position"]) < right for event in transitions)
        mode = 0 if count == 0 else 1 if count % 2 else 2
        if (human_states[left] != human_states[right]) != (mode == CHANGE_ID):
            continue
        indices.append(index); modes.append(mode)
    return indices, modes


def longest_mismatch(target: Sequence[int], predicted: Sequence[int]) -> int:
    best = active = 0
    for left, right in zip(target, predicted):
        if left == right:
            active = 0
        else:
            active += 1; best = max(best, active)
    return best


def frozen_pt_validation(
    model, pieces, stage1_rows, validation_rows, human_dataset, device: torch.device, epoch: int,
) -> dict[str, Any]:
    model.eval()
    references = build_frozen_human_reference_map(
        stage1_rows, validation_rows, [item.entry for item in human_dataset.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    grouped = references_by_piece(references)
    transition_total = transition_accumulator()
    matrices = {key: np.zeros(shape, dtype=np.int64) for key, shape in (
        ("raw_state", (2, 2)), ("vit_state", (2, 2)), ("raw_mode", (3, 3)), ("vit_mode", (3, 3)),
    )}
    distributions = {key: Counter() for key in ("raw_state", "vit_state", "raw_mode", "vit_mode")}
    identity_rows = []
    failures = []
    raw_conflicts = vit_conflicts = intervals = mode_mapping_edges = 0
    longest_raw = longest_vit = 0
    pair_count = state_pairs = 0
    output_root = RUN_ROOT / "frozen_pt_outputs" / f"epoch_{epoch:02d}"
    for piece_index, (piece, row) in enumerate(zip(pieces, stage1_rows, strict=True), 1):
        logits = collect_frozen_pt_piece_logits(model, piece, device, amp=True)
        decoded = decode_performance(logits)
        rendered = render_state_anchored_candidate(
            piece, decoded.viterbi.state_path, decoded.viterbi.mode_path,
            logits.timing_predictions, output_root / piece.piece_id / "candidate.mid",
        )
        identity_rows.append({"piece_id": piece.piece_id, "candidate": str(rendered.destination), **rendered.identity})
        raw_conflicts += decoded.raw.conflict_count
        vit_conflicts += decoded.viterbi.conflict_count
        intervals += len(decoded.viterbi.mode_path)
        for key, path in (("raw_state", decoded.raw.state_path), ("vit_state", decoded.viterbi.state_path),
                          ("raw_mode", decoded.raw.mode_path), ("vit_mode", decoded.viterbi.mode_path)):
            distributions[key].update(path.cpu().tolist())
        raw_onsets, positions = run_a_score_positions(row)
        if len(raw_onsets) != len(piece.onset_seconds) or len(positions) != len(piece.onset_seconds):
            raise RuntimeError("Frozen-PT onset/score-position cardinality changed")
        modeled = [event for event in rendered.modeled_events
                   if piece.onset_seconds[0] <= float(event["seconds"]) < piece.onset_seconds[-1]]
        candidate = common.trajectory_for_metric(modeled, str(piece.source_midi), raw_onsets, positions)
        for reference in grouped[piece.piece_id]:
            target = load_frozen_human(reference.metadata_index)
            if target is None:
                failures.append({"metadata_index": reference.metadata_index, "reason": "frozen alignment unavailable"})
                continue
            timeline = human_dataset.timeline(reference.dataset_index)
            left_ms = float(timeline.main[0].left_seconds) * 1000.0
            right_ms = float(timeline.main[-1].left_seconds) * 1000.0
            target_events = [event for event in target["transitions"]["transitions"]
                             if left_ms <= float(event["time_ms"]) < right_ms]
            sum_transition(transition_total, pooled_transition_counts(candidate, {"transitions": target_events}))
            human_states = binary_states_from_aligned_tokens(target["tokens"])
            valid = [index for index, position in enumerate(positions) if 0 <= position < len(human_states)]
            target_states = [human_states[positions[index]] for index in valid]
            raw_states = [int(decoded.raw.state_path[index]) for index in valid]
            vit_states = [int(decoded.viterbi.state_path[index]) for index in valid]
            confusion_add(matrices["raw_state"], target_states, raw_states)
            confusion_add(matrices["vit_state"], target_states, vit_states)
            longest_raw = max(longest_raw, longest_mismatch(target_states, raw_states))
            longest_vit = max(longest_vit, longest_mismatch(target_states, vit_states))
            state_pairs += len(valid)
            edge_indices, target_modes = mapped_human_modes(target_events, human_states, positions)
            if edge_indices:
                confusion_add(matrices["raw_mode"], target_modes, decoded.raw.mode_path[edge_indices])
                confusion_add(matrices["vit_mode"], target_modes, decoded.viterbi.mode_path[edge_indices])
                mode_mapping_edges += len(edge_indices)
            pair_count += 1
        log(f"FROZEN_PT_VALIDATION epoch={epoch} pieces={piece_index}/{len(pieces)}")
    if pair_count != 70 or sorted(item["metadata_index"] for item in failures) != ["856"]:
        raise RuntimeError(f"Frozen-PT canonical aligned population changed: pairs={pair_count} failures={failures}")
    if vit_conflicts:
        raise AssertionError("Frozen-PT constrained decode produced legality conflicts")
    if len(identity_rows) != 19 or not all(item["note_identity_exact"] for item in identity_rows):
        raise AssertionError("Frozen-PT non-pedal identity is not 19/19 PASS")
    if any(item["initialization_anchor_metric_event_count"] != 0 for item in identity_rows):
        raise AssertionError("serialization-only S1 anchor contaminated the metric")
    direction = finalize_transition(transition_total)
    pooled = direction["pooled"]
    result = {
        "epoch": epoch,
        "transition_precision": pooled["precision"], "transition_recall": pooled["recall"],
        "transition_f1": pooled["f1"], "predicted_transitions": pooled["candidate"],
        "reference_transitions": pooled["reference"],
        "predicted_reference_ratio": pooled["candidate"] / pooled["reference"] if pooled["reference"] else 0.0,
        "raw_state": confusion_metrics(matrices["raw_state"], ("OFF", "ON")),
        "viterbi_state": confusion_metrics(matrices["vit_state"], ("OFF", "ON")),
        "raw_mode": confusion_metrics(matrices["raw_mode"], MODE_NAMES),
        "viterbi_mode": confusion_metrics(matrices["vit_mode"], MODE_NAMES),
        "mode_mapping_edges": mode_mapping_edges,
        "raw_conflict_count": raw_conflicts, "raw_conflict_rate": raw_conflicts / intervals if intervals else 0.0,
        "viterbi_conflict_count": vit_conflicts, "viterbi_conflict_rate": 0.0,
        "raw_state_distribution": dict(distributions["raw_state"]),
        "viterbi_state_distribution": dict(distributions["vit_state"]),
        "raw_mode_distribution": dict(distributions["raw_mode"]),
        "viterbi_mode_distribution": dict(distributions["vit_mode"]),
        "longest_raw_state_mismatch_region": longest_raw,
        "longest_viterbi_state_mismatch_region": longest_vit,
        "state_comparison_points": state_pairs,
        "metric_pairs": pair_count, "alignment_failures": failures,
        "timing_active_mae": None, "timing_change_mae": None,
        "timing_return_tau1_mae": None, "timing_return_tau2_mae": None,
        "timing_unavailable_reason": "19 Frozen-PT timelines pair to multiple human performance timings",
        "nonpedal_identity_pass": 19, "nonpedal_identity_total": 19,
        "pedal_leakage_count": 0,
        "initialization_anchor_count": sum(item["initialization_anchor_count"] for item in identity_rows),
        "initialization_anchor_metric_event_count": 0,
        "primary_transition_metric_source": "modeled Run B event trajectory before MIDI serialization",
        "modeled_horizon": "[t1,tM)", "stage1_inference_regeneration": 0,
        "identity_rows": identity_rows, "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "frozen_pt_validation_epochs" / f"epoch_{epoch:02d}.json", result)
    atomic_json(RUN_ROOT / "integrity_audit.json", {
        "status": "PASS", "epoch": epoch, "pedal_leakage": 0,
        "nonpedal_identity_pass": 19, "nonpedal_identity_total": 19,
        "initialization_anchor_metric_event_count": 0,
        "stage1_regeneration": 0, "asap_test_access": 0, "rows": identity_rows,
    })
    return result


def configuration() -> dict[str, Any]:
    value = StateAnchoredModelConfig().to_dict()
    value.update({
        "experiment": RUN_ROOT.name, "run_root": str(RUN_ROOT), "created_at": now(),
        "fresh_training": True, "official_pretrained_pt": str(PRETRAINED),
        "encoder_layers": 10, "encoder_parameters": 103_271_424, "head_parameters": 5_383,
        "optimizer": "AdamW", "encoder_lr": common.ENCODER_LR, "head_lr": common.HEAD_LR,
        "weight_decay": common.WEIGHT_DECAY, "gradient_clip": common.MAX_GRAD_NORM,
        "precision": "FP16 AMP", "amp_initial_scale": AMP_INIT_SCALE,
        "window_micro_batch": WINDOW_MICRO_BATCH,
        "performances_per_optimizer_step": PERFORMANCES_PER_STEP,
        "max_epochs": MAX_EPOCHS, "early_stopping": None,
        "best_checkpoint_metric": "Frozen-PT Viterbi direction-aware Transition F1 on [t1,tM)",
        "best_checkpoint_tie_break": "earlier epoch",
        "frozen_stage1_manifest": str(STAGE1_MANIFEST),
        "frozen_stage1_manifest_sha256": EXPECTED_STAGE1_MANIFEST_SHA,
        "stage1_inference_regeneration": 0,
        "serialization_s1": "OFF=no anchor; ON=one absolute-state CC64=127 anchor at t1",
        "primary_metric_excludes_serialization_anchor": True,
        "checkpoint_retention": ["resume_last_train_state.pt", "best.pt", "last.pt"],
        "pre": False, "post": False, "final_tail": False,
        "initial_state_head": False, "teacher_forcing_state": False,
        "dynamic_target": False, "synthetic_correction": False,
        "consistency_auxiliary": False, "asap_test_metadata_access": 0,
        "asap_test_midi_access": 0,
    })
    return value


def focused_tests() -> list[dict[str, Any]]:
    files = (
        "tests/test_state_anchored_transition_v1.py",
        "tests/test_state_anchored_dataset_v1.py",
        "tests/test_state_anchored_model_v1.py",
        "tests/test_state_anchored_viterbi_v1.py",
        "tests/test_state_anchored_serialization_v1.py",
        "tests/test_state_anchored_human_mapping_fix_v1.py",
        "tests/test_binary_2slot_frozen_mapping_fix_v0.py",
        "tests/test_binary_2slot_midi_range_fix_v1.py",
    )
    rows = []
    for path in files:
        completed = subprocess.run(
            [sys.executable, path], cwd=ROOT, text=True, capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"focused test failed: {path}\n{completed.stdout}\n{completed.stderr}")
        parsed = ast.literal_eval(completed.stdout.strip())
        if not isinstance(parsed, dict) or set(parsed.values()) != {"passed"}:
            raise RuntimeError(f"focused test output malformed: {path}")
        rows.append({"path": path, "tests": len(parsed), "status": "PASS"})
    return rows


def no_duplicate_full_process() -> bool:
    current = os.getpid()
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) == current:
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_state_anchored_transition_v1_full.py" in command and (" full" in command or " resume" in command):
            return False
    return True


def preflight() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not no_duplicate_full_process():
        raise RuntimeError("duplicate Run B full-training process detected")
    smoke_report = (SMOKE_ROOT / "RUN_B_IMPLEMENTATION_SMOKE_REPORT.md").read_text(encoding="utf-8")
    smoke = json.loads((SMOKE_ROOT / "tiny_overfit_summary.json").read_text(encoding="utf-8"))
    target = json.loads((SMOKE_ROOT / "target_audit.json").read_text(encoding="utf-8"))
    if "**Verdict: PASS**" not in smoke_report or smoke["status"] != "PASS" or target["status"] != "PASS":
        raise RuntimeError("Run B smoke/target PASS artifact missing")
    if target["legality_conflicts"] or target["duplicate_owner"]:
        raise RuntimeError("Run B target legality/ownership gate failed")
    expected_weights = (0.305652382570267, 0.907267124114601, 1.787080493315132)
    if any(abs(left - right) > 1e-15 for left, right in zip(MODE_CLASS_WEIGHTS, expected_weights)):
        raise RuntimeError("frozen Run B Mode weights changed")
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
    alignment = json.loads((ALIGNMENT_ROOT / "alignment_manifest.json").read_text(encoding="utf-8"))
    if cache["cache_id"] != CANONICAL_CACHE_ID or cache["train_count"] != 2062 or cache["validation_count"] != 71:
        raise RuntimeError("canonical cache provenance changed")
    if alignment["alignment_id"] != CANONICAL_ALIGNMENT_ID:
        raise RuntimeError("canonical alignment provenance changed")
    free_bytes = shutil.disk_usage(RUN_ROOT).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"insufficient disk for retained checkpoints: free={free_bytes}")
    tests = focused_tests()
    stage1_rows, validation_rows, provenance = audit_stage1()
    train_dataset = StateAnchoredWindowDataset(CACHE_ROOT, "train", cache_input_tokens=False)
    human_dataset = StateAnchoredWindowDataset(CACHE_ROOT, "validation", cache_input_tokens=False)
    train_owner = diagnose_ownership(train_dataset)
    val_owner = diagnose_ownership(human_dataset)
    if len(train_dataset.performances) != 2062 or len(human_dataset.performances) != 71:
        raise RuntimeError("canonical train/validation population changed")
    references = build_frozen_human_reference_map(
        stage1_rows, validation_rows, [item.entry for item in human_dataset.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    unavailable = sorted(item.metadata_index for item in references if load_frozen_human(item.metadata_index) is None)
    if unavailable != ["856"]:
        raise RuntimeError(f"canonical aligned exclusion changed: {unavailable}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run B requires exactly one isolated CUDA device")
    device = torch.device("cuda:0")
    seed_everything()
    model = build_model(device)
    sample_dataset = StateAnchoredWindowDataset(
        CACHE_ROOT, "train", performance_indices=(7,), cache_input_tokens=True,
    )
    batch = move_batch(state_anchored_collate_fn([sample_dataset[0]]), device)
    with torch.no_grad():
        output = forward_batch(model, batch)
        terms = raw_loss_sums(output, batch)
        initial = {
            "state_loss": float(terms["state_sum"] / terms["state_den"]),
            "mode_loss": float(terms["mode_sum"] / terms["mode_den"]),
            "timing_loss": float(terms["timing_sum"] / terms["timing_den"]),
        }
        initial["total_loss"] = sum(initial.values())
    if not all(math.isfinite(value) for value in initial.values()):
        raise FloatingPointError("Run B initial loss is non-finite")
    representative = load_frozen_pt_piece(stage1_rows[0])
    representative_logits = collect_frozen_pt_piece_logits(model, representative, device, amp=False)
    representative_decoded = decode_performance(representative_logits)
    rendered = render_state_anchored_candidate(
        representative, representative_decoded.viterbi.state_path, representative_decoded.viterbi.mode_path,
        representative_logits.timing_predictions,
        RUN_ROOT / "preflight_frozen_pt_outputs" / representative.piece_id / "candidate.mid",
    )
    if not rendered.identity["note_identity_exact"] or rendered.identity["initialization_anchor_metric_event_count"]:
        raise AssertionError("representative S1/non-pedal integrity preflight failed")
    atomic_json(RUN_ROOT / "target_audit_reference.json", {
        "status": "PASS", "source": str(SMOKE_ROOT / "target_audit.json"),
        "source_sha256": hashlib.sha256((SMOKE_ROOT / "target_audit.json").read_bytes()).hexdigest(),
        "legality_conflicts": 0, "ownership_duplicate": 0,
        "formulation_changed": False, "asap_test_access": 0,
    })
    config = configuration()
    atomic_json(RUN_ROOT / "FULL_RUN_CONFIG.json", config)
    summary = {
        "status": "PASS", "tests": tests, "smoke_status": smoke["status"],
        "target_status": target["status"], "mode_weights": list(MODE_CLASS_WEIGHTS),
        "head_parameters": model.prediction_head_parameter_count,
        "checkpoint": model.checkpoint_path, "fresh_head_seed": model.head_init_seed,
        "hidden": model.hidden_size, "layers": int(model.encoder.config.num_hidden_layers),
        "train_count": 2062, "human_validation_count": 71, "frozen_pt_pieces": 19,
        "structural_references": 71, "canonical_aligned": 70, "excluded_metadata": unavailable,
        "train_ownership": train_owner.__dict__, "validation_ownership": val_owner.__dict__,
        "initial_loss": initial, "representative_identity": rendered.identity,
        "s1_initialization_semantics": "PASS", "transition_metric_anchor_contamination": 0,
        "pedal_leakage": 0, "stage1_provenance": provenance,
        "stage1_regeneration": 0, "free_disk_bytes": free_bytes,
        "checkpoint_peak_estimate_bytes": 5 * 2**30,
        "cuda_device_count": torch.cuda.device_count(), "cuda_device": torch.cuda.get_device_name(0),
        "asap_test_access": 0,
    }
    atomic_json(RUN_ROOT / "preflight_summary.json", summary)
    atomic_json(RUN_ROOT / "integrity_audit.json", {
        "status": "PASS", "stage": "preflight", "pedal_leakage": 0,
        "representative_nonpedal_identity": True,
        "initialization_anchor_metric_event_count": 0,
        "stage1_regeneration": 0, "asap_test_access": 0,
    })
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "PREFLIGHT_PASS", "completed_epochs": 0, "last_update": now(),
        "seed": SEED, "fresh_run_b_heads": True, "initial_loss_finite": True,
        "train_performances": 2062, "human_validation_references": 71,
        "frozen_pt_pieces": 19, "canonical_aligned_population": 70,
        "stage1_inference_regeneration": 0, "asap_test_access": 0,
    })
    del model, batch, sample_dataset, train_dataset, human_dataset
    torch.cuda.empty_cache()
    log("PREFLIGHT_PASS smoke/target/ownership/weights/head/viterbi/S1/stage1/identity/disk/finite-loss")


def flatten_epoch_rows(epoch: int, human: Mapping[str, Any], frozen: Mapping[str, Any]):
    human_row = {
        "epoch": epoch,
        "raw_state_accuracy": human["raw_state"]["accuracy"],
        "viterbi_state_accuracy": human["viterbi_state"]["accuracy"],
        "raw_mode_accuracy": human["raw_mode"]["accuracy"],
        "raw_mode_macro_f1": human["raw_mode"]["macro_f1"],
        "viterbi_mode_accuracy": human["viterbi_mode"]["accuracy"],
        "viterbi_mode_macro_f1": human["viterbi_mode"]["macro_f1"],
        "raw_conflict_count": human["raw_conflict_count"], "raw_conflict_rate": human["raw_conflict_rate"],
        "viterbi_conflict_count": human["viterbi_conflict_count"],
        "transition_precision": human["transition_precision"], "transition_recall": human["transition_recall"],
        "transition_f1": human["transition_f1"], "predicted_transitions": human["predicted_transitions"],
        "reference_transitions": human["reference_transitions"],
        "predicted_reference_ratio": human["predicted_reference_ratio"],
        "timing_active_mae": human["timing_active_mae"], "timing_change_mae": human["timing_change_mae"],
        "timing_return_tau1_mae": human["timing_return_tau1_mae"],
        "timing_return_tau2_mae": human["timing_return_tau2_mae"],
        **{f"raw_mode_{MODE_NAMES[index].lower()}": human["raw_mode_distribution"].get(index, 0) for index in range(3)},
        **{f"viterbi_mode_{MODE_NAMES[index].lower()}": human["viterbi_mode_distribution"].get(index, 0) for index in range(3)},
    }
    frozen_row = {
        "epoch": epoch, "raw_state_accuracy": frozen["raw_state"]["accuracy"],
        "viterbi_state_accuracy": frozen["viterbi_state"]["accuracy"],
        "raw_mode_accuracy": frozen["raw_mode"]["accuracy"], "raw_mode_macro_f1": frozen["raw_mode"]["macro_f1"],
        "viterbi_mode_accuracy": frozen["viterbi_mode"]["accuracy"],
        "viterbi_mode_macro_f1": frozen["viterbi_mode"]["macro_f1"],
        "mode_mapping_edges": frozen["mode_mapping_edges"],
        "transition_precision": frozen["transition_precision"], "transition_recall": frozen["transition_recall"],
        "transition_f1": frozen["transition_f1"], "predicted_transitions": frozen["predicted_transitions"],
        "reference_transitions": frozen["reference_transitions"],
        "predicted_reference_ratio": frozen["predicted_reference_ratio"],
        "raw_conflict_count": frozen["raw_conflict_count"], "raw_conflict_rate": frozen["raw_conflict_rate"],
        "viterbi_conflict_count": frozen["viterbi_conflict_count"],
        "longest_raw_state_mismatch_region": frozen["longest_raw_state_mismatch_region"],
        "longest_viterbi_state_mismatch_region": frozen["longest_viterbi_state_mismatch_region"],
        "initialization_anchor_count": frozen["initialization_anchor_count"],
        "initialization_anchor_metric_event_count": 0,
        **{f"raw_state_{name}": frozen["raw_state_distribution"].get(index, 0) for index, name in enumerate(("off", "on"))},
        **{f"viterbi_state_{name}": frozen["viterbi_state_distribution"].get(index, 0) for index, name in enumerate(("off", "on"))},
        **{f"raw_mode_{MODE_NAMES[index].lower()}": frozen["raw_mode_distribution"].get(index, 0) for index in range(3)},
        **{f"viterbi_mode_{MODE_NAMES[index].lower()}": frozen["viterbi_mode_distribution"].get(index, 0) for index in range(3)},
    }
    state_row = {
        "epoch": epoch, "human_raw_state_accuracy": human_row["raw_state_accuracy"],
        "human_viterbi_state_accuracy": human_row["viterbi_state_accuracy"],
        "frozen_raw_state_accuracy": frozen_row["raw_state_accuracy"],
        "frozen_viterbi_state_accuracy": frozen_row["viterbi_state_accuracy"],
        "frozen_longest_raw_mismatch": frozen_row["longest_raw_state_mismatch_region"],
        "frozen_longest_viterbi_mismatch": frozen_row["longest_viterbi_state_mismatch_region"],
    }
    mode_row = {
        "epoch": epoch, "human_raw_mode_accuracy": human_row["raw_mode_accuracy"],
        "human_raw_mode_macro_f1": human_row["raw_mode_macro_f1"],
        "human_viterbi_mode_accuracy": human_row["viterbi_mode_accuracy"],
        "human_viterbi_mode_macro_f1": human_row["viterbi_mode_macro_f1"],
        "frozen_raw_mode_accuracy": frozen_row["raw_mode_accuracy"],
        "frozen_raw_mode_macro_f1": frozen_row["raw_mode_macro_f1"],
        "frozen_viterbi_mode_accuracy": frozen_row["viterbi_mode_accuracy"],
        "frozen_viterbi_mode_macro_f1": frozen_row["viterbi_mode_macro_f1"],
        **{key: value for key, value in frozen_row.items() if key.startswith("raw_mode_") or key.startswith("viterbi_mode_")},
    }
    return human_row, frozen_row, state_row, mode_row


def update_report(train_rows, human_rows, frozen_rows, best_epoch, best_f1, status) -> None:
    table = []
    for train, human, frozen in zip(train_rows, human_rows, frozen_rows):
        table.append(
            f"| {train['epoch']} | {train['train_loss']:.5f} | {human['raw_state_accuracy']:.4f} | "
            f"{human['raw_mode_macro_f1']:.4f} | {human['transition_f1']:.4f} | "
            f"{frozen['raw_state_accuracy']:.4f} | {frozen['viterbi_state_accuracy']:.4f} | "
            f"{frozen['transition_precision']:.4f} | {frozen['transition_recall']:.4f} | "
            f"{frozen['transition_f1']:.4f} | {'yes' if int(train['epoch']) == best_epoch else ''} |"
        )
    best = next((row for row in frozen_rows if int(row["epoch"]) == best_epoch), None)
    if best is None:
        answers = "Pending until the first Frozen-PT validation completes."
        diagnostics = "Pending."
    else:
        state_gain = float(best["viterbi_state_accuracy"]) - RUN_A_STATE
        state_answer = "yes" if state_gain >= 0.05 else "no clear substantial gain"
        delta = float(best["transition_f1"]) - RUN_A_F1
        transition_answer = "improved" if delta > 0.01 else "degraded" if delta < -0.01 else "remained comparable"
        collapsed = any(int(best[f"viterbi_mode_{name}"]) == 0 for name in ("hold", "change", "return"))
        late = False
        if len(train_rows) >= 2 and int(best["epoch"]) < int(train_rows[-1]["epoch"]):
            late = float(frozen_rows[-1]["transition_f1"]) < float(best["transition_f1"]) and float(train_rows[-1]["train_loss"]) < float(train_rows[int(best["epoch"]) - 1]["train_loss"])
        answers = (
            f"1. State anchoring substantial absolute-state improvement: **{state_answer}** (Run B {best['viterbi_state_accuracy']:.6f}, Run A {RUN_A_STATE:.6f}).\n"
            f"2. Transition F1 vs Run A: **{transition_answer}** (Run B {best['transition_f1']:.6f}, Run A {RUN_A_F1:.6f}).\n"
            f"3. Viterbi legality conflicts eliminated: **{'yes' if int(best['viterbi_conflict_count']) == 0 else 'no'}**.\n"
            f"4. HOLD/CHANGE/RETURN collapse avoided: **{'no' if collapsed else 'yes'}**.\n"
            f"5. Late-epoch over-specialization evidence: **{'yes' if late else 'not established'}**."
        )
        diagnostics = (
            f"Best Frozen-PT P/R/F1={best['transition_precision']:.6f}/{best['transition_recall']:.6f}/{best['transition_f1']:.6f}; "
            f"raw/Viterbi State={best['raw_state_accuracy']:.6f}/{best['viterbi_state_accuracy']:.6f}; "
            f"pred/ref={best['predicted_reference_ratio']:.6f}; Viterbi conflicts={best['viterbi_conflict_count']}."
        )
    text = f"""# Full Run B State-Anchored Transition v1

| item | result |
|---|---|
| status | {status} |
| formulation | frozen State-Anchored Transition v1 MAIN-only `[t1,tM)` |
| train / human validation | 2,062 / 71 performances |
| Frozen Stage 1 | `{STAGE1_MANIFEST}`; 19 pieces; seed 42; SHA `{EXPECTED_STAGE1_MANIFEST_SHA}` |
| canonical Frozen references | 71 structural / 70 aligned; metadata 856 excluded |
| architecture | official PT 10x768 + four direct linear heads; 5,383 head parameters |
| Mode weights HOLD/CHANGE/RETURN | {MODE_CLASS_WEIGHTS[0]:.15f} / {MODE_CLASS_WEIGHTS[1]:.15f} / {MODE_CLASS_WEIGHTS[2]:.15f} |
| optimizer | AdamW; encoder 1e-5; heads 1e-4; wd 0.01; clip 1.0; FP16 AMP |
| epochs | {len(train_rows)} / 10, no early stopping |
| best criterion | Frozen-PT global-Viterbi Transition F1 on `[t1,tM)`; earlier tie |
| best epoch / F1 | {best_epoch if best_epoch is not None else 'pending'} / {best_f1 if best_f1 is not None else 'pending'} |
| PRE / POST / recurrence / Stage1 regeneration / ASAP test | 0 / 0 / 0 / 0 / 0 |

## Epoch table

| epoch | train loss | Human State | Human Mode Macro F1 | Human Transition F1 | Frozen raw State | Frozen Viterbi State | Frozen P | Frozen R | Frozen F1 | best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(table) if table else '| pending | | | | | | | | | | |'}

## Best epoch diagnostics

{diagnostics}

Frozen multi-human timing is N/A because there is no canonical one-to-one timing aggregation. Human-input CHANGE/RETURN timing MAE is recorded per epoch. Non-pedal identity, pedal masking, Viterbi legality, and S1 initialization-anchor exclusion are hard invariants.

## Direct Run A common-horizon comparison

Run A epoch 8 common `[t1,tM)`: P/R/F1={RUN_A_P:.6f}/{RUN_A_R:.6f}/{RUN_A_F1:.6f}; State agreement={RUN_A_STATE:.6f}.

{answers}
"""
    temporary = RUN_ROOT / ".FULL_STATE_ANCHORED_TRANSITION_V1_REPORT.md.tmp"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, RUN_ROOT / "FULL_STATE_ANCHORED_TRANSITION_V1_REPORT.md")


def finalize_epoch(
    *, model, optimizer, scaler, human_dataset, pieces, stage1_rows, validation_rows,
    device, epoch, global_step, train_loss, train_extras,
    train_table, human_table, frozen_table, state_table, mode_table,
) -> tuple[int, float]:
    log(f"VALIDATION_PHASE epoch={epoch} phase=HUMAN_INPUT")
    human = human_validation(model, human_dataset, device, epoch)
    log(f"VALIDATION_PHASE epoch={epoch} phase=FROZEN_PT")
    frozen = frozen_pt_validation(model, pieces, stage1_rows, validation_rows, human_dataset, device, epoch)
    train_row = {"epoch": epoch, "train_loss": train_loss["total_loss"], **train_loss, **train_extras}
    human_row, frozen_row, state_row, mode_row = flatten_epoch_rows(epoch, human, frozen)
    for table, row in ((train_table, train_row), (human_table, human_row), (frozen_table, frozen_row),
                       (state_table, state_row), (mode_table, mode_row)):
        replace_epoch(table, row)
    atomic_csv(RUN_ROOT / "training_metrics_by_epoch.csv", train_table)
    atomic_csv(RUN_ROOT / "human_input_validation_by_epoch.csv", human_table)
    atomic_csv(RUN_ROOT / "frozen_pt_validation_by_epoch.csv", frozen_table)
    atomic_csv(RUN_ROOT / "state_metrics_by_epoch.csv", state_table)
    atomic_csv(RUN_ROOT / "mode_metrics_by_epoch.csv", mode_table)
    payload = {
        "epoch": epoch, "global_step": global_step, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "grad_scaler_state": scaler.state_dict(),
        "configuration": configuration(), "seed": SEED, "human_validation": human,
        "frozen_pt_validation": frozen, "created_at": now(), "asap_test_access": 0,
    }
    atomic_torch(RUN_ROOT / "last.pt", payload)
    best_row = max(frozen_table, key=lambda row: (float(row["transition_f1"]), -int(row["epoch"])))
    best_epoch = int(best_row["epoch"]); best_f1 = float(best_row["transition_f1"])
    if best_epoch == epoch:
        atomic_torch(RUN_ROOT / "best.pt", payload)
    status = "COMPLETE" if epoch == MAX_EPOCHS else "RUNNING"
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": status, "completed_training_epoch": epoch, "completed_validation_epochs": epoch,
        "validation_pending": False, "global_step": global_step, "best_epoch": best_epoch,
        "best_frozen_pt_transition_f1": best_f1,
        "primary_metric": "Frozen-PT Viterbi Transition F1 on [t1,tM)",
        "viterbi_conflict_count": frozen["viterbi_conflict_count"],
        "nonpedal_identity_pass": 19, "initialization_anchor_metric_event_count": 0,
        "human_mapping_strategy": "direct same-performance canonical distinct-onset indices",
        "human_external_alignment_calls": 0, "mapping_fix_active": True,
        "epoch_1_train_reexecuted": False,
        "last_update": now(), "stage1_inference_regeneration": 0, "asap_test_access": 0,
    })
    update_report(train_table, human_table, frozen_table, best_epoch, best_f1, status)
    log(f"EPOCH_COMPLETE epoch={epoch} train_loss={train_loss['total_loss']:.6f} human_f1={human['transition_f1']:.6f} frozen_f1={frozen['transition_f1']:.6f} best={best_epoch}")
    return best_epoch, best_f1


def acquire_lock(mode: str) -> Path:
    lock = RUN_ROOT / ".state_anchored_full.lock"
    if lock.exists():
        try:
            prior = json.loads(lock.read_text(encoding="utf-8")); os.kill(int(prior["pid"]), 0)
        except (FileNotFoundError, ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
            os.replace(lock, RUN_ROOT / f"state_anchored_full.lock.stale.{int(time.time())}")
        else:
            raise RuntimeError(f"active Run B lock exists: {prior}")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "mode": mode, "created_at": now()}, handle); handle.write("\n")
    return lock


def full(resume_path: Path | None = None) -> None:
    preflight_value = json.loads((RUN_ROOT / "preflight_summary.json").read_text(encoding="utf-8"))
    if preflight_value["status"] != "PASS":
        raise RuntimeError("Run B preflight PASS artifact missing")
    if resume_path is None and ((RUN_ROOT / "last.pt").exists() or (RUN_ROOT / "resume_last_train_state.pt").exists()):
        raise RuntimeError("existing Run B checkpoint found; refusing fresh overwrite")
    seed_everything()
    device = torch.device("cuda:0")
    model = build_model(device)
    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=AMP_INIT_SCALE, enabled=True)
    stage1_rows, validation_rows, _ = audit_stage1()
    train_dataset = StateAnchoredWindowDataset(CACHE_ROOT, "train", cache_input_tokens=True)
    human_dataset = StateAnchoredWindowDataset(CACHE_ROOT, "validation", cache_input_tokens=True)
    pieces = [load_frozen_pt_piece(row) for row in stage1_rows]
    if len(train_dataset.performances) != 2062 or len(human_dataset.performances) != 71 or len(pieces) != 19:
        raise RuntimeError("Run B canonical population changed")
    references = build_frozen_human_reference_map(
        stage1_rows, validation_rows, [item.entry for item in human_dataset.performances],
        expected_pieces=19, expected_humans=71, verify_paths=True,
    )
    unavailable = sorted(item.metadata_index for item in references if load_frozen_human(item.metadata_index) is None)
    if unavailable != ["856"]:
        raise RuntimeError("Run B canonical aligned population changed")
    train_table = load_table(RUN_ROOT / "training_metrics_by_epoch.csv")
    human_table = load_table(RUN_ROOT / "human_input_validation_by_epoch.csv")
    frozen_table = load_table(RUN_ROOT / "frozen_pt_validation_by_epoch.csv")
    state_table = load_table(RUN_ROOT / "state_metrics_by_epoch.csv")
    mode_table = load_table(RUN_ROOT / "mode_metrics_by_epoch.csv")
    global_step = 0; completed_epoch = 0; resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        completed_epoch, global_step = restore_resumable_epoch_payload(
            resume_payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=False,
        )
    atomic_json(RUN_ROOT / "run_status.json", {
        "status": "RESUME_VALIDATION_PENDING" if resume_payload is not None else "RUNNING",
        "completed_training_epoch": completed_epoch, "completed_validation_epochs": len(frozen_table),
        "current_phase": "HUMAN_VALIDATION" if resume_payload is not None else "TRAIN_EPOCH_1",
        "global_step": global_step, "seed": SEED, "fresh_run_b_heads": resume_payload is None,
        "official_pretrained_pt_loaded": True, "train_performances": 2062,
        "human_validation_references": 71, "frozen_pt_pieces": 19,
        "canonical_aligned_population": 70, "stage1_inference_regeneration": 0,
        "human_mapping_strategy": "direct same-performance canonical distinct-onset indices",
        "human_external_alignment_calls": 0,
        "mapping_fix_active": True,
        "epoch_1_train_reexecuted": False if resume_payload is not None else None,
        "mode_weights": list(MODE_CLASS_WEIGHTS), "last_update": now(), "asap_test_access": 0,
    })
    if resume_payload is None:
        log(f"FULL_INIT checkpoint={model.checkpoint_path} seed=42 fresh_heads=1 layers=10 head_params=5383")
    else:
        log(f"RESUME_LOADED checkpoint={resume_path} completed_epoch={completed_epoch} global_step={global_step} epoch_1_train_reexecuted=0 mapping_fix=direct_same_performance")
    log("DATASET_READY train=2062 human_validation=71 frozen_pt=19 references=71 canonical_aligned=70 stage1_regeneration=0")
    update_report(train_table, human_table, frozen_table, None, None, "RUNNING")
    best_epoch = None if not frozen_table else int(max(frozen_table, key=lambda row: (float(row["transition_f1"]), -int(row["epoch"]))) ["epoch"])
    best_f1 = None if best_epoch is None else max(float(row["transition_f1"]) for row in frozen_table)
    if resume_payload is not None:
        epoch = completed_epoch
        if any(int(row["epoch"]) == epoch for row in frozen_table):
            log(f"RESUME_VALIDATION_ALREADY_COMPLETE epoch={epoch}")
        else:
            best_epoch, best_f1 = finalize_epoch(
                model=model, optimizer=optimizer, scaler=scaler, human_dataset=human_dataset,
                pieces=pieces, stage1_rows=stage1_rows, validation_rows=validation_rows,
                device=device, epoch=epoch, global_step=global_step,
                train_loss=resume_payload["train_loss"],
                train_extras={**resume_payload["train_extras"], "resumed_validation_only": True},
                train_table=train_table, human_table=human_table, frozen_table=frozen_table,
                state_table=state_table, mode_table=mode_table,
            )
        restore_resumable_epoch_payload(resume_payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=True)
        start_epoch = completed_epoch + 1
        del resume_payload
    else:
        start_epoch = 1
    consecutive_failure = 0
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
        order = list(range(len(train_dataset.performances))); random.Random(SEED + epoch).shuffle(order)
        total = empty_loss_total(); grad_norms = []
        for start in range(0, len(order), PERFORMANCES_PER_STEP):
            selected = order[start : start + PERFORMANCES_PER_STEP]
            model.train(); optimizer.zero_grad(set_to_none=True)
            result = replay_performance_group(
                model, train_dataset, selected, device, amp=True,
                backward=lambda value: scaler.scale(value).backward(),
            )
            for component in ("state", "mode", "timing"):
                denominator = result[f"denominator_{component}"]
                total[f"{component}_sum"] += result[f"{component}_loss"] * denominator
                total[f"{component}_den"] += denominator
            scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.MAX_GRAD_NORM).item())
            finite = math.isfinite(norm) and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
            scale_before = float(scaler.get_scale())
            if finite:
                scaler.step(optimizer)
            scaler.update()
            if not finite or float(scaler.get_scale()) < scale_before:
                consecutive_failure += 1
                log(f"WARNING gradient_or_amp epoch={epoch} group={start // PERFORMANCES_PER_STEP} consecutive={consecutive_failure}")
            else:
                consecutive_failure = 0; global_step += 1; grad_norms.append(norm)
            if consecutive_failure >= 3:
                raise FloatingPointError("persistent NaN/Inf or AMP overflow")
            if start == 0:
                log(f"TRAIN_INITIAL_LOSS epoch={epoch} total={result['total_loss']:.6f} finite=1 global_step={global_step}")
            if (start // PERFORMANCES_PER_STEP + 1) % 25 == 0:
                log(f"TRAIN epoch={epoch} performances={min(start + PERFORMANCES_PER_STEP,2062)}/2062 global_step={global_step} peak_gib={torch.cuda.max_memory_allocated(device)/2**30:.3f}")
        train_loss = finish_loss(total)
        train_extras = {
            "epoch_wall_seconds": time.monotonic() - started,
            "mean_gradient_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "cuda_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "resumed_validation_only": False,
        }
        resume_payload = build_resumable_epoch_payload(
            model=model, optimizer=optimizer, scaler=scaler, completed_training_epoch=epoch,
            global_step=global_step, run_configuration=configuration(),
            extra={"train_loss": train_loss, "train_extras": train_extras,
                   "created_at": now(), "asap_test_access": 0},
        )
        atomic_save_resumable_checkpoint(RUN_ROOT / "resume_last_train_state.pt", resume_payload)
        atomic_json(RUN_ROOT / "run_status.json", {
            "status": "TRAIN_EPOCH_COMPLETE_VALIDATION_PENDING", "completed_training_epoch": epoch,
            "completed_validation_epochs": epoch - 1, "validation_pending": True,
            "current_phase": "HUMAN_VALIDATION", "global_step": global_step,
            "resume_checkpoint": str(RUN_ROOT / "resume_last_train_state.pt"),
            "human_mapping_strategy": "direct same-performance canonical distinct-onset indices",
            "human_external_alignment_calls": 0, "mapping_fix_active": True,
            "epoch_1_train_reexecuted": False,
            "last_update": now(), "stage1_inference_regeneration": 0, "asap_test_access": 0,
        })
        log(f"TRAIN_EPOCH_CHECKPOINT_SAVED epoch={epoch} global_step={global_step} validation_pending=1")
        best_epoch, best_f1 = finalize_epoch(
            model=model, optimizer=optimizer, scaler=scaler, human_dataset=human_dataset,
            pieces=pieces, stage1_rows=stage1_rows, validation_rows=validation_rows,
            device=device, epoch=epoch, global_step=global_step, train_loss=train_loss,
            train_extras=train_extras, train_table=train_table, human_table=human_table,
            frozen_table=frozen_table, state_table=state_table, mode_table=mode_table,
        )
        restored_epoch, restored_step = restore_resumable_epoch_payload(
            resume_payload, model=model, optimizer=optimizer, scaler=scaler, restore_rng=True,
        )
        if (restored_epoch, restored_step) != (epoch, global_step):
            raise RuntimeError("post-validation saved train state restore changed progress")
        del resume_payload
        log(f"POST_TRAIN_RNG_RESTORED completed_epoch={epoch} global_step={global_step} next_epoch={epoch + 1}")
    update_report(train_table, human_table, frozen_table, best_epoch, best_f1, "COMPLETE")
    log(f"FULL_COMPLETE epochs=10 best_epoch={best_epoch} best_frozen_f1={best_f1:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "full", "resume"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.mode == "resume" and args.resume is None:
        parser.error("resume mode requires --resume")
    if args.mode != "resume" and args.resume is not None:
        parser.error("--resume is only valid in resume mode")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.mode) if args.mode in {"full", "resume"} else None
    try:
        if args.mode == "preflight":
            preflight()
        else:
            full(args.resume if args.mode == "resume" else None)
    except Exception as error:
        failure_path = RUN_ROOT / f"failure_{args.mode}.json"
        if failure_path.exists():
            failure_path = RUN_ROOT / f"failure_{args.mode}_{int(time.time())}.json"
        atomic_json(failure_path, {
            "mode": args.mode, "failed_at": now(), "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "asap_test_access": 0,
        })
        current_status = {}
        if (RUN_ROOT / "run_status.json").is_file():
            try:
                current_status = json.loads((RUN_ROOT / "run_status.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        atomic_json(RUN_ROOT / "run_status.json", {
            **current_status, "status": "FAILED", "mode": args.mode,
            "error": f"{type(error).__name__}: {error}", "last_update": now(),
            "asap_test_access": 0,
        })
        log(f"HARD_STOP mode={args.mode} error={type(error).__name__}: {error}")
        raise
    finally:
        if lock is not None and lock.exists():
            lock.unlink()


if __name__ == "__main__":
    main()
