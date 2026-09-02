#!/usr/bin/env python3
"""Train binary PT encoder-decoder with frozen canonical validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_binary.canonical_stage1 import sha256_file  # noqa: E402
from src.stage2_binary.canonical_validation import (  # noqa: E402
    evaluate_canonical_stage1,
    verify_canonical_bank,
)
from src.stage2_binary.full_training import (  # noqa: E402
    PedalMetricEarlyStopping,
    SharedBinaryWindowDataset,
    make_binary_loader,
    order_sha256,
    training_window_order,
)
from src.stage2_binary.training import build_binary_optimizer  # noqa: E402
from src.stage2_binary_encoder_decoder.inference import (  # noqa: E402
    infer_binary_encoder_decoder_pedals,
)
from src.stage2_binary_encoder_decoder.model import (  # noqa: E402
    BINARY_CLASSES,
    BOS_ID,
    DECODER_VOCAB_SIZE,
    PAD_ID,
    BinaryPedalEncoderDecoderModel,
)
from src.stage2_binary_encoder_decoder.training import (  # noqa: E402
    run_accumulated_training_epoch,
    run_teacher_forced_validation,
)
from src.stage2_encoder_only.train import (  # noqa: E402
    atomic_torch_save,
    capture_rng_states,
    get_gpu_identity,
)
from src.stage2_encoder_only.training import set_deterministic_seed  # noqa: E402


METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "train_binary_accuracy",
    "train_exact_pattern_accuracy",
    "teacher_forced_validation_loss",
    "teacher_forced_validation_binary_accuracy",
    "teacher_forced_validation_exact_pattern_accuracy",
    "canonical_free_running_strict_js_distance",
    "canonical_free_running_strict_intersection",
    "js_change_vs_original_pt",
    "relative_js_change_vs_original_pt",
    "intersection_change_vs_original_pt",
    "beats_original_pt",
    "js_change_vs_encoder_only_best",
    "beats_encoder_only_best",
    "canonical_steady_mass",
    "canonical_transition_containing_mass",
    "canonical_non_cc64_equality_count",
    "epoch_time_seconds",
    "global_optimizer_steps",
    "epoch_optimizer_steps",
    "amp_skipped_optimizer_steps",
    "maximum_consecutive_amp_skips",
    "encoder_learning_rate",
    "decoder_head_learning_rate",
    "peak_gpu_memory_bytes",
)
CHECKPOINT_COLUMNS = (
    "kind",
    "epoch",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "canonical_strict_js_distance",
    "canonical_strict_intersection",
)
DISTRIBUTION_COLUMNS = (
    "epoch",
    "joint_id",
    "pattern",
    "human_count",
    "human_probability",
    "original_pt_count",
    "original_pt_probability",
    "encoder_decoder_count",
    "encoder_decoder_probability",
)
EQUALITY_COLUMNS = (
    "epoch",
    "piece_id",
    "canonical_sha256",
    "candidate_sha256",
    "midi_type_exact",
    "ticks_per_beat_exact",
    "track_count_exact",
    "all_ordered_non_cc64_events_exact",
    "canonical_cc64_events",
    "candidate_cc64_events",
    "cc64_changed",
    "donor_cc64_discarded_after_canonical_eot",
    "status",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parameter_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pt_git_head(repository: Path) -> str:
    head = (repository / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference_name = head.removeprefix("ref: ")
    reference = repository / ".git" / reference_name
    if reference.is_file():
        return reference.read_text(encoding="utf-8").strip()
    packed = repository / ".git/packed-refs"
    for line in packed.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "^")):
            digest, name = line.split(" ", 1)
            if name == reference_name:
                return digest
    raise RuntimeError("cannot resolve pinned PT git revision")


def update_status(output_root: Path, **updates: Any) -> dict[str, Any]:
    path = output_root / "run_status.json"
    current = read_json(path) if path.is_file() else {}
    current.update(updates, updated_at=now())
    atomic_json(path, current)
    return current


def verify_fixed_configuration(config: Mapping[str, Any]) -> None:
    expected = {
        "seed": 42,
        "decoder_init_seed": 42,
        "window_notes": 512,
        "stride_notes": 256,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 16,
        "optimizer": "AdamW",
        "encoder_lr": 1e-5,
        "decoder_head_lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "amp_enabled": True,
        "max_epochs": 10,
        "early_stopping_patience": 3,
    }
    changed = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise RuntimeError(f"fixed controlled configuration changed: {changed}")
    if int(config["micro_batch_size"]) * int(
        config["gradient_accumulation_steps"]
    ) != int(config["effective_batch_size"]):
        raise RuntimeError("effective batch size mismatch")


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_fixed_configuration(config)
    hashes: dict[str, str] = {}
    for name, path_value in config["provenance_files"].items():
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        hashes[name] = digest
        if digest != config["expected_file_sha256"][name]:
            raise RuntimeError(f"provenance hash mismatch: {name} {path}")
    if pt_git_head(ROOT / "third_party/PianistTransformer") != config["pinned_pt_commit"]:
        raise RuntimeError("pinned Pianist Transformer revision changed")

    cache_stats = read_json(Path(config["cache_root"]) / "cache_statistics.json")
    if (
        not cache_stats.get("completed")
        or cache_stats.get("cache_id") != config["expected_cache_id"]
    ):
        raise RuntimeError("shared cache completion or ID mismatch")
    train_stats = cache_stats["splits"]["train"]
    validation_stats = cache_stats["splits"]["validation"]
    expected_inventory = (
        int(config["expected_train_performances"]),
        int(config["expected_train_notes"]),
        int(config["expected_train_windows"]),
        int(config["expected_validation_performances"]),
        int(config["expected_validation_notes"]),
        int(config["expected_validation_windows"]),
    )
    actual_inventory = (
        int(train_stats["performances"]),
        int(train_stats["notes"]),
        int(train_stats["windows"]),
        int(validation_stats["performances"]),
        int(validation_stats["notes"]),
        int(validation_stats["windows"]),
    )
    if actual_inventory != expected_inventory:
        raise RuntimeError(f"shared cache inventory mismatch: {actual_inventory}")
    if train_stats["source_statistics"] != config["expected_train_source_statistics"]:
        raise RuntimeError("natural-concatenation source statistics changed")
    train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
    validation_dataset = SharedBinaryWindowDataset(config["cache_root"], "validation")
    first_order_hash = order_sha256(
        training_window_order(len(train_dataset), int(config["seed"]), 1)
    )
    if first_order_hash != config["expected_epoch1_window_order_sha256"]:
        raise RuntimeError("seed-42 epoch-1 window order changed")

    canonical = verify_canonical_bank(
        config["canonical_manifest"],
        config["canonical_baseline"],
        tolerance=float(config["js_tie_tolerance"]),
    )
    if abs(
        canonical["js_distance"] - float(config["original_pt_strict_js_distance"])
    ) > float(config["js_tie_tolerance"]):
        raise RuntimeError("Original PT canonical JS mismatch")
    if abs(
        canonical["intersection"] - float(config["original_pt_strict_intersection"])
    ) > float(config["js_tie_tolerance"]):
        raise RuntimeError("Original PT canonical Intersection mismatch")
    return {
        "passed": True,
        "cache_id": train_dataset.cache_id,
        "train_performances": len(train_dataset.records),
        "train_notes": int(train_dataset.tokens.shape[0]),
        "train_windows": len(train_dataset),
        "validation_performances": len(validation_dataset.records),
        "validation_notes": int(validation_dataset.tokens.shape[0]),
        "validation_windows": len(validation_dataset),
        "epoch1_window_order_sha256": first_order_hash,
        "canonical_bank": canonical,
        "provenance_hashes": hashes,
        "stage1_neural_inference": 0,
        "asap_test_human_midi_access": 0,
        "canonical_test_inference": 0,
    }


def metric_row(
    config: Mapping[str, Any],
    epoch: int,
    train_metrics: Mapping[str, Any],
    train_checks: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    canonical: Mapping[str, Any],
    epoch_seconds: float,
    global_steps: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    js = float(canonical["strict_js_distance"])
    intersection = float(canonical["strict_intersection"])
    pt_js = float(config["original_pt_strict_js_distance"])
    encoder_js = float(config["encoder_only_strict_js_distance"])
    tolerance = float(config["js_tie_tolerance"])
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_binary_accuracy": train_metrics["binary_accuracy"],
        "train_exact_pattern_accuracy": train_metrics["exact_pattern_accuracy"],
        "teacher_forced_validation_loss": validation_metrics["loss"],
        "teacher_forced_validation_binary_accuracy": validation_metrics[
            "binary_accuracy"
        ],
        "teacher_forced_validation_exact_pattern_accuracy": validation_metrics[
            "exact_pattern_accuracy"
        ],
        "canonical_free_running_strict_js_distance": js,
        "canonical_free_running_strict_intersection": intersection,
        "js_change_vs_original_pt": js - pt_js,
        "relative_js_change_vs_original_pt": js / pt_js - 1.0,
        "intersection_change_vs_original_pt": intersection
        - float(config["original_pt_strict_intersection"]),
        "beats_original_pt": js < pt_js - tolerance,
        "js_change_vs_encoder_only_best": js - encoder_js,
        "beats_encoder_only_best": js < encoder_js - tolerance,
        "canonical_steady_mass": canonical["steady_mass"],
        "canonical_transition_containing_mass": canonical[
            "transition_containing_mass"
        ],
        "canonical_non_cc64_equality_count": canonical[
            "non_cc64_equality_count"
        ],
        "epoch_time_seconds": epoch_seconds,
        "global_optimizer_steps": global_steps,
        "epoch_optimizer_steps": train_checks["optimizer_steps"],
        "amp_skipped_optimizer_steps": train_checks["amp_skipped_steps"],
        "maximum_consecutive_amp_skips": train_checks[
            "maximum_consecutive_amp_skips"
        ],
        "encoder_learning_rate": optimizer.param_groups[0]["lr"],
        "decoder_head_learning_rate": optimizer.param_groups[1]["lr"],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def checkpoint_payload(
    model: BinaryPedalEncoderDecoderModel,
    run_config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "configuration": dict(run_config),
        "epoch": int(row["epoch"]),
        "canonical_strict_js_distance": float(
            row["canonical_free_running_strict_js_distance"]
        ),
        "canonical_strict_intersection": float(
            row["canonical_free_running_strict_intersection"]
        ),
    }


def write_checkpoint_manifest(
    output_root: Path,
    metrics: Sequence[Mapping[str, Any]],
    best_epoch: int,
) -> list[dict[str, Any]]:
    by_epoch = {int(row["epoch"]): row for row in metrics}
    rows: list[dict[str, Any]] = []
    items = [
        ("epoch", epoch, output_root / f"epoch_{epoch:03d}.pt")
        for epoch in sorted(by_epoch)
    ]
    items.extend(
        (
            ("best", best_epoch, output_root / "best.pt"),
            ("last", max(by_epoch), output_root / "last.pt"),
        )
    )
    for kind, epoch, path in items:
        metric = by_epoch[epoch]
        rows.append(
            {
                "kind": kind,
                "epoch": epoch,
                "checkpoint_path": str(path),
                "checkpoint_sha256": sha256_file(path),
                "checkpoint_bytes": path.stat().st_size,
                "canonical_strict_js_distance": metric[
                    "canonical_free_running_strict_js_distance"
                ],
                "canonical_strict_intersection": metric[
                    "canonical_free_running_strict_intersection"
                ],
            }
        )
    atomic_csv(output_root / "checkpoint_manifest.csv", CHECKPOINT_COLUMNS, rows)
    return rows


def transition_masses(histogram: Sequence[int]) -> tuple[float, float]:
    values = np.asarray(histogram, dtype=np.float64)
    probability = values / values.sum()
    steady = float(probability[0] + probability[15])
    return steady, 1.0 - steady


def build_report(
    config: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    best_checkpoint_sha256: str,
    baseline: Mapping[str, Any],
) -> str:
    human_steady, human_transition = transition_masses(baseline["human_histogram"])
    original_steady, original_transition = transition_masses(
        baseline["canonical_original_pt_histogram"]
    )
    lines = [
        "# Binary Encoder-Decoder Canonical-v1 Training Report",
        "",
        "## A. Setup",
        "",
        "- Training data: MAESTRO-clean 1,170 + ASAP train 892 = 2,062 performances / 9,369,095 notes / 35,573 windows (natural concatenation).",
        f"- Shared cache ID: `{config['expected_cache_id']}`",
        "- Seed 42; micro-batch 4; gradient accumulation 4; effective batch 16; window/stride 512/256.",
        "- AdamW: encoder LR 1e-5, decoder/head LR 1e-4, weight decay 0.01; gradient clip 1.0; AMP enabled.",
        "- Architecture: official trainable 10-layer PT encoder (hidden 768), native two-layer causal T5GemmaDecoder with cross-attention, shared Linear(768,2).",
        "- Decoder vocabulary: OFF=0, ON=1, BOS=2, PAD=3; note-major P1,P2,P3,P4 target order.",
        "- Training uses standard teacher forcing and unweighted binary CE. Teacher-forced validation is diagnostic only.",
        "- Checkpoint selection and patience-3 early stopping use minimum canonical free-running strict JS only; within 1e-12, higher Intersection then earliest epoch is the deterministic tie-break.",
        f"- Frozen canonical Stage 1 manifest: `{config['canonical_manifest']}`; Stage 1 neural inference regeneration: 0.",
        f"- {config['overlap_policy_caveat']}",
        "",
        "## B. Epoch trajectory",
        "",
        "| Epoch | Train loss | TF val loss | TF acc | TF exact | Free-running JS ↓ | Intersection ↑ | Beats PT? | Beats encoder-only? |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['epoch']} | {float(row['train_loss']):.9f} | "
            f"{float(row['teacher_forced_validation_loss']):.9f} | "
            f"{float(row['teacher_forced_validation_binary_accuracy']):.9f} | "
            f"{float(row['teacher_forced_validation_exact_pattern_accuracy']):.9f} | "
            f"{float(row['canonical_free_running_strict_js_distance']):.12f} | "
            f"{float(row['canonical_free_running_strict_intersection']):.12f} | "
            f"{'yes' if row['beats_original_pt'] else 'no'} | "
            f"{'yes' if row['beats_encoder_only_best'] else 'no'} |"
        )
    lines += [
        "",
        "## C. Best validation comparison",
        "",
        "| Model | Architecture | Best epoch | Validation JS ↓ | Intersection ↑ |",
        "|---|---|---:|---:|---:|",
        f"| Original PT | original PT | — | {float(config['original_pt_strict_js_distance']):.12f} | {float(config['original_pt_strict_intersection']):.12f} |",
        f"| Encoder-only Independent | encoder-only 4×2 | {int(config['encoder_only_best_epoch'])} | {float(config['encoder_only_strict_js_distance']):.12f} | {float(config['encoder_only_strict_intersection']):.12f} |",
        f"| Binary Encoder-Decoder | autoregressive encoder-decoder | {best['epoch']} | {float(best['canonical_free_running_strict_js_distance']):.12f} | {float(best['canonical_free_running_strict_intersection']):.12f} |",
        "",
        "## D. Transition diagnostics",
        "",
        "| Source | Steady mass | Transition-containing mass |",
        "|---|---:|---:|",
        f"| Human validation | {human_steady:.12f} | {human_transition:.12f} |",
        f"| Original PT | {original_steady:.12f} | {original_transition:.12f} |",
        f"| Encoder-only Independent epoch 2 | {float(config['encoder_only_steady_mass']):.12f} | {float(config['encoder_only_transition_containing_mass']):.12f} |",
        f"| Binary Encoder-Decoder best | {float(best['canonical_steady_mass']):.12f} | {float(best['canonical_transition_containing_mass']):.12f} |",
        "",
        "## E. Locked best checkpoint",
        "",
        f"- Best epoch: {best['epoch']}",
        f"- `best.pt` SHA-256: `{best_checkpoint_sha256}`",
        "- Every executed epoch checkpoint plus `best.pt` and `last.pt` is retained and hashed in `checkpoint_manifest.csv`.",
        "",
        "## F. Test isolation",
        "",
        "ASAP test was not used for training, model selection, checkpoint selection, calibration, or evaluation in this stage.",
        "",
        "- ASAP test Human MIDI access: 0",
        "- Canonical test Stage 2 inference: 0/23",
        "- Test JS / Intersection: not computed",
        "- Test audio: not generated",
        "- Scheduled sampling, class weighting, calibration, and post-processing: not used",
    ]
    return "\n".join(lines) + "\n"


def run(config: Mapping[str, Any]) -> None:
    output_root = Path(config["output_root"])
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output_root}")
    output_root.mkdir(parents=True)
    log_handle = (output_root / "train.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        print(message, file=log_handle, flush=True)

    atomic_json(
        output_root / "run_status.json",
        {
            "experiment_id": config["experiment_id"],
            "status": "running",
            "state": "preflight",
            "started_at": now(),
            "pid": os.getpid(),
            "full_training_authorized": True,
            "stage1_neural_inference": 0,
            "asap_test_human_midi_access": 0,
            "canonical_test_inference": 0,
            "scheduled_sampling_used": False,
        },
    )
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("training requires exactly one visible CUDA GPU")
        device = torch.device("cuda:0")
        gpu = get_gpu_identity()
        preflight_result = preflight(config)
        update_status(
            output_root,
            state="preflight_pass",
            preflight=preflight_result,
            preflight_pass=True,
            shared_cache_verification_pass=True,
            canonical_validation_bank_verification_pass=True,
            gpu=gpu,
        )
        log(
            f"CACHE_PASS cache_id={preflight_result['cache_id']} "
            f"train_windows={preflight_result['train_windows']}"
        )
        log(
            f"CANONICAL_BANK_PASS pieces={preflight_result['canonical_bank']['pieces']} "
            f"js={preflight_result['canonical_bank']['js_distance']:.15f}"
        )
        log("PREFLIGHT_PASS")

        set_deterministic_seed(int(config["seed"]))
        train_dataset = SharedBinaryWindowDataset(config["cache_root"], "train")
        validation_dataset = SharedBinaryWindowDataset(
            config["cache_root"], "validation"
        )
        epoch_order_hashes = {
            str(epoch): order_sha256(
                training_window_order(
                    len(train_dataset), int(config["seed"]), epoch
                )
            )
            for epoch in range(1, int(config["max_epochs"]) + 1)
        }
        model = BinaryPedalEncoderDecoderModel.from_pretrained_encoder(
            config["checkpoint_path"],
            decoder_init_seed=int(config["decoder_init_seed"]),
            freeze_encoder=False,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        encoder_hash = parameter_hash(model.encoder.parameters())
        if encoder_hash != config["expected_initial_encoder_sha256"]:
            raise RuntimeError("fresh pretrained encoder parameter hash mismatch")
        prediction_hash = parameter_hash(model.prediction_head_parameters())
        initial_model_hash = parameter_hash(model.parameters())
        run_config = dict(config)
        run_config.update(
            model_class="src.stage2_binary_encoder_decoder.model.BinaryPedalEncoderDecoderModel",
            encoder_layers=10,
            decoder_layers=2,
            hidden_size=model.hidden_size,
            output_classes=BINARY_CLASSES,
            decoder_vocab_size=DECODER_VOCAB_SIZE,
            special_tokens={"OFF": 0, "ON": 1, "BOS": BOS_ID, "PAD": PAD_ID},
            sequence_order="P1_1,P2_1,P3_1,P4_1,P1_2,...",
            initial_encoder_parameter_sha256=encoder_hash,
            initial_decoder_head_parameter_sha256=prediction_hash,
            initial_full_model_parameter_sha256=initial_model_hash,
            parameter_count=model.parameter_count,
            trainable_parameter_count=model.trainable_parameter_count,
            epoch_window_order_sha256=epoch_order_hashes,
            source_sha256_at_launch={
                "training_helper": sha256_file(
                    ROOT / "src/stage2_binary_encoder_decoder/training.py"
                ),
                "canonical_validation_helper": sha256_file(
                    ROOT / "src/stage2_binary/canonical_validation.py"
                ),
                "runner": sha256_file(Path(__file__)),
            },
            canonical_validation="greedy free-running -> CC64-only transplant -> 19/19 strict non-CC64 equality -> official MIDI-roundtrip metric",
            teacher_forced_validation="diagnostic only",
            checkpoint_selection=config["best_checkpoint_primary"],
            asap_test_access=0,
        )
        atomic_json(output_root / "config.json", run_config)
        update_status(
            output_root,
            state="model_initialized",
            model_initialized=True,
            pretrained_encoder_load_pass=True,
            fresh_encoder_decoder_initialization=True,
            initial_encoder_parameter_sha256=encoder_hash,
            initial_decoder_head_parameter_sha256=prediction_hash,
            initial_full_model_parameter_sha256=initial_model_hash,
        )
        log(
            f"MODEL_INITIALIZED encoder_sha256={encoder_hash} "
            f"decoder_head_sha256={prediction_hash}"
        )

        model.to(device)
        optimizer = build_binary_optimizer(
            model,
            encoder_lr=float(config["encoder_lr"]),
            head_lr=float(config["decoder_head_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(config["amp_enabled"]),
            init_scale=float(config["amp_init_scale"]),
        )
        validation_loader = make_binary_loader(
            validation_dataset,
            batch_size=int(config["micro_batch_size"]),
            pin_memory=bool(config["pin_memory"]),
        )
        stopping = PedalMetricEarlyStopping(
            patience=int(config["early_stopping_patience"]),
            tie_tolerance=float(config["js_tie_tolerance"]),
        )
        baseline = read_json(config["canonical_baseline"])
        metrics: list[dict[str, Any]] = []
        equality_rows: list[dict[str, Any]] = []
        distribution_rows: list[dict[str, Any]] = []
        global_steps = 0
        total_amp_skips = 0
        first_step_logged = False

        for epoch in range(1, int(config["max_epochs"]) + 1):
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            order = training_window_order(
                len(train_dataset), int(config["seed"]), epoch
            )
            train_loader = make_binary_loader(
                train_dataset,
                batch_size=int(config["micro_batch_size"]),
                pin_memory=bool(config["pin_memory"]),
                order=order,
            )

            def step_callback(step: int, details: Mapping[str, Any]) -> None:
                nonlocal first_step_logged
                if not first_step_logged and details["optimizer_step_applied"]:
                    first_step_logged = True
                    allocated = torch.cuda.memory_allocated(device)
                    reserved = torch.cuda.memory_reserved(device)
                    update_status(
                        output_root,
                        state="first_optimizer_step_pass",
                        first_optimizer_step_pass=True,
                        first_step_epoch=epoch,
                        first_step_attempt=step,
                        first_step_loss=float(details["loss"]),
                        first_step_gradient_norm=float(details["gradient_norm"]),
                        first_step_gpu_memory_allocated_bytes=allocated,
                        first_step_gpu_memory_reserved_bytes=reserved,
                    )
                    log(
                        "FIRST_OPTIMIZER_STEP_PASS "
                        f"epoch={epoch} step={step} loss={float(details['loss']):.9f} "
                        f"grad_norm={float(details['gradient_norm']):.6f} "
                        f"gpu_allocated_bytes={allocated} gpu_reserved_bytes={reserved}"
                    )
                elif step % 100 == 0:
                    update_status(
                        output_root,
                        state="training",
                        current_epoch=epoch,
                        current_epoch_attempted_steps=step,
                        current_epoch_optimizer_steps=details["optimizer_steps"],
                        current_epoch_amp_skips=details["amp_skipped_steps"],
                    )
                    log(
                        f"TRAIN_PROGRESS epoch={epoch} attempted_steps={step} "
                        f"optimizer_steps={details['optimizer_steps']} "
                        f"loss={float(details['loss']):.9f}"
                    )

            train_metrics, train_checks = run_accumulated_training_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=bool(config["amp_enabled"]),
                accumulation_steps=int(config["gradient_accumulation_steps"]),
                max_grad_norm=float(config["max_grad_norm"]),
                max_consecutive_amp_skips=int(
                    config["max_consecutive_amp_skips"]
                ),
                step_callback=step_callback,
            )
            if not first_step_logged:
                raise RuntimeError("epoch completed without a successful optimizer step")
            global_steps += int(train_checks["optimizer_steps"])
            total_amp_skips += int(train_checks["amp_skipped_steps"])

            validation_metrics = run_teacher_forced_validation(
                model,
                validation_loader,
                device=device,
                amp_enabled=bool(config["amp_enabled"]),
            )
            canonical = evaluate_canonical_stage1(
                model,
                architecture="binary_encoder_decoder",
                manifest_path=config["canonical_manifest"],
                baseline_path=config["canonical_baseline"],
                device=device,
                temporary_root=output_root / "canonical_validation_tmp",
                event_logger=log,
                inference_function=infer_binary_encoder_decoder_pedals,
            )
            if int(canonical["non_cc64_equality_count"]) != 19:
                raise AssertionError("canonical non-CC64 equality is not 19/19")
            improved, should_stop = stopping.update(
                float(canonical["strict_js_distance"]),
                float(canonical["strict_intersection"]),
                epoch,
            )
            row = metric_row(
                config,
                epoch,
                train_metrics,
                train_checks,
                validation_metrics,
                canonical,
                time.perf_counter() - epoch_started,
                global_steps,
                optimizer,
                device,
            )
            numeric = [
                key
                for key in METRIC_COLUMNS
                if key not in {"beats_original_pt", "beats_encoder_only_best"}
            ]
            if not all(math.isfinite(float(row[key])) for key in numeric):
                raise FloatingPointError("epoch metrics contain non-finite values")
            metrics.append(row)
            atomic_csv(output_root / "metrics.csv", METRIC_COLUMNS, metrics)
            atomic_csv(
                output_root / "canonical_validation_trajectory.csv",
                METRIC_COLUMNS,
                metrics,
            )
            for equality in canonical["equality_rows"]:
                equality_rows.append({"epoch": epoch, **equality})
            atomic_csv(
                output_root / "canonical_non_cc64_equality.csv",
                EQUALITY_COLUMNS,
                equality_rows,
            )
            candidate_histogram = canonical["strict_candidate_histogram"]
            for joint_id in range(16):
                human_count = int(baseline["human_histogram"][joint_id])
                original_count = int(
                    baseline["canonical_original_pt_histogram"][joint_id]
                )
                candidate_count = int(candidate_histogram[joint_id])
                distribution_rows.append(
                    {
                        "epoch": epoch,
                        "joint_id": joint_id,
                        "pattern": f"{joint_id:04b}",
                        "human_count": human_count,
                        "human_probability": human_count
                        / sum(baseline["human_histogram"]),
                        "original_pt_count": original_count,
                        "original_pt_probability": original_count
                        / sum(baseline["canonical_original_pt_histogram"]),
                        "encoder_decoder_count": candidate_count,
                        "encoder_decoder_probability": candidate_count
                        / sum(candidate_histogram),
                    }
                )
            atomic_csv(
                output_root / "canonical_joint16_distributions.csv",
                DISTRIBUTION_COLUMNS,
                distribution_rows,
            )

            epoch_path = output_root / f"epoch_{epoch:03d}.pt"
            atomic_torch_save(checkpoint_payload(model, run_config, row), epoch_path)
            last_payload = checkpoint_payload(model, run_config, row)
            last_payload.update(
                optimizer_state=optimizer.state_dict(),
                grad_scaler_state=scaler.state_dict(),
                completed_epoch=epoch,
                global_optimizer_steps=global_steps,
                best_epoch=stopping.best_epoch,
                early_stopping_counter=stopping.counter,
                amp_skipped_optimizer_steps_total=total_amp_skips,
            )
            last_payload.update(capture_rng_states())
            atomic_torch_save(last_payload, output_root / "last.pt")
            if improved:
                best_payload = checkpoint_payload(model, run_config, row)
                best_payload.update(best_epoch=epoch, selection_metric="canonical strict JS")
                atomic_torch_save(best_payload, output_root / "best.pt")
            update_status(
                output_root,
                state="training",
                current_epoch=epoch,
                completed_epochs=len(metrics),
                current_canonical_strict_js=canonical["strict_js_distance"],
                current_canonical_strict_intersection=canonical[
                    "strict_intersection"
                ],
                best_epoch=stopping.best_epoch,
                best_canonical_strict_js=stopping.best_js_distance,
                canonical_non_cc64_equality="19/19 PASS",
                global_optimizer_steps=global_steps,
            )
            log("EPOCH " + json.dumps(row, sort_keys=True))
            if should_stop:
                log(
                    f"EARLY_STOP epoch={epoch} best_epoch={stopping.best_epoch} "
                    f"best_js={stopping.best_js_distance:.15f}"
                )
                break

        manifest = write_checkpoint_manifest(
            output_root, metrics, stopping.best_epoch
        )
        best = next(
            row for row in metrics if int(row["epoch"]) == stopping.best_epoch
        )
        best_hash = sha256_file(output_root / "best.pt")
        (output_root / "ENCODER_DECODER_BINARY_TRAINING_REPORT.md").write_text(
            build_report(config, metrics, best, best_hash, baseline),
            encoding="utf-8",
        )
        update_status(
            output_root,
            status="completed",
            state="completed",
            completed_at=now(),
            completed_epochs=len(metrics),
            best_epoch=stopping.best_epoch,
            best_canonical_strict_js=stopping.best_js_distance,
            best_canonical_strict_intersection=stopping.best_intersection,
            best_checkpoint_sha256=best_hash,
            last_checkpoint_sha256=sha256_file(output_root / "last.pt"),
            checkpoint_manifest_entries=len(manifest),
            all_executed_epoch_checkpoints_preserved=True,
            stage1_neural_inference=0,
            asap_test_human_midi_access=0,
            canonical_test_inference=0,
            test_metrics_computed=0,
            test_audio_generated=0,
            scheduled_sampling_used=False,
        )
        log(
            f"RUNNER_COMPLETED epochs={len(metrics)} best_epoch={stopping.best_epoch} "
            f"best_js={stopping.best_js_distance:.15f} best_sha256={best_hash}"
        )
    except BaseException as error:
        update_status(
            output_root,
            status="failed",
            state="failed",
            failed_at=now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            stage1_neural_inference=0,
            asap_test_human_midi_access=0,
            canonical_test_inference=0,
        )
        log(f"RUNNER_FAILED error={error!r}")
        log(traceback.format_exc())
        raise
    finally:
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage2_binary_encoder_decoder_canonical_training_v1.json",
    )
    parser.add_argument("--confirm-full-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full_training:
        raise SystemExit("refusing to start without --confirm-full-training")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    run(read_json(config_path))


if __name__ == "__main__":
    main()
