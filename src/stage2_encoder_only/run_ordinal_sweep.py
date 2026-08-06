"""Safe sequential orchestration for the Stage 2 ordinal-loss sweep."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from .dataset import Stage2PedalDataset, stage2_pedal_collate_fn
from .diagnose_posterior import true_class_ranks
from .evaluate_oracle import _load_model, _make_window_sample
from .evaluate_ordinal_sweep import evaluate
from .model import Stage2PedalEncoderModel
from .ordinal_loss import squared_cdf_ordinal_loss
from .train_ordinal import (
    default_configuration,
    ordinal_evaluation_step,
    ordinal_train_step,
    train,
)
from .training import build_optimizer, create_grad_scaler, move_batch_to_device, set_deterministic_seed


EXPECTED_GPU_UUID = "GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b"
TMUX_SESSION = "stage2-ordinal-sweep-v0"
RUNS = (("lambda_0p1", 0.1), ("lambda_0p5", 0.5), ("lambda_1p0", 1.0))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class Tee:
    def __init__(self, *handles: TextIO) -> None:
        self.handles = handles

    def write(self, text: str) -> int:
        for handle in self.handles:
            handle.write(text)
            handle.flush()
        return len(text)

    def flush(self) -> None:
        for handle in self.handles:
            handle.flush()


def gpu_identity(require_idle: bool) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True,
    )
    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one container-visible GPU, found {len(rows)}")
    uuid, name, used, total, utilization = [item.strip() for item in rows[0].split(",")]
    if uuid != EXPECTED_GPU_UUID:
        raise RuntimeError(f"GPU UUID mismatch: expected {EXPECTED_GPU_UUID}, got {uuid}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    identity = {
        "uuid": uuid, "name": name, "memory_used_mib": int(used),
        "memory_total_mib": int(total), "utilization_percent": int(utilization),
        "compute_processes": processes,
    }
    if require_idle and (identity["memory_used_mib"] > 100 or processes):
        raise RuntimeError(f"expected assigned GPU to be idle, observed {identity}")
    return identity


def _overfit_evaluate(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    logits_chunks = []
    target_chunks = []
    batch_metrics = []
    for offset in range(0, 4, 2):
        batch = move_batch_to_device(stage2_pedal_collate_fn(samples[offset:offset + 2]), device)
        batch_metrics.append(ordinal_evaluation_step(model, batch, 1.0, True))
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(
                input_ids=batch["input_ids"], token_attention_mask=batch["token_attention_mask"],
                note_mask=batch["note_mask"], pedal_targets=None,
            )
        logits_chunks.append(output.logits.float())
        target_chunks.append(batch["pedal_targets"])
    logits = torch.cat(logits_chunks)
    targets = torch.cat(target_chunks)
    losses = squared_cdf_ordinal_loss(logits, targets, 1.0)
    prediction = logits.argmax(-1)
    valid = targets != -100
    exact_note = ((prediction == targets) | ~valid).all(-1)[valid.any(-1)].float().mean()
    mae = (prediction[valid].float() - targets[valid].float()).abs().mean()
    intermediate = valid & (targets >= 1) & (targets <= 126)
    intermediate_mae = (prediction[intermediate].float() - targets[intermediate].float()).abs().mean()
    intermediate_prediction_ratio = ((prediction[intermediate] >= 1) & (prediction[intermediate] <= 126)).float().mean()
    probabilities = torch.softmax(logits.float(), -1).cpu().numpy()
    rank = true_class_ranks(probabilities[intermediate.cpu().numpy()], targets[intermediate].cpu().numpy())
    return {
        "ce_loss": float(losses.ce),
        "ordinal_loss": float(losses.ordinal),
        "total_loss": float(losses.total),
        "exact_note_accuracy": float(exact_note),
        "mae": float(mae),
        "intermediate_target_mae": float(intermediate_mae),
        "intermediate_argmax_prediction_ratio": float(intermediate_prediction_ratio),
        "mean_intermediate_true_class_rank": float(rank.mean()),
        "intermediate_top10_recall": float((rank <= 10).mean()),
    }


def pedal_rich_overfit() -> dict[str, Any]:
    """Run the required in-memory four-window GPU memorization gate."""

    gpu = gpu_identity(require_idle=True)
    set_deterministic_seed(20260710)
    dataset = Stage2PedalDataset(
        "/workspace/public/ASAP/asap-dataset-v1.1",
        "/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv",
        "train", window_notes=512, stride_notes=256, cache_mode="preload",
    )
    performance_path = "Schubert/Piano_Sonatas/664-3/Lin07.mid"
    row = next(item for item in dataset.performances if item["performance_path"] == performance_path)
    tokens = dataset._token_cache[row["performance_path"]]
    starts = [0, 256, 512, 768]
    samples = [_make_window_sample(tokens, start, start + 512) for start in starts]
    device = torch.device("cuda:0")
    model = Stage2PedalEncoderModel.from_pretrained(
        "/workspace/project/checkpoints/pianist_transformer", freeze_encoder=False,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device)
    optimizer = build_optimizer(model, encoder_lr=1e-4, head_lr=1e-3, weight_decay=0.0)
    scaler = create_grad_scaler(True, "cuda", 1024.0)
    active_name, active_parameter = next(
        (name, parameter) for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "embed_tokens" not in name and parameter.ndim >= 2
    )
    encoder_before = active_parameter.detach().clone()
    head_before = {
        name: parameter.detach().clone()
        for name, parameter in model.classification_heads.named_parameters()
    }
    initial = _overfit_evaluate(model, samples, device)
    encoder_gradient_ok = True
    head_gradients_ok = True
    steps = 0
    try:
        for step in range(1, 101):
            offset = 0 if step % 2 else 2
            batch = move_batch_to_device(
                stage2_pedal_collate_fn(samples[offset:offset + 2]), device
            )
            metrics = ordinal_train_step(model, batch, optimizer, 1.0, scaler, True, 1.0)
            encoder_gradient_ok &= (
                float(metrics["encoder_gradient_norm"]) > 0
                and math_isfinite(metrics["encoder_gradient_norm"])
                and active_parameter.grad is not None
                and bool(torch.isfinite(active_parameter.grad).all())
            )
            head_gradients_ok &= all(
                any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and float(parameter.grad.abs().sum()) > 0
                    for parameter in head.parameters()
                )
                for head in model.classification_heads
            )
            steps = step
            if step % 5 == 0:
                current = _overfit_evaluate(model, samples, device)
                print(f"overfit step={step} metrics={json.dumps(current, sort_keys=True)}", flush=True)
                if current["exact_note_accuracy"] >= 0.98 and current["total_loss"] <= 0.10:
                    break
        final = _overfit_evaluate(model, samples, device)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("OOM during pedal-rich overfit gate") from exc
    encoder_changed = not torch.equal(encoder_before, active_parameter.detach())
    head_groups_changed = [
        all(
            not torch.equal(head_before[f"{slot}.{name}"], parameter.detach())
            for name, parameter in head.named_parameters()
        )
        for slot, head in enumerate(model.classification_heads)
    ]
    all_finite = all(
        math_isfinite(value)
        for group in (initial, final)
        for value in group.values()
    )
    passed = (
        final["exact_note_accuracy"] >= 0.98
        and final["total_loss"] <= 0.10
        and encoder_gradient_ok and head_gradients_ok and encoder_changed
        and all(head_groups_changed) and all_finite
    )
    result = {
        "passed": passed,
        "performance_path": performance_path,
        "window_starts": starts,
        "steps": steps,
        "initial": initial,
        "final": final,
        "active_encoder_parameter": active_name,
        "active_encoder_changed": encoder_changed,
        "encoder_gradient_finite_nonzero": encoder_gradient_ok,
        "every_head_received_finite_nonzero_gradient": head_gradients_ok,
        "every_head_parameter_group_changed": all(head_groups_changed),
        "head_groups_changed": head_groups_changed,
        "all_metrics_finite": all_finite,
        "oom": False,
        "gpu": gpu,
    }
    del model, optimizer, scaler, dataset
    torch.cuda.empty_cache()
    print("OVERFIT_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError(f"pedal-rich overfit gate failed: {result}")
    return result


def math_isfinite(value: Any) -> bool:
    return bool(np.isfinite(float(value)))


def _compatible_config(path: Path, coefficient: float) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = default_configuration(str(path.parent), coefficient)
    for key in (
        "lambda_ordinal", "seed", "batch_size", "max_epochs", "encoder_lr",
        "head_lr", "weight_decay", "max_grad_norm", "amp_init_scale",
        "amp_enabled", "freeze_encoder", "pin_memory", "num_workers",
    ):
        if config.get(key) != expected.get(key):
            raise RuntimeError(f"incompatible existing run {path.parent}: {key}")
    return config


def run_sweep(output_root: Path, overfit: dict[str, Any]) -> int:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    initial_entries = {path.name for path in output_root.iterdir()}
    if not initial_entries.issubset({"sweep.log", "sweep_status.json", *(label for label, _ in RUNS)}):
        raise FileExistsError(f"unexpected ordinal output entries: {sorted(initial_entries)}")
    status_path = output_root / "sweep_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        completed = list(status.get("completed_lambdas", []))
        start_time = status.get("start_time", utc_now())
    else:
        completed = []
        start_time = utc_now()
    status: dict[str, Any] = {
        "status": "running", "current_stage": "preflight", "current_lambda": None,
        "completed_lambdas": completed, "start_time": start_time, "end_time": None,
        "process_id": os.getpid(), "tmux_session_name": TMUX_SESSION,
        "last_completed_epoch": status.get("last_completed_epoch", 0) if 'status' in locals() else 0,
        "best_epochs": status.get("best_epochs", {}) if 'status' in locals() else {},
        "best_validation_losses": status.get("best_validation_losses", {}) if 'status' in locals() else {},
        "failure_message": None, "overfit_validation": overfit,
        "gpu": gpu_identity(require_idle=True),
    }
    atomic_json(status_path, status)
    expected_initial_hash: str | None = None
    try:
        for label, coefficient in RUNS:
            run_dir = output_root / label
            if coefficient in status["completed_lambdas"]:
                completed_config = _compatible_config(run_dir / "config.json", coefficient)
                if not (run_dir / "best.pt").is_file() or not (run_dir / "last.pt").is_file():
                    raise RuntimeError(f"completed run lacks checkpoints: {run_dir}")
                completed_hash = str(completed_config["initial_parameter_sha256"])
                if expected_initial_hash is None:
                    expected_initial_hash = completed_hash
                elif completed_hash != expected_initial_hash:
                    raise RuntimeError("completed ordinal run initial parameter hashes differ")
                print(f"skip completed compatible run lambda={coefficient}", flush=True)
                continue
            run_dir.mkdir(exist_ok=True)
            config = default_configuration(str(run_dir), coefficient)
            config_path = run_dir / "config.json"
            last_path = run_dir / "last.pt"
            if config_path.exists():
                existing = _compatible_config(config_path, coefficient)
                if not last_path.is_file():
                    raise RuntimeError(f"existing run has no resumable last.pt: {run_dir}")
                config["resume"] = str(last_path)
                print(f"resuming lambda={coefficient} from {last_path}", flush=True)
            status.update(current_stage="training", current_lambda=coefficient, last_completed_epoch=0)
            atomic_json(status_path, status)
            print(f"FIRST RUN STARTUP lambda={coefficient} output={run_dir}", flush=True)

            def callback(row: dict[str, Any]) -> None:
                status["last_completed_epoch"] = int(row["epoch"])
                status["best_epochs"][str(coefficient)] = int(row["best_epoch"])
                status["best_validation_losses"][str(coefficient)] = float(row["best_validation_loss"])
                atomic_json(status_path, status)

            with (run_dir / "train.log").open("a", encoding="utf-8") as train_log:
                tee = Tee(sys.stdout, train_log)
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    result = train(config, epoch_callback=callback)
            if expected_initial_hash is None:
                expected_initial_hash = result["initial_parameter_sha256"]
            elif result["initial_parameter_sha256"] != expected_initial_hash:
                raise RuntimeError("ordinal run initial parameter hashes differ")
            status.update(current_stage="checkpoint_reload", current_lambda=coefficient)
            atomic_json(status_path, status)
            model, _ = _load_model(run_dir / "best.pt", torch.device("cuda:0"))
            del model
            torch.cuda.empty_cache()
            status["completed_lambdas"].append(coefficient)
            status["best_epochs"][str(coefficient)] = int(result["best_epoch"])
            status["best_validation_losses"][str(coefficient)] = float(result["best_validation_loss"])
            atomic_json(status_path, status)
        status.update(current_stage="validation_comparison", current_lambda=None)
        atomic_json(status_path, status)
        comparison = evaluate(output_root, overfit)
        status.update(
            status="complete", current_stage="complete", end_time=utc_now(),
            validation_comparison=comparison,
        )
        atomic_json(status_path, status)
        return 0
    except BaseException as exc:
        status.update(
            status="failed", current_stage="failed", end_time=utc_now(),
            failure_message=f"{type(exc).__name__}: {exc}",
        )
        atomic_json(status_path, status)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    parser.add_argument("--overfit-only", action="store_true")
    parser.add_argument("--verified-overfit-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.overfit_only:
        pedal_rich_overfit()
        return 0
    verified_overfit_json = args.verified_overfit_json or os.environ.get("ORDINAL_OVERFIT_JSON")
    if verified_overfit_json is None and os.environ.get("ORDINAL_OVERFIT_B64"):
        verified_overfit_json = base64.b64decode(os.environ["ORDINAL_OVERFIT_B64"]).decode("utf-8")
    if not args.output_root or not verified_overfit_json:
        raise ValueError("full sweep requires --output-root and --verified-overfit-json")
    overfit = json.loads(verified_overfit_json)
    if not bool(overfit.get("passed")):
        raise ValueError("verified overfit result did not pass")
    return run_sweep(Path(args.output_root), overfit)


if __name__ == "__main__":
    raise SystemExit(main())
