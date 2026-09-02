#!/usr/bin/env python3
"""Regression-only frozen evaluation for Raw Huber + training-only auxiliary CE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_stage2_encoder_only_raw_cc64_huber_v0 as raw_evaluator
from scripts.run_stage2_4class_architecture_v0 import atomic_json, read_json
from scripts.run_stage2_encoder_only_raw_cc64_huber_v0 import canonical_config, make_raw_cc64_loader
from src.stage2_binary.canonical_stage1 import sha256_file
from src.stage2_binary.full_training import SharedBinaryWindowDataset
from src.stage2_four_class.raw_huber_aux_ce import RawHuberAuxCEEncoderModel
from src.stage2_four_class.representation import IGNORE_INDEX, classify_cc64
from src.stage2_four_class.validation_evaluator import classification_metrics


OBJECTIVE = "raw_huber_aux_ce"


def load_best_model(config: Mapping[str, Any], device: torch.device) -> tuple[RawHuberAuxCEEncoderModel, dict[str, Any]]:
    checkpoint = Path(config["output_root"]) / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = canonical_config(config)
    model = RawHuberAuxCEEncoderModel.from_pretrained(
        base["checkpoint_path"], delta=float(config["huber_delta_normalized"]),
        lambda_ce=float(config["lambda_ce"]), torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"hybrid checkpoint mismatch: {incompatible}")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("hybrid checkpoint contains non-finite parameters")
    metadata = {
        "path": str(checkpoint), "sha256": sha256_file(checkpoint),
        "epoch": int(payload["epoch"]),
        "validation_objective": float(payload["validation_objective"]),
        "validation_huber": float(payload["validation_huber"]),
        "validation_ce": float(payload["validation_ce"]),
        "validation_raw_cc64_mae": float(payload["validation_raw_cc64_mae"]),
        "initialization_source": base["checkpoint_path"],
        "candidate_checkpoint_used_for_initialization": False,
    }
    return model.to(device).eval(), metadata


def auxiliary_diagnostics(config: Mapping[str, Any], model: RawHuberAuxCEEncoderModel, device: torch.device) -> dict[str, Any]:
    base = canonical_config(config)
    dataset = SharedBinaryWindowDataset(base["cache_root"], "validation")
    loader = make_raw_cc64_loader(dataset, batch_size=int(base["micro_batch_size"]["encoder_only"]), pin_memory=bool(base["pin_memory"]))
    confusion = np.zeros((4, 4), dtype=np.int64)
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in cpu_batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(input_ids=batch["input_ids"], token_attention_mask=batch["token_attention_mask"], note_mask=batch["note_mask"])
            if output.auxiliary_logits is None:
                raise RuntimeError("auxiliary logits missing")
            valid = batch["pedal_targets"] != IGNORE_INDEX
            target = classify_cc64(batch["pedal_targets"], allow_ignore=True)[valid]
            predicted = output.auxiliary_logits.argmax(dim=-1)[valid]
            encoded = (target * 4 + predicted).cpu().numpy()
            confusion += np.bincount(encoded, minlength=16).reshape(4, 4)
    metrics = classification_metrics(confusion)
    return {
        "role": "training diagnostic only; excluded from candidate MIDI and main frozen evaluation",
        "universe": "canonical ASAP validation cache windows; overlapping-window weighted",
        "accuracy": metrics["token_accuracy"], "macro_f1": metrics["macro_f1"],
        "prediction_class_distribution": metrics["prediction_class_distribution"],
        "class_metrics": metrics["class_metrics"], "confusion_matrix": metrics["confusion_matrix"],
        "candidate_midi_usage_count": 0, "asap_test_access_count": 0,
    }


def run(config: Mapping[str, Any]) -> int:
    raw_evaluator.OBJECTIVE = OBJECTIVE
    raw_evaluator.load_best_model = load_best_model
    code = raw_evaluator.run(config)
    if code != 0:
        return code
    device = torch.device("cuda:0")
    model, _ = load_best_model(config, device)
    auxiliary = auxiliary_diagnostics(config, model, device)
    path = Path(config["output_root"]) / "validation_eval/evaluation.json"
    evaluation = read_json(path)
    evaluation["auxiliary_head_diagnostics"] = auxiliary
    evaluation["candidate_inference_head"] = "regression scalar only"
    evaluation["auxiliary_head_candidate_midi_usage_count"] = 0
    atomic_json(path, evaluation)
    atomic_json(Path(config["output_root"]) / "validation_eval/auxiliary_head_diagnostics.json", auxiliary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute: raise SystemExit("evaluation is guarded; pass --execute")
    return run(read_json(args.config))


if __name__ == "__main__": raise SystemExit(main())
