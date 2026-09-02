#!/usr/bin/env python3
"""CPU-only final-test MIDI generation for three frozen encoder-only models.

This runner reuses the completed Run A/B Stage-1 inventory and the frozen
validation model loading, overlap aggregation, decoding, and MIDI rendering
implementations.  It never runs Stage 1, reads Human ASAP test performances,
or computes test metrics.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_stage2_encoder_only_loss_phase3_v0 as phase3_validation  # noqa: E402
from scripts import evaluate_stage2_encoder_only_raw_huber_aux_ce_v0 as hybrid_validation  # noqa: E402
from scripts.evaluate_original_pt_early_metrics import PinnedTokenizerConfig  # noqa: E402
from scripts.run_stage2_4class_validation_eval_v0 import (  # noqa: E402
    ids_sha256,
    prediction_paths,
    render_stage2_candidate,
)
from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    assert_strict_non_cc64_equality,
    sha256_file,
    signature_sha256,
)
from src.stage2_four_class.validation_evaluator import infer_four_class_pedals  # noqa: E402
from third_party.PianistTransformer.src.utils.midi import midi_to_ids  # noqa: E402


SEED = 42
DEVICE = torch.device("cpu")
WINDOW_NOTES = 512
STRIDE_NOTES = 256

OUTPUT_ROOT = ROOT / "analysis/final_test_encoder_only_trio_inference_v0"
STATUS_PATH = OUTPUT_ROOT / "run_status.json"
CONFIG_PATH = OUTPUT_ROOT / "config.json"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
LOG_PATH = OUTPUT_ROOT / "inference.log"

RUN_AB_ROOT = ROOT / "analysis/final_test_run_ab_inference_v0"
RUN_AB_STATUS = RUN_AB_ROOT / "run_status.json"
RUN_AB_CONFIG = RUN_AB_ROOT / "config.json"
RUN_AB_MANIFEST = RUN_AB_ROOT / "manifest.csv"
CANONICAL_SOURCE_MANIFEST = (
    ROOT
    / "analysis/stage2_binary_v0/canonical_stage1_pipeline_v0/"
    / "canonical_stage1_manifest.csv"
)

PHASE3_PARENT_CONFIG = ROOT / "configs/stage2_encoder_only_4class_loss_phase3_v0.json"
HYBRID_PARENT_CONFIG = ROOT / "configs/stage2_encoder_only_raw_huber_aux_ce_v1.json"

WEIGHTED_DIR = ROOT / "analysis/stage2_encoder_only_4class_loss_phase3_v0/weighted_ce"
NTL_DIR = ROOT / "analysis/stage2_encoder_only_4class_loss_phase3_v0/ce_ntl_was_lambda_0p3"
HYBRID_DIR = ROOT / "analysis/stage2_encoder_only_raw_huber_aux_ce_v1"

MODEL_SPECS = {
    "weighted_ce": {
        "experiment_dir": WEIGHTED_DIR,
        "checkpoint": WEIGHTED_DIR / "best.pt",
        "checkpoint_sha256": "6cacf7735ac508f0218d057194807307ec47e46a46d93b7a85aed7c06e6e1322",
        "best_epoch": 2,
        "objective": "weighted_ce",
        "report": WEIGHTED_DIR / "WEIGHTED_CE_TRAINING_REPORT.md",
        "config": WEIGHTED_DIR / "config.json",
        "status": WEIGHTED_DIR / "run_status.json",
        "evaluation": WEIGHTED_DIR / "validation/evaluation.json",
        "output_suffix": "encoder_only_weighted_ce",
    },
    "ntl_was": {
        "experiment_dir": NTL_DIR,
        "checkpoint": NTL_DIR / "best.pt",
        "checkpoint_sha256": "332130fb9609b4862bd8bb0824367d3154cf5f5a7353dafc6e7a07e3801c35f1",
        "best_epoch": 2,
        "objective": "ce_ntl_was",
        "report": NTL_DIR / "NTL_WAS_TRAINING_REPORT.md",
        "config": NTL_DIR / "config.json",
        "status": NTL_DIR / "run_status.json",
        "evaluation": NTL_DIR / "validation/evaluation.json",
        "output_suffix": "encoder_only_ntl_was",
    },
    "huber_aux_ce": {
        "experiment_dir": HYBRID_DIR,
        "checkpoint": HYBRID_DIR / "best.pt",
        "checkpoint_sha256": "2a4550de0c6737ac0c9d688d0615d9e57a8a0c877eebc1dc670bec817aaa7c5e",
        "best_epoch": 5,
        "objective": "raw_huber_aux_ce",
        "report": HYBRID_DIR / "RAW_HUBER_AUX_CE_CLEAN_RERUN_REPORT.md",
        "config": HYBRID_DIR / "config.json",
        "status": HYBRID_DIR / "run_status.json",
        "evaluation": HYBRID_DIR / "validation_eval/evaluation.json",
        "output_suffix": "encoder_only_huber_aux_ce",
    },
}

MODEL_ORDER = ("weighted_ce", "ntl_was", "huber_aux_ce")
MANIFEST_FIELDS = (
    "num",
    "piece_id",
    "composer",
    "title",
    "canonical_stage1_midi_path",
    "canonical_stage1_midi_sha256",
    "canonical_non_cc64_signature_sha256",
    "weighted_ce_output_path",
    "weighted_ce_output_sha256",
    "weighted_ce_nonpedal_identity",
    "ntl_was_output_path",
    "ntl_was_output_sha256",
    "ntl_was_nonpedal_identity",
    "huber_aux_ce_output_path",
    "huber_aux_ce_output_sha256",
    "huber_aux_ce_nonpedal_identity",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def update_status(**values: Any) -> None:
    status = read_json(STATUS_PATH) if STATUS_PATH.is_file() else {}
    status.update(values)
    status["last_update"] = now()
    atomic_json(STATUS_PATH, status)


def close_enough(value: Any, expected: float) -> bool:
    return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def assert_cpu_only() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "":
        raise RuntimeError(f'CPU-only runner requires CUDA_VISIBLE_DEVICES="", found {visible!r}')
    if torch.cuda.is_available() or torch.cuda.device_count() != 0 or torch.cuda.is_initialized():
        raise RuntimeError("CUDA was not fully hidden before CPU-only inference")
    probe = torch.ones(1, device=DEVICE)
    if probe.device.type != "cpu":
        raise AssertionError("CPU tensor allocation preflight failed")
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_initialized": False,
        "device": "cpu",
        "preflight_tensor_device": str(probe.device),
    }


def assert_model_cpu(model: torch.nn.Module, label: str) -> None:
    devices = {parameter.device.type for parameter in model.parameters()}
    devices.update(buffer.device.type for buffer in model.buffers())
    if devices != {"cpu"}:
        raise RuntimeError(f"{label} model is not CPU-only: {sorted(devices)}")
    if torch.cuda.is_initialized():
        raise RuntimeError(f"CUDA initialized while loading {label}")


def _assert_head_shapes(name: str, state: Mapping[str, torch.Tensor]) -> None:
    if name in {"weighted_ce", "ntl_was"}:
        for slot in range(4):
            if tuple(state[f"classification_heads.{slot}.weight"].shape) != (4, 768):
                raise RuntimeError(f"{name} is not the encoder-only 4-class architecture")
            if tuple(state[f"classification_heads.{slot}.bias"].shape) != (4,):
                raise RuntimeError(f"{name} classification bias shape changed")
        if any(key.startswith("regression_heads.") for key in state):
            raise RuntimeError(f"{name} unexpectedly contains regression heads")
    else:
        for slot in range(4):
            if tuple(state[f"regression_heads.{slot}.weight"].shape) != (1, 768):
                raise RuntimeError("Huber+Aux regression head shape changed")
            if tuple(state[f"classification_heads.{slot}.weight"].shape) != (4, 768):
                raise RuntimeError("Huber+Aux auxiliary 4-class head shape changed")


def verify_model_artifacts() -> dict[str, Any]:
    for path in (PHASE3_PARENT_CONFIG, HYBRID_PARENT_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance: dict[str, Any] = {}
    for name in MODEL_ORDER:
        spec = MODEL_SPECS[name]
        for field in ("checkpoint", "report", "config", "status", "evaluation"):
            path = Path(spec[field])
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
        if sha256_file(spec["checkpoint"]) != spec["checkpoint_sha256"]:
            raise RuntimeError(f"fixed checkpoint SHA-256 changed: {name}")
        config = read_json(spec["config"])
        status = read_json(spec["status"])
        evaluation = read_json(spec["evaluation"])
        report = Path(spec["report"]).read_text(encoding="utf-8")
        if status.get("status") != "completed" or int(status.get("best_epoch", -1)) != spec["best_epoch"]:
            raise RuntimeError(f"training status/best epoch changed: {name}")
        payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != spec["best_epoch"] or payload.get("objective") != spec["objective"]:
            raise RuntimeError(f"checkpoint objective/epoch changed: {name}")
        state = payload.get("model_state")
        if not isinstance(state, Mapping):
            raise RuntimeError(f"checkpoint model_state is missing: {name}")
        if any(tensor.device.type != "cpu" for tensor in state.values() if isinstance(tensor, torch.Tensor)):
            raise RuntimeError(f"checkpoint did not map fully to CPU: {name}")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in state.values() if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()):
            raise FloatingPointError(f"checkpoint contains non-finite tensors: {name}")
        _assert_head_shapes(name, state)
        checkpoint_configuration = dict(payload.get("configuration", {}))

        common = {
            "experiment_directory": str(spec["experiment_dir"]),
            "training_report": str(spec["report"]),
            "training_config": str(spec["config"]),
            "training_status": str(spec["status"]),
            "validation_evaluation": str(spec["evaluation"]),
            "checkpoint_path": str(spec["checkpoint"]),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "best_epoch": spec["best_epoch"],
            "model_output_formulation": config["architecture"],
            "window_notes": int(config["window_notes"]),
            "stride_notes": int(config["stride_notes"]),
            "dtype": "torch.float32",
        }
        if common["window_notes"] != WINDOW_NOTES or common["stride_notes"] != STRIDE_NOTES:
            raise RuntimeError(f"frozen window/stride changed: {name}")

        if name == "weighted_ce":
            expected_weights = [0.875636094076258, 1.3412009506983258, 1.0660130744984522, 0.7171498807269644]
            if config.get("objective") != "weighted_ce" or config.get("class_weights") != expected_weights:
                raise RuntimeError("Weighted CE fixed train-distribution weights changed")
            if checkpoint_configuration.get("class_weights") != expected_weights:
                raise RuntimeError("Weighted CE checkpoint configuration weights changed")
            if "Weighted cross entropy with train-cache-only" not in report:
                raise RuntimeError("Weighted CE report identity changed")
            fingerprint = {
                "accuracy": evaluation["classification"]["token_accuracy"],
                "macro_f1": evaluation["classification"]["macro_f1"],
                "transition_precision": evaluation["transition"]["pooled"]["precision"],
                "transition_recall": evaluation["transition"]["pooled"]["recall"],
                "transition_f1": evaluation["transition"]["pooled"]["f1"],
                "js": evaluation["patterns"]["js_divergence_base2"],
                "intersection": evaluation["patterns"]["intersection"],
            }
            expected = (0.5335872054342352, 0.4082601893281006, 0.34097529045982655, 0.6027829206202268, 0.4355651246890612, 0.039054568580349, 0.8582923216686866)
            if not all(close_enough(value, target) for value, target in zip(fingerprint.values(), expected)):
                raise RuntimeError("Weighted CE validation fingerprint changed")
            common.update(
                objective="4-class Weighted CE with fixed train-cache inverse-sqrt weights",
                class_boundaries=config["class_boundaries"],
                representatives=config["representatives"],
                class_weights=expected_weights,
                decoding_identity="512-note/stride-256 overlap step-logit mean then deterministic argmax",
                validation_inference_implementation="src.stage2_four_class.validation_evaluator::infer_four_class_pedals + scripts.run_stage2_4class_validation_eval_v0::render_stage2_candidate",
                validation_fingerprint=fingerprint,
            )
        elif name == "ntl_was":
            if config.get("objective") != "ce_ntl_was" or not close_enough(config.get("ntl_was_lambda"), 0.3):
                raise RuntimeError("NTL-WAS objective/lambda changed")
            if not close_enough(checkpoint_configuration.get("ntl_was_lambda"), 0.3):
                raise RuntimeError("NTL-WAS checkpoint lambda changed")
            if "CE - 0.3" not in report:
                raise RuntimeError("NTL-WAS report identity changed")
            fingerprint = {
                "accuracy": evaluation["classification"]["token_accuracy"],
                "macro_f1": evaluation["classification"]["macro_f1"],
                "transition_f1": evaluation["transition"]["pooled"]["f1"],
                "js": evaluation["patterns"]["js_divergence_base2"],
                "intersection": evaluation["patterns"]["intersection"],
            }
            expected = (0.5708345432691424, 0.3951194130796312, 0.43640042223786063, 0.07140407425870704, 0.8111029884455748)
            if not all(close_enough(value, target) for value, target in zip(fingerprint.values(), expected)):
                raise RuntimeError("NTL-WAS validation fingerprint changed")
            common.update(
                objective="4-class CE + NTL-WAS",
                ntl_was_lambda=0.3,
                class_boundaries=config["class_boundaries"],
                representatives=config["representatives"],
                decoding_identity="512-note/stride-256 overlap step-logit mean then deterministic argmax",
                validation_inference_implementation="src.stage2_four_class.validation_evaluator::infer_four_class_pedals + scripts.run_stage2_4class_validation_eval_v0::render_stage2_candidate",
                validation_fingerprint=fingerprint,
            )
        else:
            if config.get("objective") != "raw_huber_aux_ce" or not close_enough(config.get("lambda_ce"), 0.1):
                raise RuntimeError("Huber+Aux objective/lambda changed")
            if not close_enough(checkpoint_configuration.get("lambda_ce"), 0.1):
                raise RuntimeError("Huber+Aux checkpoint lambda changed")
            if evaluation.get("candidate_inference_head") != "regression scalar only" or int(evaluation.get("auxiliary_head_candidate_midi_usage_count", -1)) != 0:
                raise RuntimeError("Huber+Aux validation was not regression-only")
            if "Inference: regression output only" not in report:
                raise RuntimeError("Huber+Aux report identity changed")
            common.update(
                objective="raw CC64 Huber regression + training-only auxiliary 4-class CE",
                lambda_aux=0.1,
                inference_head="regression-only",
                auxiliary_head_candidate_midi_usage_count=0,
                decoding_identity=config["inference_conversion"],
                validation_inference_implementation="scripts.evaluate_stage2_encoder_only_loss_phase3_v0::infer_huber + scripts.run_stage2_4class_validation_eval_v0::render_stage2_candidate",
            )
        provenance[name] = common
        del payload, state
        gc.collect()
    return provenance


def output_path(number: int, model_name: str) -> Path:
    suffix = MODEL_SPECS[model_name]["output_suffix"]
    return ROOT / "outputs/midi" / f"{number}_{suffix}.mid"


def canonical_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for path in (RUN_AB_STATUS, RUN_AB_CONFIG, RUN_AB_MANIFEST, CANONICAL_SOURCE_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_ab_status = read_json(RUN_AB_STATUS)
    if run_ab_status.get("status") != "completed":
        raise RuntimeError("Run A/B final inference is not completed")
    if int(run_ab_status.get("run_a_success_count", -1)) != 23 or int(run_ab_status.get("run_b_success_count", -1)) != 23:
        raise RuntimeError("Run A/B final inference did not complete 23/23 per model")
    run_ab_config = read_json(RUN_AB_CONFIG)
    run_ab_rows = read_csv(RUN_AB_MANIFEST)
    source_rows = read_csv(CANONICAL_SOURCE_MANIFEST)
    if len(run_ab_rows) != 23 or len(source_rows) != 23:
        raise RuntimeError("canonical Stage-1 inventories are not exactly 23 rows")
    config_inventory = {int(row["num"]): row for row in run_ab_config["canonical_stage1_inventory"]}
    source_by_num = {int(row["num"]): row for row in source_rows}
    if set(config_inventory) != set(range(23)) or set(source_by_num) != set(range(23)):
        raise RuntimeError("canonical Stage-1 numbering is not exactly 0..22")

    inventory: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row in sorted(run_ab_rows, key=lambda value: int(value["num"])):
        number = int(row["num"])
        cfg = config_inventory[number]
        source = source_by_num[number]
        canonical = Path(row["canonical_stage1_midi_path"])
        if row["piece_id"] != cfg["piece_id"] or row["piece_id"] != source["piece_id"]:
            raise RuntimeError(f"Stage-1 piece join mismatch for num={number}")
        if str(canonical) != cfg["path"] or str(canonical) != source["canonical_original_pt_midi_path"]:
            raise RuntimeError(f"Stage-1 path differs from Run A/B provenance for num={number}")
        digest = sha256_file(canonical)
        signature = signature_sha256(canonical)
        if digest != row["canonical_stage1_midi_sha256"] or digest != cfg["sha256"] or digest != source["canonical_midi_sha256"]:
            raise RuntimeError(f"Stage-1 MIDI SHA-256 differs from Run A/B for num={number}")
        if signature != row["canonical_non_cc64_signature_sha256"] or signature != cfg["non_cc64_signature_sha256"] or signature != source["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"Stage-1 non-CC64 signature differs from Run A/B for num={number}")
        score = Path(source["source_score_path"])
        if not score.is_file() or sha256_file(score) != source["source_score_sha256"]:
            raise RuntimeError(f"canonical score mapping source changed for num={number}")
        paths = {name: output_path(number, name) for name in MODEL_ORDER}
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite final-test outputs: {existing}")
        item = {
            "num": number,
            "piece_id": row["piece_id"],
            "composer": row["composer"],
            "title": row["title"],
            "canonical_midi_path": str(canonical),
            "canonical_midi_sha256": digest,
            "canonical_non_cc64_signature_sha256": signature,
            "source_score_path": str(score),
            "source_score_sha256": source["source_score_sha256"],
            "note_count": int(source["note_count"]),
            "outputs": {name: str(path) for name, path in paths.items()},
        }
        inventory.append(item)
        manifest.append(
            {
                "num": number,
                "piece_id": row["piece_id"],
                "composer": row["composer"],
                "title": row["title"],
                "canonical_stage1_midi_path": str(canonical),
                "canonical_stage1_midi_sha256": digest,
                "canonical_non_cc64_signature_sha256": signature,
                **{
                    f"{name}_{field}": value
                    for name in MODEL_ORDER
                    for field, value in (
                        ("output_path", str(paths[name])),
                        ("output_sha256", ""),
                        ("nonpedal_identity", "PENDING"),
                    )
                },
            }
        )
    return inventory, manifest


def build_config(cpu: Mapping[str, Any], provenance: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "experiment": OUTPUT_ROOT.name,
        "execution_started_at": now(),
        "device": "cpu",
        "seed": SEED,
        "cpu_only_preflight": dict(cpu),
        "canonical_stage1_manifest_reused_from_run_ab": str(RUN_AB_MANIFEST),
        "canonical_stage1_config_reused_from_run_ab": str(RUN_AB_CONFIG),
        "canonical_stage1_manifest_sha256": sha256_file(RUN_AB_MANIFEST),
        "canonical_stage1_count": len(inventory),
        "canonical_stage1_hash_match_with_run_ab": True,
        "canonical_stage1_inventory": [
            {
                "num": item["num"],
                "piece_id": item["piece_id"],
                "path": item["canonical_midi_path"],
                "sha256": item["canonical_midi_sha256"],
                "non_cc64_signature_sha256": item["canonical_non_cc64_signature_sha256"],
            }
            for item in inventory
        ],
        "weighted_ce": provenance["weighted_ce"],
        "ntl_was": provenance["ntl_was"],
        "huber_aux_ce": provenance["huber_aux_ce"],
        "common_inference_semantics": {
            "architecture": "PT encoder-only",
            "window_notes": WINDOW_NOTES,
            "stride_notes": STRIDE_NOTES,
            "stage2_input": "canonical Original PT Pitch/Onset-IOI/Velocity/Duration only; all four pedal token columns masked",
            "midi_reconstruction": "frozen validation render_stage2_candidate followed by CC64-only canonical transplant",
            "exact_non_pedal_identity": "assert_strict_non_cc64_equality + signature_sha256",
        },
        "output_naming": {
            "weighted_ce": "outputs/midi/{num}_encoder_only_weighted_ce.mid",
            "ntl_was": "outputs/midi/{num}_encoder_only_ntl_was.mid",
            "huber_aux_ce": "outputs/midi/{num}_encoder_only_huber_aux_ce.mid",
        },
        "stage1_inference_regeneration": 0,
        "human_test_reference_access": 0,
        "test_metric_computation": 0,
        "checkpoint_selection_during_test": 0,
        "wav_rendering": 0,
    }


def source_tokens(item: Mapping[str, Any], tokenizer: Any) -> np.ndarray:
    values = np.asarray(
        midi_to_ids(tokenizer, MidiFile(item["canonical_midi_path"])), dtype=np.int64
    )
    if values.size % 8:
        raise RuntimeError(f"canonical Stage-1 token sequence is incomplete: num={item['num']}")
    if values.size // 8 != int(item["note_count"]):
        raise RuntimeError(f"canonical Stage-1 note count changed: num={item['num']}")
    return values


def render_final(
    model_name: str,
    item: Mapping[str, Any],
    source: np.ndarray,
    classes: np.ndarray,
    candidate: np.ndarray,
    inference: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    render_row = {
        "piece_id": item["piece_id"],
        "generated_token_sha256_int64_le": ids_sha256(source),
        "selected_score_absolute_path": item["source_score_path"],
        "canonical_midi_path": item["canonical_midi_path"],
        "canonical_non_cc64_signature_sha256": item["canonical_non_cc64_signature_sha256"],
    }
    frozen_root = OUTPUT_ROOT / "frozen_validation_render"
    metadata = render_stage2_candidate(
        frozen_root,
        model_name,
        render_row,
        classes,
        candidate,
        inference,
        checkpoint_hash=MODEL_SPECS[model_name]["checkpoint_sha256"],
        tokenizer_config=tokenizer,
    )
    validation_candidate = prediction_paths(frozen_root, model_name, item["piece_id"])["midi"]
    final_path = Path(item["outputs"][model_name])
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    shutil.copy2(validation_candidate, temporary)
    os.replace(temporary, final_path)
    identity = assert_strict_non_cc64_equality(item["canonical_midi_path"], final_path)
    if signature_sha256(final_path) != item["canonical_non_cc64_signature_sha256"]:
        raise AssertionError(f"strict non-pedal signature changed: {final_path}")
    if sha256_file(item["canonical_midi_path"]) != item["canonical_midi_sha256"]:
        raise AssertionError(f"canonical Stage-1 MIDI was modified: num={item['num']}")
    return {
        "output_path": str(final_path),
        "output_sha256": sha256_file(final_path),
        "identity": identity,
        "validation_render_metadata": metadata,
    }


def run_model(
    model_name: str,
    model: torch.nn.Module,
    inventory: Sequence[Mapping[str, Any]],
    manifest: list[dict[str, Any]],
    tokenizer: Any,
) -> None:
    assert_model_cpu(model, model_name)
    for index, item in enumerate(inventory):
        number = int(item["num"])
        update_status(
            stage=f"{model_name}_inference",
            current_num=number,
            current_piece_id=item["piece_id"],
        )
        log(f"{model_name.upper()}_PIECE_START num={number} piece_id={item['piece_id']} device=cpu")
        source = source_tokens(item, tokenizer)
        if model_name == "huber_aux_ce":
            classes, candidate, inference, continuous = phase3_validation.infer_huber(
                model,
                source,
                device=DEVICE,
                window_notes=WINDOW_NOTES,
                stride_notes=STRIDE_NOTES,
            )
            continuous_path = (
                OUTPUT_ROOT
                / "frozen_validation_render"
                / "predictions"
                / model_name
                / item["piece_id"]
                / "continuous_normalized_float32.npy"
            )
            continuous_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(continuous_path, continuous.astype(np.float32), allow_pickle=False)
        else:
            classes, candidate, inference = infer_four_class_pedals(
                model,
                source,
                architecture="encoder_only",
                device=DEVICE,
                window_notes=WINDOW_NOTES,
                stride_notes=STRIDE_NOTES,
            )
        rendered = render_final(
            model_name, item, source, classes, candidate, inference, tokenizer
        )
        manifest[index][f"{model_name}_output_sha256"] = rendered["output_sha256"]
        manifest[index][f"{model_name}_nonpedal_identity"] = "PASS_STRICT"
        atomic_csv(MANIFEST_PATH, manifest)
        update_status(**{f"{model_name}_success_count": index + 1})
        log(
            f"{model_name.upper()}_PIECE_PASS num={number} "
            f"output={rendered['output_path']} identity=PASS_STRICT"
        )


def run() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite inference provenance: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    update_status(
        status="running",
        stage="initializing",
        started_at=now(),
        pid=os.getpid(),
        seed=SEED,
        device="cpu",
        canonical_stage1_count=0,
        weighted_ce_success_count=0,
        weighted_ce_failure_count=0,
        ntl_was_success_count=0,
        ntl_was_failure_count=0,
        huber_aux_ce_success_count=0,
        huber_aux_ce_failure_count=0,
        human_test_reference_access=0,
        test_metric_computation=0,
        stage1_inference_regeneration=0,
    )
    log("RUN_START final-test encoder-only trio inference-generation only")
    seed_everything()
    cpu = assert_cpu_only()
    log("CPU_ONLY_PASS visible='' cuda_available=0 device_count=0 initialized=0 device=cpu")
    provenance = verify_model_artifacts()
    log(
        "CHECKPOINTS_PASS "
        f"weighted_epoch=2 weighted={MODEL_SPECS['weighted_ce']['checkpoint']} "
        f"ntl_epoch=2 ntl_lambda=0.3 ntl={MODEL_SPECS['ntl_was']['checkpoint']} "
        f"huber_epoch=5 huber_regression_only=1 huber={MODEL_SPECS['huber_aux_ce']['checkpoint']}"
    )
    inventory, manifest = canonical_inventory()
    atomic_csv(MANIFEST_PATH, manifest)
    atomic_json(CONFIG_PATH, build_config(cpu, provenance, inventory))
    update_status(
        stage="preflight_complete",
        canonical_stage1_count=23,
        canonical_stage1_hash_match_with_run_ab=True,
        checkpoints={
            name: {
                "path": str(MODEL_SPECS[name]["checkpoint"]),
                "sha256": MODEL_SPECS[name]["checkpoint_sha256"],
                "best_epoch": MODEL_SPECS[name]["best_epoch"],
            }
            for name in MODEL_ORDER
        },
        cpu_only_preflight=cpu,
    )
    log("CANONICAL_INVENTORY_PASS count=23 nums=0..22 run_ab_path_hash_match=1 stage1_regeneration=0")

    phase3_config = read_json(PHASE3_PARENT_CONFIG)
    hybrid_config = read_json(HYBRID_PARENT_CONFIG)
    tokenizer = PinnedTokenizerConfig()

    update_status(stage="weighted_ce_model_loading")
    log("WEIGHTED_CE_MODEL_LOAD_START device=cpu dtype=torch.float32")
    weighted_model, weighted_checkpoint = phase3_validation.load_model(
        phase3_config, "weighted_ce", DEVICE
    )
    if int(weighted_checkpoint["epoch"]) != 2:
        raise RuntimeError("Weighted CE validation loader resolved the wrong checkpoint")
    assert_model_cpu(weighted_model, "weighted_ce")
    log("WEIGHTED_CE_MODEL_LOAD_PASS epoch=2 device=cpu cuda_initialized=0")
    update_status(stage="weighted_ce_inference", current_num=inventory[0]["num"])
    log(
        f"INITIAL_READY stage=WEIGHTED_CE_PIECE_START num={inventory[0]['num']} "
        f"piece_id={inventory[0]['piece_id']} device=cpu"
    )
    run_model("weighted_ce", weighted_model, inventory, manifest, tokenizer)
    del weighted_model
    gc.collect()
    update_status(stage="weighted_ce_complete", weighted_ce_success_count=23)
    log("WEIGHTED_CE_COMPLETE success=23 failure=0 identity=23/23")

    seed_everything()
    update_status(stage="ntl_was_model_loading", current_num=None, current_piece_id=None)
    log("NTL_WAS_MODEL_LOAD_START device=cpu dtype=torch.float32 lambda=0.3")
    ntl_model, ntl_checkpoint = phase3_validation.load_model(
        phase3_config, "ce_ntl_was", DEVICE
    )
    if int(ntl_checkpoint["epoch"]) != 2:
        raise RuntimeError("NTL-WAS validation loader resolved the wrong checkpoint")
    assert_model_cpu(ntl_model, "ntl_was")
    log("NTL_WAS_MODEL_LOAD_PASS epoch=2 lambda=0.3 device=cpu cuda_initialized=0")
    run_model("ntl_was", ntl_model, inventory, manifest, tokenizer)
    del ntl_model
    gc.collect()
    update_status(stage="ntl_was_complete", ntl_was_success_count=23)
    log("NTL_WAS_COMPLETE success=23 failure=0 identity=23/23")

    seed_everything()
    update_status(stage="huber_aux_ce_model_loading", current_num=None, current_piece_id=None)
    log("HUBER_AUX_CE_MODEL_LOAD_START device=cpu dtype=torch.float32 inference=regression-only lambda_aux=0.1")
    hybrid_model, hybrid_checkpoint = hybrid_validation.load_best_model(
        hybrid_config, DEVICE
    )
    if int(hybrid_checkpoint["epoch"]) != 5:
        raise RuntimeError("Huber+Aux validation loader resolved the wrong checkpoint")
    assert_model_cpu(hybrid_model, "huber_aux_ce")
    log("HUBER_AUX_CE_MODEL_LOAD_PASS epoch=5 regression_only=1 auxiliary_candidate_usage=0 device=cpu")
    run_model("huber_aux_ce", hybrid_model, inventory, manifest, tokenizer)
    del hybrid_model
    gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU-only inference")
    update_status(
        status="completed",
        stage="completed",
        completed_at=now(),
        current_num=None,
        current_piece_id=None,
        weighted_ce_success_count=23,
        weighted_ce_failure_count=0,
        ntl_was_success_count=23,
        ntl_was_failure_count=0,
        huber_aux_ce_success_count=23,
        huber_aux_ce_failure_count=0,
        nonpedal_identity_pass_count=69,
        nonpedal_identity_failure_count=0,
        cuda_initialized_during_run=False,
    )
    log("RUN_COMPLETE weighted_ce=23/23 ntl_was=23/23 huber_aux_ce=23/23 nonpedal_identity=69/69 metrics=0 human_test=0")


def main() -> None:
    try:
        run()
    except Exception as error:
        try:
            if OUTPUT_ROOT.exists():
                status = read_json(STATUS_PATH) if STATUS_PATH.is_file() else {}
                stage = str(status.get("stage", "unknown"))
                update_status(
                    status="failed",
                    failed_at=now(),
                    failure_stage=stage,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                log(f"RUN_FAILED stage={stage} error={type(error).__name__}: {error}")
                with LOG_PATH.open("a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
        finally:
            raise


if __name__ == "__main__":
    main()
