#!/usr/bin/env python3
"""CPU-only final-test MIDI generation for frozen Run A and Run B.

This runner never performs Stage-1 inference, reads Human ASAP test
performances, computes test metrics, or selects a checkpoint.  It reuses each
model's frozen validation inference and MIDI serialization implementation.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage2_binary_2slot_full_v0 as run_a_validation  # noqa: E402
from scripts import run_state_anchored_transition_v1_full as run_b_validation  # noqa: E402
from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    assert_strict_non_cc64_equality,
    sha256_file,
    signature_sha256,
)
from src.stage2_binary_2slot.frozen_pt_validation import (  # noqa: E402
    load_frozen_pt_piece,
    render_cc64_only,
    rollout_frozen_pt,
)
from src.stage2_state_anchored.inference import (  # noqa: E402
    collect_frozen_pt_piece_logits,
    decode_performance,
)
from src.stage2_state_anchored.serialization import (  # noqa: E402
    render_state_anchored_candidate,
)


SEED = 42
DEVICE = torch.device("cpu")
OUTPUT_ROOT = ROOT / "analysis/final_test_run_ab_inference_v0"
STATUS_PATH = OUTPUT_ROOT / "run_status.json"
LOG_PATH = OUTPUT_ROOT / "inference.log"
CONFIG_PATH = OUTPUT_ROOT / "config.json"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
CANONICAL_MANIFEST = (
    ROOT
    / "analysis/stage2_binary_v0/canonical_stage1_pipeline_v0/canonical_stage1_manifest.csv"
)
RUN_A_CHECKPOINT = ROOT / "analysis/stage2_binary_2slot_D_pre_main_post_full_v1/best.pt"
RUN_B_CHECKPOINT = ROOT / "analysis/stage2_state_anchored_transition_v1_full_v0/best.pt"
RUN_A_STATUS = RUN_A_CHECKPOINT.parent / "run_status.json"
RUN_B_STATUS = RUN_B_CHECKPOINT.parent / "run_status.json"
RUN_A_EXPECTED_SHA256 = "0b4f4e94eef565604b9284bbddf45aaea4adb499a21def8215cda809e77cc5d2"
RUN_B_EXPECTED_SHA256 = "2f26e91b44a80ca916f4afbc5eb3396106c60fac8721822394b73e0ab79d56ad"
RUN_A_BEST_EPOCH = 8
RUN_B_BEST_EPOCH = 9
MANIFEST_FIELDS = (
    "num",
    "piece_id",
    "composer",
    "title",
    "canonical_stage1_midi_path",
    "canonical_stage1_midi_sha256",
    "canonical_non_cc64_signature_sha256",
    "run_a_output_path",
    "run_a_output_sha256",
    "run_a_nonpedal_identity",
    "run_a_eot_extension_ticks",
    "run_b_output_path",
    "run_b_output_sha256",
    "run_b_nonpedal_identity",
    "run_b_eot_extension_ticks",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_status(**values: Any) -> None:
    status = read_json(STATUS_PATH) if STATUS_PATH.is_file() else {}
    status.update(values)
    status["last_update"] = now()
    atomic_json(STATUS_PATH, status)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def assert_cpu_only() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "":
        raise RuntimeError(
            f'CPU-only runner requires CUDA_VISIBLE_DEVICES="", found {visible!r}'
        )
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


def output_paths(number: int) -> tuple[Path, Path]:
    return (
        ROOT / "outputs/midi" / f"{number}_run_a_count2slot.mid",
        ROOT / "outputs/midi" / f"{number}_run_b_state_anchored.mid",
    )


def canonical_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not CANONICAL_MANIFEST.is_file():
        raise FileNotFoundError(CANONICAL_MANIFEST)
    rows = read_csv(CANONICAL_MANIFEST)
    if len(rows) != 23:
        raise RuntimeError(f"expected 23 canonical Stage-1 rows, found {len(rows)}")
    numbers = [int(row["num"]) for row in rows]
    if sorted(numbers) != list(range(23)) or len(set(numbers)) != 23:
        raise RuntimeError(f"canonical numbering is not exactly 0..22: {sorted(numbers)}")
    if len({row["piece_id"] for row in rows}) != 23:
        raise RuntimeError("canonical Stage-1 manifest has duplicate piece IDs")

    inventory: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["num"])):
        number = int(row["num"])
        canonical = ROOT / "outputs/midi" / f"{number}_original_pt.mid"
        recorded = Path(row["canonical_original_pt_midi_path"])
        if recorded.resolve() != canonical.resolve():
            raise RuntimeError(f"canonical path mismatch for num={number}: {recorded}")
        if not canonical.is_file() or canonical.stat().st_size <= 0:
            raise FileNotFoundError(canonical)
        digest = sha256_file(canonical)
        signature = signature_sha256(canonical)
        if digest != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical MIDI SHA-256 mismatch for num={number}")
        if signature != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"canonical non-CC64 signature mismatch for num={number}")
        run_a_output, run_b_output = output_paths(number)
        if run_a_output.exists() or run_b_output.exists():
            raise FileExistsError(
                f"refusing to overwrite final-test output for num={number}: "
                f"{run_a_output}, {run_b_output}"
            )
        item = {
            "num": number,
            "piece_id": row["piece_id"],
            "composer": row["composer"],
            "title": row["title"],
            "canonical_midi_path": str(canonical),
            "canonical_midi_sha256": digest,
            "canonical_non_cc64_signature_sha256": signature,
            "run_a_output_path": str(run_a_output),
            "run_b_output_path": str(run_b_output),
        }
        inventory.append(item)
        manifest_rows.append(
            {
                "num": number,
                "piece_id": row["piece_id"],
                "composer": row["composer"],
                "title": row["title"],
                "canonical_stage1_midi_path": str(canonical),
                "canonical_stage1_midi_sha256": digest,
                "canonical_non_cc64_signature_sha256": signature,
                "run_a_output_path": str(run_a_output),
                "run_a_output_sha256": "",
                "run_a_nonpedal_identity": "PENDING",
                "run_a_eot_extension_ticks": "",
                "run_b_output_path": str(run_b_output),
                "run_b_output_sha256": "",
                "run_b_nonpedal_identity": "PENDING",
                "run_b_eot_extension_ticks": "",
            }
        )
    return inventory, manifest_rows


def verify_fixed_artifacts() -> dict[str, Any]:
    for path in (RUN_A_CHECKPOINT, RUN_B_CHECKPOINT, RUN_A_STATUS, RUN_B_STATUS):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    run_a_status = read_json(RUN_A_STATUS)
    run_b_status = read_json(RUN_B_STATUS)
    if run_a_status.get("status") != "COMPLETE" or int(run_a_status.get("best_epoch", -1)) != 8:
        raise RuntimeError("Run A completed/best-epoch provenance changed")
    if run_b_status.get("status") != "COMPLETE" or int(run_b_status.get("best_epoch", -1)) != 9:
        raise RuntimeError("Run B completed/best-epoch provenance changed")
    run_a_sha = sha256_file(RUN_A_CHECKPOINT)
    run_b_sha = sha256_file(RUN_B_CHECKPOINT)
    if run_a_sha != RUN_A_EXPECTED_SHA256 or run_b_sha != RUN_B_EXPECTED_SHA256:
        raise RuntimeError("fixed best-checkpoint SHA-256 changed")
    return {
        "run_a": {
            "model": "Count-2Slot Condition D base-weighted",
            "checkpoint_path": str(RUN_A_CHECKPOINT),
            "checkpoint_sha256": run_a_sha,
            "best_epoch": RUN_A_BEST_EPOCH,
            "native_frozen_semantics": "PRE/MAIN/POST hard free-running Count parity",
            "modeled_regions": ["PRE", "MAIN", "POST"],
            "validation_implementation": (
                "src.stage2_binary_2slot.frozen_pt_validation::"
                "load_frozen_pt_piece+rollout_frozen_pt+render_cc64_only"
            ),
            "dtype": "torch.float32",
        },
        "run_b": {
            "model": "State-Anchored Transition v1",
            "checkpoint_path": str(RUN_B_CHECKPOINT),
            "checkpoint_sha256": run_b_sha,
            "best_epoch": RUN_B_BEST_EPOCH,
            "native_frozen_semantics": (
                "MAIN-only State/Mode owner assembly + constrained 2-state Viterbi"
            ),
            "modeled_regions": ["MAIN"],
            "viterbi_lambda_state": 1.0,
            "viterbi_lambda_mode": 1.0,
            "validation_implementation": (
                "src.stage2_state_anchored.inference::"
                "collect_frozen_pt_piece_logits+decode_performance; "
                "src.stage2_state_anchored.serialization::render_state_anchored_candidate"
            ),
            "dtype": "torch.float32",
        },
    }


def load_checkpoint(path: Path, expected_epoch: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != expected_epoch or int(payload.get("seed", -1)) != SEED:
        raise RuntimeError(f"checkpoint epoch/seed mismatch: {path}")
    if "model_state" not in payload:
        raise RuntimeError(f"checkpoint has no model_state: {path}")
    for key, tensor in payload["model_state"].items():
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
            raise RuntimeError(f"checkpoint tensor was not mapped to CPU: {key}")
    return payload


def assert_model_cpu(model: torch.nn.Module, label: str) -> None:
    devices = {parameter.device.type for parameter in model.parameters()}
    devices.update(buffer.device.type for buffer in model.buffers())
    if devices != {"cpu"}:
        raise RuntimeError(f"{label} is not CPU-only: {sorted(devices)}")
    if torch.cuda.is_initialized():
        raise RuntimeError(f"CUDA initialized while loading {label}")


def native_identity(
    canonical: Path, candidate: Path, identity: Mapping[str, Any], label: str
) -> tuple[str, int]:
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise RuntimeError(f"{label} did not create a non-empty MIDI: {candidate}")
    if not bool(identity.get("note_identity_exact")) or not bool(
        identity.get("pitch_onset_noteoff_velocity_exact")
    ):
        raise AssertionError(f"{label} note identity failed: {candidate}")
    eot_extension = int(identity.get("eot_extension_ticks", 0))
    if eot_extension == 0:
        strict = assert_strict_non_cc64_equality(canonical, candidate)
        if not strict["all_ordered_non_cc64_events_exact"]:
            raise AssertionError(f"{label} strict non-CC64 identity failed")
        return "PASS_STRICT", 0
    if label != "RUN_A" or not bool(
        identity.get("all_ordered_non_cc64_events_exact_except_eot_tick")
    ):
        raise AssertionError(f"unexpected non-CC64/EOT identity exception for {label}")
    return "PASS_NATIVE_FROZEN_EOT_EXTENSION_ONLY", eot_extension


def build_config(
    cpu: Mapping[str, Any], provenance: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "experiment": OUTPUT_ROOT.name,
        "execution_started_at": now(),
        "device": "cpu",
        "seed": SEED,
        "cpu_only_preflight": dict(cpu),
        "stage1_inference_regeneration": 0,
        "canonical_stage1_manifest": str(CANONICAL_MANIFEST),
        "canonical_stage1_count": len(inventory),
        "canonical_stage1_inventory": [
            {
                "num": row["num"],
                "piece_id": row["piece_id"],
                "path": row["canonical_midi_path"],
                "sha256": row["canonical_midi_sha256"],
                "non_cc64_signature_sha256": row[
                    "canonical_non_cc64_signature_sha256"
                ],
            }
            for row in inventory
        ],
        "run_a": provenance["run_a"],
        "run_b": provenance["run_b"],
        "native_horizon_difference": {
            "run_a": "PRE/MAIN/POST",
            "run_b": "MAIN-only",
            "inference_stage_policy": "preserve each model's native frozen validation semantics",
            "future_headline_policy": "compare on common MAIN horizon; not computed in this run",
        },
        "output_naming": {
            "run_a": "outputs/midi/{num}_run_a_count2slot.mid",
            "run_b": "outputs/midi/{num}_run_b_state_anchored.mid",
        },
        "human_test_reference_access": 0,
        "test_metric_computation": 0,
        "wav_rendering": 0,
        "checkpoint_selection_during_test": 0,
    }


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
        run_a_success_count=0,
        run_a_failure_count=0,
        run_b_success_count=0,
        run_b_failure_count=0,
        canonical_stage1_count=0,
        human_test_reference_access=0,
        test_metric_computation=0,
        stage1_inference_regeneration=0,
    )
    log("RUN_START final-test Run A/B inference-generation only")
    seed_everything()
    cpu = assert_cpu_only()
    log("CPU_ONLY_PASS visible='' cuda_available=0 device_count=0 initialized=0 device=cpu")
    provenance = verify_fixed_artifacts()
    log(
        "CHECKPOINTS_PASS "
        f"run_a_epoch={RUN_A_BEST_EPOCH} run_a={RUN_A_CHECKPOINT} "
        f"run_b_epoch={RUN_B_BEST_EPOCH} run_b={RUN_B_CHECKPOINT}"
    )
    inventory, manifest = canonical_inventory()
    atomic_csv(MANIFEST_PATH, manifest)
    config = build_config(cpu, provenance, inventory)
    atomic_json(CONFIG_PATH, config)
    update_status(
        stage="preflight_complete",
        canonical_stage1_count=len(inventory),
        run_a_checkpoint=str(RUN_A_CHECKPOINT),
        run_b_checkpoint=str(RUN_B_CHECKPOINT),
        run_a_best_epoch=RUN_A_BEST_EPOCH,
        run_b_best_epoch=RUN_B_BEST_EPOCH,
        cpu_only_preflight=cpu,
    )
    log("CANONICAL_INVENTORY_PASS count=23 nums=0..22 original_pt_unchanged=1")

    update_status(stage="run_a_model_loading")
    log("RUN_A_MODEL_LOAD_START device=cpu dtype=torch.float32 semantics=PRE/MAIN/POST")
    run_a_payload = load_checkpoint(RUN_A_CHECKPOINT, RUN_A_BEST_EPOCH)
    run_a_model = run_a_validation.build_model(DEVICE)
    run_a_model.load_state_dict(run_a_payload["model_state"], strict=True)
    del run_a_payload
    gc.collect()
    run_a_model.eval()
    assert_model_cpu(run_a_model, "RUN_A")
    log("RUN_A_MODEL_LOAD_PASS checkpoint_epoch=8 device=cpu cuda_initialized=0")
    update_status(stage="run_a_inference", current_num=inventory[0]["num"])
    log(
        f"INITIAL_READY stage=RUN_A_PIECE_START num={inventory[0]['num']} "
        f"piece_id={inventory[0]['piece_id']} device=cpu"
    )
    for index, item in enumerate(inventory):
        number = int(item["num"])
        update_status(stage="run_a_inference", current_num=number, current_piece_id=item["piece_id"])
        log(f"RUN_A_PIECE_START num={number} piece_id={item['piece_id']}")
        piece = load_frozen_pt_piece(
            {"piece_id": item["piece_id"], "canonical_midi_path": item["canonical_midi_path"]}
        )
        rollout = rollout_frozen_pt(run_a_model, piece, DEVICE, amp=False)
        output = Path(item["run_a_output_path"])
        if output.exists():
            raise FileExistsError(f"refusing to overwrite RUN A output: {output}")
        identity = render_cc64_only(rollout, output)
        result, eot_extension = native_identity(
            Path(item["canonical_midi_path"]), output, identity, "RUN_A"
        )
        if sha256_file(Path(item["canonical_midi_path"])) != item["canonical_midi_sha256"]:
            raise AssertionError(f"RUN A modified canonical Stage 1 num={number}")
        manifest[index]["run_a_output_sha256"] = sha256_file(output)
        manifest[index]["run_a_nonpedal_identity"] = result
        manifest[index]["run_a_eot_extension_ticks"] = eot_extension
        atomic_csv(MANIFEST_PATH, manifest)
        update_status(run_a_success_count=index + 1)
        log(
            f"RUN_A_PIECE_PASS num={number} output={output} identity={result} "
            f"eot_extension_ticks={eot_extension}"
        )
    del run_a_model
    gc.collect()
    update_status(stage="run_a_complete", run_a_success_count=23)
    log("RUN_A_COMPLETE success=23 failure=0")

    seed_everything()
    update_status(stage="run_b_model_loading", current_num=None, current_piece_id=None)
    log("RUN_B_MODEL_LOAD_START device=cpu dtype=torch.float32 semantics=MAIN-only+Viterbi")
    run_b_payload = load_checkpoint(RUN_B_CHECKPOINT, RUN_B_BEST_EPOCH)
    run_b_model = run_b_validation.build_model(DEVICE)
    run_b_model.load_state_dict(run_b_payload["model_state"], strict=True)
    del run_b_payload
    gc.collect()
    run_b_model.eval()
    assert_model_cpu(run_b_model, "RUN_B")
    log("RUN_B_MODEL_LOAD_PASS checkpoint_epoch=9 device=cpu cuda_initialized=0")
    for index, item in enumerate(inventory):
        number = int(item["num"])
        update_status(stage="run_b_inference", current_num=number, current_piece_id=item["piece_id"])
        log(f"RUN_B_PIECE_START num={number} piece_id={item['piece_id']}")
        piece = load_frozen_pt_piece(
            {"piece_id": item["piece_id"], "canonical_midi_path": item["canonical_midi_path"]}
        )
        logits = collect_frozen_pt_piece_logits(run_b_model, piece, DEVICE, amp=False)
        decoded = decode_performance(logits)
        if decoded.viterbi.conflict_count != 0:
            raise AssertionError(f"RUN B legal-path conflict for num={number}")
        output = Path(item["run_b_output_path"])
        if output.exists():
            raise FileExistsError(f"refusing to overwrite RUN B output: {output}")
        rendered = render_state_anchored_candidate(
            piece,
            decoded.viterbi.state_path,
            decoded.viterbi.mode_path,
            logits.timing_predictions,
            output,
        )
        result, eot_extension = native_identity(
            Path(item["canonical_midi_path"]), output, rendered.identity, "RUN_B"
        )
        if sha256_file(Path(item["canonical_midi_path"])) != item["canonical_midi_sha256"]:
            raise AssertionError(f"RUN B modified canonical Stage 1 num={number}")
        manifest[index]["run_b_output_sha256"] = sha256_file(output)
        manifest[index]["run_b_nonpedal_identity"] = result
        manifest[index]["run_b_eot_extension_ticks"] = eot_extension
        atomic_csv(MANIFEST_PATH, manifest)
        update_status(run_b_success_count=index + 1)
        log(f"RUN_B_PIECE_PASS num={number} output={output} identity={result}")
    del run_b_model
    gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU-only inference")
    update_status(
        status="completed",
        stage="completed",
        completed_at=now(),
        current_num=None,
        current_piece_id=None,
        run_a_success_count=23,
        run_a_failure_count=0,
        run_b_success_count=23,
        run_b_failure_count=0,
        nonpedal_identity_pass_count=46,
        nonpedal_identity_failure_count=0,
        cuda_initialized_during_run=False,
    )
    log("RUN_COMPLETE run_a=23/23 run_b=23/23 nonpedal_identity=46/46 metrics=0 human_test=0")


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
