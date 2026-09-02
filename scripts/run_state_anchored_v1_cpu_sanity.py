#!/usr/bin/env python3
"""CPU-only ownership and pretrained-forward sanity for State-Anchored v1."""

from __future__ import annotations

import ast
import gc
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_state_anchored.config import (  # noqa: E402
    MODE_CLASS_WEIGHTS,
    StateAnchoredModelConfig,
)
from src.stage2_state_anchored.dataset import (  # noqa: E402
    StateAnchoredWindowDataset,
    diagnose_ownership,
    state_anchored_collate_fn,
)
from src.stage2_state_anchored.decoding import (  # noqa: E402
    decode_timings,
    raw_head_decode,
    raw_head_diagnostics,
)
from src.stage2_state_anchored.losses import StateAnchoredCriterion  # noqa: E402
from src.stage2_state_anchored.model import StateAnchoredTransitionModelV1  # noqa: E402
from src.stage2_state_anchored.viterbi import constrained_viterbi_decode  # noqa: E402


CACHE_ROOT = ROOT / "analysis/custom_event_tokenizer_v1"
OUTPUT_ROOT = ROOT / "analysis/state_anchored_transition_v1"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
TEST_FILES = (
    "tests/test_state_anchored_transition_v1.py",
    "tests/test_state_anchored_dataset_v1.py",
    "tests/test_state_anchored_model_v1.py",
    "tests/test_state_anchored_viterbi_v1.py",
)


def run_focused_tests() -> dict[str, object]:
    results = {}
    for path in TEST_FILES:
        completed = subprocess.run(
            [sys.executable, path],
            cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"focused test failed: {path}\n{completed.stdout}\n{completed.stderr}")
        cases = ast.literal_eval(completed.stdout.strip())
        if not isinstance(cases, dict) or any(value != "passed" for value in cases.values()):
            raise RuntimeError(f"invalid focused result: {path}")
        results[path] = cases
    return {
        "status": "PASS",
        "commands": [f"{sys.executable} {path}" for path in TEST_FILES],
        "files": results,
        "case_count": sum(len(value) for value in results.values()),
    }


def ownership_sanity() -> dict[str, object]:
    result = {}
    expected = {
        "train": (2062, 9_009_549, 9_007_487),
        "validation": (71, 272_927, 272_856),
    }
    for split in ("train", "validation"):
        dataset = StateAnchoredWindowDataset(CACHE_ROOT, split)
        diagnostic = diagnose_ownership(dataset)
        observed = (
            diagnostic.performances,
            diagnostic.state_targets,
            diagnostic.mode_interval_targets,
        )
        if observed != expected[split]:
            raise AssertionError(f"{split} ownership population differs from tokenizer audit")
        result[split] = asdict(diagnostic) | {"status": "PASS"}
        del dataset
        gc.collect()
    return result


def canonical_forward_sanity() -> dict[str, object]:
    dataset = StateAnchoredWindowDataset(
        CACHE_ROOT,
        "train",
        cache_input_tokens=True,
        entry_limit=1,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    eligible = [sample for sample in samples if len(sample["state_targets"]) >= 2]
    if not eligible:
        raise RuntimeError("canonical sample has no window with two owned onsets")
    sample = min(eligible, key=lambda item: int(item["input_ids"].numel()))
    owned_global = sample["owned_global_onset_indices"]
    if len(owned_global) > 1 and not bool(torch.all(owned_global[1:] == owned_global[:-1] + 1)):
        raise AssertionError("owner onsets in canonical sanity window are not contiguous")
    batch = state_anchored_collate_fn((sample,))
    model = StateAnchoredTransitionModelV1.from_pretrained(
        PRETRAINED,
        head_init_seed=42,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).cpu().eval()
    criterion = StateAnchoredCriterion().cpu()
    with torch.no_grad():
        output = model(
            batch["input_ids"],
            batch["token_attention_mask"],
            batch["note_mask"],
            batch["owned_representative_positions"],
            batch["owned_onset_mask"],
        )
        losses = criterion(output, batch)
    values = (
        output.state_logits,
        output.mode_logits,
        output.timing_predictions,
        losses.state_loss,
        losses.mode_loss,
        losses.timing_loss,
        losses.total_loss,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("canonical CPU output/loss is non-finite")

    onset_count = int(batch["state_valid_mask"][0].sum().item())
    interval_count = onset_count - 1
    state_logits = output.state_logits[0, :onset_count]
    mode_logits = output.mode_logits[0, :interval_count]
    timing = output.timing_predictions[0, :interval_count]
    raw = raw_head_decode(state_logits, mode_logits, timing)
    raw_metrics = raw_head_diagnostics(
        state_logits,
        mode_logits,
        batch["state_targets"][0, :onset_count],
        batch["mode_targets"][0, :interval_count],
    )
    decoded = constrained_viterbi_decode(state_logits, mode_logits)
    decoded_timing = decode_timings(timing, decoded.mode_path)
    if decoded.conflict_count != 0 or len(decoded_timing) != interval_count:
        raise AssertionError("canonical constrained decode sanity failed")
    result = {
        "status": "PASS",
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "split": "train",
        "canonical_performance_index": int(sample["metadata"]["canonical_performance_index"]),
        "performance_id": sample["metadata"]["performance_id"],
        "window_index": int(sample["metadata"]["window_index"]),
        "window_notes": int(sample["input_ids"].numel() // 8),
        "owned_onsets": onset_count,
        "locally_decoded_intervals": interval_count,
        "output_shapes": {
            "state_logits": list(output.state_logits.shape),
            "mode_logits": list(output.mode_logits.shape),
            "timing_predictions": list(output.timing_predictions.shape),
        },
        "losses": {
            "state": float(losses.state_loss.item()),
            "mode": float(losses.mode_loss.item()),
            "timing": float(losses.timing_loss.item()),
            "total": float(losses.total_loss.item()),
        },
        "raw_decode": {
            "conflict_count": raw.conflict_count,
            "conflict_rate": raw.conflict_rate,
            "metrics": raw_metrics,
        },
        "constrained_decode": {
            "conflict_count": decoded.conflict_count,
            "score": float(decoded.score.item()),
            "state_count": len(decoded.state_path),
            "mode_count": len(decoded.mode_path),
        },
        "official_encoder_backward": "not run; finite custom-head backward covered by focused fake-encoder test",
        "parameter_counts": {
            "encoder": model.encoder_parameter_count,
            "heads": model.head_parameter_counts,
            "total": model.parameter_count,
            "trainable": model.trainable_parameter_count,
        },
        "encoder_hidden_size": model.hidden_size,
        "encoder_layers": int(model.encoder.config.num_hidden_layers),
    }
    del model, criterion, output, dataset, samples, eligible, batch
    gc.collect()
    return result


def report_markdown(payload: dict[str, object]) -> str:
    ownership = payload["ownership"]
    sanity = payload["canonical_cpu_sanity"]
    config = payload["configuration"]
    tests = payload["focused_tests"]
    return f"""# State-Anchored Transition v1 Model Implementation

**Verdict: {payload['verdict']}**

## Reused infrastructure

- Frozen State-Anchored MAIN-only target builder and audit representation (unchanged).
- Canonical Custom Event cache manifest and train/validation split only.
- Binary 2-Slot raw MIDI/timeline adapter, official PT Pitch/IOI/Velocity/Duration construction, and human-pedal masking.
- Existing 512-note/stride-256 maximum-margin onset ownership and canonical last-note representative mapping.
- Official `PianoT5Gemma.from_pretrained(...).get_encoder()` loading path and note-level encoder output.

## Dataset and ownership

State is attached to every uniquely owned onset. Mode/Timing are attached to the same owner for onset index `i < M-1`; the final onset is State-only, so no interval crosses a performance boundary.

| Split | Performances | State targets | Mode intervals | State zero/duplicate | Mode zero/duplicate | Boundary leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | {ownership['train']['performances']:,} | {ownership['train']['state_targets']:,} | {ownership['train']['mode_interval_targets']:,} | {ownership['train']['state_zero_owner']}/{ownership['train']['state_duplicate_owner']} | {ownership['train']['mode_zero_owner']}/{ownership['train']['mode_duplicate_owner']} | {ownership['train']['performance_boundary_leakage']} |
| Validation | {ownership['validation']['performances']:,} | {ownership['validation']['state_targets']:,} | {ownership['validation']['mode_interval_targets']:,} | {ownership['validation']['state_zero_owner']}/{ownership['validation']['state_duplicate_owner']} | {ownership['validation']['mode_zero_owner']}/{ownership['validation']['mode_duplicate_owner']} | {ownership['validation']['performance_boundary_leakage']} |

## Architecture and parameter counts

The pretrained PT encoder emits `h_i` with hidden size {sanity['encoder_hidden_size']}. Four direct linear heads predict State (2), Mode (3), tau1 (1), and tau2 (1). Predicted or human State is never fed into another onset's network input.

- Encoder parameters: {sanity['parameter_counts']['encoder']:,}
- State head: {sanity['parameter_counts']['heads']['state']:,}
- Mode head: {sanity['parameter_counts']['heads']['mode']:,}
- Timing heads: {sanity['parameter_counts']['heads']['timing_1']:,} + {sanity['parameter_counts']['heads']['timing_2']:,}
- All custom heads: {sanity['parameter_counts']['heads']['total']:,}
- Total parameters: {sanity['parameter_counts']['total']:,}

## Loss and exact Mode weights

`L = State CE + weighted Mode CE + masked Smooth-L1(beta=0.1)`. State CE is unweighted. Timing is averaged over valid scalar targets only.

- HOLD: {config['mode_class_weights'][0]:.15f}
- CHANGE: {config['mode_class_weights'][1]:.15f}
- RETURN: {config['mode_class_weights'][2]:.15f}
- Normalization: `sum_c train_frequency_c * weight_c = 1`; validation is not used.

## Raw heads and constrained Viterbi

Raw helpers provide State/Mode argmax, accuracy, macro F1, class precision/recall/F1, and both illegal conflict types. Viterbi maximizes State plus Mode log-probability with fixed lambdas 1/1. Different-state edges force CHANGE; same-state edges select the higher of HOLD/RETURN and preserve that Mode in backtracking.

Timing predictions are clamped to `[0,1]`. CHANGE uses head 1 only; RETURN retains both head identities as a pair and sorts their values chronologically.

## Focused tests

- Result: {tests['status']} ({tests['case_count']} cases across {len(tests['files'])} focused files).
- Covered tensor shapes, padding masks, HOLD/CHANGE/RETURN timing gradients, finite State/weighted-Mode/total losses, finite head gradients, all requested Viterbi legal transitions, raw conflict recovery, global optimum, determinism, decoded conflict zero, ownership exactly-once, and boundary isolation.

## Canonical CPU sanity

- Result: {sanity['status']}
- Device: CPU; `torch.cuda.is_available()={sanity['torch_cuda_available']}` with CUDA hidden.
- Sample: train performance index {sanity['canonical_performance_index']}, window {sanity['window_index']}, {sanity['window_notes']} notes, {sanity['owned_onsets']} owned onsets.
- Losses: State={sanity['losses']['state']:.6f}, Mode={sanity['losses']['mode']:.6f}, Timing={sanity['losses']['timing']:.6f}, Total={sanity['losses']['total']:.6f}; all finite.
- Raw conflict: {sanity['raw_decode']['conflict_count']}; constrained decoded conflict: {sanity['constrained_decode']['conflict_count']}.
- Official encoder backward was not run on CPU; custom-head finite backward is covered by focused tests.

## Safety and split gates

- PRE/POST/final-note-off tail: disabled.
- ASAP test access: 0.
- GPU use: 0.
- Tiny overfit/full training: not run.
- RUN A code/process/tmux/output: not modified by implementation or sanity commands; external liveness is checked separately.

## PASS / FAIL

{payload['verdict']}: dataset integration, model, loss, raw decoding, constrained Viterbi, focused tests, full ownership diagnostics, and canonical CPU forward/loss/decode sanity all passed.
"""


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("CPU sanity requires CUDA_VISIBLE_DEVICES empty or -1")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be unavailable to this process")
    torch.manual_seed(42)
    configuration = StateAnchoredModelConfig()
    tests = run_focused_tests()
    ownership = ownership_sanity()
    sanity = canonical_forward_sanity()
    verdict = "PASS" if tests["status"] == sanity["status"] == "PASS" else "FAIL"
    payload = {
        "schema_version": 1,
        "experiment": "state_anchored_transition_v1_model_implementation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "configuration": configuration.to_dict(),
        "mode_class_weights": list(MODE_CLASS_WEIGHTS),
        "focused_tests": tests,
        "ownership": ownership,
        "canonical_cpu_sanity": sanity,
        "asap_test_access": 0,
        "gpu_use": 0,
        "tiny_overfit": 0,
        "full_training": 0,
    }
    if any(not values["status"] == "PASS" for values in ownership.values()):
        verdict = payload["verdict"] = "FAIL"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configuration.save_json(OUTPUT_ROOT / "model_config.json")
    json_path = OUTPUT_ROOT / "model_cpu_sanity.json"
    report_path = OUTPUT_ROOT / "MODEL_IMPLEMENTATION_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(report_markdown(payload), encoding="utf-8")
    os.chmod(json_path, 0o644)
    os.chmod(report_path, 0o644)
    os.chmod(OUTPUT_ROOT / "model_config.json", 0o644)
    print(f"{verdict}: {json_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
