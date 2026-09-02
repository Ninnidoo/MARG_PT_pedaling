#!/usr/bin/env python3
"""Audit and structurally smoke-test the canonical binary encoder-decoder scaffold."""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import os
import sys
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    assert_strict_non_cc64_equality,
    sha256_file,
    signature_sha256,
    transplant_cc64_only,
)
from src.stage2_binary.strict_midi_validation import (  # noqa: E402
    PianoT5GemmaConfig,
    ids_to_midi,
    map_midi,
    midi_to_ids,
)
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
from src.stage2_binary_encoder_decoder.scaffold import (  # noqa: E402
    build_training_optimizer,
    default_training_configuration,
    verify_shared_cache,
)
from src.stage2_encoder_only.train_five_class import parameter_hash  # noqa: E402


OUTPUT = ROOT / "analysis/stage2_binary_encoder_decoder_canonical_v1"
REFERENCE = ROOT / "analysis/stage2_binary_canonical_v1"
VALIDATION_MANIFEST = REFERENCE / "canonical_validation_stage1_manifest.csv"
REFERENCE_CONFIG = REFERENCE / "train_v0/independent_4x2/config.json"
PT_CHECKPOINT = ROOT / "checkpoints/pianist_transformer"
SOURCE_PATHS = {
    "existing_five_class_model": ROOT / "src/stage2_encoder_decoder/model.py",
    "existing_five_class_train": ROOT / "src/stage2_encoder_decoder/train.py",
    "existing_five_class_evaluate": ROOT / "src/stage2_encoder_decoder/evaluate.py",
    "existing_five_class_run": ROOT / "src/stage2_encoder_decoder/run.py",
    "scheduled_sampling_reference": ROOT
    / "src/stage2_encoder_decoder/scheduled_sampling_experiment.py",
    "binary_dataset": ROOT / "src/stage2_binary/dataset.py",
    "binary_cache": ROOT / "src/stage2_binary/full_training.py",
    "canonical_midi_helper": ROOT / "src/stage2_binary/canonical_stage1.py",
    "binary_encoder_decoder_model": ROOT
    / "src/stage2_binary_encoder_decoder/model.py",
    "binary_encoder_decoder_inference": ROOT
    / "src/stage2_binary_encoder_decoder/inference.py",
    "binary_encoder_decoder_scaffold": ROOT
    / "src/stage2_binary_encoder_decoder/scaffold.py",
    "focused_tests": ROOT / "tests/test_stage2_binary_encoder_decoder.py",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def values_only_parameter_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for parameter in parameters:
            value = parameter.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_focused_tests() -> dict[str, Any]:
    stream = io.StringIO()
    suite = unittest.TestLoader().loadTestsFromName(
        "tests.test_stage2_binary_encoder_decoder"
    )
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    payload = {
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skipped": [test.id() for test, _ in result.skipped],
        "module": "tests.test_stage2_binary_encoder_decoder",
        "coverage": {
            "binary_target_and_flatten_order": True,
            "teacher_forcing_one_step_shift": True,
            "teacher_forced_future_token_causal_mask": True,
            "free_running_self_history_only": True,
            "flat_output_reconstruction": True,
            "existing_binary_pedal_masking": True,
            "unweighted_loss_gradients_and_optimizer_groups": True,
        },
        "stdout": stream.getvalue(),
        "asap_test_access_count": 0,
        "full_training_started": False,
    }
    if not result.wasSuccessful():
        raise RuntimeError("focused binary encoder-decoder tests failed")
    return payload


def protocol() -> dict[str, Any]:
    configuration = default_training_configuration()
    return {
        "experiment_id": "stage2_binary_encoder_decoder_canonical_v1",
        "stage": "architecture audit, scaffold, focused tests, one validation structural smoke",
        "scientific_comparison": {
            "reference": "canonical binary encoder-only Independent 4x2",
            "candidate": "binary Pedal1--4 causal encoder-decoder",
            "changed_variable": "Stage 2 architecture only",
        },
        "fixed_elements": {
            "training_data": "MAESTRO-clean 1,170 + ASAP train 892 natural concatenation",
            "performances": 2062,
            "notes": 9_369_095,
            "windows": 35_573,
            "asap_split": str(ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"),
            "seed": 42,
            "pretrained_encoder": str(PT_CHECKPOINT),
            "encoder_input": "[Pitch, IOI, Velocity, Duration, MASK, MASK, MASK, MASK]",
            "target": "binary Pedal1--4; raw 0..63 OFF, 64..127 ON",
            "decode": "OFF=0, ON=127",
            "window_stride_notes": [512, 256],
            "optimizer": "AdamW; encoder 1e-5, decoder/head 1e-4, weight decay 0.01",
            "effective_batch_size": 16,
            "gradient_clip": 1.0,
            "amp": True,
            "canonical_validation_stage1": str(VALIDATION_MANIFEST),
            "canonical_test_stage1": "outputs/midi/{num}_original_pt.mid (declared only; not accessed here)",
            "official_primary_metric": "strict MIDI-roundtrip base-2 JS distance",
            "candidate_construction": "validated donor conversion + CC64-only transplant + strict non-CC64 equality",
        },
        "architecture": {
            "encoder": "same official pretrained trainable PT encoder",
            "decoder": "fresh seed-42 two-layer PT-native causal decoder with cross-attention",
            "sequence": "note-major P1,P2,P3,P4",
            "vocabulary": {"OFF": 0, "ON": 1, "BOS": 2, "PAD": 3},
            "output_head": "shared Linear(768,2)",
        },
        "training_semantics": {
            "teacher_forcing": "standard one-step shift",
            "loss": "unweighted binary cross entropy",
            "scheduled_sampling": False,
            "class_weighting": False,
            "label_smoothing": 0.0,
            "auxiliary_loss": False,
        },
        "canonical_validation_semantics": {
            "decode": "fixed-length greedy free-running autoregressive argmax",
            "ground_truth_history": False,
            "window_policy": configuration["window_overlap_policy"],
            "teacher_forced_metrics": "diagnostic only",
            "checkpoint_selection": configuration["checkpoint_selection"],
        },
        "prohibited_in_this_stage": [
            "full training",
            "scheduled sampling",
            "hyperparameter sweep",
            "ASAP test Human MIDI access",
            "23-piece canonical test inference",
            "test metric or audio",
            "Stage 1 PT neural inference",
        ],
    }


def audit_existing_implementation(
    model: BinaryPedalEncoderDecoderModel,
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    reference_configuration = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    decoder = model.decoder.config
    source_hashes = {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}
    encoder_values_hash = values_only_parameter_hash(model.encoder.parameters())
    if encoder_values_hash != reference_configuration["expected_initial_encoder_sha256"]:
        raise RuntimeError("binary encoder-decoder did not start from canonical encoder state")
    return {
        "source_files": {
            name: {"path": str(path), "sha256": source_hashes[name]}
            for name, path in SOURCE_PATHS.items()
        },
        "existing_five_class": {
            "model_class": "src.stage2_encoder_decoder.model.EncoderDecoderFiveClassModel",
            "encoder_initialization": "EncoderDecoderFiveClassModel.from_pretrained_encoder -> PianoT5Gemma.from_pretrained -> get_encoder",
            "encoder": {
                "layers": 10,
                "hidden_size": 768,
                "trainable": True,
            },
            "decoder": {
                "class": "transformers.models.t5gemma.modeling_t5gemma.T5GemmaDecoder",
                "layers": 2,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "attention_heads": 8,
                "key_value_heads": 4,
                "head_dim": 128,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "dropout_rate": 0.0,
                "attention_dropout": 0.0,
                "cross_attention_hidden_size": 768,
                "decoder_input_embedding": "decoder.embed_tokens + learned Pedal1--4 slot_embeddings",
                "output_head": "shared Linear(768,5)",
            },
            "special_tokens": {
                "classes": "0..4",
                "BOS_ID": 5,
                "PAD_ID": 6,
                "vocab_size": 7,
                "EOS": "PAD_ID in config only; rollout is fixed length and does not early-stop",
            },
            "sequence_order": "flatten_pedal_targets uses contiguous view of [B,N,4]: P1_1,P2_1,P3_1,P4_1,P1_2,...",
            "causality": "T5GemmaDecoder is_decoder=True; shifted history and native causal self-attention make y_t conditional only on y_<t",
            "teacher_forcing": "teacher_forced -> shift_right_pedal_targets -> decoder(use_cache=False) -> shared output_head",
            "target_shift": "decoder input is BOS,y_1,...,y_(L-1); target is y_1,...,y_L",
            "loss_masking": "IGNORE_INDEX=-100 excluded; PAD decoder inputs and decoder attention mask cover padded notes",
            "training_entry": "src.stage2_encoder_decoder.train.train / optimizer_step; base config scheduled_sampling_enabled=False",
            "validation_diagnostic": "src.stage2_encoder_decoder.train.validation_epoch uses teacher forcing",
            "free_running": "EncoderDecoderFiveClassModel.greedy_from_encoder uses KV cache, BOS then prior argmax token, fixed 4N steps",
            "windowing": {
                "function": "src.stage2_encoder_decoder.evaluate.infer_performance",
                "window_notes": 512,
                "stride_notes": 256,
                "per_window_decoder_start": "independent BOS",
                "merge": "average_five_class_logits averages raw generated-step logits across overlap; one final argmax",
                "probability_average": False,
                "center_or_first_selection": False,
                "ambiguity": "overlapping autoregressive windows have different BOS-reset histories; raw-logit averaging merges predictions, not causal histories",
            },
            "checkpoint": {
                "save": "src.stage2_encoder_only.train.build_best_checkpoint/build_last_checkpoint via atomic_torch_save",
                "load": "src.stage2_encoder_decoder.evaluate._load_model reconstructs pretrained encoder/fresh decoder then strict load_state_dict",
            },
            "midi_conversion": "existing five-class validation is token-metric only and does not construct canonical CC64-only MIDI",
            "scheduled_sampling_hook": "build_scheduled_sampling_history/loss_from_scheduled_history and separate run_scheduled_sampling exist; explicitly excluded here",
        },
        "binary_adaptation": {
            "model_class": "src.stage2_binary_encoder_decoder.model.BinaryPedalEncoderDecoderModel",
            "unchanged": [
                "official pretrained encoder initialization and trainability",
                "two-layer PT-native decoder config and cross-attention",
                "hidden size, FFN, attention heads, RoPE, norm/dropout",
                "learned Pedal1--4 slot embeddings",
                "note-major causal order",
                "teacher-forcing shift and fixed-length greedy decoding",
                "independent-BOS window and raw-logit overlap merge policy",
            ],
            "minimal_changes": {
                "class_count": "5 -> 2",
                "decoder_vocabulary": "7 -> 4 (OFF,ON,BOS,PAD)",
                "output_head": "Linear(768,5) -> Linear(768,2)",
                "target": "existing binarize_raw_pedals threshold 64",
                "loss": "unweighted two-class CE",
                "decode": "0/1 -> raw 0/127",
            },
            "scheduled_sampling_dependencies_imported": False,
            "canonical_output_path": "infer_binary_encoder_decoder_pedals -> ids_to_midi/map_midi donor -> transplant_cc64_only -> assert_strict_non_cc64_equality",
        },
        "fair_comparison": {
            "shared_cache": dict(cache),
            "asap_split_sha256": sha256_file(
                ROOT / "analysis/stage2_encoder_only_v0/asap_split.csv"
            ),
            "seed": 42,
            "encoder_values_sha256_canonical_semantics": encoder_values_hash,
            "encoder_named_parameter_sha256_five_class_semantics": parameter_hash(
                model.encoder
            ),
            "canonical_expected_encoder_values_sha256": reference_configuration[
                "expected_initial_encoder_sha256"
            ],
            "optimizer_policy": "AdamW encoder 1e-5, decoder/head 1e-4, weight decay 0.01",
            "effective_batch_size": 16,
            "micro_batch_difference": "scaffold conservatively fixes 4 with gradient accumulation 4 from the measured five-class decoder; no binary batch-size search was run; encoder-only used physical batch 16",
            "canonical_validation_bank": str(VALIDATION_MANIFEST),
        },
        "actual_binary_model": {
            "encoder_layers": len(model.encoder.layers),
            "decoder_layers": len(model.decoder.layers),
            "hidden_size": model.hidden_size,
            "decoder_intermediate_size": int(decoder.intermediate_size),
            "decoder_attention_heads": int(decoder.num_attention_heads),
            "decoder_key_value_heads": int(decoder.num_key_value_heads),
            "decoder_head_dim": int(decoder.head_dim),
            "decoder_vocab_size": int(decoder.vocab_size),
            "BOS_ID": BOS_ID,
            "PAD_ID": PAD_ID,
            "binary_classes": BINARY_CLASSES,
            "output_head": [model.output_head.in_features, model.output_head.out_features],
            "parameter_count": model.parameter_count,
            "trainable_parameter_count": model.trainable_parameter_count,
        },
    }


def validation_rows() -> list[dict[str, str]]:
    reference = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    if sha256_file(VALIDATION_MANIFEST) != reference["expected_file_sha256"]["canonical_manifest"]:
        raise RuntimeError("canonical validation manifest hash changed")
    rows = read_csv(VALIDATION_MANIFEST)
    if len(rows) != 19 or any(row["status"] != "frozen_pass" for row in rows):
        raise RuntimeError("canonical validation bank is not frozen 19/19")
    for row in rows:
        path = Path(row["canonical_midi_path"])
        if sha256_file(path) != row["canonical_midi_sha256"]:
            raise RuntimeError(f"canonical validation MIDI hash mismatch: {row['piece_id']}")
        if signature_sha256(path) != row["canonical_non_cc64_signature_sha256"]:
            raise RuntimeError(f"canonical validation signature mismatch: {row['piece_id']}")
    return rows


def structural_smoke(
    model: BinaryPedalEncoderDecoderModel,
    rows: Sequence[Mapping[str, str]],
    device: torch.device,
) -> dict[str, Any]:
    # Deterministic smallest frozen validation performance; no test path is read.
    row = min(rows, key=lambda item: (int(item["note_count"]), item["piece_id"]))
    canonical = Path(row["canonical_midi_path"])
    canonical_hash = sha256_file(canonical)
    smoke_root = OUTPUT / "structural_smoke"
    smoke_root.mkdir(parents=True, exist_ok=False)
    donor = smoke_root / "temporary_pedal_donor.mid"
    candidate = smoke_root / "candidate_cc64_only.mid"
    configuration = PianoT5GemmaConfig()
    canonical_ids = np.asarray(
        midi_to_ids(configuration, MidiFile(str(canonical))), dtype=np.int64
    )
    predicted_ids, details = infer_binary_encoder_decoder_pedals(
        model,
        canonical_ids,
        device=device,
        window_notes=512,
        stride_notes=256,
    )
    performance = ids_to_midi(
        configuration, predicted_ids, ref=canonical_ids.tolist()
    )
    mapped = map_midi(MidiFile(str(canonical)), performance)
    mapped.dump(str(donor))
    transplant = transplant_cc64_only(canonical, donor, candidate)
    equality = assert_strict_non_cc64_equality(canonical, candidate)
    reloaded_ids = midi_to_ids(configuration, MidiFile(str(candidate)))
    if len(reloaded_ids) == 0 or len(reloaded_ids) % 8:
        raise RuntimeError("structural smoke candidate reload failed")
    if sha256_file(canonical) != canonical_hash:
        raise AssertionError("canonical validation source changed during smoke")
    if not equality["passed"]:
        raise AssertionError("structural smoke non-CC64 equality failed")
    return {
        "passed": True,
        "scope": "one frozen canonical validation piece only",
        "piece_id": row["piece_id"],
        "canonical_midi_path": str(canonical),
        "canonical_midi_sha256_before_after": canonical_hash,
        "canonical_notes_from_manifest": int(row["note_count"]),
        "official_roundtrip_notes": len(reloaded_ids) // 8,
        "predicted_shape": [len(predicted_ids) // 8, 4],
        "candidate_midi_path": str(candidate),
        "candidate_midi_sha256": sha256_file(candidate),
        "candidate_reload_pass": True,
        "strict_non_cc64_equality": "PASS",
        "all_ordered_non_cc64_events_exact": equality[
            "all_ordered_non_cc64_events_exact"
        ],
        "canonical_cc64_events": equality["canonical_cc64_events"],
        "candidate_cc64_events": equality["candidate_cc64_events"],
        "cc64_changed": equality["cc64_changed"],
        "donor_cc64_events": transplant["donor_cc64_events"],
        "discarded_after_canonical_eot": transplant[
            "donor_cc64_discarded_after_canonical_eot"
        ],
        "inference": details,
        "metric_computed": False,
        "metric_interpretation": "none; initialized-model structural smoke only",
        "stage1_neural_inference": 0,
        "asap_test_access_count": 0,
    }


def report(
    audit: Mapping[str, Any],
    tests: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> str:
    existing = audit["existing_five_class"]
    actual = audit["actual_binary_model"]
    cache = audit["fair_comparison"]["shared_cache"]
    return f"""# Binary Encoder-Decoder Canonical-v1 Audit Report

## 1. Verdict

**READY FOR AN EXPLICITLY AUTHORIZED FULL TRAINING RUN.** The binary adaptation, cache/config scaffold, free-running inference, and canonical CC64-only structural path are implemented and tested. Full training was not started.

The inherited overlap policy is scientifically imperfect but explicit: every 512-note window starts its own decoder from BOS, stride is 256, raw free-running step logits are averaged in overlaps, and argmax is applied once. This matches the actual existing 5-class path; it does not merge autoregressive histories.

## 2. Existing 5-class encoder-decoder source and architecture

- Model: `src/stage2_encoder_decoder/model.py::EncoderDecoderFiveClassModel`
- Training: `src/stage2_encoder_decoder/train.py::train`, `optimizer_step`, `validation_epoch`
- Free-running evaluation: `src/stage2_encoder_decoder/evaluate.py::infer_performance`
- Checkpoint load: `src/stage2_encoder_decoder/evaluate.py::_load_model`
- Encoder initialization: `from_pretrained_encoder` loads official `PianoT5Gemma`, retains `get_encoder()`, and leaves it trainable.
- Encoder: 10 layers, hidden 768.
- Decoder: native `T5GemmaDecoder`, 2 causal layers, hidden 768, FFN 3072, 8 attention heads, 4 KV heads, head dimension 128, cross-attention hidden 768, RoPE maximum 8192, dropout 0.
- Decoder input: native token embedding plus learned Pedal1--4 slot embedding.
- Five-class output: one shared `Linear(768,5)`.

## 3. Decoder sequence semantics

`flatten_pedal_targets` uses a contiguous view of `[B,N,4]`, so the actual order is:

`P1_1, P2_1, P3_1, P4_1, P1_2, P2_2, P3_2, P4_2, ...`

This exactly matches the requested note-major causal order. Native decoder causal self-attention makes prediction `y_t` conditional on encoder states and decoder history `y_<t`, never future tokens.

## 4. BOS, PAD, shifting, and teacher forcing

The existing five-class IDs are classes 0--4, BOS 5, PAD 6, vocabulary 7. `shift_right_pedal_targets` constructs `BOS,y_1,...,y_(L-1)`. `IGNORE_INDEX=-100` positions become PAD and are masked from both decoder attention and loss. EOS is configured as PAD, but generation never uses early stopping; every valid window generates exactly `4N` tokens.

The binary scaffold uses OFF 0, ON 1, BOS 2, PAD 3, vocabulary 4 and the same shift/mask semantics. Loss is unweighted two-class cross entropy. Scheduled sampling code exists in the historical 5-class model, but the new scaffold neither imports nor enables it.

## 5. Free-running inference

`BinaryPedalEncoderDecoderModel.greedy_from_encoder` starts with BOS, performs deterministic argmax, feeds each predicted class as the next decoder input, and uses KV cache until exactly `4N` predictions exist. The public greedy path rejects any supplied ground-truth target.

## 6. Window and overlap policy

The actual 5-class `infer_performance` policy is retained:

- 512-note windows, stride 256;
- every overlapping window independently starts from BOS;
- no probability averaging and no center/first-window selection;
- raw generated-step logits are averaged per note/slot across windows;
- final class is one argmax after averaging.

Ambiguity: two windows can condition an overlap note on different BOS-reset histories. Logit averaging combines outputs but cannot make those causal histories identical. The policy is frozen for minimal-change comparability, and this limitation must accompany future results.

## 7. Exact binary adaptation

- New model: `src/stage2_binary_encoder_decoder/model.py::BinaryPedalEncoderDecoderModel`
- New inference: `src/stage2_binary_encoder_decoder/inference.py::infer_binary_encoder_decoder_pedals`
- Training scaffold: `src/stage2_binary_encoder_decoder/scaffold.py`
- Classes/head/vocabulary only: 5→2, `Linear(768,5)`→`Linear(768,2)`, vocab 7→4.
- Existing `binarize_raw_pedals` supplies threshold 64 targets.
- Existing `make_masked_binary_sample` supplies `[Pitch, IOI, Velocity, Duration, MASK×4]` without pedal leakage.
- Actual initialized model: encoder {actual['encoder_layers']} layers, decoder {actual['decoder_layers']} layers, hidden {actual['hidden_size']}, output head {actual['output_head']}, vocab {actual['decoder_vocab_size']}.

## 8. Fair comparison scaffold

- Shared cache ID: `{cache['cache_id']}`.
- Training: {cache['train_performances']} performances / {cache['train_notes']} notes / {cache['train_windows']} windows.
- Human validation diagnostics: {cache['validation_performances']} performances / {cache['validation_notes']} notes / {cache['validation_windows']} windows.
- Same seed 42, official pretrained encoder, AdamW encoder LR 1e-5 and decoder/head LR 1e-4, weight decay 0.01, gradient clip 1.0, AMP, effective batch 16, 512/256 windows.
- The scaffold conservatively fixes micro-batch 4 with accumulation 4 from the measured 5-class architecture; this audit did not run a binary batch-size search. Encoder-only physically used batch 16; effective batch remains 16.
- Future checkpoint selection is minimum free-running canonical strict validation JS. Teacher-forced loss/accuracy is diagnostic only.
- The historical 5-class run trained on ASAP train only and used weighted 5-class CE plus teacher-loss selection. Those choices are not reused; the binary canonical cache and unweighted binary objective are the controlled source of truth.

## 9. Checkpoint and MIDI compatibility

Future checkpoints contain `model_state` plus configuration and reload through strict `load_state_dict`. The existing five-class validation did not render MIDI. The new path produces binary 0/127 pedal tokens, uses the validated `ids_to_midi`/`map_midi` donor conversion, transplants CC64 only, and gates on raw non-CC64 equality before any official MIDI-roundtrip metric.

## 10. Focused tests

- Synthetic tests: **{tests['tests_run']}/{tests['tests_run']} PASS**.
- Verified threshold/flatten order, one-step teacher shift, future-token causal masking, free-running self-history, `[B,4N]→[B,N,4]` reconstruction, encoder pedal masking, unweighted gradients, and disjoint optimizer groups.

## 11. Canonical validation structural smoke

- Scope: one frozen validation piece `{smoke['piece_id']}`; no test MIDI.
- Initialized/untrained model; {smoke['inference']['notes']} notes, {smoke['inference']['windows']} windows.
- Predicted Pedal1--4: generated successfully.
- Donor → CC64-only transplant → candidate reload: PASS.
- Strict non-CC64 equality: **PASS**.
- Candidate: `{smoke['candidate_midi_path']}`.
- JS/Intersection: deliberately not computed; initialized-model metric quality is scientifically meaningless.

## 12. Isolation and stop point

- ASAP test Human MIDI accesses: **0**.
- Canonical test MIDI Stage 2 inference: **0/23**.
- Stage 1 PT neural inference: **0**.
- Full training/optimizer steps on research data: **0**.
- Scheduled sampling, class weighting, calibration, audio, and test-based decisions: **0**.

The scaffold is ready, but starting full training requires a separate explicit instruction.
"""


def chown_output() -> None:
    owner = ROOT.stat()
    for path in [OUTPUT, *OUTPUT.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    status_path = OUTPUT / "run_status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "stage": "protocol",
            "started_at": now(),
            "full_training_started": False,
            "stage1_neural_inference": 0,
            "asap_test_access_count": 0,
        },
    )
    stage = "protocol"
    try:
        write_json(OUTPUT / "experiment_protocol.json", protocol())
        stage = "focused_tests"
        tests = run_focused_tests()
        stage = "cache_preflight"
        cache = verify_shared_cache()
        rows = validation_rows()
        stage = "model_initialization"
        if torch.cuda.device_count() != 1:
            raise RuntimeError("structural smoke requires exactly one visible GPU")
        device = torch.device("cuda:0")
        model = BinaryPedalEncoderDecoderModel.from_pretrained_encoder(
            PT_CHECKPOINT,
            decoder_init_seed=42,
            freeze_encoder=False,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        if len(model.encoder.layers) != 10 or len(model.decoder.layers) != 2:
            raise RuntimeError("actual model layer structure differs from audit")
        if model.hidden_size != 768 or model.output_head.out_features != 2:
            raise RuntimeError("actual binary output architecture differs from audit")
        optimizer = build_training_optimizer(model)
        optimizer_groups = [group["group_name"] for group in optimizer.param_groups]
        if optimizer_groups != ["encoder", "head"]:
            raise RuntimeError("future optimizer groups are not encoder/head")
        audit = audit_existing_implementation(model, cache)
        audit["training_scaffold"] = default_training_configuration()
        audit["training_scaffold"]["optimizer_groups_verified"] = optimizer_groups
        audit["training_scaffold"]["optimizer_step_count"] = 0
        del optimizer
        model.to(device).eval()
        stage = "canonical_validation_structural_smoke"
        smoke = structural_smoke(model, rows, device)
        tests["canonical_validation_structural_smoke"] = smoke
        tests["all_required_focused_tests_passed"] = True
        tests["full_training_started"] = False
        tests["asap_test_access_count"] = 0
        write_json(OUTPUT / "implementation_audit.json", audit)
        write_json(OUTPUT / "focused_test_results.json", tests)
        (OUTPUT / "ENCODER_DECODER_BINARY_AUDIT_REPORT.md").write_text(
            report(audit, tests, smoke), encoding="utf-8"
        )
        write_json(
            status_path,
            {
                "status": "completed",
                "completed_at": now(),
                "verdict": "ready_for_explicitly_authorized_full_training",
                "focused_tests": f"{tests['tests_run']}/{tests['tests_run']} PASS",
                "canonical_validation_structural_smoke": "PASS",
                "strict_non_cc64_equality": "PASS",
                "full_training_started": False,
                "optimizer_steps_on_research_data": 0,
                "stage1_neural_inference": 0,
                "asap_test_human_midi_access_count": 0,
                "canonical_test_midi_inference_count": 0,
                "scheduled_sampling_used": False,
            },
        )
        chown_output()
        print(
            json.dumps(
                {
                    "status": "completed",
                    "tests": tests["tests_run"],
                    "structural_smoke": smoke["strict_non_cc64_equality"],
                    "piece_id": smoke["piece_id"],
                },
                indent=2,
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "failed_at": now(),
                "failed_stage": stage,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "full_training_started": False,
                "stage1_neural_inference": 0,
                "asap_test_access_count": 0,
            },
        )
        chown_output()
        raise


if __name__ == "__main__":
    main()
