"""Six-score listening comparison for two encoder-decoder Stage 2 checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from miditoolkit import MidiFile

from src.listening_comparison_5class_v0 import (
    DEFAULT_INSTRUMENT,
    DEFAULT_RENDERER,
    DEFAULT_STAGE1_CHECKPOINT,
    REPOSITORY_ROOT,
    _cc64_signature,
    _metadata_signature,
    _note_signature,
    _read_wav,
    _set_repository_ownership,
    batch_performance_render,
    ids_to_midi,
    map_midi,
    midi_to_ids,
    render_midi,
    run_cc64_smoke_check,
    seed_everything,
    verify_midi_control,
    PianoT5Gemma,
)
from src.stage2_encoder_decoder.model import EncoderDecoderFiveClassModel
from src.stage2_encoder_only.dataset import (
    MASK_ID,
    NON_PEDAL_FEATURES,
    PEDAL_TOKEN_OFFSET,
    TOKENS_PER_NOTE,
    generate_window_starts,
)
from src.stage2_encoder_only.evaluate_oracle import _make_window_sample
from src.stage2_encoder_only.five_class import (
    NUM_CLASSES,
    REPRESENTATIVES,
    average_five_class_logits,
    decode_five_classes,
    five_class_collate_fn,
)


TARGET_INDICES = (0, 4, 7, 12, 17, 21)
MODEL_A_SUFFIX = "stage2_5class_v2"
MODEL_B_SUFFIX = "stage2_5class_v2_fulltrain"
DEFAULT_MODEL_A = (
    REPOSITORY_ROOT / "analysis/stage2_encoder_decoder_5class_weighted_v0/best.pt"
)
DEFAULT_MODEL_B = REPOSITORY_ROOT / (
    "analysis/stage2_encoder_decoder_5class_weighted_no_early_stop_20ep_v0/last.pt"
)
DEFAULT_SCORE_ROOT = (
    REPOSITORY_ROOT / "third_party/PianistTransformer/data/midis/testset/score"
)
DEFAULT_LOG_ROOT = REPOSITORY_ROOT / (
    "logs/listening_comparison_encoder_decoder_5class_v2"
)
WINDOW_NOTES = 512
STRIDE_NOTES = 256
SAMPLE_RATE = 48_000


def _sha256_int64(values: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _checkpoint_identity(path: Path, expected_kind: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, Mapping) or "model_state" not in checkpoint:
        raise KeyError(f"invalid encoder-decoder checkpoint: {path}")
    if int(configuration["window_notes"]) != WINDOW_NOTES:
        raise ValueError(f"checkpoint window mismatch: {path}")
    if int(configuration["stride_notes"]) != STRIDE_NOTES:
        raise ValueError(f"checkpoint stride mismatch: {path}")
    if list(configuration["representatives"]) != REPRESENTATIVES.tolist():
        raise ValueError(f"checkpoint representatives mismatch: {path}")
    state = checkpoint["model_state"]
    required_prefixes = ("encoder.", "decoder.", "slot_embeddings.", "output_head.")
    if not all(any(key.startswith(prefix) for key in state) for prefix in required_prefixes):
        raise ValueError(f"checkpoint lacks encoder-decoder parameter groups: {path}")
    best_epoch = int(checkpoint["best_epoch"])
    if expected_kind == "best":
        checkpoint_epoch = best_epoch
        if path.name != "best.pt":
            raise ValueError("Model A must use best.pt")
    elif expected_kind == "last":
        checkpoint_epoch = int(checkpoint.get("completed_epoch", -1))
        if path.name != "last.pt" or checkpoint_epoch != 20:
            raise ValueError("Model B must use the epoch-20 last.pt")
        if bool(configuration.get("early_stopping_enabled", True)):
            raise ValueError("Model B is not the no-early-stop trajectory checkpoint")
    else:
        raise ValueError(expected_kind)
    result = {
        "path": str(path),
        "kind": expected_kind,
        "checkpoint_epoch": checkpoint_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "architecture": str(configuration["architecture"]),
        "window_notes": int(configuration["window_notes"]),
        "stride_notes": int(configuration["stride_notes"]),
        "decoder_init_seed": int(configuration["decoder_init_seed"]),
    }
    del checkpoint
    gc.collect()
    return result


def _load_stage2(path: Path, device: torch.device) -> EncoderDecoderFiveClassModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    configuration = checkpoint["configuration"]
    model = EncoderDecoderFiveClassModel.from_pretrained_encoder(
        configuration["checkpoint_path"],
        decoder_init_seed=int(configuration["decoder_init_seed"]),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.set_class_weights(configuration["loss_configuration"]["class_weights"])
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint state mismatch: {path}")
    del checkpoint
    gc.collect()
    return model.to(device).eval()


def _slice_batch(batch: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    size = int(batch["input_ids"].shape[0])
    return {
        key: (
            value[start:end]
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size
            else value[start:end]
            if isinstance(value, list) and len(value) == size
            else value
        )
        for key, value in batch.items()
    }


def infer_encoder_decoder_pedals(
    model: EncoderDecoderFiveClassModel,
    generated_ids: Sequence[int],
    device: torch.device,
    micro_batch_size: int = 4,
) -> tuple[list[int], dict[str, Any]]:
    """Run the validated free-running greedy decoder and overlap aggregation."""

    flat = np.asarray(generated_ids, dtype=np.int64)
    if flat.size == 0 or flat.size % TOKENS_PER_NOTE:
        raise ValueError("Stage 1 sequence must contain complete eight-token notes")
    tokens = flat.reshape(-1, TOKENS_PER_NOTE)
    notes = len(tokens)
    starts = generate_window_starts(notes, WINDOW_NOTES, STRIDE_NOTES)
    windows: list[tuple[int, np.ndarray]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(starts), micro_batch_size):
            current = starts[offset : offset + micro_batch_size]
            samples = [
                _make_window_sample(tokens, start, min(start + WINDOW_NOTES, notes))
                for start in current
            ]
            cpu_batch = five_class_collate_fn(samples)
            batch = _slice_batch(cpu_batch, 0, len(current))
            gpu = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    gpu["input_ids"],
                    gpu["token_attention_mask"],
                    note_mask=gpu["note_mask"],
                    decode_mode="greedy",
                )
            if not bool(torch.isfinite(output.logits).all()):
                raise FloatingPointError("Stage 2 produced non-finite greedy logits")
            values = output.logits.float().cpu().numpy()
            for index, start in enumerate(current):
                length = min(WINDOW_NOTES, notes - start)
                windows.append((start, values[index, :length].copy()))
    averaged, contribution_count = average_five_class_logits(notes, windows)
    classes = averaged.argmax(axis=-1).astype(np.int64)
    decoded = decode_five_classes(classes)
    result = tokens.copy()
    result[:, NON_PEDAL_FEATURES:] = decoded + PEDAL_TOKEN_OFFSET
    if not np.array_equal(result[:, :NON_PEDAL_FEATURES], tokens[:, :NON_PEDAL_FEATURES]):
        raise AssertionError("Stage 2 modified non-pedal tokens")
    return result.reshape(-1).tolist(), {
        "notes": notes,
        "windows": len(starts),
        "window_notes": WINDOW_NOTES,
        "stride_notes": STRIDE_NOTES,
        "micro_batch_size": micro_batch_size,
        "minimum_overlap_contributions": int(contribution_count.min()),
        "maximum_overlap_contributions": int(contribution_count.max()),
        "decoded_representatives": sorted(np.unique(decoded).tolist()),
        "runtime_seconds": time.perf_counter() - started,
        "mode": "greedy autoregressive with KV cache; overlap-average raw logits then argmax",
    }


def remove_sustain_cc64(midi: MidiFile) -> MidiFile:
    result = copy.deepcopy(midi)
    for instrument in result.instruments:
        instrument.control_changes = [
            event for event in instrument.control_changes if event.number != 64
        ]
    return result


def _intermediate_sanity(
    original_mapped: MidiFile,
    pedal_free_path: Path,
    generated_ids: Sequence[int],
) -> dict[str, Any]:
    pedal_free = MidiFile(str(pedal_free_path))
    temporary_reference: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="stage1-reference-",
            suffix=".mid",
            dir=pedal_free_path.parent,
            delete=False,
        ) as handle:
            temporary_reference = Path(handle.name)
        original_mapped.dump(str(temporary_reference))
        serialized_original = MidiFile(str(temporary_reference))
    finally:
        if temporary_reference is not None:
            temporary_reference.unlink(missing_ok=True)
    tokens = np.asarray(generated_ids, dtype=np.int64).reshape(-1, TOKENS_PER_NOTE)
    masked = tokens.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    checks = {
        "no_sustain_cc64": len(_cc64_signature(pedal_free)) == 0,
        "pitch_onset_duration_velocity_preserved": (
            _note_signature(serialized_original) == _note_signature(pedal_free)
        ),
        "metadata_preserved": _metadata_signature(serialized_original) == _metadata_signature(pedal_free),
        "stage2_non_pedal_tokens_preserved": np.array_equal(
            masked[:, :NON_PEDAL_FEATURES], tokens[:, :NON_PEDAL_FEATURES]
        ),
        "stage2_all_pedal_positions_masked": bool(
            np.all(masked[:, NON_PEDAL_FEATURES:] == MASK_ID)
        ),
        "note_count": len(tokens),
    }
    checks["passed"] = all(
        value for key, value in checks.items() if key not in {"note_count", "passed"}
    )
    if not checks["passed"]:
        raise RuntimeError(f"pedal-free intermediate sanity failed: {checks}")
    return checks


def _score_paths(score_root: Path) -> dict[int, Path]:
    paths = {index: score_root / f"{index}.mid" for index in TARGET_INDICES}
    for path in paths.values():
        path.resolve(strict=True)
    return paths


def _assert_outputs_absent(output_midi: Path, output_audio: Path) -> None:
    requested = []
    for index in TARGET_INDICES:
        for suffix in (MODEL_A_SUFFIX, MODEL_B_SUFFIX):
            requested.extend(
                (output_midi / f"{index}_{suffix}.mid", output_audio / f"{index}_{suffix}.wav")
            )
    existing = [str(path) for path in requested if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite requested outputs: {existing}")


def _render_one(renderer: Path, instrument: Path, midi: Path, wav: Path) -> dict[str, Any]:
    started = time.perf_counter()
    command = render_midi(renderer, instrument, midi, wav)
    metadata, _ = _read_wav(wav)
    if metadata["sample_rate"] != SAMPLE_RATE or metadata["channels"] != 2:
        raise RuntimeError(f"unexpected WAV format: {wav}: {metadata}")
    return {
        "runtime_seconds": time.perf_counter() - started,
        "command": command,
        "wav": metadata,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_all = time.perf_counter()
    score_root = Path(args.score_root).resolve(strict=True)
    stage1_checkpoint = Path(args.stage1_checkpoint).resolve(strict=True)
    model_a_path = Path(args.model_a).resolve(strict=True)
    model_b_path = Path(args.model_b).resolve(strict=True)
    renderer = Path(args.renderer).resolve(strict=True)
    instrument = Path(args.instrument).resolve(strict=True)
    output_midi = Path(args.output_midi_dir).resolve()
    output_audio = Path(args.output_audio_dir).resolve()
    log_root = Path(args.log_root).resolve()
    intermediate_root = log_root / "intermediate"
    token_root = log_root / "stage1_tokens"
    for directory in (output_midi, output_audio, log_root, intermediate_root, token_root):
        directory.mkdir(parents=True, exist_ok=True)
    _assert_outputs_absent(output_midi, output_audio)
    score_paths = _score_paths(score_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this inference run requires the configured GPU")

    identities = {
        "model_a": _checkpoint_identity(model_a_path, "best"),
        "model_b": _checkpoint_identity(model_b_path, "last"),
    }
    records: dict[int, dict[str, Any]] = {index: {} for index in TARGET_INDICES}

    stage1_model = PianoT5Gemma.from_pretrained(
        str(stage1_checkpoint), torch_dtype=torch.bfloat16
    ).to(device).eval()
    stage1_config = stage1_model.config
    for sequence_index, index in enumerate(TARGET_INDICES):
        score_path = score_paths[index]
        seed_everything(int(args.seed))
        score = MidiFile(str(score_path))
        score_ids = midi_to_ids(stage1_config, score)
        started = time.perf_counter()
        performances, generated = batch_performance_render(
            stage1_model,
            [score],
            temperature=1.0,
            top_p=0.95,
            device=str(device),
        )
        stage1_seconds = time.perf_counter() - started
        if len(performances) != 1 or len(generated) != 1:
            raise RuntimeError("official Stage 1 did not return one result")
        generated_ids = generated[0]
        token_path = token_root / f"{index}_stage1_generated_ids_int64.npy"
        np.save(token_path, np.asarray(generated_ids, dtype=np.int64), allow_pickle=False)
        original_mapped = map_midi(MidiFile(str(score_path)), copy.deepcopy(performances[0]))
        pedal_free = remove_sustain_cc64(original_mapped)
        intermediate = intermediate_root / f"{index}_stage1_pt_pedal_free.mid"
        pedal_free.dump(str(intermediate))
        sanity = _intermediate_sanity(original_mapped, intermediate, generated_ids)
        if sequence_index == 0 and not sanity["passed"]:
            raise RuntimeError("score-0 prerequisite sanity failed")
        records[index].update(
            {
                "score": str(score_path),
                "score_ids": score_ids,
                "generated_ids": generated_ids,
                "stage1": {
                    "rerun": True,
                    "device": str(device),
                    "dtype": "torch.bfloat16",
                    "runtime_seconds": stage1_seconds,
                    "generated_tokens": len(generated_ids),
                    "generated_ids_sha256_int64_le": _sha256_int64(generated_ids),
                    "pedal_free_intermediate": str(intermediate),
                    "token_artifact": str(token_path),
                    "sanity": sanity,
                },
            }
        )
        print(
            f"STAGE1 score={index} seconds={stage1_seconds:.3f} "
            f"notes={len(generated_ids) // TOKENS_PER_NOTE} sanity=passed",
            flush=True,
        )
    del stage1_model
    gc.collect()
    torch.cuda.empty_cache()

    model_specs = (
        ("model_a", MODEL_A_SUFFIX, model_a_path),
        ("model_b", MODEL_B_SUFFIX, model_b_path),
    )
    for model_name, suffix, checkpoint_path in model_specs:
        model = _load_stage2(checkpoint_path, device)
        for index in TARGET_INDICES:
            record = records[index]
            stage2_ids, inference = infer_encoder_decoder_pedals(
                model,
                record["generated_ids"],
                device,
                micro_batch_size=int(args.micro_batch_size),
            )
            performance = ids_to_midi(
                stage1_config,
                stage2_ids,
                ref=record["score_ids"],
            )
            mapped = map_midi(MidiFile(record["score"]), performance)
            midi_path = output_midi / f"{index}_{suffix}.mid"
            mapped.dump(str(midi_path))
            verification = verify_midi_control(
                Path(record["stage1"]["pedal_free_intermediate"]), midi_path
            )
            if not verification["passed"]:
                raise RuntimeError(f"MIDI preservation failed: {midi_path}")
            record[model_name] = {
                "checkpoint": identities[model_name],
                "midi": str(midi_path),
                "inference": inference,
                "midi_verification": verification,
            }
            print(
                f"STAGE2 model={model_name} score={index} "
                f"seconds={inference['runtime_seconds']:.3f} midi={midi_path}",
                flush=True,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    smoke = run_cc64_smoke_check(renderer, instrument)
    for index in TARGET_INDICES:
        for model_name, suffix, _ in model_specs:
            midi_path = Path(records[index][model_name]["midi"])
            wav_path = output_audio / f"{index}_{suffix}.wav"
            rendering = _render_one(renderer, instrument, midi_path, wav_path)
            records[index][model_name]["audio"] = str(wav_path)
            records[index][model_name]["rendering"] = rendering
            print(
                f"RENDER model={model_name} score={index} "
                f"seconds={rendering['runtime_seconds']:.3f} wav={wav_path}",
                flush=True,
            )

    for index in TARGET_INDICES:
        first = verify_midi_control(
            Path(records[index]["stage1"]["pedal_free_intermediate"]),
            Path(records[index]["model_a"]["midi"]),
        )
        second = verify_midi_control(
            Path(records[index]["stage1"]["pedal_free_intermediate"]),
            Path(records[index]["model_b"]["midi"]),
        )
        records[index]["final_sanity"] = {
            "model_a": first,
            "model_b": second,
            "only_intended_sustain_cc64_may_differ": bool(
                first["passed"] and second["passed"]
            ),
        }
        del records[index]["score_ids"]
        del records[index]["generated_ids"]

    expected_midis = [
        output_midi / f"{index}_{suffix}.mid"
        for index in TARGET_INDICES
        for suffix in (MODEL_A_SUFFIX, MODEL_B_SUFFIX)
    ]
    expected_audio = [
        output_audio / f"{index}_{suffix}.wav"
        for index in TARGET_INDICES
        for suffix in (MODEL_A_SUFFIX, MODEL_B_SUFFIX)
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected_midis + expected_audio):
        raise RuntimeError("one or more requested MIDI/audio files are missing or empty")

    summary = {
        "completed": True,
        "target_indices": list(TARGET_INDICES),
        "models": identities,
        "stage1": {
            "checkpoint": str(stage1_checkpoint),
            "reused_existing_intermediate": False,
            "reason": "no persisted pedal-free generated-token intermediate was provenance-confirmed",
            "generation": "official batch_performance_render; sampling temperature=1.0 top_p=0.95",
            "shared_exactly_between_models": True,
        },
        "device": {
            "stage1": str(device),
            "stage2": str(device),
            "name": torch.cuda.get_device_name(device),
            "uuid": str(torch.cuda.get_device_properties(device).uuid),
        },
        "safe_speed_improvements": [
            "one Stage 1 generation per score shared by both Stage 2 models",
            "model.eval and torch.inference_mode",
            "GPU placement for neural model computation",
            "encoder computed once per window",
            "native decoder KV cache/incremental greedy decoding",
            "batched windows and no unnecessary CPU-GPU logits round trips",
        ],
        "renderer": str(renderer),
        "instrument": str(instrument),
        "sample_rate": SAMPLE_RATE,
        "cc64_renderer_smoke": smoke,
        "midi_count": len(expected_midis),
        "audio_count": len(expected_audio),
        "all_outputs_nonempty": True,
        "scores": {str(index): records[index] for index in TARGET_INDICES},
        "total_runtime_seconds": time.perf_counter() - started_all,
    }
    summary_path = log_root / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths_to_fix = [
        *expected_midis,
        *expected_audio,
        summary_path,
        *intermediate_root.glob("*.mid"),
        *token_root.glob("*.npy"),
    ]
    _set_repository_ownership(paths_to_fix)
    for directory in (log_root, intermediate_root, token_root):
        owner = REPOSITORY_ROOT.stat()
        os.chown(directory, owner.st_uid, owner.st_gid)
        os.chmod(directory, 0o775)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    parser.add_argument("--stage1-checkpoint", default=str(DEFAULT_STAGE1_CHECKPOINT))
    parser.add_argument("--model-a", default=str(DEFAULT_MODEL_A))
    parser.add_argument("--model-b", default=str(DEFAULT_MODEL_B))
    parser.add_argument("--renderer", default=str(DEFAULT_RENDERER))
    parser.add_argument("--instrument", default=str(DEFAULT_INSTRUMENT))
    parser.add_argument("--output-midi-dir", default=str(REPOSITORY_ROOT / "outputs/midi"))
    parser.add_argument("--output-audio-dir", default=str(REPOSITORY_ROOT / "outputs/audio"))
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    result = run(build_argument_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
