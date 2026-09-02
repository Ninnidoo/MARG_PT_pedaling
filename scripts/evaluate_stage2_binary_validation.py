#!/usr/bin/env python3
"""Evaluate a future binary Stage 2 checkpoint on the fixed Stage 1 cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from miditoolkit import MidiFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_binary.validation_evaluator import (  # noqa: E402
    JOINT_PATTERNS,
    infer_cached_binary_pedals,
    joint16_histogram_from_ids,
    load_binary_stage2_checkpoint,
    normalized_joint16,
    official_pt_pedal_similarity,
)
from third_party.PianistTransformer.src import model as _official_model_package  # noqa: E402
from third_party.PianistTransformer.src import utils as _official_utils_package  # noqa: E402
from third_party.PianistTransformer.src.model import pianoformer as _official_pianoformer  # noqa: E402
from third_party.PianistTransformer.src.utils import midi as _official_midi  # noqa: E402

sys.modules.setdefault("src.model", _official_model_package)
sys.modules.setdefault("src.model.pianoformer", _official_pianoformer)
sys.modules.setdefault("src.utils", _official_utils_package)
sys.modules.setdefault("src.utils.midi", _official_midi)

from third_party.PianistTransformer.src.model.generate import map_midi  # noqa: E402
from third_party.PianistTransformer.src.model.pianoformer import (  # noqa: E402
    PianoT5GemmaConfig,
)
from third_party.PianistTransformer.src.utils.midi import (  # noqa: E402
    ids_to_midi,
    midi_to_ids,
)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run(args: argparse.Namespace) -> dict[str, object]:
    cache_manifest = _resolve(args.cache_manifest)
    baseline_metrics = _resolve(args.baseline_metrics)
    output_root = _resolve(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    midi_root = output_root / "candidate_midis"
    midi_root.mkdir()
    device = torch.device(args.device)
    model = load_binary_stage2_checkpoint(
        _resolve(args.checkpoint),
        architecture=args.architecture,
        encoder_checkpoint=_resolve(args.encoder_checkpoint),
        device=device,
    )
    tokenizer_config = PianoT5GemmaConfig.from_pretrained(
        str(_resolve(args.encoder_checkpoint))
    )
    with cache_manifest.open(newline="", encoding="utf-8") as handle:
        cache_rows = list(csv.DictReader(handle))
    if len(cache_rows) != 19:
        raise AssertionError("fixed Stage 1 cache manifest must contain 19 pieces")
    human_histogram = np.asarray(
        json.loads(baseline_metrics.read_text(encoding="utf-8"))["human_histogram"],
        dtype=np.int64,
    )
    candidate_histogram = np.zeros(16, dtype=np.int64)
    preservation = []
    for row in cache_rows:
        cached_ids = np.load(row["generated_ids_path"], allow_pickle=False)
        candidate_ids, inference = infer_cached_binary_pedals(
            model,
            cached_ids,
            architecture=args.architecture,
            device=device,
        )
        if not inference["non_pedal_tokens_preserved"]:
            raise AssertionError("Stage 2 changed a cached non-pedal token")
        score_path = Path(row["source_score_path"])
        score_ids = midi_to_ids(tokenizer_config, MidiFile(str(score_path)))
        performance = ids_to_midi(tokenizer_config, candidate_ids, ref=score_ids)
        mapped = map_midi(MidiFile(str(score_path)), performance)
        midi_path = midi_root / f"{row['piece_id']}.mid"
        mapped.dump(str(midi_path))
        retokenized = midi_to_ids(tokenizer_config, MidiFile(str(midi_path)))
        candidate_histogram += joint16_histogram_from_ids(retokenized)
        preservation.append(
            {
                "piece_id": row["piece_id"],
                "non_pedal_tokens_preserved": True,
                "cached_notes": inference["notes"],
                "windows": inference["windows"],
                "candidate_midi": str(midi_path),
            }
        )
    probability = normalized_joint16(candidate_histogram)
    with (output_root / "candidate_joint16.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("joint_id", "pattern", "count", "probability")
        )
        writer.writeheader()
        for joint_id in range(16):
            writer.writerow(
                {
                    "joint_id": joint_id,
                    "pattern": JOINT_PATTERNS[joint_id],
                    "count": int(candidate_histogram[joint_id]),
                    "probability": float(probability[joint_id]),
                }
            )
    result = {
        "architecture": args.architecture,
        "checkpoint": str(_resolve(args.checkpoint)),
        "stage1_cache_manifest": str(cache_manifest),
        "stage1_piece_count": len(cache_rows),
        "metrics": official_pt_pedal_similarity(
            human_histogram, candidate_histogram
        ),
        "non_pedal_preservation": preservation,
        "all_non_pedal_tokens_preserved": all(
            row["non_pedal_tokens_preserved"] for row in preservation
        ),
        "model_selection": {
            "primary": "js_distance_base2 (lower is better)",
            "secondary": "histogram_intersection (higher is better)",
            "cross_architecture_validation_ce_used": False,
        },
    }
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture",
        required=True,
        choices=("independent_4x2", "joint_16"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--cache-manifest",
        default="analysis/stage2_binary_v0/validation_eval_v0/stage1_cache_manifest.csv",
    )
    parser.add_argument(
        "--baseline-metrics",
        default="analysis/stage2_binary_v0/validation_eval_v0/original_pt_validation_metrics.json",
    )
    parser.add_argument(
        "--encoder-checkpoint", default="checkpoints/pianist_transformer"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
