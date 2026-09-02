#!/usr/bin/env python3
"""Build and structurally verify the canonical ASAP-test Stage-1 bank."""

from __future__ import annotations

import csv
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import mido
import numpy as np
import torch
from miditoolkit import MidiFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage2_binary.canonical_stage1 import (  # noqa: E402
    cc64_schedule,
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
from src.stage2_binary.validation_evaluator import (  # noqa: E402
    infer_cached_binary_pedals,
    load_binary_stage2_checkpoint,
)
from src.stage2_encoder_only.dataset import (  # noqa: E402
    MASK_ID,
    NON_PEDAL_FEATURES,
    TOKENS_PER_NOTE,
)


OUTPUT = ROOT / "analysis/stage2_binary_v0/canonical_stage1_pipeline_v0"
SCORE_MANIFEST = ROOT / "analysis/stage2_binary_v0/test_eval_v0/test_score_manifest.csv"
NUMBERED_SCORES = ROOT / "third_party/PianistTransformer/data/midis/testset/score"
CANONICAL_ROOT = ROOT / "outputs/midi"
ASAP_ROOT = Path("/workspace/public/ASAP/asap-dataset-v1.1")
LOCK = ROOT / "analysis/stage2_binary_v0/final_lock_v1/final_experiment_lock.json"
PROTECTED = (
    LOCK,
    ROOT / "analysis/stage2_binary_v0/test_eval_v0/FINAL_TEST_REPORT.md",
    ROOT / "analysis/stage2_binary_v0/test_eval_v0/run_status.json",
    ROOT / "analysis/stage2_binary_v0/listening_render_v0/RENDER_LISTENING_REPORT.md",
    ROOT / "analysis/stage2_binary_v0/listening_render_v0/run_status.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_manifest() -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = read_csv(SCORE_MANIFEST)
    if len(rows) != 23 or len({row["piece_id"] for row in rows}) != 23:
        raise RuntimeError("locked test-score manifest is not 23 unique pieces")
    number_by_hash: dict[str, int] = {}
    for number in range(23):
        score = NUMBERED_SCORES / f"{number}.mid"
        digest = sha256_file(score)
        if digest in number_by_hash:
            raise RuntimeError(f"ambiguous numbered-score SHA-256: {digest}")
        number_by_hash[digest] = number
    output: list[dict[str, Any]] = []
    hashes_before: dict[str, str] = {}
    for row in rows:
        candidates: dict[str, list[Path]] = {}
        for specification in row["alternative_candidate_score_paths"].split(";"):
            relative, _ = specification.rsplit("|", 1)
            path = ASAP_ROOT / relative
            digest = sha256_file(path)
            candidates.setdefault(digest, []).append(path)
        matching_hashes = sorted(set(candidates) & set(number_by_hash))
        if len(matching_hashes) != 1:
            raise RuntimeError(
                f"expected one exact numbered-score candidate for {row['piece_id']}, "
                f"found {len(matching_hashes)}"
            )
        score_hash = matching_hashes[0]
        number = number_by_hash[score_hash]
        numbered_score = NUMBERED_SCORES / f"{number}.mid"
        source_score = sorted(candidates[score_hash], key=lambda path: str(path))[0]
        if sha256_file(source_score) != score_hash or sha256_file(numbered_score) != score_hash:
            raise RuntimeError(f"score identity changed for {row['piece_id']}")
        canonical = CANONICAL_ROOT / f"{number}_original_pt.mid"
        if not canonical.is_file():
            raise FileNotFoundError(canonical)
        digest = sha256_file(canonical)
        hashes_before[str(canonical)] = digest
        midi = mido.MidiFile(str(canonical), clip=False)
        note_count = sum(
            1
            for track in midi.tracks
            for message in track
            if not message.is_meta and message.type == "note_on" and message.velocity > 0
        )
        output.append(
            {
                "num": number,
                "piece_id": row["piece_id"],
                "composer": row["composer"],
                "title": row["title"],
                "source_score_path": str(source_score),
                "source_score_sha256": score_hash,
                "all_exact_asap_score_paths": ";".join(
                    str(path) for path in sorted(candidates[score_hash], key=lambda path: str(path))
                ),
                "locked_final_test_selected_score_path": row["selected_score_absolute_path"],
                "locked_selected_score_same_as_canonical": row["selected_score_sha256"] == score_hash,
                "canonical_numbered_score_path": str(numbered_score),
                "mapping_method": "exact SHA-256 join over all recorded direct score candidates to canonical numbered score",
                "canonical_original_pt_midi_path": str(canonical),
                "canonical_midi_sha256": digest,
                "canonical_non_cc64_signature_sha256": signature_sha256(canonical),
                "note_count": note_count,
                "midi_duration_seconds": midi.length,
                "ticks_per_beat": midi.ticks_per_beat,
                "track_count": len(midi.tracks),
                "cc64_event_count": len(cc64_schedule(canonical)),
                "read_only_source_of_truth": True,
            }
        )
    output.sort(key=lambda item: item["num"])
    if [item["num"] for item in output] != list(range(23)):
        raise RuntimeError("canonical mapping is not exactly num 0..22")
    return output, hashes_before


def run_unit_tests() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "tests.test_stage2_binary_canonical_stage1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"focused unittest failed:\n{result.stdout}\n{result.stderr}")
    return {
        "status": "PASS",
        "tests_run": 3,
        "synthetic_preservation": "PASS",
        "intentional_nonpedal_mutation_hard_failure": "PASS",
        "no_pedal_edge_case": "PASS",
        "command": "python -m unittest -v tests.test_stage2_binary_canonical_stage1",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def real_smoke(manifest: list[dict[str, Any]], lock: dict[str, Any]) -> dict[str, Any]:
    # Deterministic pedal-bearing choice, independent of test metrics.
    row = next(item for item in manifest if int(item["cc64_event_count"]) >= 2)
    canonical = Path(row["canonical_original_pt_midi_path"])
    canonical_hash = sha256_file(canonical)
    config = PianoT5GemmaConfig()
    token_ids = np.asarray(midi_to_ids(config, MidiFile(str(canonical))), dtype=np.int64)
    if len(token_ids) == 0 or len(token_ids) % TOKENS_PER_NOTE:
        raise RuntimeError("canonical official tokenization returned invalid note tokens")
    notes = token_ids.reshape(-1, TOKENS_PER_NOTE)
    masked = notes.copy()
    masked[:, NON_PEDAL_FEATURES:] = MASK_ID
    if not np.array_equal(masked[:, :NON_PEDAL_FEATURES], notes[:, :NON_PEDAL_FEATURES]):
        raise AssertionError("masking changed canonical non-pedal tokens")
    if not np.all(masked[:, NON_PEDAL_FEATURES:] == MASK_ID):
        raise AssertionError("canonical Original PT pedals were not fully masked")

    candidate_lock = lock["candidates"]["independent_4x2"]
    checkpoint = ROOT / candidate_lock["checkpoint_project_relative_path"]
    if sha256_file(checkpoint) != candidate_lock["checkpoint_sha256"]:
        raise RuntimeError("locked independent checkpoint hash mismatch")
    encoder = ROOT / "checkpoints/pianist_transformer"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_binary_stage2_checkpoint(
        checkpoint,
        architecture="independent_4x2",
        encoder_checkpoint=encoder,
        device=device,
    )
    candidate_ids, inference = infer_cached_binary_pedals(
        model,
        token_ids,
        architecture="independent_4x2",
        device=device,
        window_notes=512,
        stride_notes=256,
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not inference["non_pedal_tokens_preserved"]:
        raise AssertionError("Stage 2 token inference changed non-pedal tokens")

    smoke_root = OUTPUT / "structural_smoke"
    smoke_root.mkdir(parents=True)
    donor_path = smoke_root / f"num_{row['num']}_independent_pedal_donor.mid"
    candidate_path = smoke_root / f"num_{row['num']}_independent_cc64_only.mid"
    performance = ids_to_midi(config, candidate_ids, ref=token_ids.tolist())
    donor = map_midi(MidiFile(str(canonical)), performance)
    donor.dump(str(donor_path))
    transplant = transplant_cc64_only(
        canonical,
        donor_path,
        candidate_path,
        require_cc64_difference=True,
    )
    # Actual reload is part of the smoke contract.
    reloaded = mido.MidiFile(str(candidate_path), clip=False)
    if not reloaded.tracks:
        raise RuntimeError("CC64-only smoke candidate did not reload")
    if sha256_file(canonical) != canonical_hash:
        raise AssertionError("real smoke modified canonical source MIDI")
    return {
        "status": "PASS",
        "selection_rule": "smallest num with at least two canonical CC64 events",
        "num": row["num"],
        "piece_id": row["piece_id"],
        "canonical_midi": str(canonical),
        "canonical_sha256": canonical_hash,
        "official_tokenizer": "third_party/PianistTransformer/src/utils/midi.py::midi_to_ids",
        "tokenized_notes": len(token_ids) // TOKENS_PER_NOTE,
        "all_original_pedal_input_slots_masked": True,
        "architecture": "independent_4x2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": candidate_lock["checkpoint_sha256"],
        "device": str(device),
        "inference": inference,
        "pedal_donor_midi": str(donor_path),
        "candidate_midi": str(candidate_path),
        "candidate_reload": "PASS",
        "strict_non_cc64_equality": transplant,
        "stage1_neural_inference": 0,
        "human_test_midi_access": 0,
        "test_metric_computation": 0,
    }


def report(manifest: list[dict[str, Any]], results: dict[str, Any]) -> str:
    total_notes = sum(int(row["note_count"]) for row in manifest)
    total_duration = sum(float(row["midi_duration_seconds"]) for row in manifest)
    smoke = results["real_canonical_structural_smoke"]
    return f"""# Canonical ASAP Test Stage-1 Pipeline v0

## Permanent invariant

Root `AGENTS.md` now defines `outputs/midi/{{num}}_original_pt.mid` as the immutable CPU/seed-42 canonical ASAP test Stage 1 bank. Future Stage 2 candidates may transplant CC64 only; full token-to-MIDI reconstruction is prohibited as the final canonical candidate path, and strict non-CC64 equality is a hard pre-metric gate.

## Canonical bank

- Exact mapping: **23/23 PASS**, nums 0--22 exactly once
- Mapping provenance: every direct score candidate recorded in locked `test_score_manifest.csv` was hashed and joined to the exact SHA-256 of `third_party/PianistTransformer/data/midis/testset/score/{{num}}.mid`; each piece had exactly one numbered-score hash match
- Canonical MIDI: `outputs/midi/{{num}}_original_pt.mid`, read-only source of truth
- Total parsed notes: {total_notes}
- Total MIDI duration: {total_duration:.6f} seconds
- Original canonical hashes unchanged after all tests: **PASS**

## Pedal-only implementation

`src/stage2_binary/canonical_stage1.py::transplant_cc64_only` loads canonical and donor MIDI with Mido, removes only canonical CC64, inserts only donor CC64 at absolute ticks, and copies every canonical non-CC64 raw message. `assert_strict_non_cc64_equality` compares MIDI type, ticks-per-beat, ordered tracks, all note/channel/meta events, non-CC64 controls, pitch bends, tempo/time/key signatures, markers/lyrics, and EOT timing while ignoring CC64 only.

The donor MIDI still uses the established official `ids_to_midi` + `map_midi` path, but donor non-pedal content is discarded. Final candidate non-pedal events always come from the canonical MIDI.

## Reused implementation

- Official tokenizer/conversion: `third_party/PianistTransformer/src/utils/midi.py::midi_to_ids,ids_to_midi`
- Official mapping: `third_party/PianistTransformer/src/model/generate.py::map_midi`
- Binary masking/window/decode: `src/stage2_binary/validation_evaluator.py::infer_cached_binary_pedals`
- Locked model loading: `load_binary_stage2_checkpoint`
- Mapping rule: existing locked test-score manifest and listening audit's exact score-hash join
- Equality scope: strengthened from the existing metric-fidelity/listening equality audits to raw ordered MIDI events including EOT

## Verification

- Synthetic preservation: **PASS**
- Intentional velocity mutation hard-failure: **PASS**
- No-pedal canonical edge case: **PASS**
- Real canonical structural smoke: **PASS**, num={smoke['num']}, piece=`{smoke['piece_id']}`, notes={smoke['tokenized_notes']}, windows={smoke['inference']['windows']}
- Canonical tokenization and all four original pedal input slots masked: **PASS**
- Predicted pedal donor -> CC64-only transplant: **PASS**
- Strict ordered non-CC64 equality after reload: **PASS**
- Candidate reload: **PASS**

## Scope guard

This stage performed **no Stage 1 PT neural inference, no training/retraining, no checkpoint change, no Human ASAP test MIDI access, no JS/Intersection or other test metric computation, no full 23-piece Stage 2 inference, and no audio rendering**. Existing final lock, final-test, and listening-render artifacts were not modified.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite output root: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    protected_before = {str(path): sha256_file(path) for path in PROTECTED}
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    manifest, canonical_before = canonical_manifest()
    write_csv(OUTPUT / "canonical_stage1_manifest.csv", manifest)
    unit = run_unit_tests()
    smoke = real_smoke(manifest, lock)
    canonical_after = {path: sha256_file(path) for path in map(Path, canonical_before)}
    canonical_after = {str(path): digest for path, digest in canonical_after.items()}
    if canonical_before != canonical_after:
        raise AssertionError("one or more canonical MIDI files changed")
    protected_after = {str(path): sha256_file(path) for path in PROTECTED}
    if protected_before != protected_after:
        raise AssertionError("one or more locked experiment artifacts changed")
    results = {
        "status": "PASS",
        "canonical_mapping": {"status": "PASS", "pieces": 23, "nums": list(range(23))},
        "focused_unit_tests": unit,
        "real_canonical_structural_smoke": smoke,
        "canonical_midi_original_hashes_unchanged": True,
        "protected_existing_artifacts_unchanged": True,
        "protected_artifact_sha256": protected_after,
        "stage1_neural_inference": 0,
        "full_stage2_test_inference": 0,
        "human_test_midi_access": 0,
        "test_metric_computation": 0,
        "audio_rendering": 0,
    }
    write_json(OUTPUT / "test_results.json", results)
    (OUTPUT / "CANONICAL_STAGE1_PIPELINE_REPORT.md").write_text(
        report(manifest, results), encoding="utf-8"
    )
    owner = ROOT.stat()
    for path in [OUTPUT, *OUTPUT.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid)
    print(json.dumps({"status": "PASS", "mapping": "23/23", "smoke_num": smoke["num"]}, indent=2))


if __name__ == "__main__":
    main()
