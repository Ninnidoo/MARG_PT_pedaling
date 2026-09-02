#!/usr/bin/env python3
"""Matched-human-input validation inference for the frozen Hybrid finalist."""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import mido
import numpy as np
import torch
from miditoolkit import MidiFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_original_pt_early_metrics import PinnedTokenizerConfig
from scripts.run_custom_event_model_v0_canonical_val_inference_v1 import load_frozen_human
from scripts.run_stage2_4class_validation_eval_v0 import (
    SPLIT_CSV, STAGE1_MANIFEST, align_and_cache, finalize_transition,
    sum_transition, top_patterns,
)
from scripts.run_stage2_loss_decoding_fusion_phase4b_v0 import (
    infer_hybrid_final_outputs, regression_cc64_integer,
)
from src.stage2_binary.canonical_stage1 import (
    assert_strict_non_cc64_equality, sha256_file, signature_sha256,
    transplant_cc64_only,
)
from src.stage2_binary.validation_evaluator import replace_pedal_tokens
from src.stage2_four_class.loss_objectives import HUBER_DELTA_NORMALIZED
from src.stage2_four_class.raw_huber_aux_ce import LAMBDA_CE, RawHuberAuxCEEncoderModel
from src.stage2_four_class.validation_evaluator import (
    PATTERN_COUNT, canonical_classes_from_tokens, classification_metrics,
    confusion_from_pairs, pattern_ids, pattern_metrics, pooled_transition_counts,
)
from third_party.PianistTransformer.src import model as _official_model_package
from third_party.PianistTransformer.src import utils as _official_utils_package
from third_party.PianistTransformer.src.model import pianoformer as _official_pianoformer
from third_party.PianistTransformer.src.utils import midi as _official_midi

sys.modules.setdefault("src.model", _official_model_package)
sys.modules.setdefault("src.model.pianoformer", _official_pianoformer)
sys.modules.setdefault("src.utils", _official_utils_package)
sys.modules.setdefault("src.utils.midi", _official_midi)

from third_party.PianistTransformer.src.model.generate import map_midi
from third_party.PianistTransformer.src.utils.midi import ids_to_midi, midi_to_ids


OUTPUT = ROOT / "analysis/custom_vs_hybrid_matched_input_val_v1"
CUSTOM_ROOT = ROOT / "analysis/custom_event_model_v0_canonical_val_inference_v1"
HYBRID_ROOT = ROOT / "analysis/stage2_encoder_only_raw_huber_aux_ce_v1"
CHECKPOINT = HYBRID_ROOT / "best.pt"
PRETRAINED = ROOT / "checkpoints/pianist_transformer"
EXPECTED_CHECKPOINT_SHA = "2a4550de0c6737ac0c9d688d0615d9e57a8a0c877eebc1dc670bec817aaa7c5e"
TOKENIZER_CACHE_ID = "3a5520155b5db1e9a1da7f8148556aa3e1da852655c9adde25d3dbbd1d966263"
ALIGNMENT_ID = "98de59ef7a41fba26a2c89fe686c273f6c8ec7f27979922e71813624ac1a4822"
DECODER_ID = "938ca7e20e1d398aa530938bbcede404fd3a04af23842fbcf351ba02df0268f8"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_checkpoint() -> dict[str, Any]:
    from numpy.core.multiarray import _reconstruct
    safe = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe):
        value = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if int(value["epoch"]) != 5 or sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen Hybrid checkpoint identity mismatch")
    return value


def load_model(device: torch.device) -> tuple[RawHuberAuxCEEncoderModel, dict[str, Any]]:
    payload = safe_checkpoint()
    model = RawHuberAuxCEEncoderModel.from_pretrained(
        PRETRAINED, delta=HUBER_DELTA_NORMALIZED, lambda_ce=LAMBDA_CE,
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    model.load_state_dict(payload["model_state"], strict=True)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("Hybrid checkpoint contains non-finite parameters")
    metadata = {
        "path": str(CHECKPOINT), "sha256": EXPECTED_CHECKPOINT_SHA,
        "epoch": 5, "validation_objective": float(payload["validation_objective"]),
        "validation_huber": float(payload["validation_huber"]),
        "validation_ce": float(payload["validation_ce"]),
    }
    return model.to(device).eval(), metadata


def audit_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    custom = json.loads((CUSTOM_ROOT / "candidate_manifest.json").read_text())["candidates"]
    cache_manifest = json.loads((ROOT / "analysis/custom_event_tokenizer_v1/cache_manifest.json").read_text())
    cache_entries = {row["performance_path"]: row for row in cache_manifest["entries"] if row["split"] == "validation"}
    if len(custom) != 71 or len(cache_entries) != 71 or cache_manifest["cache_id"] != TOKENIZER_CACHE_ID:
        raise RuntimeError("Custom matched-input universe mismatch")
    rows = []
    for item in custom:
        path = Path(item["source_midi"])
        entry = cache_entries[item["performance_path"]]
        source_sha = sha256_file(path)
        if source_sha != item["source_sha256"] or source_sha != entry["source_sha256"]:
            raise RuntimeError("Custom/Hybrid source identity mismatch")
        tokens = np.asarray(midi_to_ids(PinnedTokenizerConfig(), MidiFile(str(path))), dtype=np.int64)
        if tokens.size % 8 or tokens.size // 8 != int(entry["notes"]):
            raise RuntimeError("matched-input PT note-count mismatch")
        rows.append({
            "performance_index": int(item["performance_index"]),
            "performance_id": item["performance_id"], "performance_path": item["performance_path"],
            "piece_id": item["piece_id"], "source_midi": str(path),
            "source_sha256": source_sha, "non_pedal_projection_sha256": signature_sha256(path),
            "note_count": int(entry["notes"]), "custom_source_exact": True,
        })
    rows.sort(key=lambda row: row["performance_index"])
    if [row["performance_index"] for row in rows] != list(range(71)):
        raise RuntimeError("matched-input performance indices changed")
    return rows, {
        "status": "passed", "matched_performances": 71,
        "source_sha_exact": 71, "non_pedal_projection_exact": 71,
        "tokenizer_cache_id": TOKENIZER_CACHE_ID, "note_alignment_id": ALIGNMENT_ID,
        "custom_decoder_id": DECODER_ID, "asap_test_access_count": 0,
    }


def make_layout_compatible_donor(source: Path, mapped_donor: Path, destination: Path) -> int:
    canonical = mido.MidiFile(str(source), clip=False)
    mapped = mido.MidiFile(str(mapped_donor), clip=False)
    source_cc = []
    for track_index, track in enumerate(canonical.tracks):
        tick = 0
        for message in track:
            tick += int(message.time)
            if not message.is_meta and message.type == "control_change" and int(message.control) == 64:
                source_cc.append((track_index, int(message.channel)))
    target_track, target_channel = Counter(source_cc).most_common(1)[0][0] if source_cc else (0, 0)
    mapped_cc = []
    for track in mapped.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if not message.is_meta and message.type == "control_change" and int(message.control) == 64:
                mapped_cc.append((tick, int(message.value)))
    output = copy.deepcopy(canonical); discarded = 0
    for track_index, track in enumerate(canonical.tracks):
        tick = 0; base = []
        for order, message in enumerate(track):
            tick += int(message.time)
            if not message.is_meta and message.type == "control_change" and int(message.control) == 64:
                continue
            base.append((tick, order, message))
        eot = [value for value, _, message in base if message.is_meta and message.type == "end_of_track"]
        if len(eot) != 1: raise ValueError("canonical source track must have one EOT")
        inserted = defaultdict(list)
        if track_index == target_track:
            for event_tick, value in mapped_cc:
                if event_tick > eot[0]: discarded += 1; continue
                inserted[event_tick].append(mido.Message("control_change",channel=target_channel,control=64,value=value,time=0))
        rebuilt=[]
        grouped=defaultdict(list)
        for event_tick, order, message in base: grouped[event_tick].append((order,message))
        for event_tick in sorted(set(grouped)|set(inserted)):
            for index,message in enumerate(inserted.get(event_tick,[])): rebuilt.append((event_tick,-1000000+index,message))
            rebuilt.extend((event_tick,order,message) for order,message in grouped.get(event_tick,[]))
        new_track=mido.MidiTrack(); previous=0
        for event_tick,_,message in sorted(rebuilt,key=lambda row:(row[0],row[1])):
            new_track.append(message.copy(time=int(event_tick)-previous)); previous=int(event_tick)
        output.tracks[track_index]=new_track
    output.save(str(destination)); return discarded


def render_candidate(source: Path, candidate_ids: np.ndarray, destination: Path, tokenizer: Any) -> dict[str, Any]:
    source_midi = MidiFile(str(source))
    source_ids = np.asarray(midi_to_ids(tokenizer, source_midi), dtype=np.int64)
    performance = ids_to_midi(tokenizer, candidate_ids, ref=source_ids.tolist())
    mapped = map_midi(source_midi, performance)
    donor = OUTPUT / "donor_midi" / destination.name
    donor.parent.mkdir(parents=True, exist_ok=True)
    mapped_tmp = donor.with_name("." + donor.name + ".mapped.tmp")
    mapped.dump(str(mapped_tmp))
    donor_tmp = donor.with_name("." + donor.name + ".tmp")
    discarded = make_layout_compatible_donor(source, mapped_tmp, donor_tmp)
    mapped_tmp.unlink(); os.replace(donor_tmp, donor)
    candidate_tmp = destination.with_name("." + destination.name + ".tmp")
    if candidate_tmp.exists(): candidate_tmp.unlink()
    transplant_cc64_only(source, donor, candidate_tmp)
    os.replace(candidate_tmp, destination)
    identity = assert_strict_non_cc64_equality(source, destination)
    if signature_sha256(source) != signature_sha256(destination):
        raise AssertionError("Hybrid matched candidate changed non-pedal MIDI")
    identity["mapped_cc64_discarded_after_source_eot"] = discarded
    return identity


def infer(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not torch.cuda.is_available(): raise RuntimeError("matched inference requires CUDA")
    device = torch.device("cuda:0"); tokenizer = PinnedTokenizerConfig()
    model, checkpoint = load_model(device)
    output_rows = []; finite = 0; deterministic = 0
    for index, row in enumerate(rows):
        source = Path(row["source_midi"])
        tokens = np.asarray(midi_to_ids(tokenizer, MidiFile(str(source))), dtype=np.int64)
        regression, auxiliary, info = infer_hybrid_final_outputs(
            model, tokens, device=device, window_notes=512, stride_notes=256,
        )
        if not np.isfinite(regression).all() or not np.isfinite(auxiliary).all():
            raise FloatingPointError("non-finite Hybrid matched prediction")
        integer = regression_cc64_integer(regression)
        candidate_ids = replace_pedal_tokens(tokens.reshape(-1, 8), integer)
        classes = np.select((integer <= 25, integer <= 63, integer <= 103), (0, 1, 2), default=3).astype(np.int64)
        if index < 2:
            repeated, _, _ = infer_hybrid_final_outputs(model, tokens, device=device, window_notes=512, stride_notes=256)
            np.testing.assert_array_equal(regression, repeated); deterministic += 1
        base = OUTPUT / "hybrid_candidate_midi" / f"{index:04d}_{row['performance_id'].replace('/', '_')}.mid"
        identity = render_candidate(source, candidate_ids, base, tokenizer)
        cache_root = OUTPUT / "prediction_cache" / f"{index:04d}"
        atomic_npy(cache_root / "regression_normalized_float32.npy", regression)
        atomic_npy(cache_root / "classes_int64.npy", classes)
        output_rows.append({**row, "candidate_midi": str(base), "candidate_sha256": sha256_file(base),
                            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                            "non_pedal_identity_pass": True, "midi_identity": identity,
                            "prediction_finite": True, "inference": info})
        finite += 1
        if (index + 1) % 10 == 0 or index == 70:
            print(f"INFERENCE_PROGRESS {index+1}/71", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    result = {"checkpoint": checkpoint, "candidates": 71, "finite_predictions": finite,
              "non_pedal_identity_pass": 71, "determinism_sample_pass": deterministic,
              "auxiliary_head_candidate_usage": 0, "asap_test_access_count": 0,
              "repedal_execution_count": 0}
    atomic_json(OUTPUT / "hybrid_candidate_manifest.json", {"candidates": output_rows})
    return output_rows, result


def evaluate(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation = {row["performance_path"]: row for row in read_csv(SPLIT_CSV) if row["split"] == "validation"}
    stage1 = {row["piece_id"]: row for row in read_csv(STAGE1_MANIFEST)}
    confusion = np.zeros((4, 4), np.int64); candidate_hist = np.zeros(PATTERN_COUNT, np.int64); target_hist = np.zeros(PATTERN_COUNT, np.int64)
    transition = {scope: {key: 0 for key in ("candidate", "reference", "tp", "fp", "fn")} for scope in ("pooled", "up", "down")}
    per_rows=[]; failures=[]; aligned_notes=0
    for index, row in enumerate(candidates):
        human = validation[row["performance_path"]]; identifier = human["metadata_index"]
        target = load_frozen_human(identifier)
        if target is None:
            failures.append({"metadata_index": identifier, "performance_path": row["performance_path"], "reason": "frozen shared-human alignment unavailable"}); continue
        candidate = align_and_cache(OUTPUT, kind="hybrid_matched", identifier=identifier,
                                    score_path=Path(stage1[row["piece_id"]]["selected_score_absolute_path"]),
                                    performance_path=Path(row["candidate_midi"]))
        pc=canonical_classes_from_tokens(candidate["tokens"]); tc=canonical_classes_from_tokens(target["tokens"])
        if pc.shape != tc.shape: raise RuntimeError("aligned candidate/reference shape mismatch")
        matrix=confusion_from_pairs(pc,tc); metrics=classification_metrics(matrix)
        counts=pooled_transition_counts(candidate["transitions"],target["transitions"])
        confusion += matrix; candidate_hist += np.bincount(pattern_ids(pc),minlength=PATTERN_COUNT); target_hist += np.bincount(pattern_ids(tc),minlength=PATTERN_COUNT); sum_transition(transition,counts); aligned_notes += len(pc)
        final=finalize_transition({scope:dict(value) for scope,value in counts.items()})
        per_rows.append({"metadata_index":identifier,"performance_id":row["performance_id"],"performance_path":row["performance_path"],"piece_id":row["piece_id"],"aligned_notes":len(pc),"four_class_accuracy":metrics["token_accuracy"],"macro_f1":metrics["macro_f1"],"transition_precision":final["pooled"]["precision"],"transition_recall":final["pooled"]["recall"],"transition_f1":final["pooled"]["f1"],"predicted_transitions":final["pooled"]["candidate"],"reference_transitions":final["pooled"]["reference"]})
        if (index+1)%10==0 or index==70: print(f"EVALUATION_PROGRESS {index+1}/71 successful={len(per_rows)}",flush=True)
    if len(per_rows)!=70 or len(failures)!=1 or aligned_notes!=272053: raise RuntimeError("frozen 70-pair universe regression failed")
    aggregate={"candidate_performances":71,"common_successful_pairs":70,"aligned_notes":aligned_notes,"classification":classification_metrics(confusion),"transition":finalize_transition(transition),"patterns":{**pattern_metrics(candidate_hist,target_hist),"top_candidate_patterns":top_patterns(candidate_hist),"top_target_patterns":top_patterns(target_hist)},"evaluator":{"source":"src/stage2_four_class/validation_evaluator.py + scripts/run_stage2_4class_validation_eval_v0.py","patterns":"4^4=256, base-2 JS, intersection=sum(min(P,Q))","transition":"threshold 64, direction-aware one-to-one, +/-1 distinct onset"}}
    atomic_json(OUTPUT/"hybrid_aggregate_metrics.json",aggregate); atomic_json(OUTPUT/"alignment_failures.json",{"failures":failures})
    return aggregate,per_rows


def comparisons(hybrid: Mapping[str,Any], hybrid_per: list[dict[str,Any]]) -> dict[str,Any]:
    custom=json.loads((CUSTOM_ROOT/"aggregate_metrics.json").read_text()); historical=json.loads((HYBRID_ROOT/"validation_eval/evaluation.json").read_text())
    def flat(value): return {"accuracy":value["classification"]["token_accuracy"],"macro_f1":value["classification"]["macro_f1"],"transition_precision":value["transition"]["pooled"]["precision"],"transition_recall":value["transition"]["pooled"]["recall"],"transition_f1":value["transition"]["pooled"]["f1"],"candidate_transitions":value["transition"]["pooled"]["candidate"],"reference_transitions":value["transition"]["pooled"]["reference"],"js":value["patterns"]["js_divergence_base2"],"intersection":value["patterns"]["intersection"]}
    a,b,c=flat(custom),flat(hybrid),flat(historical)
    keys=("accuracy","macro_f1","transition_precision","transition_recall","transition_f1","js","intersection")
    result={"custom_matched_human":a,"hybrid_matched_human":b,"hybrid_historical_original_pt":c,"custom_minus_matched_hybrid":{k:a[k]-b[k] for k in keys},"matched_minus_historical_hybrid":{k:b[k]-c[k] for k in keys}}
    custom_per={row["metadata_index"]:row for row in csv.DictReader((CUSTOM_ROOT/"per_performance_metrics.csv").open())}
    paired=[]
    for row in hybrid_per:
        x=custom_per[row["metadata_index"]]
        paired.append({"metadata_index":row["metadata_index"],"performance_id":row["performance_id"],"piece_id":row["piece_id"],"custom_accuracy":float(x["four_class_accuracy"]),"hybrid_accuracy":row["four_class_accuracy"],"custom_macro_f1":float(x["macro_f1"]),"hybrid_macro_f1":row["macro_f1"],"custom_transition_precision":float(x["transition_precision"]),"hybrid_transition_precision":row["transition_precision"],"custom_transition_recall":float(x["transition_recall"]),"hybrid_transition_recall":row["transition_recall"],"custom_transition_f1":float(x["transition_f1"]),"hybrid_transition_f1":row["transition_f1"],"custom_predicted_transitions":int(x["candidate_transitions"]),"hybrid_predicted_transitions":row["predicted_transitions"],"reference_transitions":row["reference_transitions"],"custom_minus_hybrid_transition_f1":float(x["transition_f1"])-row["transition_f1"]})
    with (OUTPUT/"per_performance_comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    pieces=defaultdict(list)
    for row in paired: pieces[row["piece_id"]].append(row)
    piece_rows=[]
    for piece,values in sorted(pieces.items()):
        piece_rows.append({"piece_id":piece,"aligned_performances":len(values),"mean_custom_accuracy":float(np.mean([x["custom_accuracy"] for x in values])),"mean_hybrid_accuracy":float(np.mean([x["hybrid_accuracy"] for x in values])),"mean_custom_transition_f1":float(np.mean([x["custom_transition_f1"] for x in values])),"mean_hybrid_transition_f1":float(np.mean([x["hybrid_transition_f1"] for x in values])),"custom_minus_hybrid_transition_f1":float(np.mean([x["custom_minus_hybrid_transition_f1"] for x in values]))})
    with (OUTPUT/"piece_comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(piece_rows[0]));writer.writeheader();writer.writerows(piece_rows)
    atomic_json(OUTPUT/"matched_comparison.json",result); return result


def report(provenance:Mapping[str,Any],comparison:Mapping[str,Any])->None:
    a,b,c=comparison["custom_matched_human"],comparison["hybrid_matched_human"],comparison["hybrid_historical_original_pt"]
    ab,bc=comparison["custom_minus_matched_hybrid"],comparison["matched_minus_historical_hybrid"]
    retained=ab["transition_f1"]>0
    decision="EVENT REPRESENTATION RETAINS A TRANSITION ADVANTAGE UNDER MATCHED INPUTS" if retained else "APPARENT TRANSITION ADVANTAGE WAS SUBSTANTIALLY INPUT-CONSTRUCTION DEPENDENT"
    lines=["# Custom vs Hybrid Matched-Input ASAP Validation","",f"- Hybrid checkpoint: `{CHECKPOINT}`; epoch 5; SHA `{EXPECTED_CHECKPOINT_SHA}`.","- Architecture/objective: pretrained PT Encoder-only; raw CC64/127 Smooth-L1 Huber delta=14/127; auxiliary canonical 4C CE lambda=0.1; inference regression-only.","- Inference: Pitch/IOI/Velocity/Duration input with pedal masked; 512/256 windows; regression scalar overlap mean; clip [0,1], x127, half-up integer.","- Primary input identity: 71/71 exact Custom source SHA/non-pedal projection/note count. ASAP test=0; Repedal=0.","- Frozen evaluator: 70 successful pairs / 272,053 notes; Tysman07M excluded exactly as before.","","## Three-way metrics","","| Model/input | 4C Acc | Macro F1 | Trans P | Trans R | Trans F1 | Candidates | JS | Intersection |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label,x in [("Custom Event / matched human",a),("Hybrid regression-only / matched human",b),("Hybrid regression-only / historical Original-PT",c)]: lines.append(f"| {label} | {x['accuracy']:.6f} | {x['macro_f1']:.6f} | {x['transition_precision']:.6f} | {x['transition_recall']:.6f} | {x['transition_f1']:.6f} | {x['candidate_transitions']:,} | {x['js']:.6f} | {x['intersection']:.6f} |")
    lines += ["","A vs B is primary; C is read-only provenance. Historical C used 19 Original-PT candidates and is not the primary matched comparison.","","## Exact deltas","",f"- Custom - matched Hybrid: `{ab}`",f"- Matched - historical Hybrid: `{bc}`",f"- Transition density: Custom {a['candidate_transitions']}/{a['reference_transitions']}={a['candidate_transitions']/a['reference_transitions']:.6f}; matched Hybrid {b['candidate_transitions']}/{b['reference_transitions']}={b['candidate_transitions']/b['reference_transitions']:.6f}; historical Hybrid {c['candidate_transitions']}/{c['reference_transitions']}={c['candidate_transitions']/c['reference_transitions']:.6f}.","","## Interpretation","",f"- Under exact matched inputs, Custom Transition F1 advantage is {ab['transition_f1']:+.6f}; precision delta {ab['transition_precision']:+.6f}, recall delta {ab['transition_recall']:+.6f}.",f"- Custom 4-state fidelity deltas remain: accuracy {ab['accuracy']:+.6f}, Macro F1 {ab['macro_f1']:+.6f}, JS {ab['js']:+.6f} (lower is better), Intersection {ab['intersection']:+.6f}.",f"- Matched-input construction changes Hybrid Transition F1 by {bc['transition_f1']:+.6f} from historical; this is an observed association, not a causal estimate.",f"- Decision: **{decision}**",("- Recommended next task: Custom Event 4-state/depth-state distribution failure audit; do not tune decoder first." if retained else "- Recommended next task: input provenance / Stage1 dependency analysis."),"- No training, checkpoint selection, calibration, smoothing, threshold change, or post-hoc tuning was performed."]
    (OUTPUT/"CUSTOM_VS_HYBRID_MATCHED_INPUT_VAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main()->None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows,input_audit=audit_inputs(); atomic_json(OUTPUT/"input_identity_audit.json",{**input_audit,"performances":rows})
    config=json.loads((HYBRID_ROOT/"config.json").read_text())
    provenance={"hybrid_checkpoint":str(CHECKPOINT),"checkpoint_sha256":EXPECTED_CHECKPOINT_SHA,"checkpoint_epoch":5,"architecture":config["architecture"],"training_data":config["training_data"],"target_pipeline":config["target_pipeline"],"lambda_ce":config["lambda_ce"],"huber_delta_normalized":config["huber_delta_normalized"],"window_notes":512,"stride_notes":256,"inference_conversion":config["inference_conversion"],"matched_inputs":71,"successful_pairs_expected":70,"asap_test_access_count":0,"repedal_execution_count":0,"training_steps":0,"post_hoc_tuning_count":0}
    atomic_json(OUTPUT/"hybrid_matched_inference_config.json",provenance); atomic_json(OUTPUT/"provenance.json",provenance)
    candidates,infer_info=infer(rows); aggregate,per_rows=evaluate(candidates); comparison=comparisons(aggregate,per_rows); report(provenance,comparison)
    atomic_json(OUTPUT/"run_status.json",{"status":"completed","candidates":71,"successful_pairs":70,"asap_test_access_count":0,"repedal_execution_count":0,"training_steps":0,"post_hoc_tuning_count":0,"inference":infer_info})
    print(json.dumps({"status":"completed","hybrid_matched":comparison["hybrid_matched_human"],"custom_minus_hybrid":comparison["custom_minus_matched_hybrid"]},sort_keys=True))


if __name__=="__main__": main()
