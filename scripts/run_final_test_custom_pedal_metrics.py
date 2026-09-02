#!/usr/bin/env python3
from __future__ import annotations
import argparse, inspect, json, math, os, sys, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import scripts.harmonic_pure_hall_negative_core as harmonic
from scripts.run_harmonic_metric_pure_hall_negative_v0 import synthetic_cases
from scripts.evaluate_bass_connectivity_v0_validation import evaluate_performance as bass_eval, required_logic_tests
from scripts.run_final_test_dual_reference_pedal_evaluation import (
    AB_MANIFEST, TRIO_MANIFEST, SPLIT_CSV, CANDIDATE_SYSTEMS, cpu_preflight,
    discover_candidates, discover_humans, sha256_file, update_status, write_csv, write_json,
)
from src.bass_connectivity_v0 import (
    GROUP_WINDOW_SECONDS, LOCAL_IOI_RADIUS, NEXT_LOW_MAX_PITCH, R_LOW, R_HIGH,
    BETA_OCTAVE, STRUCTURAL_BASS_THRESHOLD,
)

OUT = ROOT / "analysis/final_test_custom_pedal_metrics_v0"
HARMONIC_FILE = ROOT / "scripts/harmonic_pure_hall_negative_core.py"
BASS_FILE = ROOT / "src/bass_connectivity_v0.py"
HUMAN_LABEL = {"setA": "Human — Set A (104)", "setB": "Human — Set B (166)"}
ORDER = tuple(HUMAN_LABEL.values()) + tuple(CANDIDATE_SYSTEMS)
STAGE2 = tuple(CANDIDATE_SYSTEMS[1:])

def now(): return datetime.now(timezone.utc).isoformat()

def log(out: Path, message: str):
    line = f"{now()} {message}"
    with (out / "evaluation.log").open("a", encoding="utf-8") as f: f.write(line + "\n")
    print(line, flush=True)

def frozen_tests():
    import tests.test_harmonic_muddiness_final as ht
    funcs = [f for n, f in inspect.getmembers(ht, inspect.isfunction) if n.startswith("test_")]
    for f in funcs: f()
    syn = synthetic_cases()
    cols = ["HALL_NEG_MEAN","NEGATIVE_PAIR_FRACTION","CONDITIONAL_NEGATIVE","MAX_NEGATIVE"]
    assert len(syn) == 8 and np.isfinite(syn[cols].to_numpy()).all()
    bass = required_logic_tests()
    assert len(bass) == 8 and all(r["passed"] for r in bass)
    return {"harmonic_synthetic":"8/8 PASS","harmonic_aggregation":f"{len(funcs)}/4 PASS","bass":"8/8 PASS"}

def preflight(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.log").touch(exist_ok=True)
    cpu, tests = cpu_preflight(), frozen_tests()
    inv, _, pieces, configs = discover_candidates()
    humans = discover_humans(pieces)
    assert len(inv) == 138 and len(humans["setA"]) == 104 and len(humans["setB"]) == 166
    write_csv(out / "candidate_inventory.csv", inv)
    write_csv(out / "human_inventory_setA.csv", humans["setA"])
    write_csv(out / "human_inventory_setB.csv", humans["setB"])
    provenance = {
        "experiment":"final_test_custom_pedal_metrics_v0","created_at":now(),
        "measurement_only":True,"inference_count":0,"checkpoint_load_count":0,
        "human_candidate_alignment_count":0,"cpu_only_preflight":cpu,
        "harmonic":{
            "implementation_file":str(HARMONIC_FILE),"implementation_sha256":sha256_file(HARMONIC_FILE),
            "final_aggregation_function":"aggregate_final_harmonic_metric",
            "midi_entry_point":"evaluate_final_harmonic_metric",
            "hall_weights_interval_1_to_12":harmonic.pilot.HALL_WEIGHTS[1:].tolist(),
            "cc64_threshold":64,"M_harm_coefficient":harmonic.FINAL_ACCUMULATION_COEFFICIENT,
            "formula":"H_mean(valid Q only) + 0.05 * A_acc(all onsets)",
            "synthetic_tests":tests["harmonic_synthetic"],"aggregation_tests":tests["harmonic_aggregation"],
        },
        "bass":{
            "implementation_file":str(BASS_FILE),"implementation_sha256":sha256_file(BASS_FILE),
            "grouping_seconds":GROUP_WINDOW_SECONDS,"grouping":"earliest-anchor inclusive",
            "local_radius":LOCAL_IOI_RADIUS,"next_low_cutoff":NEXT_LOW_MAX_PITCH,
            "R_low":R_LOW,"R_high":R_HIGH,"B_threshold":STRUCTURAL_BASS_THRESHOLD,
            "BETA_OCTAVE":BETA_OCTAVE,"cc64_threshold":64,
            "sigma_left":"d/2","sigma_right":"d/4",
            "same_bass":"exact MIDI pitch equality","tests":tests["bass"],
        },
        "human_set_A":{"source":str(SPLIT_CSV),"source_sha256":sha256_file(SPLIT_CSV),"pieces":23,"performances":104},
        "human_set_B":{"source":str(ROOT/"third_party/PianistTransformer/data/midis/testset/human"),"pieces":23,"performances":166,"piece_match":"23/23 PASS"},
        "generated":{"systems":list(CANDIDATE_SYSTEMS),"pieces_each":23,
            "run_ab_manifest":str(AB_MANIFEST),"run_ab_sha256":sha256_file(AB_MANIFEST),
            "trio_manifest":str(TRIO_MANIFEST),"trio_sha256":sha256_file(TRIO_MANIFEST),
            "nonpedal_identity":configs["identity_counts"],
            "run_a":"PRE/MAIN/POST; 13 strict + 10 frozen EOT-extension-only; note signature PASS 23/23",
            "run_b":"State-Anchored constrained Viterbi"},
        "aggregation":"Human performance -> piece mean -> piece macro; generated one MIDI -> piece macro; Bass NA excluded and counted",
    }
    write_json(out/"metric_provenance.json", provenance)
    update_status(out,status="prepared",state="preflight_pass",
        harmonic_tests="8/8 + 4/4 PASS",bass_tests="8/8 PASS",
        set_a_inventory="23/104 PASS",set_b_inventory="23/166; match 23/23 PASS",
        generated_inventory="6x23 PASS",generated_nonpedal_identity="PASS",cpu_only=True,
        first_generated_finite=False,first_human_setA_finite=False,first_human_setB_finite=False,
        error=None,traceback=None)
    log(out,"PREFLIGHT_PASS harmonic=8/8+4/4 bass=8/8 generated=6x23 nonpedal=PASS setA=23/104 setB=23/166 cpu=PASS")
    return {"inventory":inv,"pieces":pieces,"humans":humans,"provenance":provenance}

def evaluate(row: Mapping[str,Any], scope: str):
    path = Path(str(row["midi_path"]))
    onset, hp = harmonic.evaluate_midi(path)
    h = harmonic.aggregate_final_harmonic_metric(onset)
    inp = {"system":str(row.get("system",scope)),"system_label":str(row.get("system",scope)),
        "piece_id":row["piece_id"],"composer":row["composer"],"title":row["title"],
        "performance_id":row.get("performance_id",row["piece_id"]),"midi_path":str(path)}
    b, transitions, structural, _ = bass_eval(inp)
    bs = b["C_Bass_performance"]
    assert all(math.isfinite(float(h[k])) and float(h[k]) >= 0 for k in ("H_mean","A_acc","M_harm"))
    assert bs is None or (math.isfinite(float(bs)) and 0 <= float(bs) <= 1)
    octpass = sum(int(x["O_oct"])==1 and float(x["L_n"])*float(x["S_n"]) < .5 <= float(x["B_n"]) for x in structural)
    base = {"scope":scope,"piece_id":row["piece_id"],"composer":row["composer"],"title":row["title"],
        "system":str(row.get("system",scope)),"performance_id":str(row.get("performance_id",row["piece_id"])),
        "midi_path":str(path),"sha256":row["sha256"],**h,"Bass_Connectivity":bs,
        "structural_bass_count":b["structural_bass_count"],"valid_bass_pair_count":b["valid_transition_count"],
        "excluded_keyheld_pair_count":b["excluded_keyoff_at_or_after_next_count"],
        "excluded_missing_keyoff_pair_count":b["excluded_missing_keyoff_count"],
        "same_bass_pair_count":b["same_bass_transition_count"],
        "different_bass_pair_count":b["different_bass_transition_count"],
        "octave_bonus_threshold_pass_count":octpass}
    hd = {"scope":scope,"piece_id":base["piece_id"],"system":base["system"],
        "performance_id":base["performance_id"],"midi_path":str(path),**h,
        "total_RAW":math.fsum(float(x["negative_hall_mass_n"]) for x in onset),
        "total_pair_count":sum(int(x["N_pair_n"]) for x in onset),
        "total_negative_pair_count":sum(int(x["N_neg_n"]) for x in onset),**hp}
    bd = {"scope":scope,"piece_id":base["piece_id"],"system":base["system"],
        "performance_id":base["performance_id"],"midi_path":str(path),"Bass_Connectivity":bs,
        "structural_bass_count":b["structural_bass_count"],"valid_bass_pair_count":b["valid_transition_count"],
        "excluded_keyheld_pair_count":b["excluded_keyoff_at_or_after_next_count"],
        "same_bass_pair_count":b["same_bass_transition_count"],"different_bass_pair_count":b["different_bass_transition_count"],
        "octave_bonus_threshold_pass_count":octpass,"transition_diagnostic_count":len(transitions)}
    return base,hd,bd

def stats(values, prefix):
    a=np.asarray([float(x) for x in values if pd.notna(x)],float)
    if not len(a): return {prefix+"_valid_piece_count":0}
    return {prefix+"_valid_piece_count":len(a),prefix+"_mean":a.mean(),prefix+"_median":np.median(a),
        prefix+"_std":a.std(),prefix+"_IQR":np.quantile(a,.75)-np.quantile(a,.25),
        prefix+"_min":a.min(),prefix+"_max":a.max()}

def aggregate(out: Path, d, gen, hum, hd, bd):
    pieces=pd.DataFrame(d["pieces"])[["piece_id","composer","title"]]
    human_piece={}
    for key in ("setA","setB"):
        f=pd.DataFrame(hum[key]); rows=[]
        for _,p in pieces.iterrows():
            q=f[f.piece_id==p.piece_id]
            bass=q.Bass_Connectivity.dropna()
            rows.append({**p.to_dict(),"human_performance_count":len(q),
                "scored_harmonic_performance_count":q.M_harm.notna().sum(),
                "scored_bass_performance_count":len(bass),"H_mean":q.H_mean.mean(),
                "A_acc":q.A_acc.mean(),"M_harm":q.M_harm.mean(),
                "Bass_Connectivity":bass.mean() if len(bass) else None})
        human_piece[key]=pd.DataFrame(rows)
    gf=pd.DataFrame(gen)
    summaries=[]
    for key in ("setA","setB"):
        q=human_piece[key]
        row={"system":HUMAN_LABEL[key],"category":"human","performance_count":len(hum[key]),"piece_count":23,
            "scored_harmonic_piece_count":q.M_harm.notna().sum(),"scored_bass_piece_count":q.Bass_Connectivity.notna().sum(),
            "Harmonic_Muddiness":q.M_harm.mean(),"Bass_Connectivity":q.Bass_Connectivity.mean()}
        row.update(stats(q.M_harm,"M_harm")); row.update(stats(q.Bass_Connectivity,"Bass")); summaries.append(row)
    for system in CANDIDATE_SYSTEMS:
        q=gf[gf.system==system]; assert len(q)==23
        row={"system":system,"category":"generated","performance_count":23,"piece_count":23,
            "scored_harmonic_piece_count":q.M_harm.notna().sum(),"scored_bass_piece_count":q.Bass_Connectivity.notna().sum(),
            "Harmonic_Muddiness":q.M_harm.mean(),"Bass_Connectivity":q.Bass_Connectivity.mean()}
        row.update(stats(q.M_harm,"M_harm")); row.update(stats(q.Bass_Connectivity,"Bass")); summaries.append(row)
    sf=pd.DataFrame(summaries); sf["order"]=sf.system.map({x:i for i,x in enumerate(ORDER)}); sf=sf.sort_values("order").drop(columns="order")
    idx={(r.piece_id,r.system):r for r in gf.itertuples()}; deltas=[]
    for piece in pieces.piece_id:
        o=idx[(piece,"Original PT")]
        for system in STAGE2:
            m=idx[(piece,system)]; db=None if pd.isna(o.Bass_Connectivity) or pd.isna(m.Bass_Connectivity) else m.Bass_Connectivity-o.Bass_Connectivity
            deltas.append({"piece_id":piece,"composer":m.composer,"title":m.title,"system":system,
                "delta_M_harm":m.M_harm-o.M_harm,"M_harm_improved":m.M_harm<o.M_harm,
                "delta_BassConnectivity":db,"BassConnectivity_improved":None if db is None else db>0})
    df=pd.DataFrame(deltas); ds=[]
    for s in STAGE2:
        q=df[df.system==s]; qb=q.dropna(subset=["delta_BassConnectivity"])
        ds.append({"system":s,"M_harm_improved_piece_count":q.M_harm_improved.sum(),"M_harm_valid_piece_count":len(q),
            "BassConnectivity_improved_piece_count":qb.BassConnectivity_improved.sum(),"BassConnectivity_valid_piece_count":len(qb)})
    comp=[]; sm={r.system:r for r in sf.itertuples()}
    for s in CANDIDATE_SYSTEMS:
        for hk in ("setA","setB"):
            g,h=sm[s],sm[HUMAN_LABEL[hk]]
            comp.append({"system":s,"human_baseline":HUMAN_LABEL[hk],
                "delta_M_harm_generated_minus_human":g.Harmonic_Muddiness-h.Harmonic_Muddiness,
                "delta_BassConnectivity_generated_minus_human":g.Bass_Connectivity-h.Bass_Connectivity})
    gf.to_csv(out/"generated_per_piece_scores.csv",index=False)
    human_piece["setA"].to_csv(out/"human_per_piece_setA_asap104.csv",index=False)
    human_piece["setB"].to_csv(out/"human_per_piece_setB_pt166.csv",index=False)
    pd.DataFrame(hum["setA"]).to_csv(out/"human_performance_scores_setA.csv",index=False)
    pd.DataFrame(hum["setB"]).to_csv(out/"human_performance_scores_setB.csv",index=False)
    sf.to_csv(out/"summary.csv",index=False); df.to_csv(out/"delta_vs_original_pt.csv",index=False)
    pd.DataFrame(ds).to_csv(out/"delta_summary.csv",index=False); pd.DataFrame(comp).to_csv(out/"comparison_to_human_baselines.csv",index=False)
    pd.DataFrame(hd).to_csv(out/"harmonic_diagnostics.csv",index=False); pd.DataFrame(bd).to_csv(out/"bass_connectivity_diagnostics.csv",index=False)
    lines=["# Final Test Custom Pedal Metrics","","## Overall final test","",
        "| System | Harmonic Muddiness ↓ | Bass Connectivity ↑ |","|---|---:|---:|"]
    for r in sf.itertuples(): lines.append(f"| {r.system} | {r.Harmonic_Muddiness:.6f} | {r.Bass_Connectivity:.6f} |")
    lines += ["","## Accounting","",
        "- Human: performance score → within-piece arithmetic mean → 23-piece macro; Set A and Set B were not pooled.",
        "- Generated: each saved MIDI measured once; 6 × 23 = 138. No Human alignment was used.",
        "- RUN A note provenance: 13 strict + 10 frozen EOT-extension-only; pitch/onset/offset/velocity PASS 23/23.",
        "- Undefined Bass performance/piece counts are recorded in CSV.","","## Original PT improvement counts","",
        "| System | M_harm improved | Bass improved |","|---|---:|---:|"]
    for r in ds: lines.append(f"| {r['system']} | {r['M_harm_improved_piece_count']}/{r['M_harm_valid_piece_count']} | {r['BassConnectivity_improved_piece_count']}/{r['BassConnectivity_valid_piece_count']} |")
    lines += ["","## Metric provenance","","Frozen definitions are recorded in metric_provenance.json. No weighted sum was created."]
    (out/"FINAL_TEST_CUSTOM_PEDAL_METRICS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def run(out: Path):
    d=preflight(out); update_status(out,status="running",state="evaluating",started_at=now(),evaluation_progress=0,evaluation_total=408)
    log(out,"RUN_START total=408 measurement_only=1")
    inv=list(d["inventory"]); hs=d["humans"]
    tasks=[("generated",inv[0]),("setA",hs["setA"][0]),("setB",hs["setB"][0])]
    tasks += [("generated",x) for x in inv[1:]]+[("setA",x) for x in hs["setA"][1:]]+[("setB",x) for x in hs["setB"][1:]]
    assert len(tasks)==408
    gen=[]; hum={"setA":[],"setB":[]}; hd=[]; bd=[]; flags={"generated":False,"setA":False,"setB":False}
    for i,(scope,row) in enumerate(tasks,1):
        log(out,f"EVAL_START {i}/408 scope={scope} piece={row['piece_id']} path={row['midi_path']}")
        result,h,b=evaluate(row,scope); (gen if scope=="generated" else hum[scope]).append(result); hd.append(h); bd.append(b); flags[scope]=True
        update_status(out,status="running",state="evaluating",evaluation_progress=i,current_scope=scope,current_piece=row["piece_id"],
            first_generated_finite=flags["generated"],first_human_setA_finite=flags["setA"],first_human_setB_finite=flags["setB"],
            latest_M_harm=result["M_harm"],latest_Bass_Connectivity=result["Bass_Connectivity"])
        log(out,f"EVAL_PASS {i}/408 scope={scope} M_harm={result['M_harm']:.12g} Bass={result['Bass_Connectivity']}")
    assert len(gen)==138 and len(hum["setA"])==104 and len(hum["setB"])==166
    aggregate(out,d,gen,hum,hd,bd)
    prov=d["provenance"]; prov["completed_at"]=now(); prov["actual_counts"]={"generated":138,"setA":104,"setB":166}; write_json(out/"metric_provenance.json",prov)
    update_status(out,status="completed",state="completed",completed_at=now(),evaluation_progress=408,report_generated=True,error=None,traceback=None)
    log(out,"RUN_COMPLETED generated=138 setA=104 setB=166")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,default=OUT); p.add_argument("--preflight-only",action="store_true"); p.add_argument("--execute",action="store_true"); a=p.parse_args()
    if a.preflight_only==a.execute: p.error("choose exactly one mode")
    out=a.output_root if a.output_root.is_absolute() else ROOT/a.output_root
    try:
        d=preflight(out) if a.preflight_only else None
        if a.preflight_only:
            print(json.dumps({"status":"prepared","generated":len(d["inventory"]),"setA":len(d["humans"]["setA"]),"setB":len(d["humans"]["setB"])})); return
        run(out)
    except Exception as e:
        out.mkdir(parents=True,exist_ok=True); tr=traceback.format_exc(); update_status(out,status="failed",state="failed",error=repr(e),traceback=tr); log(out,f"RUN_FAILED {e!r}\n{tr}"); raise
if __name__=="__main__": main()
