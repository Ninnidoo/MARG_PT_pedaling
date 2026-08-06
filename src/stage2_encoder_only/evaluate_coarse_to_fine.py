"""Validation-only overlap-aware evaluation for CTF v0."""
from __future__ import annotations
import csv, json, math, os, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .coarse_to_fine import Stage2CoarseToFineModel, decode_pedals, pedal_regions
from .dataset import MASK_ID, NON_PEDAL_FEATURES, PEDAL_TOKEN_OFFSET, Stage2PedalDataset, generate_window_starts, stage2_pedal_collate_fn
from .evaluate_oracle import _make_window_sample
from .calibrate_decoder import aggregate_candidate_metrics, performance_candidate_metrics

def _atomic(path:Path,text:str)->None:
    tmp=path.parent/f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"; tmp.write_text(text,encoding="utf-8"); os.replace(tmp,path)
def _average(n:int, windows:list[tuple[int,np.ndarray]])->np.ndarray:
    shape=windows[0][1].shape[1:]; total=np.zeros((n,*shape),np.float64); count=np.zeros(n,np.int64)
    for start, values in windows:
        total[start:start+len(values)]+=np.asarray(values); count[start:start+len(values)]+=1
    if np.any(count==0): raise RuntimeError("overlap reconstruction left uncovered notes")
    return (total/count.reshape((n,)+(1,)*len(shape))).astype(np.float32)
def _infer(model:Any,tokens:np.ndarray,device:torch.device)->dict[str,np.ndarray]:
    n=len(tokens); starts=generate_window_starts(n,512,256); regions=[]; depths=[]
    with torch.inference_mode():
      for offset in range(0,len(starts),16):
        ss=starts[offset:offset+16]; batch=stage2_pedal_collate_fn([_make_window_sample(tokens,s,min(s+512,n)) for s in ss]); ids=batch["input_ids"].to(device); attention=batch["token_attention_mask"].to(device); notes=batch["note_mask"].to(device)
        with torch.amp.autocast(device_type="cuda",dtype=torch.float16,enabled=True): out=model(ids,attention,notes)
        for j,s in enumerate(ss):
          length=min(512,n-s); regions.append((s,out.region_logits[j,:length].float().cpu().numpy().copy())); depths.append((s,out.depth_logits[j,:length].float().cpu().numpy().copy()))
    return {"region_logits":_average(n,regions),"depth_logits":_average(n,depths),"targets":tokens[:,NON_PEDAL_FEATURES:].astype(np.int64)-PEDAL_TOKEN_OFFSET}
def _confusion(pred:np.ndarray,true:np.ndarray)->dict[str,Any]:
    matrix=np.zeros((3,3),np.int64)
    for a,b in zip(true.ravel(),pred.ravel()): matrix[a,b]+=1
    out={"matrix":matrix.tolist()}; f=[]
    for i,name in enumerate(("zero","intermediate","full")):
      tp=matrix[i,i]; precision=tp/matrix[:,i].sum() if matrix[:,i].sum() else 0.; recall=tp/matrix[i].sum() if matrix[i].sum() else 0.; score=2*precision*recall/(precision+recall) if precision+recall else 0.; out[name]={"precision":precision,"recall":recall,"f1":score}; f.append(score)
    out["accuracy"]=np.trace(matrix)/matrix.sum(); out["macro_f1"]=float(np.mean(f)); return out
def evaluate(output_dir:str|Path)->dict[str,Any]:
    out=Path(output_dir); config=json.loads((out/"config.json").read_text()); ckpt=torch.load(out/"best.pt",map_location="cpu",weights_only=False); device=torch.device("cuda:0")
    model=Stage2CoarseToFineModel.from_pretrained(config["checkpoint_path"],torch_dtype=torch.float32,attn_implementation="eager"); model.load_state_dict(ckpt["model_state"],strict=True); model.to(device).eval()
    data=Stage2PedalDataset(config["asap_root"],config["split_csv"],"validation",512,256,cache_mode="preload")
    if data.performance_count!=71 or len({r["piece_id"] for r in data.performances})!=19: raise RuntimeError("validation scope mismatch")
    records=[]; perf=[]; all_true=[]; all_pred=[]; all_region_true=[]; all_region_pred=[]; depth_true=[]; depth_pred=[]; slot_rows=[]
    for row in data.performances:
      result=_infer(model,data._token_cache[row["performance_path"]],device); rlog=torch.from_numpy(result["region_logits"]); dlog=torch.from_numpy(result["depth_logits"]); pred=decode_pedals(rlog,dlog).numpy(); truth=result["targets"]; regions=pedal_regions(torch.from_numpy(truth)).numpy(); rp=result["region_logits"].argmax(-1)
      rec=performance_candidate_metrics(pred,truth,row["performance_path"]); records.append(rec); flat=dict(rec["flat"]); flat.update(performance_path=row["performance_path"],piece_id=row["piece_id"],region_accuracy=float((rp==regions).mean())); perf.append(flat)
      all_true.append(truth);all_pred.append(pred);all_region_true.append(regions);all_region_pred.append(rp)
      mask=(regions==1); depth_true.append(truth[mask]); depth_pred.append((1+np.rint(125/(1+np.exp(-result["depth_logits"][mask])))).clip(1,126))
      for slot in range(4): slot_rows.append({"performance_path":row["performance_path"],"slot":slot+1,"mae":float(np.abs(pred[:,slot]-truth[:,slot]).mean()),"exact_accuracy":float((pred[:,slot]==truth[:,slot]).mean()),"region_accuracy":float((rp[:,slot]==regions[:,slot]).mean())})
    aggregate=aggregate_candidate_metrics(records); true=np.concatenate(all_true);pred=np.concatenate(all_pred); reg_true=np.concatenate(all_region_true);reg_pred=np.concatenate(all_region_pred); depths_t=np.concatenate(depth_true); depths_p=np.concatenate(depth_pred); conf=_confusion(reg_pred,reg_true)
    oracle={"intermediate_depth_mae":float(np.abs(depths_p-depths_t).mean()),"tolerance_accuracy_5":float((np.abs(depths_p-depths_t)<=5).mean()),"tolerance_accuracy_10":float((np.abs(depths_p-depths_t)<=10).mean()),"tolerance_accuracy_20":float((np.abs(depths_p-depths_t)<=20).mean()),"predicted_mean":float(depths_p.mean()),"predicted_median":float(np.median(depths_p)),"predicted_std":float(depths_p.std()),"predicted_q05":float(np.quantile(depths_p,.05)),"predicted_q25":float(np.quantile(depths_p,.25)),"predicted_q75":float(np.quantile(depths_p,.75)),"predicted_q95":float(np.quantile(depths_p,.95)),"unique_rounded_intermediate_values":int(len(np.unique(depths_p)))}
    hist=[]
    for value in range(1,127): hist.append({"value":value,"target_count":int((depths_t==value).sum()),"prediction_count":int((depths_p==value).sum())})
    ctf={"model":"coarse_to_fine_v0","decoder":"predicted_region_conditional_depth","overall_mae":aggregate["micro_mae"],"intermediate_mae":aggregate["micro_intermediate_mae"],"exact_token_accuracy":aggregate["micro_exact_accuracy"],"exact_note_accuracy":aggregate["micro_exact_note_accuracy"],"predicted_zero_ratio":aggregate["micro_predicted_zero_ratio"],"predicted_intermediate_ratio":aggregate["micro_predicted_intermediate_ratio"],"predicted_full_ratio":aggregate["micro_predicted_full_ratio"],"intermediate_endpoint_collapse":aggregate["micro_intermediate_endpoint_collapse_ratio"],"transition_f1":aggregate["micro_transition_detection_f1"],"region_accuracy":conf["accuracy"],"region_macro_f1":conf["macro_f1"],**{f"region_{name}_{metric}":value for name in ("zero","intermediate","full") for metric,value in conf[name].items()},**oracle}
    reference=[]
    with open("/workspace/project/analysis/stage2_encoder_only_ordinal_v0/validation_comparison.csv",newline="") as handle:
      for r in csv.DictReader(handle):
       if (r["model"],r["decoder"]) in {("ce_baseline","posterior_median"),("ordinal_1p0","posterior_median")}:
        reference.append({"model":r["model"],"decoder":r["decoder"],"overall_mae":r["micro_mae"],"intermediate_mae":r["micro_intermediate_mae"],"exact_token_accuracy":r["micro_exact_accuracy"],"exact_note_accuracy":r["micro_exact_note_accuracy"],"predicted_zero_ratio":r["micro_predicted_zero_ratio"],"predicted_intermediate_ratio":r["micro_predicted_intermediate_ratio"],"predicted_full_ratio":r["micro_predicted_full_ratio"],"intermediate_endpoint_collapse":r["micro_intermediate_endpoint_collapse_ratio"],"transition_f1":r["micro_transition_detection_f1"]})
    val=out/"validation"; val.mkdir(exist_ok=True)
    def write_csv(path:Path,rows:list[dict[str,Any]]):
      fields=sorted({k for row in rows for k in row}); from io import StringIO; b=StringIO(); w=csv.DictWriter(b,fieldnames=fields);w.writeheader();w.writerows(rows);_atomic(path,b.getvalue())
    write_csv(val/"validation_comparison.csv",reference+[ctf]);write_csv(val/"per_performance_metrics.csv",perf);write_csv(val/"per_slot_metrics.csv",slot_rows);write_csv(val/"region_confusion.csv",[{"true_region":a,"pred_zero":conf["matrix"][i][0],"pred_intermediate":conf["matrix"][i][1],"pred_full":conf["matrix"][i][2]} for i,a in enumerate(("zero","intermediate","full"))]);write_csv(val/"intermediate_histogram.csv",hist);_atomic(val/"oracle_region_metrics.json",json.dumps(oracle,indent=2,sort_keys=True)+"\n")
    baseline={"mae":28.197258,"intermediate":36.245111,"ratio":.334542,"collapse":.5101,"transition":.305097}; A=ctf["overall_mae"]<=baseline["mae"]-1;B=ctf["intermediate_mae"]<=baseline["intermediate"]-3 and ctf["overall_mae"]<=baseline["mae"]+.5;safeguards=.15<=ctf["predicted_intermediate_ratio"]<=.55 and ctf["intermediate_endpoint_collapse"]<.5 and ctf["transition_f1"]>=baseline["transition"]-.02 and all(math.isfinite(float(v)) for v in ctf.values() if isinstance(v,(int,float))) and aggregate["non_degenerate"]; result={"ctf":ctf,"oracle":oracle,"success_criterion_a":A,"success_criterion_b":B,"safeguards":safeguards,"passes":(A or B) and safeguards}; del model;torch.cuda.empty_cache();return result
