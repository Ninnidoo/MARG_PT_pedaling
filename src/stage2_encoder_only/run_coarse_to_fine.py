"""Detached-safe orchestration and pedal-rich gate for CTF v0."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Any
import torch
from .coarse_to_fine import Stage2CoarseToFineModel, coarse_to_fine_loss, decode_pedals, pedal_regions
from .dataset import Stage2PedalDataset, stage2_pedal_collate_fn
from .evaluate_oracle import _make_window_sample
from .evaluate_coarse_to_fine import evaluate
from .train import get_gpu_identity
from .train_coarse_to_fine import build_optimizer, default_config, evaluation_step, train, train_step
from .training import create_grad_scaler, move_batch_to_device, set_deterministic_seed

def atomic_json(path:Path,payload:dict[str,Any])->None:
    tmp=path.parent/f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"; tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def overfit(output:Path)->dict[str,Any]:
    set_deterministic_seed(20260710); data=Stage2PedalDataset("/workspace/public/ASAP/asap-dataset-v1.1","/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv","train",512,256,cache_mode="preload"); row=next(r for r in data.performances if r["performance_path"]=="Schubert/Piano_Sonatas/664-3/Lin07.mid"); tokens=data._token_cache[row["performance_path"]]; samples=[_make_window_sample(tokens,s,s+512) for s in (0,256,512,768)]; device=torch.device("cuda:0"); model=Stage2CoarseToFineModel.from_pretrained("/workspace/project/checkpoints/pianist_transformer",torch_dtype=torch.float32,attn_implementation="eager").to(device);opt=build_optimizer(model,1e-4,1e-3,0.); scaler=create_grad_scaler(True,"cuda",1024.)
    active_name,active=next((n,p) for n,p in model.encoder.named_parameters() if "embed_tokens" not in n and p.ndim>=2); before=active.detach().clone(); heads={n:p.detach().clone() for n,p in list(model.region_heads.named_parameters())+list(model.depth_heads.named_parameters())}
    def measure()->dict[str,float]:
      logits=[];depth=[];targets=[]
      for off in (0,2):
       b=move_batch_to_device(stage2_pedal_collate_fn(samples[off:off+2]),device)
       with torch.no_grad(),torch.amp.autocast("cuda",dtype=torch.float16): o=model(b["input_ids"],b["token_attention_mask"],b["note_mask"])
       logits.append(o.region_logits.float());depth.append(o.depth_logits.float());targets.append(b["pedal_targets"])
      r=torch.cat(logits);d=torch.cat(depth);t=torch.cat(targets);p=decode_pedals(r,d); region=pedal_regions(t); inter=region==1; return {"region_accuracy":float(((r.argmax(-1)==region)&(t!=-100)).float().sum()/(t!=-100).sum()),"intermediate_oracle_region_mae":float((p[inter].float()-t[inter].float()).abs().mean()),"end_to_end_mae":float((p[t!=-100].float()-t[t!=-100].float()).abs().mean())}
    initial=measure(); grad_ok=True; steps=0
    for step in range(1,301):
      b=move_batch_to_device(stage2_pedal_collate_fn(samples[0 if step%2 else 2:2 if step%2 else 4]),device); m=train_step(model,b,opt,scaler,True,1.); grad_ok &= m["encoder_gradient_norm"]>0 and m["head_gradient_norm"]>0;steps=step
      if step%5==0:
       final=measure()
       if final["region_accuracy"]>=.98 and final["intermediate_oracle_region_mae"]<=0.10 and final["end_to_end_mae"]<=0.10: break
    final=measure(); changed=not torch.equal(before,active.detach()) and all(not torch.equal(heads[n],p.detach()) for n,p in list(model.region_heads.named_parameters())+list(model.depth_heads.named_parameters())); result={"passed":final["region_accuracy"]>=.98 and final["intermediate_oracle_region_mae"]<=0.10 and final["end_to_end_mae"]<=0.10 and grad_ok and changed,"performance_path":row["performance_path"],"window_starts":[0,256,512,768],"steps":steps,"initial":initial,"final":final,"active_encoder_parameter":active_name,"encoder_and_all_head_parameters_changed":changed,"finite_nonzero_gradients_all_groups":grad_ok,"oom":False}; atomic_json(output/"tests"/"pedal_rich_overfit.json",result); del model;torch.cuda.empty_cache();
    if not result["passed"]: raise RuntimeError(f"CTF overfit failed: {result}")
    return result
def report(output:Path, result:dict[str,Any], overfit:dict[str,Any])->None:
    metrics=list(__import__('csv').DictReader((output/"metrics.csv").open())); best=int(result["best_epoch"]); bestrow=next(r for r in metrics if int(r["epoch"])==best); c=result["ctf"]; text=f"""# Encoder-only Coarse-to-Fine Pedal Head v0

## 1. Hypothesis

Separating endpoint-region selection from conditional intermediate depth can avoid the flat 128-class argmax endpoint collapse without changing encoder, data, or training controls.

## 2. Exact architecture

Official pretrained bidirectional PT encoder (unfrozen, hidden size 768), with four independent Linear(768,3) region heads and four independent Linear(768,1) depth heads; no MLP, conditioning, smoothing, calibration, or post-processing.

## 3. Target construction

0→ZERO, 1–126→INTERMEDIATE, 127→FULL; intermediate z=(y−1)/125.

## 4. Loss and decoding formulas

Total loss is unweighted region CE plus SmoothL1(beta=0.05) on true-intermediate positions. Raw region and depth logits are averaged over windows before argmax/sigmoid decoding.

## 5. Implementation/test results

See `tests/test_results.json`.

## 6. Overfit result

```json
{json.dumps(overfit,indent=2,sort_keys=True)}
```

## 7. Training summary

Completed {result['completed_epoch']} epochs; best epoch {best}; best validation total loss {result['best_validation_total_loss']:.6f}. Best-epoch region CE/depth/total: {bestrow['validation_region_ce_loss']}/{bestrow['validation_depth_smooth_l1_loss']}/{bestrow['validation_total_loss']}.

## 8. Full validation comparison

Overall MAE {c['overall_mae']:.6f}; intermediate MAE {c['intermediate_mae']:.6f}; exact token accuracy {c['exact_token_accuracy']:.6f}; exact-note accuracy {c['exact_note_accuracy']:.6f}.

## 9. Region classification analysis

Region accuracy {c['region_accuracy']:.6f}, macro F1 {c['region_macro_f1']:.6f}; detailed confusion and per-region P/R/F1 are in `validation/`.

## 10. Oracle-region depth analysis

Oracle-region intermediate MAE {result['oracle']['intermediate_depth_mae']:.6f}; detailed tolerance and distribution diagnostics are in `validation/oracle_region_metrics.json`.

## 11. Intermediate concentration/collapse analysis

Predicted intermediate ratio {c['predicted_intermediate_ratio']:.6f}; endpoint collapse {c['intermediate_endpoint_collapse']:.6f}; unique rounded intermediate values {result['oracle']['unique_rounded_intermediate_values']}.

## 12. Transition analysis

Transition F1 {c['transition_f1']:.6f}; detailed transition metrics are in the validation comparison.

## 13. Success-criteria decision

Criterion A={result['success_criterion_a']}; Criterion B={result['success_criterion_b']}; safeguards={result['safeguards']}; final pass={result['passes']}.

## 14. Interpretation

Region-selection failure is indicated by low region accuracy/F1; conditional-depth failure by poor oracle-region MAE or narrow intermediate support; temporal-transition failure by transition F1. This decision uses validation only, never training loss alone.

## 15. Recommended next step

{'Freeze this candidate for one later untouched test evaluation.' if result['passes'] else 'Do not automatically continue to MAESTRO or another architecture; inspect the measured region/depth/transition failure mode first.'}
"""; (output/"COARSE_TO_FINE_REPORT.md").write_text(text,encoding="utf-8")
def run(output:Path)->int:
    status={"stage":"training","current_epoch":0,"best_epoch":0,"best_validation_total_loss":None,"elapsed_seconds":0.,"peak_memory_bytes":0,"completed":False,"failed":False,"failure_reason":None,"gpu_name":get_gpu_identity()["name"],"tmux_session_name":"stage2-ctf-v0","launch_command":os.environ.get("CTF_LAUNCH_COMMAND","unknown")}; atomic_json(output/"run_status.json",status)
    def cb(row:dict[str,Any]): status.update(current_epoch=int(row["epoch"]),best_epoch=int(row["best_epoch"]),best_validation_total_loss=float(row["best_validation_total_loss"]),elapsed_seconds=float(row["elapsed_seconds"]),peak_memory_bytes=int(row["peak_gpu_memory_bytes"]));atomic_json(output/"run_status.json",status)
    try:
      result=train(default_config(str(output)),cb); status.update(stage="validation",current_epoch=result["completed_epoch"]);atomic_json(output/"run_status.json",status); evaluation=evaluate(output); over=json.loads((output/"tests"/"pedal_rich_overfit.json").read_text()); report(output,result|evaluation,over); status.update(stage="complete",completed=True,elapsed_seconds=result["elapsed_seconds"]);atomic_json(output/"run_status.json",status);return 0
    except BaseException as exc: status.update(stage="failed",failed=True,failure_reason=f"{type(exc).__name__}: {exc}");atomic_json(output/"run_status.json",status);raise
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True);p.add_argument("--overfit",action="store_true");a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/"tests").mkdir(exist_ok=True);return (overfit(out) and 0) if a.overfit else run(out)
if __name__=="__main__":raise SystemExit(main())
