"""Full controlled trainer for the Stage 2 coarse-to-fine v0 head."""

from __future__ import annotations

import argparse, base64, csv, json, math, os, time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .coarse_to_fine import Stage2CoarseToFineModel, coarse_to_fine_loss, decode_pedals, pedal_regions
from .dataset import Stage2PedalDataset
from .train import (EarlyStopping, GracefulStop, PINNED_PT_COMMIT, atomic_torch_save, build_best_checkpoint, build_last_checkpoint, deterministic_train_order, get_gpu_identity, make_loader, prepare_output_directory, restore_training_state, write_configuration)
from .train_ordinal import parameter_hash
from .training import _gradient_norm, _parameters_with_gradients, _validate_finite_gradients, create_grad_scaler, move_batch_to_device, set_deterministic_seed


METRICS = ["total_loss", "region_ce_loss", "depth_smooth_l1_loss", "intermediate_target_count", "region_accuracy", "pedal_token_accuracy", "exact_note_accuracy", "pedal_value_mae", "valid_target_count", "valid_note_count", "encoder_gradient_norm", "head_gradient_norm", "total_gradient_norm", "encoder_learning_rate", "head_learning_rate", "amp_scale"]
LOSS_CONFIG = {"region_loss":"unweighted_cross_entropy","depth_loss":"smooth_l1","smooth_l1_beta":0.05,"lambda_depth":1.0,"depth_target":"(target-1)/125 for true intermediate only"}


def build_optimizer(model: Stage2CoarseToFineModel, encoder_lr: float, head_lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": encoder_lr, "weight_decay": weight_decay, "group_name":"encoder"},
        {"params": list(model.region_heads.parameters()) + list(model.depth_heads.parameters()), "lr": head_lr, "weight_decay": weight_decay, "group_name":"heads"},
    ])


def _forward(model: torch.nn.Module, batch: Mapping[str, Any], amp: bool) -> Any:
    dtype = batch["input_ids"].device.type
    with torch.amp.autocast(device_type=dtype, dtype=torch.float16 if dtype == "cuda" else torch.bfloat16, enabled=amp and dtype == "cuda"):
        return model(batch["input_ids"], batch["token_attention_mask"], batch.get("note_mask"))


def _metrics(output: Any, targets: torch.Tensor, losses: Any, optimizer: torch.optim.Optimizer | None = None, gradients: tuple[float,float,float] = (0.,0.,0.), amp_scale: float = 1.) -> dict[str, float|int]:
    valid = targets != -100
    pred = decode_pedals(output.region_logits.detach(), output.depth_logits.detach())
    correct = (pred == targets) & valid
    note_valid = valid.any(-1)
    regions = pedal_regions(targets)
    reg_acc = ((output.region_logits.detach().argmax(-1) == regions) & valid).sum().item() / valid.sum().item()
    rates = {g.get("group_name"):float(g["lr"]) for g in optimizer.param_groups} if optimizer else {}
    return {"total_loss":float(losses.total.detach()),"region_ce_loss":float(losses.region_ce.detach()),"depth_smooth_l1_loss":float(losses.depth_smooth_l1.detach()),"intermediate_target_count":losses.intermediate_count,"region_accuracy":reg_acc,"pedal_token_accuracy":correct.sum().item()/valid.sum().item(),"exact_note_accuracy":(((pred==targets)|~valid).all(-1)&note_valid).sum().item()/note_valid.sum().item(),"pedal_value_mae":(pred[valid].float()-targets[valid].float()).abs().mean().item(),"valid_target_count":int(valid.sum()),"valid_note_count":int(note_valid.sum()),"encoder_gradient_norm":gradients[0],"head_gradient_norm":gradients[1],"total_gradient_norm":gradients[2],"encoder_learning_rate":rates.get("encoder",0.),"head_learning_rate":rates.get("heads",0.),"amp_scale":amp_scale}


def train_step(model: Stage2CoarseToFineModel, batch: Mapping[str,Any], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, amp: bool, max_grad_norm: float) -> dict[str,float|int]:
    model.train(); optimizer.zero_grad(set_to_none=True)
    output = _forward(model,batch,amp); losses=coarse_to_fine_loss(output.region_logits,output.depth_logits,batch["pedal_targets"])
    if not all(bool(torch.isfinite(x)) for x in losses[:3]): raise FloatingPointError("non-finite CTF loss")
    use=amp and batch["input_ids"].is_cuda
    (scaler.scale(losses.total).backward() if use else losses.total.backward())
    if use: scaler.unscale_(optimizer)
    enc=list(model.encoder.parameters()); heads=list(model.region_heads.parameters())+list(model.depth_heads.parameters())
    if not _parameters_with_gradients(enc): raise RuntimeError("encoder received no gradients")
    for prefix, modules in (("region",model.region_heads),("depth",model.depth_heads)):
        for i, head in enumerate(modules):
            if not _parameters_with_gradients(head.parameters()): raise RuntimeError(f"{prefix} head {i+1} received no gradients")
    allp=[p for p in model.parameters() if p.requires_grad]; _validate_finite_gradients(allp)
    norms=(_gradient_norm(enc),_gradient_norm(heads),float(torch.nn.utils.clip_grad_norm_(allp,max_grad_norm,error_if_nonfinite=True)))
    if use: scaler.step(optimizer); scaler.update()
    else: optimizer.step()
    return _metrics(output,batch["pedal_targets"],losses,optimizer,norms,float(scaler.get_scale()) if use else 1.)


@torch.no_grad()
def evaluation_step(model: Stage2CoarseToFineModel,batch:Mapping[str,Any],amp:bool)->dict[str,float|int]:
    model.eval(); output=_forward(model,batch,amp); losses=coarse_to_fine_loss(output.region_logits,output.depth_logits,batch["pedal_targets"])
    return _metrics(output,batch["pedal_targets"],losses)


def run_epoch(model: Stage2CoarseToFineModel, loader: Any, device: torch.device, training: bool, config:dict[str,Any], optimizer:Any=None, scaler:Any=None, step:int=0) -> tuple[dict[str,float|int],int]:
    sums={key:0. for key in METRICS}; targets=notes=0
    for cpu in loader:
        batch=move_batch_to_device(cpu,device)
        metrics=train_step(model,batch,optimizer,scaler,config["amp_enabled"],config["max_grad_norm"]) if training else evaluation_step(model,batch,config["amp_enabled"])
        if training: step+=1
        weight=int(metrics["valid_target_count"]); nweight=int(metrics["valid_note_count"]); targets+=weight; notes+=nweight
        for key,value in metrics.items():
            if key in {"valid_target_count","valid_note_count"}: continue
            sums[key]+=float(value)*(nweight if key=="exact_note_accuracy" else weight)
    result={key:(sums[key]/(notes if key=="exact_note_accuracy" else targets)) for key in sums}; result["intermediate_target_count"]=int(sums["intermediate_target_count"]); result.update(valid_target_count=targets,valid_note_count=notes)
    return result,step


def default_config(output_dir:str)->dict[str,Any]:
    return {"asap_root":"/workspace/public/ASAP/asap-dataset-v1.1","split_csv":"/workspace/project/analysis/stage2_encoder_only_v0/asap_split.csv","checkpoint_path":"/workspace/project/checkpoints/pianist_transformer","output_dir":output_dir,"batch_size":16,"max_epochs":20,"seed":20260710,"encoder_lr":1e-5,"head_lr":1e-4,"weight_decay":.01,"max_grad_norm":1.,"amp_enabled":True,"amp_init_scale":1024.,"pin_memory":True,"early_stopping_patience":4,"early_stopping_min_delta":.0001,"expected_gpu_uuid":"GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b","resume":None}


def train(config:dict[str,Any], callback:Callable[[dict[str,Any]],None]|None=None)->dict[str,Any]:
    out=Path(config["output_dir"]).resolve(); resume=Path(config["resume"]).resolve() if config.get("resume") else None
    if resume is None and out.exists():
        allowed={"tests","run_status.json","train.log"}
        if not {p.name for p in out.iterdir()}.issubset(allowed): raise FileExistsError(f"refusing to overwrite output directory: {out}")
    else: prepare_output_directory(out,resume,True)
    for key in ("asap_root","split_csv","checkpoint_path"): config[key]=str(Path(config[key]).resolve())
    config.update(output_dir=str(out),pianist_transformer_commit=PINNED_PT_COMMIT,loss_configuration=LOSS_CONFIG,architecture="official PT encoder + four Linear(768,3) region heads + four Linear(768,1) depth heads")
    config["project_git_commit"]=os.environ.get("PROJECT_GIT_COMMIT","unavailable"); config["project_git_dirty_status"]=base64.b64decode(os.environ.get("PROJECT_GIT_STATUS_B64","")).decode("utf-8") if os.environ.get("PROJECT_GIT_STATUS_B64") else "unavailable"
    set_deterministic_seed(config["seed"]); gpu=get_gpu_identity()
    if gpu["uuid"]!=config["expected_gpu_uuid"]: raise RuntimeError("GPU UUID mismatch")
    trainset=Stage2PedalDataset(config["asap_root"],config["split_csv"],"train",512,256,cache_mode="preload"); valset=Stage2PedalDataset(config["asap_root"],config["split_csv"],"validation",512,256,cache_mode="preload")
    device=torch.device("cuda:0"); model=Stage2CoarseToFineModel.from_pretrained(config["checkpoint_path"],torch_dtype=torch.float32,attn_implementation="eager").to(device)
    config["encoder_initial_sha256"]=parameter_hash(model.encoder); config["region_head_initial_sha256"]=[parameter_hash(head) for head in model.region_heads]; config["depth_head_initial_sha256"]=[parameter_hash(head) for head in model.depth_heads]; config["project_git_status"]=os.environ.get("PROJECT_GIT_STATUS","unknown")
    if resume is None: write_configuration(out/"config.json",config)
    else:
        old=json.loads((out/"config.json").read_text());
        if old["loss_configuration"]!=LOSS_CONFIG: raise RuntimeError("incompatible resume loss config")
        config=old
    print("CTF RUN START",json.dumps(config,sort_keys=True),flush=True)
    opt=build_optimizer(model,config["encoder_lr"],config["head_lr"],config["weight_decay"]); scaler=create_grad_scaler(config["amp_enabled"],"cuda",config["amp_init_scale"]); early=EarlyStopping(config["early_stopping_patience"],config["early_stopping_min_delta"]); start,step=1,0
    if resume is not None: start,step=restore_training_state(torch.load(resume,map_location=device,weights_only=False),model,opt,scaler,early)
    val_loader=make_loader(valset,config["batch_size"],config["pin_memory"]); fields=["epoch","global_optimizer_step","epoch_seconds","peak_gpu_memory_bytes"]+[f"{p}_{m}" for p in ("train","validation") for m in METRICS]; handle=(out/"metrics.csv").open("a" if resume else "w",newline="",encoding="utf-8"); writer=csv.DictWriter(handle,fieldnames=fields); 
    if not resume: writer.writeheader()
    started=time.perf_counter(); completed=start-1
    try:
        for epoch in range(start,config["max_epochs"]+1):
            torch.cuda.reset_peak_memory_stats(device); began=time.perf_counter(); order=deterministic_train_order(len(trainset),config["seed"],epoch); train_loader=make_loader(trainset,config["batch_size"],config["pin_memory"],order)
            print(f"epoch {epoch} start",flush=True); tr,step=run_epoch(model,train_loader,device,True,config,opt,scaler,step); va,step=run_epoch(model,val_loader,device,False,config,step=step); completed=epoch; improved,stop=early.update(va["total_loss"],epoch); row={"epoch":epoch,"global_optimizer_step":step,"epoch_seconds":time.perf_counter()-began,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device))}; row.update(**{f"train_{k}":v for k,v in tr.items()},**{f"validation_{k}":v for k,v in va.items()}); writer.writerow(row); handle.flush(); os.fsync(handle.fileno()); atomic_torch_save(build_last_checkpoint(model,opt,scaler,completed,step,early,config),out/"last.pt");
            if improved: atomic_torch_save(build_best_checkpoint(model,config,early.best_epoch,early.best_loss),out/"best.pt")
            if callback: callback({**row,"best_epoch":early.best_epoch,"best_validation_total_loss":early.best_loss,"elapsed_seconds":time.perf_counter()-started})
            print(f"epoch {epoch} complete validation={va} best={early.best_loss}",flush=True)
            if stop: break
    finally: handle.close()
    return {"completed_epoch":completed,"best_epoch":early.best_epoch,"best_validation_total_loss":early.best_loss,"global_step":step,"elapsed_seconds":time.perf_counter()-started}


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",required=True); parser.add_argument("--resume"); args=parser.parse_args(); config=default_config(args.output_dir); config["resume"]=args.resume; train(config); return 0
if __name__=="__main__": raise SystemExit(main())
