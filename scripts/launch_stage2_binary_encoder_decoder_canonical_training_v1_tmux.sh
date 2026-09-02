#!/usr/bin/env bash
set -euo pipefail

repo=/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling
container=ilkyun-marg-pedaling-dev
session=stage2-binary-encdec-canonical-v1
runner=run_stage2_binary_encoder_decoder_canonical_training_v1.py
config=stage2_binary_encoder_decoder_canonical_training_v1.json
ready_channel=stage2_binary_encdec_canonical_v1_ready
go_channel=stage2_binary_encdec_canonical_v1_go
status=/workspace/project/analysis/stage2_binary_encoder_decoder_canonical_v1/train_v0/run_status.json
pipeline_log=/workspace/project/analysis/stage2_binary_encoder_decoder_canonical_v1/encoder_decoder_canonical_training.log

if [[ "$(pwd)" != "$repo" ]]; then
  echo "launch from project root: $repo" >&2
  exit 1
fi
if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -Fx true >/dev/null; then
  echo "required container is not running: $container" >&2
  exit 1
fi
if [[ "$(docker exec "$container" pwd)" != "/workspace/project" ]]; then
  echo "container working directory is not /workspace/project" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "duplicate tmux session exists: $session" >&2
  exit 1
fi
if docker top "$container" | grep -F "$runner" | grep -v grep >/dev/null; then
  echo "duplicate encoder-decoder training process exists" >&2
  exit 1
fi
if [[ -e analysis/stage2_binary_encoder_decoder_canonical_v1/train_v0 ]]; then
  echo "training output already exists; refusing overwrite" >&2
  exit 1
fi
if [[ -e analysis/stage2_binary_encoder_decoder_canonical_v1/encoder_decoder_canonical_training.log ]]; then
  echo "pipeline log already exists; refusing overwrite" >&2
  exit 1
fi

container_gpu_count=$(docker exec "$container" nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
if [[ "$container_gpu_count" -ne 1 ]]; then
  echo "container must expose exactly one GPU; found $container_gpu_count" >&2
  exit 1
fi
assigned_gpu_uuid=$(docker exec "$container" nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
assigned_gpu_name=$(docker exec "$container" nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | awk -F, -v uuid="$assigned_gpu_uuid" '$1 == uuid {print}')
if [[ -n "$gpu_processes" ]]; then
  echo "assigned GPU already has compute processes:" >&2
  echo "$gpu_processes" >&2
  exit 1
fi
docker exec "$container" python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1' >/dev/null

tmux new-session -d -s "$session" \
  "tmux wait-for '$go_channel'; exec docker exec '$container' bash -lc 'set -o pipefail; cd /workspace/project && exec python -u scripts/$runner --config configs/$config --confirm-full-training 2>&1 | tee -a $pipeline_log'"

tmux pipe-pane -t "$session" -o \
  "awk '/FIRST_OPTIMIZER_STEP_PASS|RUNNER_FAILED/ { system(\"tmux wait-for -S $ready_channel\"); exit }'"
tmux wait-for -S "$go_channel"
tmux wait-for "$ready_channel"

runner_status=$(docker exec "$container" python -c \
  "import json; print(json.load(open('$status'))['status'])")
runner_state=$(docker exec "$container" python -c \
  "import json; print(json.load(open('$status'))['state'])")
if [[ "$runner_status" != "running" || "$runner_state" != "first_optimizer_step_pass" ]]; then
  echo "runner failed initial confirmation: status=$runner_status state=$runner_state" >&2
  exit 1
fi

printf 'session=%s status=%s state=%s gpu_uuid=%s gpu_name=%s log=%s\n' \
  "$session" "$runner_status" "$runner_state" "$assigned_gpu_uuid" "$assigned_gpu_name" \
  "$repo/analysis/stage2_binary_encoder_decoder_canonical_v1/encoder_decoder_canonical_training.log"
