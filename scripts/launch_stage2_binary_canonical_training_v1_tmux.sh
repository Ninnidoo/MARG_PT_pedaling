#!/usr/bin/env bash
set -euo pipefail

repo=/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling
container=ilkyun-marg-pedaling-dev
session=stage2-binary-canonical-v1-train
ready_channel=stage2_binary_canonical_v1_train_ready
go_channel=stage2_binary_canonical_v1_train_go
runner=run_stage2_binary_canonical_training_v1.py
status=/workspace/project/analysis/stage2_binary_canonical_v1/training_run_status.json
pipeline_log=/workspace/project/analysis/stage2_binary_canonical_v1/canonical_binary_training.log
assigned_gpu_uuid=GPU-6982dbee-fbaf-f359-d7ef-a22d0e83400b

if [[ "$(pwd)" != "$repo" ]]; then
  echo "launch from the project root: $repo" >&2
  exit 1
fi
if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -Fx true >/dev/null; then
  echo "required container is not running: $container" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "duplicate tmux session exists: $session" >&2
  exit 1
fi
if docker top "$container" | grep -F "$runner" | grep -v grep >/dev/null; then
  echo "duplicate canonical training runner exists in $container" >&2
  exit 1
fi
if [[ -e analysis/stage2_binary_canonical_v1/train_v0 || -e analysis/stage2_binary_canonical_v1/training_run_status.json ]]; then
  echo "canonical training output/status already exists; refusing overwrite" >&2
  exit 1
fi

gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | awk -F, -v uuid="$assigned_gpu_uuid" '$1 == uuid {print}')
if [[ -n "$gpu_processes" ]]; then
  echo "assigned GPU already has compute processes:" >&2
  echo "$gpu_processes" >&2
  exit 1
fi
docker exec "$container" python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))' >/dev/null

tmux new-session -d -s "$session" \
  "tmux wait-for '$go_channel'; exec docker exec '$container' bash -lc 'cd /workspace/project && exec python -u scripts/$runner --config configs/stage2_binary_canonical_training_v1.json --confirm-canonical-training 2>&1 | tee -a $pipeline_log'"

tmux pipe-pane -t "$session" -o \
  "awk '/FIRST_TRAIN_STEP_PASS architecture=independent_4x2|RUNNER_FAILED/ { system(\"tmux wait-for -S $ready_channel\"); exit }'"
tmux wait-for -S "$go_channel"
tmux wait-for "$ready_channel"

runner_status=$(docker exec "$container" python -c \
  "import json; print(json.load(open('$status'))['status'])")
runner_state=$(docker exec "$container" python -c \
  "import json; print(json.load(open('$status'))['state'])")
if [[ "$runner_status" != "running" || "$runner_state" != "independent_first_training_step_pass" ]]; then
  echo "runner failed initial confirmation: status=$runner_status state=$runner_state" >&2
  exit 1
fi

echo "session=$session status=$runner_status state=$runner_state log=$repo/analysis/stage2_binary_canonical_v1/canonical_binary_training.log"
