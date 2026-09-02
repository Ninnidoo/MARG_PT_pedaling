#!/usr/bin/env bash
set -euo pipefail

session_name="stage2-4class-arch-v0"
container_name="ilkyun-marg-pedaling-dev"
config_path="/workspace/project/configs/stage2_4class_architecture_v0.json"
output_root="/workspace/project/analysis/stage2_4class_architecture_v0"
runner_pattern="[r]un_stage2_4class_architecture_v0.py"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "Refusing duplicate tmux session: $session_name" >&2
  exit 1
fi
if docker exec "$container_name" bash -lc "pgrep -af '$runner_pattern'" >/dev/null; then
  echo "Refusing duplicate four-class training process" >&2
  docker exec "$container_name" bash -lc "pgrep -af '$runner_pattern'"
  exit 1
fi
if docker exec "$container_name" test -e "$output_root/run_status.json"; then
  echo "Refusing duplicate canonical output root: $output_root" >&2
  exit 1
fi

docker exec "$container_name" nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv,noheader

tmux new-session -d -s "$session_name"   "docker exec $container_name bash -lc 'cd /workspace/project && exec python scripts/run_stage2_4class_architecture_v0.py --config $config_path --sequential'"

echo "Started tmux session: $session_name"
