#!/usr/bin/env bash
set -euo pipefail

repo=/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling
container=ilkyun-marg-pedaling-dev
session=stage2_binary_canonical_v1
ready_channel=stage2_binary_canonical_v1_ready
go_channel=stage2_binary_canonical_v1_go
log=/workspace/project/analysis/stage2_binary_canonical_v1/canonical_validation.log

if tmux has-session -t "$session" 2>/dev/null; then
  echo "duplicate tmux session exists: $session" >&2
  exit 1
fi

if docker top "$container" | grep -F 'run_stage2_binary_canonical_validation_v1.py' | grep -v grep >/dev/null; then
  echo "duplicate canonical validation runner exists in $container" >&2
  exit 1
fi

tmux new-session -d -s "$session" \
  "tmux wait-for '$go_channel'; exec docker exec '$container' bash -lc 'cd /workspace/project && exec python -u scripts/run_stage2_binary_canonical_validation_v1.py 2>&1 | tee -a $log'"

tmux pipe-pane -t "$session" -o \
  "awk '/FIRST_INFERENCE_ACTIVE|RUNNER_FAILED/ { system(\"tmux wait-for -S $ready_channel\"); exit }'"
tmux wait-for -S "$go_channel"
tmux wait-for "$ready_channel"

status=$(docker exec "$container" python -c \
  'import json; print(json.load(open("/workspace/project/analysis/stage2_binary_canonical_v1/run_status.json"))["state"])')
if [[ "$status" != "first_inference_active" ]]; then
  echo "runner did not reach first inference: state=$status" >&2
  exit 1
fi

echo "session=$session status=$status log=$repo/analysis/stage2_binary_canonical_v1/canonical_validation.log"
