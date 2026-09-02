#!/usr/bin/env bash
set -euo pipefail

session_name="stage2-enc-only-loss-phase3-v0"
config_path="/workspace/project/configs/stage2_encoder_only_4class_loss_phase3_v0.json"

if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 1
fi

tmux new-session -d -s "${session_name}" \
  "docker exec ilkyun-marg-pedaling-dev bash -lc 'cd /workspace/project && python scripts/run_stage2_encoder_only_loss_phase3_v0.py --config ${config_path} --overnight'"

echo "started tmux session: ${session_name}"
