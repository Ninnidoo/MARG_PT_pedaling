#!/usr/bin/env bash
set -euo pipefail
session_name="final-test-custom-pedal-metrics"
container_name="ilkyun-marg-pedaling-dev"
project_dir="/workspace/project"
output_dir="analysis/final_test_custom_pedal_metrics_v0"
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "session already exists: ${session_name}" >&2
  exit 2
fi
tmux new-session -d -s "${session_name}" "exec docker exec -u $(id -u):$(id -g) -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib-custom-pedal-metrics ${container_name} bash -lc 'cd ${project_dir} && python scripts/run_final_test_custom_pedal_metrics.py --output-root ${output_dir} --execute'"
echo "started ${session_name}"
