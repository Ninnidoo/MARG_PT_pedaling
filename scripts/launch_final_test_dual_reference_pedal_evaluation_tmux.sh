#!/usr/bin/env bash
set -euo pipefail

session_name="stage2-final-test-dual-reference-eval"
container_name="ilkyun-marg-pedaling-dev"
project_dir="/workspace/project"
output_dir="analysis/final_test_dual_reference_pedal_evaluation_v0"

if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "session already exists: ${session_name}" >&2
    exit 2
fi

tmux new-session -d -s "${session_name}" \
    "exec docker exec -u $(id -u):$(id -g) -e CUDA_VISIBLE_DEVICES= ${container_name} bash -lc 'cd ${project_dir} && python scripts/run_final_test_dual_reference_pedal_evaluation.py --output-root ${output_dir} --execute'"

echo "started ${session_name}"
