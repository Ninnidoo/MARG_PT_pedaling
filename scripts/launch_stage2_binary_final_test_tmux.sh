#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="stage2-binary-final-test-v0"
READY_CHANNEL="stage2-binary-final-test-v0-ready"
CONTAINER_NAME="ilkyun-marg-pedaling-dev"
OUTPUT_ROOT="/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling/analysis/stage2_binary_v0/test_eval_v0"
RESUME_ARGUMENT=""
if [[ "${1:-}" == "--resume-after-stage1-failure" ]]; then
    RESUME_ARGUMENT="--resume-after-stage1-failure"
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--resume-after-stage1-failure]" >&2
    exit 2
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing duplicate tmux session: $SESSION_NAME" >&2
    exit 2
fi
if pgrep -f '[r]un_stage2_binary_final_test.py' >/dev/null; then
    echo "refusing duplicate final-test runner process" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" && -z "$RESUME_ARGUMENT" ]]; then
    echo "refusing to overwrite final-test output: $OUTPUT_ROOT" >&2
    exit 2
fi
if [[ ! -e "$OUTPUT_ROOT" && -n "$RESUME_ARGUMENT" ]]; then
    echo "resume requested but final-test output does not exist: $OUTPUT_ROOT" >&2
    exit 2
fi
if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "required existing container is not running: $CONTAINER_NAME" >&2
    exit 2
fi

PIPELINE_COMMAND="set -o pipefail; docker exec $CONTAINER_NAME bash -lc 'cd /workspace/project && exec python scripts/run_stage2_binary_final_test.py $RESUME_ARGUMENT' 2>&1 | awk -v channel='$READY_CHANNEL' '{ print; fflush(); if (!ready && index(\$0, \"INITIAL_READY\")) { system(\"tmux wait-for -S \" channel); ready=1 } } END { if (!ready) system(\"tmux wait-for -S \" channel) }'"
tmux new-session -d -s "$SESSION_NAME" "bash -lc $(printf '%q' "$PIPELINE_COMMAND")"
tmux set-option -t "$SESSION_NAME" remain-on-exit off

printf 'session=%s\nready_channel=%s\noutput_root=%s\n' \
    "$SESSION_NAME" "$READY_CHANNEL" "$OUTPUT_ROOT"
