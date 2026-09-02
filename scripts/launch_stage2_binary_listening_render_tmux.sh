#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="stage2-binary-listening-render-v0"
READY_CHANNEL="stage2-binary-listening-render-v0-ready"
CONTAINER_NAME="ilkyun-marg-pedaling-dev"
OUTPUT_ROOT="/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling/analysis/stage2_binary_v0/listening_render_v0"
RUNNER_PATTERN="[r]un_stage2_binary_listening_render.py"

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing duplicate tmux session: $SESSION_NAME" >&2
    exit 2
fi
if pgrep -f "$RUNNER_PATTERN" >/dev/null; then
    echo "refusing duplicate listening-render runner process" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "refusing to overwrite listening-render output: $OUTPUT_ROOT" >&2
    exit 2
fi
if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "required existing container is not running: $CONTAINER_NAME" >&2
    exit 2
fi

PIPELINE_COMMAND="set -o pipefail; docker exec $CONTAINER_NAME bash -lc 'cd /workspace/project && exec python scripts/run_stage2_binary_listening_render.py' 2>&1 | awk -v channel='$READY_CHANNEL' '{ print; fflush(); if (!ready && index(\$0, \"INITIAL_READY\")) { system(\"tmux wait-for -S \" channel); ready=1 } } END { if (!ready) system(\"tmux wait-for -S \" channel) }'"
tmux new-session -d -s "$SESSION_NAME" "bash -lc $(printf '%q' "$PIPELINE_COMMAND")"
tmux set-option -t "$SESSION_NAME" remain-on-exit off

printf 'session=%s\nready_channel=%s\noutput_root=%s\n' \
    "$SESSION_NAME" "$READY_CHANNEL" "$OUTPUT_ROOT"
