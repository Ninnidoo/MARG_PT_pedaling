#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="stage2-final-test-encoder-only-trio"
READY_CHANNEL="stage2-final-test-encoder-only-trio-ready"
CONTAINER_NAME="ilkyun-marg-pedaling-dev"
PROJECT_HOST="/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling"
PROJECT_CONTAINER="/workspace/project"
OUTPUT_ROOT="$PROJECT_HOST/analysis/final_test_encoder_only_trio_inference_v0"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "refusing duplicate tmux session: $SESSION_NAME" >&2
    exit 2
fi
if pgrep -f '[r]un_final_test_encoder_only_trio_inference.py' >/dev/null; then
    echo "refusing duplicate final-test encoder-only trio process" >&2
    exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "refusing to overwrite final-test encoder-only trio provenance: $OUTPUT_ROOT" >&2
    exit 2
fi
if find "$PROJECT_HOST/outputs/midi" -maxdepth 1 -type f \
    \( -name '*_encoder_only_weighted_ce.mid' \
       -o -name '*_encoder_only_ntl_was.mid' \
       -o -name '*_encoder_only_huber_aux_ce.mid' \) \
    -print -quit | grep -q .; then
    echo "refusing to overwrite existing encoder-only trio output MIDI" >&2
    exit 2
fi
if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    echo "required existing container is not running: $CONTAINER_NAME" >&2
    exit 2
fi

PIPELINE_COMMAND="set -o pipefail; docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONUNBUFFERED=1 $CONTAINER_NAME bash -lc 'cd $PROJECT_CONTAINER && exec python scripts/run_final_test_encoder_only_trio_inference.py' 2>&1 | awk -v channel='$READY_CHANNEL' '{ print; fflush(); if (!ready && index(\$0, \"INITIAL_READY\")) { system(\"tmux wait-for -S \" channel); ready=1 } } END { if (!ready) system(\"tmux wait-for -S \" channel) }'"
tmux new-session -d -s "$SESSION_NAME" "bash -lc $(printf '%q' "$PIPELINE_COMMAND")"
tmux set-option -t "$SESSION_NAME" remain-on-exit on

printf 'session=%s\nready_channel=%s\noutput_root=%s\nstatus=%s\nlog=%s\n' \
    "$SESSION_NAME" \
    "$READY_CHANNEL" \
    "$OUTPUT_ROOT" \
    "$OUTPUT_ROOT/run_status.json" \
    "$OUTPUT_ROOT/inference.log"
