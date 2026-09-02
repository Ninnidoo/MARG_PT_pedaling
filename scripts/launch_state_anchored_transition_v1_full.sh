#!/usr/bin/env bash
set -uo pipefail

PROJECT_HOST=/home/intern_2026_summer/yimilkyun/MARG_PT_pedaling
PROJECT_CONTAINER=/workspace/project
CONTAINER=ilkyun-marg-pedaling-dev
RUN_REL=analysis/stage2_state_anchored_transition_v1_full_v0
DRIVER=scripts/run_state_anchored_transition_v1_full.py
STATUS="$PROJECT_HOST/$RUN_REL/run_status.json"
RESUME="$PROJECT_CONTAINER/$RUN_REL/resume_last_train_state.pt"

mkdir -p "$PROJECT_HOST/$RUN_REL"

echo "$(date --iso-8601=seconds) TMUX_DRIVER_START container=$CONTAINER logical_cuda=0"
docker exec \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc \
  "cd '$PROJECT_CONTAINER' && set -o pipefail && python '$DRIVER' full 2>&1 | tee -a '$PROJECT_CONTAINER/$RUN_REL/tmux_driver.log'"
code=$?

if [[ $code -ne 0 ]] && [[ -f "$STATUS" ]] && grep -q '"validation_pending": true' "$STATUS"; then
  echo "$(date --iso-8601=seconds) TMUX_DRIVER_RESUME_PENDING_VALIDATION checkpoint=$RESUME"
  docker exec \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONUNBUFFERED=1 \
    "$CONTAINER" bash -lc \
    "cd '$PROJECT_CONTAINER' && set -o pipefail && python '$DRIVER' resume --resume '$RESUME' 2>&1 | tee -a '$PROJECT_CONTAINER/$RUN_REL/tmux_driver.log'"
  code=$?
fi

echo "$(date --iso-8601=seconds) TMUX_DRIVER_EXIT code=$code"
exit "$code"
