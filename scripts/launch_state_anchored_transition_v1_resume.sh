#!/usr/bin/env bash
set -uo pipefail

PROJECT_CONTAINER=/workspace/project
CONTAINER=ilkyun-marg-pedaling-dev
RUN_REL=analysis/stage2_state_anchored_transition_v1_full_v0
DRIVER=scripts/run_state_anchored_transition_v1_full.py
RESUME="$PROJECT_CONTAINER/$RUN_REL/resume_last_train_state.pt"
LOG="$PROJECT_CONTAINER/$RUN_REL/tmux_driver.log"

docker exec \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc \
  "cd '$PROJECT_CONTAINER' && \
   echo \"\$(date --iso-8601=seconds) TMUX_DRIVER_RESUME_START checkpoint=$RESUME logical_cuda=0\" | tee -a '$LOG' && \
   set -o pipefail && \
   python '$DRIVER' resume --resume '$RESUME' 2>&1 | tee -a '$LOG'; \
   code=\${PIPESTATUS[0]}; \
   echo \"\$(date --iso-8601=seconds) TMUX_DRIVER_RESUME_EXIT code=\$code\" | tee -a '$LOG'; \
   exit \$code"
