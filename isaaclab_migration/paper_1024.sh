#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 1024 robots with the paper's 25% camera coverage and unchanged training
# network, AMP, world-model, and depth-predictor hyperparameters.
exec ./isaaclab_migration/run.sh \
  --num_envs 1024 --camera_envs 256 --camera_backend raycast --headless \
  --terrain mixed --episode_seconds 20 --rollout_steps 24 \
  --wm_batch 16 --wm_length 64 --wm_updates 10 --wm_start 10000 \
  --amp_preload 2000000 --amp_replay_size 1000000 \
  --depth_batch 1024 --depth_iters 1000 \
  --reward_schedule_scale 4 --save_interval 1000 \
  --iterations 40000 \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/paper_1024 "$@"
