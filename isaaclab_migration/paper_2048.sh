#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 2048 robots, 512 geometric depth cameras: same 25% coverage as the paper.
# 48 steps per robot match the paper's 4096*24 PPO samples per iteration.
# All model, AMP and predictor training sizes below are from the original A1 run.
exec ./isaaclab_migration/run.sh \
  --num_envs 2048 --camera_envs 512 --camera_backend raycast --headless \
  --terrain mixed --episode_seconds 20 --rollout_steps 48 \
  --wm_batch 16 --wm_length 64 --wm_updates 10 --wm_start 10000 \
  --amp_preload 2000000 --amp_replay_size 1000000 \
  --depth_batch 1024 --depth_iters 1000 \
  --reward_schedule_scale 1 --save_interval 1000 \
  --iterations 10000 \
  --log_dir /home/qyw/WMP/isaaclab_migration/logs/paper_2048 "$@"
