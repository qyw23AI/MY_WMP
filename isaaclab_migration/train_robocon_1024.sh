#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
log_dir="${1:-/home/qyw/WMP/isaaclab_migration/logs/robocon_1024_20261019}"
if [[ -e "$log_dir/config.json" ]]; then
  echo "Refusing to overwrite an existing run: $log_dir" >&2
  exit 1
fi
mkdir -p "$log_dir"
exec ./isaaclab_migration/run.sh --terrain robocon --compact_courses \
 --competition_stage learn --num_envs 1024 --camera_envs 256 \
 --iterations "${ITERATIONS:-10000}" --rollout_steps 48 --episode_seconds 45 \
 --checkpoint /home/qyw/WMP/isaaclab_migration/logs/paper_2048/model_10000.pt \
 --runtime_config --wm_batch 4 --wm_length 64 --wm_updates 10 --wm_start 256 \
 --depth_batch 32 --depth_iters 1 --amp_preload 100000 --amp_replay_size 100000 \
 --save_interval 300 --video_interval 300 --video_steps 1000 --follow_camera \
 --require_wm_updates --log_dir "$log_dir" > "$log_dir/console.log" 2>&1
