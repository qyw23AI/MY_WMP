#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

checkpoint=/home/qyw/WMP/isaaclab_migration/logs/paper_2048/model_10000.pt
log_dir="${1:-/home/qyw/WMP/isaaclab_migration/logs/robocon_2048_no_rails_from_10000_20261019}"
if [[ ! -f "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ -e "$log_dir/config.json" ]]; then
  echo "Refusing to overwrite an existing run: $log_dir" >&2
  exit 1
fi
mkdir -p "$log_dir"
echo "Starting at model_10000.pt: 2048 environments, 512 real depth cameras, no side rails."
./isaaclab_migration/run.sh \
  --terrain robocon --compact_courses --competition_stage learn \
  --num_envs 2048 --camera_envs 512 --episode_seconds 45 \
  --iterations 10000 --rollout_steps 48 \
  --checkpoint "$checkpoint" --runtime_config --reset_course_stats \
  --wm_batch 4 --wm_length 64 --wm_updates 10 --wm_start 256 \
  --depth_batch 32 --depth_iters 1 \
  --amp_preload 100000 --amp_replay_size 100000 \
  --save_interval 300 --progress_interval 300 --headless \
  --require_wm_updates --log_dir "$log_dir" \
  2>&1 | tee "$log_dir/console.log"
