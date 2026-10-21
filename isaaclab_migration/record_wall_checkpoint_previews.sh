#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if ! nvidia-smi >/dev/null 2>&1; then
  echo 'GPU driver unavailable; replay capture cannot start.' >&2
  exit 1
fi
root=/home/qyw/WMP/isaaclab_migration/logs/robocon_2048_paper25_guardrails_headless_10600_20261019
for round in 900 1200; do
  iteration=$((10000+round))
  checkpoint="$root/model_${iteration}.pt"
  output="$root/checkpoint_videos/wall_L9_${round}_realtime20"
  mkdir -p "$output"
  ./isaaclab_migration/run.sh \
    --play --terrain robocon --compact_courses --competition_stage eval \
    --eval_terrain_col 2 --eval_terrain_level 9 --fixed_forward_speed 0.2 \
    --num_envs 4 --camera_envs 1 --play_steps 1000 \
    --capture_fps 20 --follow_camera \
    --checkpoint "$checkpoint" --capture_dir "$output" --log_dir "$output/log" \
    > "$output/run.log" 2>&1
  /home/qyw/miniconda3/envs/wmp_lab/bin/python isaaclab_migration/scripts/make_preview.py "$output" --fps 20 --mp4_only
done
echo "900 and 1200 high-wall L9 replay videos are ready."
