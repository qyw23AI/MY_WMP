#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:-/home/qyw/WMP/isaaclab_migration/logs/robocon_2048_no_rails_from_10000_20261019}"
if [[ $# -gt 0 && "$1" != -* ]]; then shift; else run_dir=/home/qyw/WMP/isaaclab_migration/logs/robocon_2048_no_rails_from_10000_20261019; fi
exec /home/qyw/miniconda3/envs/wmp_lab/bin/python \
  isaaclab_migration/scripts/record_course_checkpoints.py "$run_dir" "$@"
