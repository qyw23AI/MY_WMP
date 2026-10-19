#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/qyw/miniconda3/envs/wmp_lab/bin/python isaaclab_migration/scripts/preview_terrains.py "$@"
