"""Manually record six real-time course clips for each saved 300-round checkpoint."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
TERRAINS = ('plum_stones', 'tire_array', 'high_wall')
LEVELS = (0, 9)
FPS = 20
STEPS = 1000


def valid_clip(directory):
    video = directory / 'preview.mp4'
    state = directory / 'log' / 'state.csv'
    if not video.is_file() or not state.is_file():
        return False
    try:
        reader = imageio.get_reader(video)
        metadata = reader.get_meta_data()
        reader.close()
        with state.open(newline='') as stream:
            steps = sum(1 for _ in csv.DictReader(stream))
        return metadata['fps'] == FPS and abs(metadata['duration'] - 20.) < .05 and steps == STEPS
    except (OSError, ValueError, KeyError):
        return False


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path, help='Training directory containing model_*.pt')
parser.add_argument('rounds', nargs='*', type=int, help='Cumulative rounds to record; default: all saved 300-round checkpoints')
parser.add_argument('--list', action='store_true', help='Show available checkpoints without starting Isaac Sim')
args = parser.parse_args()
run_dir = args.run_dir.resolve()
available = {int(path.stem[6:]) - 10000: path for path in run_dir.glob('model_*.pt')
             if path.stem[6:].isdigit() and int(path.stem[6:]) > 10000
             and (int(path.stem[6:]) - 10000) % 300 == 0}
rounds = sorted(args.rounds or available)
if not rounds:
    parser.error('No saved 300-round checkpoints found')
for training_round in rounds:
    if training_round not in available:
        parser.error(f'Round {training_round} has no matching model_{10000 + training_round}.pt')
if args.list:
    for training_round in rounds:
        print(training_round, available[training_round])
    raise SystemExit(0)
if subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
    parser.error('NVIDIA driver unavailable; cannot record GUI replay')

index_path = run_dir / 'checkpoint_videos' / 'index.json'
index = json.loads(index_path.read_text()) if index_path.exists() else []
for training_round in rounds:
    checkpoint = available[training_round]
    output = run_dir / 'checkpoint_videos' / f'round_{training_round:06d}'
    for column, terrain in enumerate(TERRAINS):
        for level in LEVELS:
            clip = output / f'{terrain}_L{level}'
            clip.mkdir(parents=True, exist_ok=True)
            video = clip / 'preview.mp4'
            if not valid_clip(clip):
                for stale in clip.glob('frame_*.png'):
                    stale.unlink()
                video.unlink(missing_ok=True)
                command = [str(ROOT / 'isaaclab_migration/run.sh'),
                           '--play', '--terrain', 'robocon', '--compact_courses',
                           '--competition_stage', 'eval', '--num_envs', '4', '--camera_envs', '1',
                           '--episode_seconds', '45', '--play_steps', str(STEPS),
                           '--capture_fps', str(FPS), '--follow_camera',
                           '--eval_terrain_col', str(column), '--eval_terrain_level', str(level),
                           '--fixed_forward_speed', '.2', '--checkpoint', str(checkpoint),
                           '--capture_dir', str(clip), '--log_dir', str(clip / 'log')]
                print(f'Recording round {training_round}: {terrain} L{level}', flush=True)
                with (clip / 'run.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([str(PYTHON), str(ROOT / 'isaaclab_migration/scripts/make_preview.py'),
                                str(clip), '--fps', str(FPS), '--mp4_only'], cwd=ROOT, check=True)
                if not valid_clip(clip) or len(list(clip.glob('frame_*.png'))) != FPS * 20:
                    raise RuntimeError(f'Incomplete replay: {clip}; inspect run.log')
                for frame in clip.glob('frame_*.png'):
                    frame.unlink()
            else:
                print(f'Already complete: round {training_round}: {terrain} L{level}', flush=True)
            entry = {'round': training_round, 'checkpoint': checkpoint.name, 'terrain': terrain,
                     'level': level, 'fps': FPS, 'duration_seconds': 20,
                     'camera_eye': [-3., 0., 1.65], 'camera_lookat': [1.2, 0., .25],
                     'video': str(video.relative_to(run_dir))}
            index = [old for old in index if not (old['round'] == training_round and
                     old['terrain'] == terrain and old['level'] == level)]
            index.append(entry)
            index.sort(key=lambda item: (item['round'], TERRAINS.index(item['terrain']), item['level']))
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
print(f'Completed {len(index)} clips. Index: {index_path}', flush=True)
