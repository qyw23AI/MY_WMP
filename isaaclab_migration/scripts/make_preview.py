"""Create a GIF and curve from actual captured training/playback data."""
from pathlib import Path
import argparse
from bisect import bisect_left
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--frame_steps',type=int,default=20);p.add_argument('--fps',type=int,default=20);p.add_argument('--mp4_only',action='store_true');a=p.parse_args()
if a.fps<1 or a.fps>50: p.error('FPS must be between 1 and 50')
frames=[]
files=sorted(a.directory.glob('frame_*.png'))
steps=[int(f.stem.split('_')[-1]) for f in files]
for f in files:
    im=Image.open(f).convert('RGB');im.thumbnail((960,540))
    d=a.directory/('depth_'+f.stem.split('_')[1]+'.png')
    if d.exists():
        dep=Image.open(d).resize((128,128),Image.Resampling.NEAREST).convert('RGB');im.paste(dep,(im.width-138,10));ImageDraw.Draw(im).text((im.width-138,140),'Depth 0-2 m',fill='white')
    frames.append(im)
if not frames:raise SystemExit('No actual viewport frames found')
# The simulation/control loop is 50 Hz. Select a source frame at each output
# timestamp so changing encoded FPS never speeds up or slows down the robot.
output_count=steps[-1]*a.fps//50
timeline=[frames[min(bisect_left(steps, (i+1)*50/a.fps), len(frames)-1)] for i in range(output_count)]
if not a.mp4_only:
    timeline[0].save(a.directory/'preview.gif',save_all=True,append_images=timeline[1:],duration=1000/a.fps,loop=0)
with imageio.get_writer(a.directory/'preview.mp4',fps=a.fps,codec='libx264',quality=8,macro_block_size=2) as video:
    for frame in timeline:video.append_data(np.asarray(frame))
print(a.directory/'preview.mp4',len(frames),'source frames,',len(timeline),'output frames,',len(timeline)/a.fps,'simulated seconds')
