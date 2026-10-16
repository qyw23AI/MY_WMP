"""Create a GIF and curve from actual captured training/playback data."""
from pathlib import Path
import argparse
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
frames=[]
for f in sorted(a.directory.glob('frame_*.png')):
    im=Image.open(f).convert('RGB');im.thumbnail((960,540))
    d=a.directory/('depth_'+f.stem.split('_')[1]+'.png')
    if d.exists():
        dep=Image.open(d).resize((128,128),Image.Resampling.NEAREST).convert('RGB');im.paste(dep,(im.width-138,10));ImageDraw.Draw(im).text((im.width-138,140),'Depth 0-2 m',fill='white')
    frames.append(im)
if not frames:raise SystemExit('No actual viewport frames found')
frames[0].save(a.directory/'preview.gif',save_all=True,append_images=frames[1:],duration=400,loop=0)
print(a.directory/'preview.gif',len(frames))
