"""Legacy integer-grid terrain rules with Isaac Lab's supported mesh conversion.
Grid sizes/difficulty schedule deliberately remain those in the local project.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline

class SubTerrain:
    def __init__(self,name,width,length,vertical_scale,horizontal_scale):
        self.width=width; self.length=length
        self.vertical_scale=vertical_scale; self.horizontal_scale=horizontal_scale
        self.height_field_raw=np.zeros((width,length),dtype=np.int16)

def convert_heightfield_to_trimesh(height_field_raw,horizontal_scale,vertical_scale,slope_threshold=None):
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
    return convert_height_field_to_mesh(height_field_raw,horizontal_scale,vertical_scale,slope_threshold)

def pyramid_stairs_terrain(t, step_width, step_height, platform_size=1.):
    width=int(step_width/t.horizontal_scale); height=int(step_height/t.vertical_scale)
    platform=int(platform_size/t.horizontal_scale)
    x0=y0=0; x1=t.width; y1=t.length; h=0
    while x1-x0>platform and y1-y0>platform:
        x0+=width; y0+=width; x1-=width; y1-=width; h+=height
        t.height_field_raw[x0:x1,y0:y1]=h
    return t

def pyramid_sloped_terrain(t,slope=1.,platform_size=1.):
    x=np.arange(t.width); y=np.arange(t.length)
    xx=(t.width/2-np.abs(t.width/2-x))/(t.width/2)
    yy=(t.length/2-np.abs(t.length/2-y))/(t.length/2)
    max_h=int(slope*(t.horizontal_scale/t.vertical_scale)*(t.width/2))
    t.height_field_raw += (max_h*xx[:,None]*yy[None,:]).astype(np.int16)
    half=int(platform_size/t.horizontal_scale/2)
    h=t.height_field_raw[t.width//2-half,t.length//2-half]
    t.height_field_raw=np.clip(t.height_field_raw,min(h,0),max(h,0))
    return t

def random_uniform_terrain(t,min_height,max_height,step=1,downsampled_scale=None):
    scale=downsampled_scale or t.horizontal_scale
    heights=np.arange(int(min_height/t.vertical_scale),int(max_height/t.vertical_scale)+int(step/t.vertical_scale),int(step/t.vertical_scale))
    nx=int(t.width*t.horizontal_scale/scale); ny=int(t.length*t.horizontal_scale/scale)
    coarse=np.random.choice(heights,(nx,ny))
    x=np.linspace(0,t.width*t.horizontal_scale,nx); y=np.linspace(0,t.length*t.horizontal_scale,ny)
    # Equivalent linear interpolation to legacy scipy.interp2d (removed in recent SciPy).
    interp=RectBivariateSpline(x,y,coarse,kx=1,ky=1)
    t.height_field_raw += np.rint(interp(np.linspace(x[0],x[-1],t.width),np.linspace(y[0],y[-1],t.length))).astype(np.int16)
    return t

def discrete_obstacles_terrain(t,max_height,min_size,max_size,num_rects,platform_size=1.):
    h=int(max_height/t.vertical_scale)
    sizes=list(range(int(min_size/t.horizontal_scale),int(max_size/t.horizontal_scale),4))
    for _ in range(num_rects):
        w=np.random.choice(sizes); l=np.random.choice(sizes)
        x=np.random.choice(range(0,t.width-w,4)); y=np.random.choice(range(0,t.length-l,4))
        t.height_field_raw[x:x+w,y:y+l]=np.random.choice([-h,-h//2,h//2,h])
    p=int(platform_size/t.horizontal_scale)
    t.height_field_raw[(t.width-p)//2:(t.width+p)//2,(t.length-p)//2:(t.length+p)//2]=0
    return t

def stepping_stones_terrain(t,stone_size,stone_distance,max_height,platform_size=1.,depth=-10):
    s=int(stone_size/t.horizontal_scale); d=int(stone_distance/t.horizontal_scale); h=int(max_height/t.vertical_scale)
    t.height_field_raw[:]=int(depth/t.vertical_scale)
    heights=np.arange(-h-1,h,1)
    for y in range(0,t.length,s+d):
        x=np.random.randint(0,s)
        t.height_field_raw[:max(0,x-d),y:y+s]=np.random.choice(heights)
        while x<t.width:
            t.height_field_raw[x:x+s,y:y+s]=np.random.choice(heights); x+=s+d
    p=int(platform_size/t.horizontal_scale)
    t.height_field_raw[(t.width-p)//2:(t.width+p)//2,(t.length-p)//2:(t.length+p)//2]=0
    return t
