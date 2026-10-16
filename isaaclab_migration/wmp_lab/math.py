"""Tensor math in legacy xyzw convention; simulator boundary explicitly uses wxyz."""
import torch
import numpy as np

def to_torch(x, device='cpu', dtype=torch.float): return torch.as_tensor(x,device=device,dtype=dtype)
def get_axis_params(value, axis):
    a=[0.,0.,0.]; a[axis]=value; return a
def torch_rand_float(lower, upper, shape, device): return (upper-lower)*torch.rand(*shape,device=device)+lower
def normalize(x): return x / x.norm(p=2,dim=-1,keepdim=True).clamp(min=1e-9)
def quat_apply(q,v):
    shape=v.shape; q=q.reshape(-1,4); v=v.reshape(-1,3)
    t=2*torch.cross(q[:,:3],v,dim=-1)
    return (v+q[:,3:]*t+torch.cross(q[:,:3],t,dim=-1)).reshape(shape)
def quat_rotate_inverse(q,v):
    qi=q.clone(); qi[...,:3]*=-1; return quat_apply(qi,v)
def quat_apply_yaw(q,v):
    q=q.clone().reshape(-1,4); q[:,:2]=0; return quat_apply(normalize(q),v)
def wrap_to_pi(angles):
    angles %= 2*np.pi; angles -= 2*np.pi*(angles>np.pi); return angles
def get_euler_xyz(q):
    x,y,z,w=q.unbind(-1)
    return (torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))%(2*np.pi),
            torch.asin((2*(w*y-z*x)).clamp(-1,1))%(2*np.pi),
            torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))%(2*np.pi))
def tf_inverse(q,t):
    qi=q.clone(); qi[...,:3]*=-1; return qi,-quat_apply(qi,t)
def tf_apply(q,t,v): return quat_apply(q,v)+t
