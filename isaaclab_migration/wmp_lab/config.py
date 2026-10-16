from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
def class_to_dict(obj):
    if isinstance(obj, dict): return {k: class_to_dict(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [class_to_dict(v) for v in obj]
    if not hasattr(obj, '__dict__'): return obj
    return {k: class_to_dict(getattr(obj,k)) for k in dir(obj)
            if not k.startswith('_') and not callable(getattr(obj,k))}

def asset_contract(cfg):
    path = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT)))
    if path.suffix != '.urdf':raise ValueError('This migration supports the original A1 URDF only')
    tree = ET.parse(path)
    joints = [j for j in tree.getroot().findall('joint') if j.attrib['type'] != 'fixed']
    # Gym loads articulations depth-first; this repository's URDF sibling order defines the policy order.
    children = {}
    child_links = set()
    for j in tree.getroot().findall('joint'):
        children.setdefault(j.find('parent').attrib['link'],[]).append(j)
        child_links.add(j.find('child').attrib['link'])
    roots = [l.attrib['name'] for l in tree.getroot().findall('link') if l.attrib['name'] not in child_links]
    ordered=[]
    def walk(link):
        for j in children.get(link,[]):
            if j.attrib['type']!='fixed': ordered.append(j)
            walk(j.find('child').attrib['link'])
    walk(roots[0])
    names=[j.attrib['name'] for j in ordered]
    if len(names)!=cfg.env.num_actions or set(names)!=set(cfg.init_state.default_joint_angles):
        raise ValueError(f'URDF/algorithm joint mismatch: {names}; actions={cfg.env.num_actions}')
    limits={j.attrib['name']: {k:float(v) for k,v in j.find('limit').attrib.items()} for j in ordered}
    meshes=[]
    for m in tree.iter('mesh'):
        name=m.attrib['filename']
        if name.startswith('package://'):
            name=name.split('/',3)[-1]; candidate=path.parent.parent/name
        else: candidate=path.parent/name
        if not candidate.exists(): raise FileNotFoundError(candidate)
        meshes.append(candidate.resolve())
    return path, names, limits, roots[0]


def scale_reward_schedule(schedule, scale):
    """Keep reward milestones aligned with collected samples when N changes."""
    import math
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('reward_schedule_scale must be finite and positive')
    result=[]
    for start,end,low,high in schedule:
        a,b=round(start*scale),round(end*scale)
        if b <= a:raise ValueError('Scaled reward schedule has no duration')
        result.append([a,b,low,high])
    return result
