"""ROBOCON geometry shared by training, height queries and the terrain-only viewer.

No simulator imports. Coordinates are local to an 8 m terrain tile; +X is travel.
Tires are fixed rigid elliptical rings, not a deformable rubber model.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .trimesh import box_trimesh

LEGACY_TYPES = ('wave', 'slope', 'stairup', 'stairdown', 'discrete', 'gap', 'pit', 'tilt', 'crawl', 'roughflat')
COMPETITION_TYPES = ('plum_stones', 'tire_array', 'high_wall')
ALL_TYPES = LEGACY_TYPES + COMPETITION_TYPES
RING_SEGMENTS = 64
RING_SECTION = 32
WALL_HEIGHTS = (.03, .05, .08, .11, .14, .17, .20, .24, .27, .30)
STONE_SIZES = (.40, .36, .32, .295, .27, .25, .235, .22, .21, .20)
STONE_HEIGHTS = (.05, .075, .10, .12, .14, .16, .175, .185, .195, .20)
TIRE_HEIGHTS = (.03, .045, .06, .075, .09, .11, .125, .14, .155, .165)
FILL_FACTORS = (1., .8, .6, .45, .3, .15, 0., 0., 0., 0.)

@dataclass
class Course:
    kind: str
    level: int
    boxes: list  # [center_x, center_y, size_x, size_y, height], stones first
    rings: list  # [center_x, center_y, height, fill_height]
    spawn: tuple
    entry: float
    exit: float
    half_width: float
    height: float

    def to_dict(self):
        return asdict(self)


def make_course(kind, level):
    if kind not in COMPETITION_TYPES or level not in range(10):
        raise ValueError(f'Invalid competition course: {kind}, {level}')
    if kind == 'plum_stones':
        size, h = STONE_SIZES[level], STONE_HEIGHTS[level]
        stretch = max(.65 + .35 * level / 9, (size + .02) / .6)
        boxes = [[x * stretch, y, size, size, h] for y, xs in
                 ((-.2, (-.75, -.15, .45)), (.2, (-.45, .15, .75))) for x in xs]
        platform = .95 * stretch + .4
        boxes += [[-platform, 0., .8, .8, h], [platform, 0., .8, .8, h]]
        # Begin on the ground 0.7 m before the near platform, including ascent.
        return Course(kind, level, boxes, [], (-platform - .4 - .7, 0., 0.),
                      -platform - .4, platform + .4, .4, h)
    if kind == 'tire_array':
        h = TIRE_HEIGHTS[level]
        rings = [[x, y, h, h * FILL_FACTORS[level]] for y in (-.275, .275)
                 for x in (-.825, -.275, .275, .825)]
        return Course(kind, level, [], rings, (-1.75, 0., 0.), -1.1, 1.1, .55, h)
    h = WALL_HEIGHTS[level]
    return Course(kind, level, [[0., 0., .08, .8, h]], [],
                  (-1.1, 0., 0.), -.04, .04, .4, h)


def join_meshes(meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v); faces.append(f.astype(np.int64) + offset); offset += len(v)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.uint32)


def ring_mesh(x, y, height):
    a = np.arange(RING_SEGMENTS)[:, None] * (2 * np.pi / RING_SEGMENTS)
    b = np.arange(RING_SECTION)[None, :] * (2 * np.pi / RING_SECTION)
    radius = .22 + .055 * np.cos(b)
    v = np.stack(np.broadcast_arrays(radius * np.cos(a) + x, radius * np.sin(a) + y,
                                    height / 2 * (1 + np.sin(b))), axis=-1).reshape(-1, 3)
    f = []
    for i in range(RING_SEGMENTS):
        for j in range(RING_SECTION):
            p = i * RING_SECTION + j
            q = ((i + 1) % RING_SEGMENTS) * RING_SECTION + j
            r = ((i + 1) % RING_SEGMENTS) * RING_SECTION + (j + 1) % RING_SECTION
            s = i * RING_SECTION + (j + 1) % RING_SECTION
            f.extend(((p, q, r), (p, r, s)))
    return v.astype(np.float32), np.array(f, dtype=np.uint32)


def cylinder_mesh(x, y, radius, height):
    # Radius uses the same polygon as the ring, so query and geometry agree.
    a = np.arange(RING_SEGMENTS) * (2 * np.pi / RING_SEGMENTS)
    v = [(x, y, 0.), (x, y, height)]
    for z in (0., height):
        v.extend(zip(x + radius * np.cos(a), y + radius * np.sin(a), np.full(len(a), z)))
    f = []
    for i in range(RING_SEGMENTS):
        j = (i + 1) % RING_SEGMENTS
        a, b, c, d = 2 + i, 2 + j, 2 + RING_SEGMENTS + i, 2 + RING_SEGMENTS + j
        f.extend(((0, b, a), (1, c, d), (a, b, c), (b, d, c)))
    return np.array(v, dtype=np.float32), np.array(f, dtype=np.uint32)


def course_mesh(course, ground=False):
    parts = [box_trimesh(np.array([sx, sy, h]), np.array([x, y, h / 2]))
             for x, y, sx, sy, h in course.boxes]
    for x, y, h, fill in course.rings:
        parts.append(ring_mesh(x, y, h))
        if fill > 0:
            parts.append(cylinder_mesh(x, y, .22, fill))
    if ground:
        parts.append(box_trimesh(np.array([8., 4., .02]), np.array([0., 0., -.01])))
    return join_meshes(parts)


def column_types(cfg):
    """Use column identity rather than environment index ranges for new terrain."""
    cumulative = np.cumsum(cfg.terrain_proportions)
    return [ALL_TYPES[min(int(np.searchsorted(cumulative, j / cfg.num_cols + .001)), len(ALL_TYPES)-1)]
            for j in range(cfg.num_cols)]


def tile_center(level, type_index, length=8., width=8.):
    """Physical layout: constant Y = one type lane, increasing X = difficulty."""
    return ((level + .5) * length, (type_index + .5) * width)


def configure_competition(cfg, profile, stage='learn'):
    if profile not in ('robocon', 'robocon_mixed', *COMPETITION_TYPES):
        raise ValueError(profile)
    if stage not in ('learn', 'standard', 'eval'):
        raise ValueError(stage)
    if profile == 'robocon_mixed':
        counts = [0, 1, 3, 3, 0, 6, 6, 1, 1, 1, 8, 6, 4]
    elif profile == 'robocon':
        counts = [0] * 10 + [9, 7, 4]
    else:
        counts = [0] * 13; counts[ALL_TYPES.index(profile)] = 4
    cfg.terrain.num_cols = sum(counts)
    cfg.terrain.num_rows = 10
    cfg.terrain.terrain_proportions = [n / sum(counts) for n in counts]
    cfg.terrain.competition = True
    cfg.terrain.competition_stage = stage
    cfg.terrain.curriculum = stage != 'eval'
    cfg.terrain.competition_speed = (.15, .30)
    cfg.terrain.competition_window = 50
    cfg.terrain.competition_promote = .8
    cfg.terrain.competition_demote = .4
    cfg.terrain.competition_standard_fraction = .60
    cfg.terrain.competition_near_fraction = .25


class HeightQuery:
    """Batched exact top-surface queries of the piecewise planar course meshes.

    Fixed-size per-cell tensors allow querying 187/525 points without CPU transfers.
    """
    def __init__(self, cells, rows, cols, device):
        import torch
        self.torch = torch
        boxes = np.zeros((rows, cols, 8, 5), dtype=np.float32)
        rings = np.zeros((rows, cols, 8, 4), dtype=np.float32)
        for (r, c), spec in cells.items():
            if spec.boxes:boxes[r, c, :len(spec.boxes)] = spec.boxes
            if spec.rings:rings[r, c, :len(spec.rings)] = spec.rings
        self.boxes = torch.tensor(boxes, device=device)
        self.rings = torch.tensor(rings, device=device)
        p = torch.linspace(math.pi, 0., RING_SECTION // 2 + 1, device=device)
        self.radii = (.22 + .055 * p.cos()).contiguous()
        self.zfrac = (1 + p.sin()) / 2

    def __call__(self, xy, rows, cols):
        t = self.torch
        b, rings = self.boxes[rows, cols], self.rings[rows, cols]
        p = xy[..., None, :]
        inside = ((p - b[:, None, :, :2]).abs() <= b[:, None, :, 2:4] / 2 + 1e-7).all(-1)
        h = t.where(inside, b[:, None, :, 4], 0.).amax(-1)
        delta = p - rings[:, None, :, :2]
        angle = t.atan2(delta[..., 1], delta[..., 0])
        step = 2 * math.pi / RING_SEGMENTS
        mid = (t.floor(angle / step) + .5) * step
        radius = (delta[..., 0] * mid.cos() + delta[..., 1] * mid.sin()) / math.cos(step / 2)
        idx = t.searchsorted(self.radii, radius.contiguous()).clamp(1, len(self.radii)-1)
        blend = ((radius - self.radii[idx-1]) / (self.radii[idx] - self.radii[idx-1])).clamp(0, 1)
        top = (self.zfrac[idx-1] + blend * (self.zfrac[idx] - self.zfrac[idx-1])) * rings[:, None, :, 2]
        valid = (radius >= .165) & (radius <= .275) & (rings[:, None, :, 2] > 0)
        top = t.where(valid, top, 0.)
        fill = t.where(radius <= .22, rings[:, None, :, 3], 0.)
        return t.maximum(h, t.maximum(top, fill).amax(-1))
