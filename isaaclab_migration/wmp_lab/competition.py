"""ROBOCON geometry shared by training, height queries and the terrain-only viewer.

No simulator imports. Coordinates are local to an 8 m terrain tile; +X is travel.
Tires are fixed rigid elliptical rings, not a deformable rubber model.
"""
from dataclasses import dataclass, asdict, field
import math
import numpy as np
from .trimesh import box_trimesh

LEGACY_TYPES = ('wave', 'slope', 'stairup', 'stairdown', 'discrete', 'gap', 'pit', 'tilt', 'crawl', 'roughflat')
COMBO_TYPES = ('stair_gap', 'stair_gap_ramp')
PLUM_TYPES = ('plum_stones', 'plum_stones_flat')
COMPETITION_TYPES = ('plum_stones', 'tire_array', 'high_wall', 'platform', 'ramp', 'platform_gap', *COMBO_TYPES, 'plum_stones_flat')
ALL_TYPES = LEGACY_TYPES + COMPETITION_TYPES
RING_SEGMENTS = 64
RING_SECTION = 32
WALL_HEIGHTS = (.05, .08, .12, .15, .20, .25, .30, .32, .34, .36)
STONE_SIZE = .20
STONE_HEIGHT = .20
PLUM_LEVELS = 15
PLUM_STANDARD_LEVEL = 10
# Edge gaps between consecutive, alternating-row stones, not same-row gaps.
# Negative X gaps overlap projections only; the positive Y gap separates solids.
# L10 exactly reproduces the competition; L14 reaches the requested target.
STONE_LONG_GAPS = tuple(float(x) for x in np.r_[np.linspace(-.075, .10, 11), np.linspace(.10, .15, 5)[1:]])
STONE_ROW_GAPS = tuple(float(x) for x in np.r_[np.linspace(.025, .20, 11), np.linspace(.20, .25, 5)[1:]])
# Road/platform edges must stay disconnected from every stone even at L0.
STONE_PLATFORM_GAPS = tuple(float(x) for x in np.r_[np.linspace(.015, .10, 11), np.linspace(.10, .15, 5)[1:]])
TIRE_HEIGHTS = (.03, .05, .07, .09, .115, .14, .165, .176, .187, .198)
TREAD_DEPTH_FRACTION = .04  # fixed tread shape; only overall tire height is curricular
PLATFORM_HEIGHTS = tuple(np.linspace(.05, .30, 10))
RAMP_ANGLES = tuple(np.linspace(5., 30., 10))
RAMP_RUN = .8  # horizontal length of each slope; crest is also 0.8 m long
PLATFORM_GAPS = tuple(np.linspace(.05, .45, 10))
GAP_PLATFORM_COUNT = 6
GAP_FLOOR_HEIGHT = -5.  # legacy gap: -1000 height samples * 0.005 m
COMBO_RISES = (.04, .05, .06, .07, .08, .10, .12, .14, .16, .18)
COMBO_GAPS = (.05, .10, .15, .20, .25, .30, .36, .42, .48, .54)
COMBO_ANGLES = (5., 8., 10., 12., 15., 18., 20., 23., 26., 30.)


def standard_level(kind):
    if kind in PLUM_TYPES:return PLUM_STANDARD_LEVEL
    return 6 if kind in (*COMBO_TYPES, 'high_wall', 'tire_array') else 9


def level_count(kind):
    return PLUM_LEVELS if kind in (*PLUM_TYPES, 'platform_plum_combo') else 10


def maximum_level(kind):
    return level_count(kind)-1


def course_revision(kind):
    if kind in PLUM_TYPES:return 5
    if kind in COMBO_TYPES:return 4  # updated rewards and independent 30s combo timeout
    return 2 if kind in (*PLUM_TYPES, 'high_wall', 'tire_array') else 1

@dataclass
class Course:
    kind: str
    level: int
    boxes: list  # [center_x, center_y, size_x, size_y, height], stones first
    rings: list  # [center_x, center_y, height, reserved_zero]; holes are always open
    spawn: tuple
    entry: float
    exit: float
    half_width: float
    height: float
    ramps: list = field(default_factory=list)  # [x_start, x_end, half_width, z_start, z_end]
    floor_height: float = 0.
    parameters: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def make_course(kind, level):
    if kind == 'platform_plum_combo':kind = 'plum_stones'  # old ID/CLI stays compatible
    if kind not in COMPETITION_TYPES or level not in range(level_count(kind)):
        raise ValueError(f'Invalid competition course: {kind}, {level}')
    if kind in COMBO_TYPES:
        rise, gap, angle = COMBO_RISES[level], COMBO_GAPS[level], COMBO_ANGLES[level]
        height = 3 * rise
        landing_end = 2.24 + gap
        run = height / math.tan(math.radians(angle)) if kind == 'stair_gap_ramp' else 0.
        route_end = landing_end + run + (1.2 if run else 0.)
        shift = (route_end - 1.2) / 2
        # Fixed box identities: approach, step 1, step 2, launch, landing, finish.
        boxes = [[-.6, 0., 1.2, .8, 0.], [.16, 0., .32, .8, rise],
                 [.48, 0., .32, .8, 2 * rise], [1.04, 0., .8, .8, height],
                 [1.84 + gap, 0., .8, .8, height]]
        ramps = []
        if run:
            ramps = [[landing_end - shift, landing_end + run - shift, .4, height, 0.]]
            boxes.append([landing_end + run + .6, 0., 1.2, .8, 0.])
        for box in boxes:box[0] -= shift
        return Course(kind, level, boxes, [], (-.7 - shift, 0., 0.), -shift,
                      landing_end + run - shift, .4, height, ramps, GAP_FLOOR_HEIGHT,
                      dict(step_rise=rise, platform_height=height, gap=gap, slope_degrees=angle if run else 0.,
                           slope_run=run, route_start=-1.2-shift, route_end=route_end-shift,
                           platform_size=.8, step_depth=.32, turn_degrees=0))
    if kind == 'platform_gap':
        spacing = .8 + float(PLATFORM_GAPS[level])
        boxes = [[(index - (GAP_PLATFORM_COUNT - 1) / 2) * spacing, 0., .8, .8, 0.]
                 for index in range(GAP_PLATFORM_COUNT)]
        return Course(kind, level, boxes, [], (boxes[0][0], 0., 0.),
                      boxes[0][0] - .4, boxes[-1][0] + .4, .4, 0.,
                      floor_height=GAP_FLOOR_HEIGHT)
    if kind == 'platform':
        h = PLATFORM_HEIGHTS[level]
        return Course(kind, level, [[0., 0., .8, .8, h]], [],
                      (-1.1, 0., 0.), -.4, .4, .4, h)
    if kind == 'ramp':
        h = RAMP_RUN * math.tan(math.radians(RAMP_ANGLES[level]))
        edge = .4 + RAMP_RUN
        return Course(kind, level, [[0., 0., .8, .8, h]], [],
                      (-edge - .7, 0., 0.), -edge, edge, .4, h,
                      [[-edge, -.4, .4, 0., h], [.4, edge, .4, h, 0.]])
    if kind in PLUM_TYPES:
        size, h = STONE_SIZE, STONE_HEIGHT
        # Same-row centers are two alternating steps apart.
        pitch = 2 * (size + STONE_LONG_GAPS[level])
        row_y = (size + STONE_ROW_GAPS[level]) / 2
        boxes = [[x*pitch, y, size, size, h] for y, xs in
                 ((-row_y, (-1.25, -.25, .75)), (row_y, (-.75, .25, 1.25))) for x in xs]
        stone_edge = 1.25*pitch + size/2
        has_platforms = kind == 'plum_stones'
        if has_platforms:
            platform = stone_edge + STONE_PLATFORM_GAPS[level] + .4
            boxes += [[-platform, 0., .8, .8, h], [platform, 0., .8, .8, h]]
            entry, exit_x = -platform-.4, platform+.4
        else:
            road_gap = STONE_PLATFORM_GAPS[level]
            entry, exit_x = -stone_edge-road_gap, stone_edge+road_gap
            # Carve the stone section out of a level road: approach, tops and
            # finish share one elevation; the recessed gaps remain 20 cm lower.
            boxes += [[entry-.55, 0., 1.1, .8, h],
                      [exit_x+.55, 0., 1.1, .8, h]]
        road_edge = exit_x + 1.1
        boxes.append([0., 0., 2*road_edge, .8, 0.])
        road_height = 0. if has_platforms else h
        return Course(kind, level, boxes, [], (entry-.7, 0., road_height), entry, exit_x, .4, h,
                      floor_height=GAP_FLOOR_HEIGHT,
                      parameters=dict(stone_size=size, stone_height=h, longitudinal_gap=STONE_LONG_GAPS[level],
                                      row_gap=STONE_ROW_GAPS[level], platform_gap=STONE_PLATFORM_GAPS[level],
                                      has_platforms=has_platforms, stone_entry=-stone_edge, stone_exit=stone_edge,
                                      road_start=-road_edge, road_end=road_edge, road_height=road_height,
                                      gap_depth=h, standard_level=PLUM_STANDARD_LEVEL,
                                      gap_reference='consecutive_alternating_stone_edges',
                                      same_row_gap=pitch-size))
    if kind == 'tire_array':
        h = TIRE_HEIGHTS[level]
        rings = [[x, y, h, 0.] for y in (-.275, .275)
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


def tire_z_fraction(a, b):
    """Physical, shallow dimples in the upper rubber surface, identical at every level."""
    s = np.sin(b)
    dimples = TREAD_DEPTH_FRACTION * (.5+.5*np.cos(12*a)) * (.5+.5*np.cos(4*b)) * np.maximum(s, 0.)**2
    return (1+s)/2 - dimples


def ring_mesh(x, y, height):
    a = np.arange(RING_SEGMENTS)[:, None] * (2 * np.pi / RING_SEGMENTS)
    b = np.arange(RING_SECTION)[None, :] * (2 * np.pi / RING_SECTION)
    radius = .22 + .055 * np.cos(b)
    v = np.stack(np.broadcast_arrays(radius * np.cos(a) + x, radius * np.sin(a) + y,
                                    height * tire_z_fraction(a, b)), axis=-1).reshape(-1, 3)
    f = []
    for i in range(RING_SEGMENTS):
        for j in range(RING_SECTION):
            p = i * RING_SECTION + j
            q = ((i + 1) % RING_SEGMENTS) * RING_SECTION + j
            r = ((i + 1) % RING_SEGMENTS) * RING_SECTION + (j + 1) % RING_SECTION
            s = i * RING_SECTION + (j + 1) % RING_SECTION
            f.extend(((p, q, r), (p, r, s)))
    return v.astype(np.float32), np.array(f, dtype=np.uint32)


def ramp_mesh(start, end, half_width, z_start, z_end):
    """Closed triangular prism with an exact planar slope, independent of grid scale."""
    h = max(z_start, z_end)
    v = np.array([[start, -half_width, 0.], [start, half_width, 0.],
                  [end, -half_width, 0.], [end, half_width, 0.],
                  [end, -half_width, h], [end, half_width, h]], dtype=np.float32)
    f = np.array([(0, 1, 3), (0, 3, 2), (0, 4, 5), (0, 5, 1),
                  (2, 3, 5), (2, 5, 4), (0, 2, 4), (1, 5, 3)], dtype=np.uint32)
    if z_start > z_end:
        v[:, 0] = start + end - v[:, 0]
        f = f[:, ::-1].copy()
    return v, f


def course_mesh(course, ground=False):
    parts = [box_trimesh(np.array([sx, sy, h - course.floor_height]),
                        np.array([x, y, (h + course.floor_height) / 2]))
             for x, y, sx, sy, h in course.boxes if sx > 0 and sy > 0]
    parts.extend(ramp_mesh(*ramp) for ramp in course.ramps)
    if course.floor_height < 0:
        # The slope sits on a solid base down to the pit floor, like the box sections.
        parts.extend(box_trimesh(np.array([end-start, 2*width, -course.floor_height]),
                                 np.array([(start+end)/2, 0., course.floor_height/2]))
                     for start, end, width, _, _ in course.ramps)
    for x, y, h, _ in course.rings:
        parts.append(ring_mesh(x, y, h))
    if ground:
        parts.append(box_trimesh(np.array([8., 8., .02]), np.array([0., 0., course.floor_height - .01])))
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
    if profile == 'platform_plum_combo':profile = 'plum_stones'
    if profile not in ('robocon', 'robocon_mixed', 'gap_combos', 'gap_training', *COMPETITION_TYPES):
        raise ValueError(profile)
    if stage not in ('learn', 'standard', 'eval'):
        raise ValueError(stage)
    if profile == 'gap_training':
        weights = dict(stair_gap=6, stair_gap_ramp=6, slope=3, stairup=3, gap=2)
        counts = [weights.get(name, 0) for name in ALL_TYPES]
    elif profile == 'robocon_mixed':
        counts = [0, 1, 3, 3, 0, 6, 6, 1, 1, 1, 8, 6, 4, 4, 4, 4, 4, 4, 4]
    elif profile == 'robocon':
        counts = [0] * 10 + [9, 7, 4, 4, 4, 4, 4, 4, 4]
    elif profile == 'gap_combos':
        counts = [4 if name in COMBO_TYPES else 0 for name in ALL_TYPES]
    else:
        counts = [0] * len(ALL_TYPES); counts[ALL_TYPES.index(profile)] = 4
    cfg.terrain.num_cols = sum(counts)
    cfg.terrain.num_rows = max(level_count(name) for name,count in zip(ALL_TYPES, counts) if count)
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
        boxes = np.zeros((rows, cols, 9, 5), dtype=np.float32)
        rings = np.zeros((rows, cols, 8, 4), dtype=np.float32)
        ramps = np.zeros((rows, cols, 2, 5), dtype=np.float32)
        bounds = np.zeros((rows, cols, 3), dtype=np.float32)
        floors = np.zeros((rows, cols), dtype=np.float32)
        for (r, c), spec in cells.items():
            if spec.boxes:boxes[r, c, :len(spec.boxes)] = spec.boxes
            if spec.rings:rings[r, c, :len(spec.rings)] = spec.rings
            if spec.ramps:ramps[r, c, :len(spec.ramps)] = spec.ramps
            bounds[r, c] = [spec.entry, spec.exit, spec.half_width]
            floors[r, c] = spec.floor_height
        self.boxes = torch.tensor(boxes, device=device)
        self.rings = torch.tensor(rings, device=device)
        self.has_rings = torch.tensor((rings[..., 2] > 0).any(-1), device=device)
        self.any_rings = bool((rings[..., 2] > 0).any())
        self.ramps = torch.tensor(ramps, device=device)
        self.bounds = torch.tensor(bounds, device=device)
        self.floors = torch.tensor(floors, device=device)
        p = torch.linspace(math.pi, 0., RING_SECTION // 2 + 1, device=device)
        self.radii = (.22 + .055 * p.cos()).contiguous()
        angles = np.arange(RING_SEGMENTS)[:, None] * (2*np.pi/RING_SEGMENTS)
        self.sector_directions = torch.tensor(np.concatenate((np.cos(angles), np.sin(angles)), axis=1),
                                              device=device, dtype=torch.float32)
        profile = np.linspace(np.pi, 0., RING_SECTION//2+1)[None]
        self.tread_z = torch.tensor(tire_z_fraction(angles, profile), device=device, dtype=torch.float32)

    def __call__(self, xy, rows, cols):
        t = self.torch
        b, rings = self.boxes[rows, cols], self.rings[rows, cols]
        floor = self.floors[rows, cols][:, None, None]
        p = xy[..., None, :]
        inside = ((p - b[:, None, :, :2]).abs() <= b[:, None, :, 2:4] / 2 + 1e-7).all(-1)
        inside &= (b[:, None, :, 2:4] > 0).all(-1)
        h = t.where(inside, b[:, None, :, 4], floor).amax(-1)
        ramps = self.ramps[rows, cols][:, None]
        run = ramps[..., 1] - ramps[..., 0]
        along = (xy[..., 0, None] - ramps[..., 0]) / run.clamp(min=1e-7)
        on_ramp = (run > 0) & (along >= 0) & (along <= 1) & (xy[..., 1, None].abs() <= ramps[..., 2])
        ramp_h = ramps[..., 3] + along.clamp(0, 1) * (ramps[..., 4] - ramps[..., 3])
        h = t.maximum(h, t.where(on_ramp, ramp_h, floor).amax(-1))
        if not self.any_rings:return h
        ring_ids = self.has_rings[rows, cols].nonzero(as_tuple=False).flatten()
        if not len(ring_ids):return h
        floor = floor[ring_ids];rings = rings[ring_ids]
        delta = p[ring_ids] - rings[:, None, :, :2]
        angle = t.atan2(delta[..., 1], delta[..., 0])
        step = 2 * math.pi / RING_SEGMENTS
        sector = t.floor(angle / step).long()
        s0 = sector % RING_SEGMENTS;s1 = (s0+1) % RING_SEGMENTS
        d0 = self.sector_directions[s0];d1 = self.sector_directions[s1]
        u = (delta[..., 0]*d1[..., 1] - delta[..., 1]*d1[..., 0]) / math.sin(step)
        v = (-delta[..., 0]*d0[..., 1] + delta[..., 1]*d0[..., 0]) / math.sin(step)
        radius = u + v
        idx = t.searchsorted(self.radii, radius.contiguous()).clamp(1, len(self.radii)-1)
        inner, outer = self.radii[idx-1], self.radii[idx]
        width = outer-inner
        z00 = self.tread_z[s0, idx];z10 = self.tread_z[s1, idx]
        z01 = self.tread_z[s0, idx-1];z11 = self.tread_z[s1, idx-1]
        # Exact barycentric height on the same two annular triangles as ring_mesh.
        wp = u/outer;wr = (outer-radius)/width
        upper = wp*z00 + (1-wp-wr)*z10 + wr*z11
        wr = v/inner;wp = (radius-inner)/width
        lower = wp*z00 + wr*z11 + (1-wp-wr)*z01
        top = t.where(u/outer+v/inner >= 1., upper, lower) * rings[:, None, :, 2]
        valid = (radius >= .165) & (radius <= .275) & (rings[:, None, :, 2] > 0)
        top = t.where(valid, top, floor)
        h[ring_ids] = t.maximum(h[ring_ids], top.amax(-1))
        return h
