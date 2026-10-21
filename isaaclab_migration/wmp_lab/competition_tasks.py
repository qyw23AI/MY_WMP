"""Course state, strict completion metrics and success-rate curricula for A1.

Runs before reward/reset. No privileged target or coverage bits enter the actor.
Original Logic remains unchanged; this mixin is inactive for legacy profiles.
"""
import math
import torch
from .competition import COMPETITION_TYPES, COMBO_TYPES, PLUM_TYPES, GAP_PLATFORM_COUNT, PLUM_LEVELS, HeightQuery, standard_level, maximum_level, level_count, course_revision
from .math import quat_apply_yaw, torch_rand_float


class CourseState:
    def __init__(self, core):
        self.c = c = core
        self.ids = torch.tensor([i for i, col in enumerate(c.terrain_types.tolist())
                                 if c.terrain.column_types[col] in COMPETITION_TYPES], device=c.device, dtype=torch.long)
        self.kind = torch.full((c.num_envs,), -1, device=c.device, dtype=torch.long)
        for k, name in enumerate(COMPETITION_TYPES):
            cols = [i for i, n in enumerate(c.terrain.column_types) if n == name]
            for col in cols:
                self.kind[c.terrain_types == col] = k
        self.mask = self.kind >= 0
        self.query = HeightQuery(c.terrain.competition_cells, c.cfg.terrain.num_rows, c.cfg.terrain.num_cols, c.device)
        self.centers = torch.stack(((c.terrain_levels + .5) * c.terrain.env_length,
                                   (c.terrain_types + .5) * c.terrain.env_width), -1)
        n, dev = c.num_envs, c.device
        self.coverage = torch.zeros(n, 6, dtype=torch.bool, device=dev)
        self.platforms = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.jump_stage = torch.zeros(n, dtype=torch.long, device=dev)
        self.jump_airborne = torch.zeros(n, dtype=torch.bool, device=dev)
        self.jump_ready = torch.zeros(n, dtype=torch.bool, device=dev)
        self.landing_feet = torch.zeros(n, 4, dtype=torch.bool, device=dev)
        self.entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self.visited = torch.zeros_like(self.entered)
        self.failed = torch.zeros_like(self.entered)
        self.success = torch.zeros_like(self.entered)
        self.strict = torch.zeros_like(self.entered)
        self.hold = torch.zeros(n, device=dev)
        self.max_x = torch.zeros(n, device=dev)
        self.reward = torch.zeros(n, device=dev)
        self.frontier = torch.zeros(len(COMPETITION_TYPES), dtype=torch.long, device=dev)
        self.attempts = torch.zeros(len(COMPETITION_TYPES), PLUM_LEVELS, dtype=torch.long, device=dev)
        self.wins = torch.zeros_like(self.attempts)
        self.strict_wins = torch.zeros_like(self.attempts)
        self.window_n = torch.zeros_like(self.attempts)
        self.window_wins = torch.zeros_like(self.attempts)
        self.violated = torch.zeros_like(self.entered)
        self.totals = torch.zeros(len(COMPETITION_TYPES),PLUM_LEVELS,12,dtype=torch.long,device=dev)
        self.last_metrics = {}
        self.reset(self.ids)

    def state_dict(self):
        names=('frontier','attempts','wins','strict_wins','window_n','window_wins','totals')
        return {'version':3, 'course_types':list(COMPETITION_TYPES),
                'course_revisions':[course_revision(name) for name in COMPETITION_TYPES],
                **{name:getattr(self,name).detach().cpu().clone() for name in names}}

    def load_state_dict(self, state):
        if state.get('version') not in (1, 2, 3):raise ValueError('Unknown competition curriculum checkpoint version')
        count = state['frontier'].shape[0]
        source_names = state.get('course_types', list(COMPETITION_TYPES[:count]))
        revisions = state.get('course_revisions', [1]*count)
        if len(source_names)!=count or len(revisions)!=count:raise ValueError('Invalid curriculum course metadata')
        for name in ('frontier','attempts','wins','strict_wins','window_n','window_wins','totals'):
            if name not in state:continue
            target=getattr(self,name)
            source = state[name].to(target.device)
            valid_shape = source.ndim == target.ndim and source.shape[0] == count
            if target.ndim > 1:
                valid_shape &= 0 < source.shape[1] <= target.shape[1] and source.shape[2:] == target.shape[2:]
            if not valid_shape:
                raise ValueError(f'Invalid curriculum tensor {name}')
            target.zero_()
            for old, kind in enumerate(source_names):
                if kind in COMPETITION_TYPES and revisions[old] == course_revision(kind):
                    index = COMPETITION_TYPES.index(kind)
                    if target.ndim == 1:target[index].copy_(source[old].clamp(0, maximum_level(kind)))
                    else:
                        levels=min(source.shape[1],level_count(kind))
                        target[index, :levels].copy_(source[old, :levels])

    def reset_statistics(self):
        """Reset curriculum counters after changing course geometry or reward rules."""
        for name in ('frontier','attempts','wins','strict_wins','window_n','window_wins','totals'):
            getattr(self,name).zero_()

    def refresh_centers(self, ids):
        c = self.c
        self.centers[ids, 0] = (c.terrain_levels[ids] + .5) * c.terrain.env_length
        self.centers[ids, 1] = (c.terrain_types[ids] + .5) * c.terrain.env_width

    def reset(self, ids):
        self.refresh_centers(ids)
        self.coverage[ids] = False; self.entered[ids] = False; self.visited[ids] = False
        self.platforms[ids] = False
        self.jump_stage[ids] = 0; self.jump_airborne[ids] = False; self.jump_ready[ids] = False
        self.landing_feet[ids] = False
        self.violated[ids] = False
        self.failed[ids] = False; self.success[ids] = False; self.strict[ids] = False
        self.hold[ids] = 0.; self.reward[ids] = 0.
        self.max_x[ids] = self.c.env_origins[ids, 0] - self.centers[ids, 0]

    def heights(self, world_xy, ids):
        return self.query(world_xy - self.centers[ids, None], self.c.terrain_levels[ids], self.c.terrain_types[ids])

    def update(self):
        c, ids = self.c, self.ids
        self.reward.zero_()
        jumping = self.kind[ids] == COMPETITION_TYPES.index('platform_gap')
        self._update_platform_gap(ids[jumping])
        combo = (self.kind[ids] == COMPETITION_TYPES.index(COMBO_TYPES[0])) | (self.kind[ids] == COMPETITION_TYPES.index(COMBO_TYPES[1]))
        self._update_combos(ids[combo])
        plum = (self.kind[ids] == 0) | (self.kind[ids] == COMPETITION_TYPES.index('plum_stones_flat'))
        self._update_plum_stones(ids[plum])
        ids = ids[~jumping & ~combo & ~plum]
        if not len(ids):return
        rows, cols = c.terrain_levels[ids], c.terrain_types[ids]
        root = c.root_states[ids, :3].clone(); root[:, :2] -= self.centers[ids]
        feet = c.rigid_body_pos[ids][:, c.feet_indices].clone(); feet[:, :, :2] -= self.centers[ids, None]
        # A1 foot link origin is the spherical foot center (radius 0.02 m).
        sole_z = feet[:, :, 2] - .02
        contact = c.contact_forces[ids][:, c.feet_indices, 2] > 1.
        k = self.kind[ids]
        boxes = self.query.boxes[rows, cols]
        level_height = torch.where(k == 0, boxes[:, 0, 4],
                                   torch.where(k == 1, self.query.rings[rows, cols, 0, 2], boxes[:, 0, 4]))
        entry, exit_x, half_width = self.query.bounds[rows, cols].unbind(-1)
        ground = contact & (sole_z < .025)
        stones = k == 0; tires = k == 1; wall = k == 2
        paired = (k == 3) | (k == 4)
        ramps = k == 4
        # Require real support on the uphill, crest and downhill, in that order.
        surface = self.query(feet[:, :, :2], rows, cols)
        support = contact & ((sole_z - surface).abs() < .02)
        in_lane = feet[:, :, 1].abs() < half_width[:, None] - .005
        uphill = (support & in_lane & (feet[:, :, 0] > entry[:, None] + .05)
                  & (feet[:, :, 0] < -.45) & (surface > .02)).any(-1) & ramps
        downhill = (support & in_lane & (feet[:, :, 0] > .45)
                    & (feet[:, :, 0] < exit_x[:, None] - .05) & (surface > .02)).any(-1)
        downhill &= ramps & self.platforms[ids].all(-1)
        slope_touch = torch.stack((uphill, downhill), -1)
        new_slope = slope_touch & ~self.coverage[ids, :2]
        self.coverage[ids, :2] |= slope_touch
        # Record real top support on both platforms; side contact is insufficient.
        platform_boxes = boxes[:, 6:8]
        platform_top = ((feet[:, :, None, :2] - platform_boxes[:, None, :, :2]).abs()
                        <= platform_boxes[:, None, :, 2:4] / 2 - .005).all(-1)
        platform_top &= (sole_z[:, :, None] - level_height[:, None, None]).abs() < .02
        platform_touch = (platform_top & contact[:, :, None]).any(1) & stones[:, None]
        crest = support & ((feet[:, :, :2] - boxes[:, None, 0, :2]).abs()
                           <= boxes[:, None, 0, 2:4] / 2 - .005).all(-1)
        near_top = (crest & (feet[:, :, 0] <= 0)).any(-1) & paired
        near_top &= ~ramps | self.coverage[ids, 0]
        far_top = (crest & (feet[:, :, 0] >= 0)).any(-1) & paired
        far_top &= self.platforms[ids, 0] | near_top
        platform_touch |= torch.stack((near_top, far_top), -1)
        new_platform = platform_touch & ~self.platforms[ids]
        self.platforms[ids] |= platform_touch
        # During ascent rear feet may remain on the approach ground. Arm only
        # after entry-platform support and all four soles clear ground height.
        fully_inside = (feet[:, :, 0] > entry[:, None]).all(-1)
        on_course = (sole_z >= level_height[:, None] - .02).all(-1)
        self.entered[ids] |= fully_inside & (~stones | (self.platforms[ids, 0] & on_course)) & ~paired
        self.entered[ids] |= paired & self.platforms[ids, 0]
        inside_array = (feet[:, :, 0].abs() <= 1.1) & (feet[:, :, 1].abs() <= .55)
        self.visited[ids] |= (inside_array & contact).any(-1) & tires
        on_top = ((feet[:, :, None, :2] - boxes[:, None, :6, :2]).abs()
                  <= (boxes[:, None, :6, 2:4] / 2 - .005)).all(-1)
        on_top &= ((sole_z[:, :, None] - level_height[:, None, None]).abs() < .02)
        touched = (on_top & contact[:, :, None]).any(1) & stones[:, None]
        new = touched & ~self.coverage[ids]
        self.coverage[ids] |= touched
        # Contacts beyond the exit are permitted once leaving the far platform.
        leaving_platform = self.platforms[ids, 1] & (feet[:, :, 0] > (boxes[:, 7, 0] - .4)[:, None]).all(-1)
        bad_stones = stones & ~leaving_platform & (ground.sum(-1) >= 2)
        outside = (feet[:, :, 1].abs() > .55) | (feet[:, :, 0] < -1.1)
        bad_tires = tires & ((ground & outside).sum(-1) >= 2)
        # Reject crossing around the wall end without changing physical geometry.
        bad_wall = wall & (root[:, 0].abs() < .55) & (root[:, 1].abs() > .30)
        side_fall = stones & (root[:, 1].abs() > .60)
        skipped_entry = stones & (root[:, 0] > boxes[:, 6, 0] + .45) & ~self.entered[ids]
        paired_side = paired & (root[:, 0] > entry - .3) & (root[:, 0] < exit_x + .3) & (root[:, 1].abs() > half_width - .05)
        skipped_crest = paired & (root[:, 0] > .45) & ~self.platforms[ids, 0]
        bypass = bad_wall | side_fall | skipped_entry | (self.entered[ids] & bad_tires) | paired_side | skipped_crest
        bad = (self.entered[ids] & bad_stones) | bypass
        self.failed[ids] |= bad | (c.reset_buf[ids] & ~c.time_out_buf[ids])
        cleared = (feet[:, :, 0] > exit_x[:, None] + .05).all(-1)
        cleared &= root[:, 0] > exit_x + .35
        stable = (contact.sum(-1) >= 2) & (c.projected_gravity[ids, 2] < -.8) & (c.root_states[ids, 9].abs() < .35)
        # Allow a normal walking gait: two grounded support feet suffice once all
        # four feet have passed the exit; swinging feet need not touch together.
        landed = (contact & (sole_z.abs() < .025)).sum(-1) >= 2
        stable &= ~paired | (landed & (feet[:, :, 1].abs() < half_width[:, None]).all(-1))
        self.hold[ids] = torch.where(cleared & stable, self.hold[ids] + c.dt, 0.)
        reached_exit = (self.hold[ids] >= .30) & self.entered[ids] & ~self.failed[ids]
        complete = reached_exit & (~tires | self.visited[ids])
        complete &= ~stones | self.platforms[ids].all(-1)
        traversed = self.platforms[ids].all(-1) & (~ramps | self.coverage[ids, :2].all(-1))
        complete &= ~paired | traversed
        full = self.coverage[ids].all(-1)
        strict = complete & (~stones | full)
        # Early stone levels teach traversal; levels 3-9 require all six tops.
        success = complete & (~stones | full | (rows < 3))
        # Reaching the exit while missing a required stone is a failed attempt.
        missed = reached_exit & stones & (~self.platforms[ids].all(-1) | (~full & (rows >= 3)))
        missed |= paired & (self.hold[ids] >= .30) & ~traversed
        self.violated[ids] |= bad | missed
        self.failed[ids] |= missed
        new_success = success & ~self.success[ids]
        self.success[ids] |= success; self.strict[ids] |= strict
        progress = (root[:, 0] - self.max_x[ids]).clamp(0., .06)
        self.max_x[ids] = torch.maximum(self.max_x[ids], root[:, 0])
        self.reward[ids] = (2. * progress + .5 * new.sum(-1) + new_platform.sum(-1) + .5 * new_slope.sum(-1)
                            + 8 * new_success.float() - 5 * (bad | missed).float()
                            - 5 * bypass.float())
        c.reset_buf[ids] |= self.failed[ids] | self.success[ids]
        c.time_out_buf[ids] &= ~(self.failed[ids] | self.success[ids])

    def _update_platform_gap(self, ids):
        """Five ordered takeoff/landing events; finish on the last platform, over a deep pit."""
        if not len(ids):return
        c = self.c
        rows, cols = c.terrain_levels[ids], c.terrain_types[ids]
        boxes = self.query.boxes[rows, cols, :GAP_PLATFORM_COUNT]
        root = c.root_states[ids, :3].clone();root[:, :2] -= self.centers[ids]
        feet = c.rigid_body_pos[ids][:, c.feet_indices].clone();feet[:, :, :2] -= self.centers[ids, None]
        sole_z = feet[:, :, 2] - .02
        contact = c.contact_forces[ids][:, c.feet_indices, 2] > 1.
        inside = ((feet[:, :, None, :2] - boxes[:, None, :, :2]).abs()
                  <= boxes[:, None, :, 2:4] / 2 - .005).all(-1)
        on_top = inside & ((sole_z[:, :, None] - boxes[:, None, :, 4]).abs() < .02)
        support = on_top & contact[:, :, None]
        counts = support.sum(1)
        stage = self.jump_stage[ids]
        source = boxes.gather(1, stage[:, None, None].expand(-1, 1, 5)).squeeze(1)
        source_support = counts.gather(1, stage[:, None]).squeeze(1) >= 2
        source_inside = inside.gather(2, stage[:, None, None].expand(-1, 4, 1)).squeeze(-1).all(-1)
        first_support = (stage == 0) & source_support
        self.coverage[ids, 0] |= first_support
        self.platforms[ids, 0] |= first_support
        self.entered[ids] |= first_support
        self.jump_ready[ids] |= source_support & source_inside
        flight = ~contact.any(-1) & self.jump_ready[ids] & (root[:, 2] > source[:, 4] + .10)
        self.jump_airborne[ids] |= flight
        target = (stage + 1).clamp(max=GAP_PLATFORM_COUNT - 1)
        target_feet = support.gather(2, target[:, None, None].expand(-1, 4, 1)).squeeze(-1)
        target_inside = inside.gather(2, target[:, None, None].expand(-1, 4, 1)).squeeze(-1).all(-1)
        self.landing_feet[ids] |= target_feet & self.jump_airborne[ids, None]
        target_support = (target_feet.sum(-1) >= 2) & (stage < GAP_PLATFORM_COUNT - 1)
        source_cleared = (feet[:, :, 0] > (source[:, 0] + source[:, 2] / 2 + .01)[:, None]).all(-1)
        landed = target_support & self.landing_feet[ids].all(-1) & target_inside & source_cleared
        landed &= self.jump_airborne[ids]
        wrong_landing = self.jump_airborne[ids] & (contact & ~target_feet).any(-1)
        # A foot can drop into the gap without touching anything. Detect that
        # immediately, rather than waiting for the whole body to fall into the pit.
        foot_in_gap = (~inside.any(-1) & (sole_z < source[:, None, 4] - .02)).any(-1)
        foot_in_gap &= self.jump_ready[ids]
        # A stride without a flight phase, or skipping any platform, is not a jump traversal.
        skipped = ((counts > 0) & (torch.arange(GAP_PLATFORM_COUNT, device=c.device)[None] > target[:, None])).any(-1)
        walked = target_feet.any(-1) & ~self.jump_airborne[ids] & (stage < GAP_PLATFORM_COUNT - 1)
        bad_contact = (contact & ~on_top.any(-1)).any(-1)
        side = root[:, 1].abs() > .40
        dropped = root[:, 2] < source[:, 4] - .15
        bad = skipped | walked | bad_contact | side | dropped | wrong_landing | foot_in_gap
        self.violated[ids] |= bad
        self.failed[ids] |= bad | (c.reset_buf[ids] & ~c.time_out_buf[ids])
        landed &= ~self.failed[ids]
        new_stage = torch.where(landed, target, stage)
        self.jump_stage[ids] = new_stage
        self.coverage[ids, target] |= landed
        self.landing_feet[ids[landed]] = False
        self.jump_airborne[ids] &= ~(landed | source_support)
        final_support = counts[:, -1] >= 2
        self.platforms[ids, 1] |= landed & (new_stage == GAP_PLATFORM_COUNT - 1)
        stable = final_support & inside[:, :, -1].all(-1)
        stable &= (c.projected_gravity[ids, 2] < -.8) & (c.root_states[ids, 9].abs() < .35)
        stable &= (new_stage == GAP_PLATFORM_COUNT - 1) & self.coverage[ids].all(-1)
        self.hold[ids] = torch.where(stable & ~self.failed[ids], self.hold[ids] + c.dt, 0.)
        success = (self.hold[ids] >= .30) & ~self.failed[ids]
        new_success = success & ~self.success[ids]
        self.success[ids] |= success;self.strict[ids] |= success
        progress = (root[:, 0] - self.max_x[ids]).clamp(0., .06)
        self.max_x[ids] = torch.maximum(self.max_x[ids], root[:, 0])
        self.reward[ids] = 2. * progress + landed.float() + 8. * new_success.float() - 5. * bad.float()
        c.reset_buf[ids] |= self.failed[ids] | self.success[ids]
        c.time_out_buf[ids] &= ~(self.failed[ids] | self.success[ids])

    def _update_plum_stones(self, ids):
        """Fixed-height tops, spacing curriculum; any foot landing between stones fails."""
        if not len(ids):return
        c = self.c
        rows, cols = c.terrain_levels[ids], c.terrain_types[ids]
        b = self.query.boxes[rows, cols]
        combo = self.kind[ids] == 0
        root = c.root_states[ids, :3].clone();root[:, :2] -= self.centers[ids]
        feet = c.rigid_body_pos[ids][:, c.feet_indices].clone();feet[:, :, :2] -= self.centers[ids, None]
        sole = feet[:, :, 2]-.02
        contact = c.contact_forces[ids][:, c.feet_indices, 2] > 1.
        h = b[:, 0, 4]
        inside = ((feet[:, :, None, :2]-b[:, None, :8, :2]).abs() <= b[:, None, :8, 2:4]/2-.005).all(-1)
        top = inside & ((sole[:, :, None]-h[:, None, None]).abs() < .02)
        touch = (top & contact[:, :, None]).any(1)
        new_stones = touch[:, :6] & ~self.coverage[ids]
        self.coverage[ids] |= touch[:, :6]
        platform_touch = touch[:, 6:8] & combo[:, None]
        new_platforms = platform_touch & ~self.platforms[ids]
        self.platforms[ids] |= platform_touch
        bounds = self.query.bounds[rows, cols]
        entry, exit_x = bounds[:, 0], bounds[:, 1]
        stone_entry = (b[:, :6, 0]-b[:, :6, 2]/2).amin(-1)
        stone_exit = (b[:, :6, 0]+b[:, :6, 2]/2).amax(-1)
        # Rear feet may remain on legal approach ground while the front feet ascend.
        armed_combo = self.platforms[ids, 0] & (feet[:, :, 0] > entry[:, None]).all(-1)
        armed_combo &= (sole >= h[:, None]-.02).all(-1)
        self.entered[ids] |= torch.where(combo, armed_combo, touch[:, :6].any(-1))
        complete_tops = self.coverage[ids].all(-1)
        in_stones = (feet[:, :, 0] >= stone_entry[:, None]) & (feet[:, :, 0] <= stone_exit[:, None])
        on_raised_route = (feet[:, :, 0] > entry[:, None]) & (feet[:, :, 0] <= exit_x[:, None])
        # The flat course includes both curricular gaps between road and stones,
        # even before the first stone has been touched.
        protected = in_stones | ((~combo | self.entered[ids])[:, None] & on_raised_route)
        ground_miss = (protected & contact & (sole < h[:, None]-.02)).any(-1)
        dropped_foot = (protected & ~inside.any(-1) & (sole < h[:, None]-.025)).any(-1)
        exit_height = torch.where(combo, 0., h)
        premature_exit = ((feet[:, :, 0] > exit_x[:, None]) & contact
                          & (sole < exit_height[:, None]+.025)).any(-1)
        premature_exit &= ~complete_tops | (combo & ~self.platforms[ids, 1])
        skipped_entry = combo & (root[:, 0] > b[:, 6, 0]+.45) & ~self.entered[ids]
        side = root[:, 1].abs() > .4
        side |= ((feet[:, :, 1].abs() > .4) & contact).any(-1)
        bad = ground_miss | dropped_foot | premature_exit | skipped_entry | side
        self.failed[ids] |= bad | (c.reset_buf[ids] & ~c.time_out_buf[ids])
        cleared = (feet[:, :, 0] > exit_x[:, None]+.05).all(-1) & (root[:, 0] > exit_x+.35)
        valid = complete_tops & (~combo | self.platforms[ids].all(-1)) & self.entered[ids]
        stable = cleared & valid & ((contact & ((sole-exit_height[:, None]).abs() < .025)).sum(-1) >= 2)
        stable &= (c.projected_gravity[ids, 2] < -.8) & (c.root_states[ids, 9].abs() < .35)
        self.hold[ids] = torch.where(stable & ~self.failed[ids], self.hold[ids]+c.dt, 0.)
        missed = cleared & ~valid
        self.violated[ids] |= bad | missed
        self.failed[ids] |= missed
        success = (self.hold[ids] >= .30) & ~self.failed[ids]
        new_success = success & ~self.success[ids]
        self.success[ids] |= success;self.strict[ids] |= success
        progress = (root[:, 0]-self.max_x[ids]).clamp(0., .06)
        self.max_x[ids] = torch.maximum(self.max_x[ids], root[:, 0])
        self.reward[ids] = 2*progress + .5*new_stones.sum(-1) + new_platforms.sum(-1)
        self.reward[ids] += 8*new_success.float() - 5*(bad|missed).float()
        c.reset_buf[ids] |= self.failed[ids] | self.success[ids]
        c.time_out_buf[ids] &= ~(self.failed[ids] | self.success[ids])

    def _update_combos(self, ids):
        """Straight stairs -> one jump -> optional downhill, with pits on both sides."""
        if not len(ids):return
        c = self.c
        rows, cols = c.terrain_levels[ids], c.terrain_types[ids]
        boxes = self.query.boxes[rows, cols]
        ramps = self.query.ramps[rows, cols, 0]
        downhill = self.kind[ids] == COMPETITION_TYPES.index('stair_gap_ramp')
        root = c.root_states[ids, :3].clone();root[:, :2] -= self.centers[ids]
        feet = c.rigid_body_pos[ids][:, c.feet_indices].clone();feet[:, :, :2] -= self.centers[ids, None]
        sole = feet[:, :, 2] - .02
        contact = c.contact_forces[ids][:, c.feet_indices, 2] > 1.
        inside = ((feet[:, :, None, :2] - boxes[:, None, :, :2]).abs()
                  <= boxes[:, None, :, 2:4] / 2 - .005).all(-1)
        inside &= (boxes[:, None, :, 2:4] > 0).all(-1)
        top = inside & ((sole[:, :, None] - boxes[:, None, :, 4]).abs() < .02)
        support = top & contact[:, :, None]
        touched = support.any(1)
        old_coverage = self.coverage[ids].clone()
        self.coverage[ids, 0] |= touched[:, 1]
        self.coverage[ids, 1] |= touched[:, 2] & self.coverage[ids, 0]
        stairs_done = self.coverage[ids, :2].all(-1)
        self.coverage[ids, 2] |= touched[:, 3] & stairs_done
        ready = stairs_done & inside[:, :, 3].all(-1) & (support[:, :, 3].sum(-1) >= 2)
        self.jump_ready[ids] |= ready
        self.entered[ids] |= self.coverage[ids, 2]
        self.platforms[ids, 0] |= self.coverage[ids, 2]
        already_landed = self.coverage[ids, 3].clone()
        flight = ~contact.any(-1) & self.jump_ready[ids] & ~already_landed
        flight &= root[:, 2] > boxes[:, 3, 4] + .10
        self.jump_airborne[ids] |= flight
        target_feet = support[:, :, 4]
        self.landing_feet[ids] |= target_feet & self.jump_airborne[ids, None]
        cleared = (feet[:, :, 0] > (boxes[:, 3, 0] + .41)[:, None]).all(-1)
        landed = self.jump_airborne[ids] & self.landing_feet[ids].all(-1) & inside[:, :, 4].all(-1)
        landed &= cleared & (target_feet.sum(-1) >= 2)
        wrong_landing = self.jump_airborne[ids] & (contact & ~target_feet).any(-1)
        walked = target_feet.any(-1) & ~self.jump_airborne[ids] & ~already_landed
        skipped_stairs = (root[:, 0] > boxes[:, 3, 0] - .2) & ~stairs_done
        surface = self.query(feet[:, :, :2], rows, cols)
        on_slope = (feet[:, :, 0] > ramps[:, None, 0] + .03) & (feet[:, :, 0] < ramps[:, None, 1] - .03)
        on_slope &= feet[:, :, 1].abs() < .395
        slope_support = on_slope & ((sole - surface).abs() < .02) & contact & downhill[:, None]
        # Off-track feet below road height, and gap feet below the raised tops, fail immediately.
        over_gap = (feet[:, :, 0] > (boxes[:, 3, 0] + .4)[:, None]) & (feet[:, :, 0] < (boxes[:, 4, 0] - .4)[:, None])
        gap_drop = (over_gap & (sole < boxes[:, None, 3, 4] - .02)).any(-1)
        centerline = torch.stack((feet[:, :, 0], torch.zeros_like(sole)), -1)
        lane_surface = self.query(centerline, rows, cols).clamp(min=0.)
        side_drop = ((feet[:, :, 1].abs() > .4) & (sole < lane_surface - .02)).any(-1)
        bad_contact = (contact & ~(top.any(-1) | slope_support)).any(-1)
        side = root[:, 1].abs() > .4
        dropped = root[:, 2] < -.15
        # Do not continue beyond a no-ramp finish platform into the pit.
        off_end = ~downhill & (root[:, 0] > boxes[:, 4, 0] + .4)
        bad = wrong_landing | walked | skipped_stairs | gap_drop | side_drop | bad_contact | side | dropped | off_end
        self.violated[ids] |= bad
        self.failed[ids] |= bad | (c.reset_buf[ids] & ~c.time_out_buf[ids])
        landed &= ~self.failed[ids]
        self.coverage[ids, 3] |= landed
        self.platforms[ids, 1] |= landed
        self.jump_stage[ids] = self.coverage[ids, 3].long()
        self.jump_airborne[ids] &= ~landed
        self.coverage[ids, 4] |= slope_support.any(-1) & self.coverage[ids, 3]
        self.coverage[ids, 4] |= ~downhill  # no slope is required in combination A
        final_platform = inside[:, :, 4].all(-1) & (target_feet.sum(-1) >= 2)
        ground_exit = inside[:, :, 5].all(-1) & (support[:, :, 5].sum(-1) >= 2)
        ground_exit &= (feet[:, :, 0] > ramps[:, None, 1] + .05).all(-1) & (root[:, 0] > ramps[:, 1] + .35)
        at_finish = torch.where(downhill, ground_exit & self.coverage[ids, 4], final_platform)
        at_finish &= self.coverage[ids, :4].all(-1)
        # Stop advancing once the complete route has reached its finish support.
        c.commands[ids[at_finish], :3] = 0.
        stable = at_finish & (c.projected_gravity[ids, 2] < -.8) & (c.root_states[ids, 9].abs() < .35)
        self.coverage[ids, 5] |= stable
        self.hold[ids] = torch.where(stable & ~self.failed[ids], self.hold[ids] + c.dt, 0.)
        success = (self.hold[ids] >= .30) & ~self.failed[ids]
        new_success = success & ~self.success[ids]
        self.success[ids] |= success;self.strict[ids] |= success
        new = self.coverage[ids] & ~old_coverage
        self.reward[ids] = .5*new[:, :3].sum(-1) + 2*landed.float()
        self.reward[ids] += downhill * new[:, 4].float() + 4*new_success.float() - 5*bad.float()
        c.reset_buf[ids] |= self.failed[ids] | self.success[ids]
        c.time_out_buf[ids] &= ~(self.failed[ids] | self.success[ids])

    def finish(self, ids):
        c = self.c
        ids = ids[self.mask[ids] & (c.episode_length_buf[ids] > 0)]
        self.last_metrics = {}
        if hasattr(self,'evaluation_events') and (ids==0).any():
            self.evaluation_events.append(dict(level=int(c.terrain_levels[0]),success=bool(self.success[0]),
                strict_success=bool(self.strict[0]),failure=bool(self.failed[0]),violation=bool(self.violated[0]),
                timeout=bool(c.time_out_buf[0]),stones=self.coverage[0].tolist(),platforms=self.platforms[0].tolist()))
        for k, name in enumerate(COMPETITION_TYPES):
            selected = ids[self.kind[ids] == k]
            if not len(selected):continue
            level = c.terrain_levels[selected]
            ones = torch.ones_like(level)
            for target, values in ((self.attempts, ones), (self.wins, self.success[selected].long()),
                                   (self.strict_wins, self.strict[selected].long()), (self.window_n, ones),
                                   (self.window_wins, self.success[selected].long())):
                target[k].scatter_add_(0, level, values)
            values = torch.cat((self.failed[selected,None],self.violated[selected,None],c.time_out_buf[selected,None],
                                self.platforms[selected],self.coverage[selected],self.coverage[selected].all(-1,keepdim=True)),1).long()
            self.totals[k].index_add_(0,level,values)
            self.last_metrics[f'{name}/violation'] = self.violated[selected].float().mean()
            standard = standard_level(name)
            self.last_metrics[f'{name}/standard_strict_rate'] = self.strict_wins[k,standard].float()/self.attempts[k,standard].clamp(min=1)
            target = maximum_level(name)
            self.last_metrics[f'{name}/target_attempts'] = self.attempts[k,target].float()
            self.last_metrics[f'{name}/target_strict_rate'] = self.strict_wins[k,target].float()/self.attempts[k,target].clamp(min=1)
            self.last_metrics[f'{name}/success'] = self.success[selected].float().mean()
            self.last_metrics[f'{name}/strict_success'] = self.strict[selected].float().mean()
            self.last_metrics[f'{name}/coverage'] = self.coverage[selected].float().mean()
            if k == 0 or name in (*COMBO_TYPES, 'platform', 'ramp', 'platform_gap'):
                self.last_metrics[f'{name}/entry_platform'] = self.platforms[selected, 0].float().mean()
                self.last_metrics[f'{name}/exit_platform'] = self.platforms[selected, 1].float().mean()
            if k == 4:
                self.last_metrics[f'{name}/uphill_contact'] = self.coverage[selected, 0].float().mean()
                self.last_metrics[f'{name}/downhill_contact'] = self.coverage[selected, 1].float().mean()
            if name == 'platform_gap':
                self.last_metrics[f'{name}/jumps_completed'] = self.jump_stage[selected].float().mean()
            if name in COMBO_TYPES:
                self.last_metrics[f'{name}/stairs_completed'] = self.coverage[selected, :2].all(-1).float().mean()
                self.last_metrics[f'{name}/jump_completed'] = self.coverage[selected, 3].float().mean()
                if name == 'stair_gap_ramp':
                    self.last_metrics[f'{name}/downhill_contact'] = self.coverage[selected, 4].float().mean()
            self.last_metrics[f'{name}/failure'] = self.failed[selected].float().mean()
            self.last_metrics[f'{name}/timeout'] = c.time_out_buf[selected].float().mean()
            top = selected[level == standard]
            self.last_metrics[f'{name}/standard_attempts'] = self.attempts[k, standard].float()
            if len(top):self.last_metrics[f'{name}/standard_success'] = self.strict[top].float().mean()
            frontier = int(self.frontier[k])
            count = int(self.window_n[k, frontier])
            if count >= c.cfg.terrain.competition_window:
                rate = float(self.window_wins[k, frontier]) / count
                if rate >= c.cfg.terrain.competition_promote:self.frontier[k] = min(maximum_level(name), frontier+1)
                elif rate < c.cfg.terrain.competition_demote:self.frontier[k] = max(0, frontier-1)
                self.window_n[k, frontier] = 0; self.window_wins[k, frontier] = 0
            self.last_metrics[f'{name}/frontier'] = self.frontier[k].float()

    def select_levels(self, ids):
        c = self.c
        ids = ids[self.mask[ids]]
        if not len(ids):return
        stage = c.cfg.terrain.competition_stage
        if stage == 'learn':
            c.terrain_levels[ids] = self.frontier[self.kind[ids]]
        elif stage == 'standard':
            u = torch.rand(len(ids), device=c.device)
            cfg = c.cfg.terrain
            levels = torch.randint(0, 7, (len(ids),), device=c.device)
            levels = torch.where(u < cfg.competition_standard_fraction + cfg.competition_near_fraction,
                                 torch.randint(7, 9, (len(ids),), device=c.device), levels)
            c.terrain_levels[ids] = torch.where(u < cfg.competition_standard_fraction, 9, levels)
            standard = torch.tensor([standard_level(name) for name in COMPETITION_TYPES], device=c.device)
            maximum = torch.tensor([maximum_level(name) for name in COMPETITION_TYPES], device=c.device)
            base, top = standard[self.kind[ids]], maximum[self.kind[ids]]
            strengthened = base < top
            between = base+1+(torch.rand(len(ids),device=c.device)*(top-base-1).clamp(min=1)).long()
            sampled = torch.where(u < .4, base, torch.where(u < .8, top, between))
            c.terrain_levels[ids[strengthened]] = sampled[strengthened]
        # eval stays at the requested fixed level, never changes on reset.
        c.env_origins[ids] = c.terrain_origins[c.terrain_levels[ids], c.terrain_types[ids]]
        self.refresh_centers(ids)


class CompetitionMixin:
    def init_competition(self):
        self.course = None
        if self.terrain is not None and getattr(self.cfg.terrain, 'competition', False):
            self.course = CourseState(self)
            self.course.select_levels(self.course.ids)
            self.course.reset(self.course.ids)

    def _get_env_origins(self):
        super()._get_env_origins()
        if self.terrain is None:return
        competition = getattr(self.cfg.terrain, 'competition', False)
        if competition:
            self.max_terrain_level = 10  # legacy lanes keep their ten-level curriculum
            maxima = torch.tensor([maximum_level(name) for name in self.terrain.column_types],device=self.device)
            self.terrain_levels = torch.minimum(self.terrain_levels, maxima[self.terrain_types])
        if hasattr(self.cfg.terrain, 'eval_terrain_col'):
            if competition and not 0 <= self.cfg.terrain.eval_terrain_level <= int(maxima[self.cfg.terrain.eval_terrain_col]):
                raise ValueError('Requested evaluation level is outside this terrain type')
            self.terrain_levels.fill_(self.cfg.terrain.eval_terrain_level)
            if competition:self.terrain_types.fill_(self.cfg.terrain.eval_terrain_col)
            else:self.terrain_types = (torch.arange(self.num_envs,device=self.device)+self.cfg.terrain.eval_terrain_col)%self.cfg.terrain.num_cols
        elif competition and self.cfg.terrain.competition_stage == 'eval':
            self.terrain_levels[:] = maxima[self.terrain_types]
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _sample_course_heights(self, points, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        world = quat_apply_yaw(self.base_quat[ids].repeat(1, points.shape[1]), points[ids]) + self.root_states[ids, :3].unsqueeze(1)
        # Legacy minimum-of-three height query, preserving existing terrain behavior.
        grid = ((world[:, :, :2] + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale).long()
        px = grid[..., 0].clamp(0, self.height_samples.shape[0]-2)
        py = grid[..., 1].clamp(0, self.height_samples.shape[1]-2)
        h = torch.minimum(self.height_samples[px, py], torch.minimum(self.height_samples[px+1, py], self.height_samples[px, py+1])).float() * self.cfg.terrain.vertical_scale
        chosen = self.course.mask[ids]
        h[chosen] = self.course.heights(world[chosen, :, :2], ids[chosen])
        return h

    def _get_heights(self, env_ids=None):
        if getattr(self, 'course', None) is None:return super()._get_heights(env_ids)
        return self._sample_course_heights(self.height_points, env_ids)

    def _get_forward_heights(self, env_ids=None):
        if getattr(self, 'course', None) is None:return super()._get_forward_heights(env_ids)
        return self._sample_course_heights(self.forward_height_points, env_ids)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if getattr(self, 'course', None) is None:return
        ids = env_ids[self.course.mask[env_ids]]
        low, high = self.cfg.terrain.competition_speed
        self.commands[ids, 0] = torch_rand_float(low, high, (len(ids),), self.device)
        self.commands[ids, 1:] = 0.

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if getattr(self, 'course', None) is not None:
            ids = self.course.ids
            # Unlike old flat terrain, all new courses hold +X heading.
            from .math import quat_apply, wrap_to_pi
            forward = quat_apply(self.base_quat[ids], self.forward_vec[ids])
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[ids, 2] = (-.5 * wrap_to_pi(heading)).clamp(-1., 1.)

    def check_termination(self):
        super().check_termination()
        if getattr(self, 'course', None) is None:return
        # Wall contact may be part of climbing. Retain falls/timeouts and detect a grounded base,
        # rather than treating every brief base/wall contact as an immediate fall.
        ids = self.course.ids[self.course.kind[self.course.ids] == 2]
        grounded = self.root_states[ids, 2] < .10
        self.reset_buf[ids] = self.time_out_buf[ids] | self.fall[ids] | self.vel_violate[ids] | grounded
        self.course.update()

    def compute_reward(self):
        super().compute_reward()
        if getattr(self, 'course', None) is not None:self.rew_buf += self.course.reward

    def _reward_lin_vel_z(self):
        penalty = super()._reward_lin_vel_z()
        if getattr(self, 'course', None) is not None:
            ids = self.course.ids
            combo = (self.course.kind[ids] == COMPETITION_TYPES.index('stair_gap')) | (self.course.kind[ids] == COMPETITION_TYPES.index('stair_gap_ramp'))
            jumping = combo & self.course.jump_ready[ids] & ~self.course.coverage[ids, 3]
            penalty[ids[jumping]] *= .2
        return penalty

    def _update_terrain_curriculum(self, env_ids):
        if getattr(self, 'course', None) is None:return super()._update_terrain_curriculum(env_ids)
        legacy = env_ids[~self.course.mask[env_ids]]
        if len(legacy):super()._update_terrain_curriculum(legacy)
        self.course.select_levels(env_ids)

    def reset_idx(self, env_ids):
        if getattr(self, 'course', None) is not None:self.course.finish(env_ids)
        super().reset_idx(env_ids)
        if getattr(self, 'course', None) is not None and len(env_ids):
            self.extras['episode'].update(self.course.last_metrics)
            self.course.reset(env_ids)
