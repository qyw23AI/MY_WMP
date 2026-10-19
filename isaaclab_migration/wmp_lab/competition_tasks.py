"""Course state, strict completion metrics and success-rate curricula for A1.

Runs before reward/reset. No privileged target or coverage bits enter the actor.
Original Logic remains unchanged; this mixin is inactive for legacy profiles.
"""
import math
import torch
from .competition import COMPETITION_TYPES, HeightQuery
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
        self.entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self.visited = torch.zeros_like(self.entered)
        self.failed = torch.zeros_like(self.entered)
        self.success = torch.zeros_like(self.entered)
        self.strict = torch.zeros_like(self.entered)
        self.hold = torch.zeros(n, device=dev)
        self.max_x = torch.zeros(n, device=dev)
        self.reward = torch.zeros(n, device=dev)
        self.frontier = torch.zeros(3, dtype=torch.long, device=dev)
        self.attempts = torch.zeros(3, 10, dtype=torch.long, device=dev)
        self.wins = torch.zeros_like(self.attempts)
        self.strict_wins = torch.zeros_like(self.attempts)
        self.window_n = torch.zeros_like(self.attempts)
        self.window_wins = torch.zeros_like(self.attempts)
        self.violated = torch.zeros_like(self.entered)
        self.totals = torch.zeros(3,10,12,dtype=torch.long,device=dev)
        self.last_metrics = {}
        self.reset(self.ids)

    def state_dict(self):
        names=('frontier','attempts','wins','strict_wins','window_n','window_wins','totals')
        return {'version':1, **{name:getattr(self,name).detach().cpu().clone() for name in names}}

    def load_state_dict(self, state):
        if state.get('version') != 1:raise ValueError('Unknown competition curriculum checkpoint version')
        for name in ('frontier','attempts','wins','strict_wins','window_n','window_wins','totals'):
            if name not in state:continue
            target=getattr(self,name)
            if state[name].shape != target.shape:raise ValueError(f'Invalid curriculum tensor {name}')
            target.copy_(state[name].to(target.device))

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
        self.violated[ids] = False
        self.failed[ids] = False; self.success[ids] = False; self.strict[ids] = False
        self.hold[ids] = 0.; self.reward[ids] = 0.
        self.max_x[ids] = self.c.env_origins[ids, 0] - self.centers[ids, 0]

    def heights(self, world_xy, ids):
        return self.query(world_xy - self.centers[ids, None], self.c.terrain_levels[ids], self.c.terrain_types[ids])

    def update(self):
        c, ids = self.c, self.ids
        self.reward.zero_()
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
        entry = torch.where(k == 0, boxes[:, 6, 0] - .4, torch.where(k == 1, -1.1, -.04))
        exit_x = torch.where(k == 0, boxes[:, 7, 0] + .4, torch.where(k == 1, 1.1, .04))
        ground = contact & (sole_z < .025)
        stones = k == 0; tires = k == 1; wall = k == 2
        # Record real top support on both platforms; side contact is insufficient.
        platform_boxes = boxes[:, 6:8]
        platform_top = ((feet[:, :, None, :2] - platform_boxes[:, None, :, :2]).abs()
                        <= platform_boxes[:, None, :, 2:4] / 2 - .005).all(-1)
        platform_top &= (sole_z[:, :, None] - level_height[:, None, None]).abs() < .02
        platform_touch = (platform_top & contact[:, :, None]).any(1) & stones[:, None]
        new_platform = platform_touch & ~self.platforms[ids]
        self.platforms[ids] |= platform_touch
        # During ascent rear feet may remain on the approach ground. Arm only
        # after entry-platform support and all four soles clear ground height.
        fully_inside = (feet[:, :, 0] > entry[:, None]).all(-1)
        on_course = (sole_z >= level_height[:, None] - .02).all(-1)
        self.entered[ids] |= fully_inside & (~stones | (self.platforms[ids, 0] & on_course))
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
        bypass = bad_wall | side_fall | skipped_entry | (self.entered[ids] & bad_tires)
        bad = (self.entered[ids] & bad_stones) | bypass
        self.failed[ids] |= bad | (c.reset_buf[ids] & ~c.time_out_buf[ids])
        cleared = (feet[:, :, 0] > exit_x[:, None] + .05).all(-1)
        cleared &= root[:, 0] > exit_x + .35
        stable = (contact.sum(-1) >= 2) & (c.projected_gravity[ids, 2] < -.8) & (c.root_states[ids, 9].abs() < .35)
        self.hold[ids] = torch.where(cleared & stable, self.hold[ids] + c.dt, 0.)
        reached_exit = (self.hold[ids] >= .30) & self.entered[ids] & ~self.failed[ids]
        complete = reached_exit & (~tires | self.visited[ids])
        complete &= ~stones | self.platforms[ids].all(-1)
        full = self.coverage[ids].all(-1)
        strict = complete & (~stones | full)
        # Early stone levels teach traversal; levels 3-9 require all six tops.
        success = complete & (~stones | full | (rows < 3))
        # Reaching the exit while missing a required stone is a failed attempt.
        missed = reached_exit & stones & (~self.platforms[ids].all(-1) | (~full & (rows >= 3)))
        self.violated[ids] |= bad | missed
        self.failed[ids] |= missed
        new_success = success & ~self.success[ids]
        self.success[ids] |= success; self.strict[ids] |= strict
        progress = (root[:, 0] - self.max_x[ids]).clamp(0., .06)
        self.max_x[ids] = torch.maximum(self.max_x[ids], root[:, 0])
        self.reward[ids] = (2. * progress + .5 * new.sum(-1) + new_platform.sum(-1)
                            + 8 * new_success.float() - 5 * (bad | missed).float()
                            - 5 * bypass.float())
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
            self.last_metrics[f'{name}/standard_strict_rate'] = self.strict_wins[k,9].float()/self.attempts[k,9].clamp(min=1)
            self.last_metrics[f'{name}/success'] = self.success[selected].float().mean()
            self.last_metrics[f'{name}/strict_success'] = self.strict[selected].float().mean()
            self.last_metrics[f'{name}/coverage'] = self.coverage[selected].float().mean()
            if k == 0:
                self.last_metrics[f'{name}/entry_platform'] = self.platforms[selected, 0].float().mean()
                self.last_metrics[f'{name}/exit_platform'] = self.platforms[selected, 1].float().mean()
            self.last_metrics[f'{name}/failure'] = self.failed[selected].float().mean()
            self.last_metrics[f'{name}/timeout'] = c.time_out_buf[selected].float().mean()
            top = selected[level == 9]
            self.last_metrics[f'{name}/standard_attempts'] = self.attempts[k, 9].float()
            if len(top):self.last_metrics[f'{name}/standard_success'] = self.strict[top].float().mean()
            frontier = int(self.frontier[k])
            count = int(self.window_n[k, frontier])
            if count >= c.cfg.terrain.competition_window:
                rate = float(self.window_wins[k, frontier]) / count
                if rate >= c.cfg.terrain.competition_promote:self.frontier[k] = min(9, frontier+1)
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
        if hasattr(self.cfg.terrain, 'eval_terrain_col'):
            self.terrain_levels.fill_(self.cfg.terrain.eval_terrain_level)
            if competition:self.terrain_types.fill_(self.cfg.terrain.eval_terrain_col)
            else:self.terrain_types = (torch.arange(self.num_envs,device=self.device)+self.cfg.terrain.eval_terrain_col)%self.cfg.terrain.num_cols
        elif competition and self.cfg.terrain.competition_stage == 'eval':
            self.terrain_levels.fill_(9)
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
