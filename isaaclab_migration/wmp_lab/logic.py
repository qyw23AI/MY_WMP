# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import math

import math, numpy as np, torch
from .math import *
from .config import class_to_dict
COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET



class Logic:
    def normalize_depth_image(self, depth_image):
        depth_image = depth_image * -1
        depth_image = (depth_image - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip)  - 0.5
        return depth_image
    def process_depth_image(self, depth_image, env_id):
        # These operations are replicated on the hardware
        # depth_image = self.crop_depth_image(depth_image)
        depth_image += self.cfg.depth.dis_noise * 2 * (torch.rand(1)-0.5)[0]
        depth_image = torch.clip(depth_image, -self.cfg.depth.far_clip, -self.cfg.depth.near_clip)
        # depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self.normalize_depth_image(depth_image)
        return depth_image
    def crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]
    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        return policy_obs
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        vel_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
        self.vel_violate = ((vel_error > 1.5) & (self.commands[:, 0] < 0.)) | ((vel_error < -1.5) & (self.commands[:, 0] > 0.))
        self.vel_violate *= (self.terrain_levels > 3)

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.vel_violate

        self.fall = (self.root_states[:, 9] < -3.) | (self.projected_gravity[:, 2] > 0.)
        self.reset_buf |= self.fall
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)


        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0
        self.latency_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]

            self.extras["episode"]["max_command_flat_x"] = self.command_ranges["flat_lin_vel_x"][1]
            self.extras["episode"]["max_command_flat_yaw"] = self.command_ranges["flat_ang_vel_yaw"][1]

            self.extras["episode"]["push_interval_s"] = self.cfg.domain_rand.push_interval_s
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # reward curriculum
            if self.cfg.rewards.reward_curriculum:
                for j in range(len(self.cfg.rewards.reward_curriculum_term)):
                    if(name == self.cfg.rewards.reward_curriculum_term[j]):
                        rew *= self.reward_curriculum_coef[j]
                # print('reward:', name, ' coef:', self.reward_curriculum_coef)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    def compute_observations(self):
        """ Computes observations
        """
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

        if (self.cfg.env.privileged_obs):
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

            if self.cfg.domain_rand.randomize_friction:
                self.privileged_obs_buf= torch.cat((self.randomized_frictions, self.privileged_obs_buf), dim=-1)

            if self.cfg.domain_rand.randomize_restitution:
                self.privileged_obs_buf = torch.cat((self.randomized_restitutions, self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_base_mass):
                self.privileged_obs_buf = torch.cat((self.randomized_added_masses ,self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_com_pos):
                self.privileged_obs_buf = torch.cat((self.randomized_com_pos * self.obs_scales.com_pos ,self.privileged_obs_buf), dim=-1)

            if (self.cfg.domain_rand.randomize_gains):
                self.privileged_obs_buf = torch.cat(((self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains ,self.privileged_obs_buf), dim=-1)
                self.privileged_obs_buf = torch.cat(((self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains, self.privileged_obs_buf),
                                                    dim=-1)

            contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
            self.privileged_obs_buf = torch.cat((contact_force, self.privileged_obs_buf), dim=-1)
            contact_flag = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1
            self.privileged_obs_buf = torch.cat((contact_flag, self.privileged_obs_buf), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec


        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 6:
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        elif self.num_obs == self.num_privileged_obs - 3:
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)
    def get_amp_observations(self):
        joint_pos = self.dof_pos
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        # if (self.cfg.terrain.measure_heights):
        #     z_pos = z_pos - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
    def get_full_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        pos = self.root_states[:, :3]
        if (self.cfg.terrain.measure_heights):
            pos[:, 2:3] = pos[:, 2:3] - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        rot = self.root_states[:, 3:7]
        foot_vel = torch.zeros_like(foot_pos)
        return torch.cat((pos, rot, joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, foot_vel), dim=-1)
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
            self.commands[:self.roughflat_start_idx, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:self.roughflat_start_idx, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.measured_forward_heights = self._get_forward_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)


        #resample commands for rough flat terrain
        flat_env_ids = env_ids[torch.where(env_ids >= self.roughflat_start_idx)]
        if(len(flat_env_ids) > 0):
            self.commands[flat_env_ids, 0] = torch_rand_float(self.command_ranges["flat_lin_vel_x"][0],
                                                         self.command_ranges["flat_lin_vel_x"][1], (len(flat_env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[flat_env_ids, 1] = torch_rand_float(self.command_ranges["flat_lin_vel_y"][0],
                                                         self.command_ranges["flat_lin_vel_y"][1], (len(flat_env_ids), 1),
                                                         device=self.device).squeeze(1)
            self.commands[flat_env_ids, 2] = torch_rand_float(self.command_ranges["flat_ang_vel_yaw"][0],
                                                         self.command_ranges["flat_ang_vel_yaw"][1], (len(flat_env_ids), 1),
                                                         device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # set heading command for tilt envs to zero
        self.commands[self.tilt_start_idx:self.tilt_end_idx, 3] = 0
        self.commands[self.pit_start_idx:self.pit_end_idx, 3] = 0
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    def update_reward_curriculum(self, current_iter):
        for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
            percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / \
                         (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
            percentage = max(min(percentage, 1), 0)
            self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + \
                                          percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above a certain percentage of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / (self.max_episode_length * self.reward_scales["tracking_lin_vel"]) +
                torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / (self.max_episode_length * self.reward_scales["tracking_ang_vel"]) > 1.65
                ):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.05,
                                                          -self.cfg.commands.max_lin_vel_backward_x_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_lin_vel_forward_x_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.05,
                                                          -self.cfg.commands.max_lin_vel_y_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_lin_vel_y_curriculum)

            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.025,
                                                          -self.cfg.commands.max_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.025, 0.,
                                                          self.cfg.commands.max_ang_vel_yaw_curriculum)

            self.command_ranges["flat_lin_vel_x"][0] = np.clip(self.command_ranges["flat_lin_vel_x"][0] - 0.05,
                                                          -self.cfg.commands.max_flat_lin_vel_backward_x_curriculum, 0.)
            self.command_ranges["flat_lin_vel_x"][1] = np.clip(self.command_ranges["flat_lin_vel_x"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_flat_lin_vel_forward_x_curriculum)
            self.command_ranges["flat_lin_vel_y"][0] = np.clip(self.command_ranges["flat_lin_vel_y"][0] - 0.05,
                                                          -self.cfg.commands.max_flat_lin_vel_y_curriculum, 0.)
            self.command_ranges["flat_lin_vel_y"][1] = np.clip(self.command_ranges["flat_lin_vel_y"][1] + 0.05, 0.,
                                                          self.cfg.commands.max_flat_lin_vel_y_curriculum)

            self.command_ranges["flat_ang_vel_yaw"][0] = np.clip(self.command_ranges["flat_ang_vel_yaw"][0] - 0.1,
                                                          -self.cfg.commands.max_flat_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["flat_ang_vel_yaw"][1] = np.clip(self.command_ranges["flat_ang_vel_yaw"][1] + 0.1, 0.,
                                                          self.cfg.commands.max_flat_ang_vel_yaw_curriculum)

            self.cfg.domain_rand.push_interval_s = max(self.cfg.domain_rand.push_interval_s - 0.5, self.cfg.domain_rand.min_push_interval_s)
            self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_start_dim = self.privileged_dim - 3 # last 3-dim is the linear vel
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[noise_start_dim:noise_start_dim+3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[noise_start_dim+3:noise_start_dim+6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[noise_start_dim+6:noise_start_dim+9] = noise_scales.gravity * noise_level
        noise_vec[noise_start_dim+9:noise_start_dim+12] = 0. # commands
        noise_vec[noise_start_dim+12:noise_start_dim+24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[noise_start_dim+24:noise_start_dim+36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[noise_start_dim+36:noise_start_dim+48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[noise_start_dim+48:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec
    def compute_randomized_gains(self, num_envs):
        p_mult = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        d_mult = torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1],
                                  (num_envs, self.num_actions), device=self.device)
        return p_mult * self.p_gains, d_mult * self.d_gains
    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)
    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            # self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
            self.terrain_levels = torch.fmod(torch.arange(self.num_envs, device=self.device), max_init_level + 1)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    def _init_forward_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_forward_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_forward_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_forward_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_forward_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    def _get_forward_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_forward_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_forward_height_points), self.forward_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_forward_height_points), self.forward_height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    def get_forward_map(self):
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_forward_heights, -1,
                             1.) * self.obs_scales.height_measurements
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    def _reward_torques_distribution(self):
        # Penalize torques
        return torch.var(torch.abs(self.torques), dim=1)
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    def _reward_dof_pos_dif(self):
        return torch.sum(torch.square(self.last_dof_pos - self.dof_pos), dim=1)
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

        # clipping tracking reward
        lin_vel = self.base_lin_vel[:, :2].clone()
        lin_vel_upper_bound = torch.where(self.commands[:, :2] < 0, 1e5, self.commands[:, :2] + self.cfg.rewards.lin_vel_clip)
        lin_vel_lower_bound = torch.where(self.commands[:, :2] > 0, -1e5, self.commands[:, :2] - self.cfg.rewards.lin_vel_clip)
        clip_lin_vel = torch.clip(lin_vel, lin_vel_lower_bound, lin_vel_upper_bound)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - clip_lin_vel), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact = self.sensor_forces[:, :, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    def _reward_action_magnitude(self):
        return torch.sum(torch.square(torch.maximum(torch.abs(self.actions[:,[0,3,6,9]]) - 1.0,torch.zeros_like(self.actions[:,[0,3,6,9]]))), dim=1)
    def _reward_power(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
    def _reward_power_distribution(self):
        return torch.var(torch.abs(self.torques * self.dof_vel), dim=1)
    def _reward_smoothness(self):
        return torch.sum(torch.square(self.last_last_actions - 2*self.last_actions + self.actions), dim=1)
    def _reward_clearance(self):
        # foot_pos = self.rigid_body_pos[:, self.feet_indices,:]
        #
        # foot_pos -= self.root_states[:,:3].unsqueeze(1)
        # foot_pos = torch.reshape(foot_pos,(self.num_envs * 4,-1))
        # foot_pos_base = quat_rotate_inverse(self.base_quat.repeat(4, 1), foot_pos)
        # foot_pos_base = torch.reshape(foot_pos_base,(self.num_envs,4,-1))
        # foot_heights = foot_pos_base[:,:,2]

        if self.cfg.terrain.mesh_type == 'plane':
            foot_heights = self.rigid_body_pos[:, self.feet_indices, 2]
        else:
            points = self.rigid_body_pos[:, self.feet_indices,:]

            #measure ground height under the foot
            points += self.terrain.cfg.border_size
            points = (points / self.terrain.cfg.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py]
            heights3 = self.height_samples[px, py + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            ground_heights = torch.reshape(heights, (self.num_envs, -1)) * self.terrain.cfg.vertical_scale
            foot_heights = self.rigid_body_pos[:, self.feet_indices, 2] - ground_heights

        foot_lateral_vel = torch.norm(self.rigid_body_lin_vel[:, self.feet_indices,:2], dim = -1)
        # return torch.sum(foot_lateral_vel * torch.maximum(-foot_heights + self.cfg.rewards.foot_height_target, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(foot_lateral_vel * torch.square(foot_heights - self.cfg.rewards.foot_height_target), dim = -1)
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0,3,6,9]] - self.default_dof_pos[:, [0,3,6,9]]), dim=1)
    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:self.roughflat_start_idx, 1], forward[:self.roughflat_start_idx, 0])
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        cheat_penalty[:self.roughflat_start_idx] = cheat
        return cheat_penalty
    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices,
                        :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)

        edge_reward = torch.zeros_like(rew)
        edge_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        return edge_reward
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew * (self.terrain_levels > 3)

        rew = rew.float()
        stumble_reward = torch.zeros_like(rew)
        stumble_reward[self.gap_start_idx:self.pit_end_idx] = rew[self.gap_start_idx:self.pit_end_idx]
        return stumble_reward
    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    def _reward_stuck(self):
        # Penalize stuck
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)

    def init_tensors(self):
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros_like(self.last_actions)
        # for latency
        self.latency_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                           requires_grad=False)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.forward_height_points = self._init_forward_height_points()
        self.measured_heights = 0
        self.measured_forward_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

        if self.cfg.depth.use_camera:
            self.depth_buffer = torch.zeros(self.cfg.depth.camera_num_envs,
                                            self.cfg.depth.buffer_len,
                                            self.cfg.depth.resized[0],
                                            self.cfg.depth.resized[1]).to(self.device)
