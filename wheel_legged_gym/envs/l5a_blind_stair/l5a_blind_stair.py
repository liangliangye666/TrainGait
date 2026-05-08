import math
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from wheel_legged_gym.envs.l5a_step.l5a_step import L5A_STEP
from wheel_legged_gym.utils.math import quat_from_euler_zyx

from .blind_stair_terrain import BlindStairTerrain
from .l5a_blind_stair_config import L5A_BLIND_STAIR_Cfg


class L5A_BLIND_STAIR(L5A_STEP):
    def __init__(self, cfg: L5A_BLIND_STAIR_Cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = BlindStairTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("bad terrain mesh type")
        self._create_envs()

    def _init_buffers(self):
        super()._init_buffers()
        n = int(self.cfg.commands.contact_window_size)
        self.xy_contact_force_history = torch.zeros(self.num_envs, n, 2, dtype=torch.bool, device=self.device)
        self.raw_xy_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.stable_xy_contact = torch.zeros_like(self.raw_xy_contact)
        self.swing_timer = torch.full((self.num_envs, 2), -1.0, dtype=torch.float, device=self.device)
        self.primary_swing_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.triggered_this_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.spawn_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.heading_vec = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.heading_vec[:, 0] = 1.0
        self.forward_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_forward_progress = torch.zeros_like(self.forward_progress)
        self.initial_base_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_base_height_for_reward = torch.zeros_like(self.initial_base_height)

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        if len(env_ids) == 0:
            return
        yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        zyx = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
        zyx[:, 2] = yaw
        self.root_states[env_ids, 3:7] = quat_from_euler_zyx(zyx)
        self.root_states[env_ids, 7:13] = 0.0
        self.spawn_yaw[env_ids] = yaw
        self.heading_vec[env_ids, 0] = torch.cos(yaw)
        self.heading_vec[env_ids, 1] = torch.sin(yaw)
        ids = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(ids), len(ids))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.xy_contact_force_history[env_ids] = False
        self.raw_xy_contact[env_ids] = False
        self.stable_xy_contact[env_ids] = False
        self.swing_timer[env_ids] = -1.0
        self.triggered_this_episode[env_ids] = False
        self.initial_base_height[env_ids] = self.root_states[env_ids, 2]
        self.last_base_height_for_reward[env_ids] = self.root_states[env_ids, 2]
        self.forward_progress[env_ids] = 0.0
        self.last_forward_progress[env_ids] = 0.0

    def _range_values(self, name, env_ids):
        value = self.command_ranges[name]
        if torch.is_tensor(value):
            return value[env_ids, 0], value[env_ids, 1]
        return value[0], value[1]

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        low, high = self._range_values("lin_vel_x", env_ids)
        if not torch.is_tensor(low):
            low = torch.full((len(env_ids),), float(low), device=self.device)
            high = torch.full((len(env_ids),), float(high), device=self.device)
        self.commands[env_ids, :] = 0.0
        self.commands[env_ids, 0] = (high - low) * torch.rand(len(env_ids), device=self.device) + low
        h_low, h_high = self._range_values("height", env_ids)
        if not torch.is_tensor(h_low):
            h_low = torch.full((len(env_ids),), float(h_low), device=self.device)
            h_high = torch.full((len(env_ids),), float(h_high), device=self.device)
        self.commands[env_ids, 2] = (h_high - h_low) * torch.rand(len(env_ids), device=self.device) + h_low
        self.commands[env_ids, 3] = 1.0
        self.commands[env_ids, 6] = self.cfg.terrain.blind_stair_step_height
        self.wheel_mode = self.commands[:, 3].unsqueeze(-1)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        xy = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=-1)
        self.raw_xy_contact = xy > self.cfg.commands.contact_force_threshold
        self.xy_contact_force_history = torch.cat((self.xy_contact_force_history[:, 1:, :], self.raw_xy_contact.unsqueeze(1)), dim=1)
        self.stable_xy_contact = torch.all(self.xy_contact_force_history, dim=1)
        duration = self.cfg.commands.swing_duration
        delay = self.cfg.commands.contralateral_delay
        valid = self.swing_timer > -0.5
        self.swing_timer[valid] += self.dt
        self.swing_timer[self.swing_timer > duration] = -1.0
        idle = torch.all(self.swing_timer < 0.0, dim=1)
        trigger = idle & torch.any(self.stable_xy_contact, dim=1)
        if torch.any(trigger):
            left = self.stable_xy_contact[:, 0]
            right = self.stable_xy_contact[:, 1]
            big = torch.argmax(xy, dim=1)
            pick = torch.where(left & ~right, torch.zeros_like(big), torch.where(right & ~left, torch.ones_like(big), big))
            ids = trigger.nonzero(as_tuple=False).flatten()
            p = pick[ids]
            s = 1 - p
            self.primary_swing_leg[ids] = p
            self.swing_timer[ids, p] = 0.0
            self.swing_timer[ids, s] = -delay
            self.triggered_this_episode[ids] = True
        delayed = (self.swing_timer < 0.0) & (self.swing_timer > -0.5)
        self.swing_timer[delayed] += self.dt
        active = (self.swing_timer >= 0.0) & (self.swing_timer <= duration)
        progress = torch.clamp(self.swing_timer / duration, min=0.0, max=1.0)
        self.desired_swing = active
        self.desired_stance = torch.logical_not(active)
        self.leg_phase = torch.where(active, progress, torch.zeros_like(progress))
        self.phase = torch.max(self.leg_phase, dim=1).values

    def _feet_contact(self):
        return self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.min_wheel_contact_force

    def _projected_forward_progress(self):
        return torch.sum((self.root_states[:, :2] - self.env_origins[:, :2]) * self.heading_vec, dim=1)

    def _reward_contact_trigger_swing(self):
        c = self._feet_contact()
        s = self.desired_swing
        count = torch.clamp(torch.sum(s.float(), dim=1), min=1.0)
        return torch.sum((s & (~c)).float(), dim=1) / count

    def _reward_contact_trigger_clearance(self):
        s = self.desired_swing.float()
        clearance = torch.clamp(self.feet_pos[:, :, 2] - self.cfg.asset.wheel_radius, min=0.0, max=0.3)
        target = self.cfg.terrain.blind_stair_step_height + 0.03
        score = torch.exp(-torch.square(clearance - target) / 0.005) * s
        return torch.sum(score, dim=1) / torch.clamp(torch.sum(s, dim=1), min=1.0)

    def _reward_contact_trigger_sequence(self):
        n = torch.sum(self.desired_swing.float(), dim=1)
        return (n <= 1.1).float() * (0.5 + 0.5 * self.triggered_this_episode.float())

    def _reward_climb_progress(self):
        p = self._projected_forward_progress()
        d = p - self.last_forward_progress
        self.forward_progress[:] = p
        return torch.clamp(d, min=-0.05, max=0.10) / 0.10

    def _reward_stair_height_progress(self):
        dz = self.root_states[:, 2] - self.last_base_height_for_reward
        total = self.root_states[:, 2] - self.initial_base_height
        return torch.clamp(dz, min=-0.02, max=0.08) / 0.08 + 0.1 * torch.clamp(total, min=0.0, max=0.8)

    def _reward_wheel_zero_velocity_in_swing(self):
        s = self.desired_swing.float()
        w = torch.abs(self.dof_vel[:, self.wheel_indices[[0, 1]]])
        score = torch.exp(-0.05 * w ** 2) * s
        return torch.sum(score, dim=1) / torch.clamp(torch.sum(s, dim=1), min=1.0)

    def post_physics_step(self):
        super().post_physics_step()
        self.last_forward_progress[:] = self.forward_progress[:]
        self.last_base_height_for_reward[:] = self.root_states[:, 2]
