import numpy as np
import torch

from isaacgym.torch_utils import torch_rand_float

from wheel_legged_gym.envs.l5a_balance.l5a_balance import L5A_BALANCE
from .l5a_step_config import L5A_STEP_Cfg


class L5A_STEP(L5A_BALANCE):
    def __init__(self, cfg: L5A_STEP_Cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        self.step_commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
                self.obs_scales.mode,
                self.obs_scales.step_rise_height,
            ],
            device=self.device,
            requires_grad=False,
        )

        self.gait_phase_offset = torch.rand(
            self.num_envs,
            device=self.device,
            requires_grad=False,
        )

        self.desired_stance = torch.ones(
            self.num_envs,
            2,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.desired_swing = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        if not hasattr(self, "raw_base_mass"):
            self.raw_base_mass = self.base_mass.mean()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if hasattr(self, "gait_phase_offset"):
            self.gait_phase_offset[env_ids] = torch.rand(
                len(env_ids),
                device=self.device,
                requires_grad=False,
            )

        super().reset_idx(env_ids)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, :] = 0.0

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["height"][0],
            self.command_ranges["height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 3] = float(self.command_ranges["mode"][0])
        self.commands[env_ids, 4] = 0.0
        self.commands[env_ids, 5] = 0.0

        self.commands[env_ids, 6] = torch_rand_float(
            self.command_ranges["step_rise_height"][0],
            self.command_ranges["step_rise_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.wheel_mode = self.commands[:, 3].unsqueeze(-1)

        if hasattr(self, "jump"):
            self.jump[:] = False
        if hasattr(self, "jump_cmd"):
            self.jump_cmd[env_ids] = 0.0
        if hasattr(self, "jump_request"):
            self.jump_request[env_ids] = False

    def update_stage(self):
        if hasattr(self, "stage_buf"):
            self.stage_buf[:, :] = 0.0
            self.stage_buf[:, 0] = 1.0

        if hasattr(self, "stage_time_buf"):
            self.stage_time_buf += self.dt

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        period = self.cfg.commands.gait_period
        stance_ratio = self.cfg.commands.stance_ratio

        phase_time = self.episode_length_buf.float() * self.dt / period
        '''
        通过 torch.remainder(..., 1.0) 取小数部分，得到一个 [0, 1) 之间的数值。这代表机器人在当前周期内的进度
            remainder就是取余操作
        '''
        self.phase = torch.remainder(phase_time + self.gait_phase_offset, 1.0)

        self.phase_left = self.phase
        self.phase_right = torch.remainder(self.phase + 0.5, 1.0)

        self.leg_phase = torch.cat(
            [
                self.phase_left.unsqueeze(1),
                self.phase_right.unsqueeze(1),
            ],
            dim=-1,
        )

        '''
        后面会被奖励函数使用，来鼓励机器人"在正确的时间做正确的事"——该踩地时踩地，该抬腿时抬腿。
        '''
        self.desired_stance = self.leg_phase < stance_ratio# 左右腿是否应该支撑地面
        self.desired_swing = torch.logical_not(self.desired_stance)# 左右腿是否应该抬起摆动

    def _feet_contact(self):
        contact_threshold = self.cfg.commands.contact_force_threshold
        return self.contact_forces[:, self.feet_indices, 2] > contact_threshold

    def compute_proprioception_observations(self):
        sin_phase = torch.sin(2.0 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2.0 * np.pi * self.phase).unsqueeze(1)

        command_obs = torch.stack(
            [
                self.commands[:, 0],
                self.commands[:, 1],
                self.commands[:, 2],
                self.commands[:, 3],
                self.commands[:, 6],
            ],
            dim=1,
        ) * self.step_commands_scale

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_quat_local * self.obs_scales.quat,
                command_obs,
                self.dof_pos[:, self.joint_indices] * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                sin_phase,
                cos_phase,
                self.desired_swing.float(),
            ),
            dim=-1,
        )

        return obs_buf

    def compute_observations(self):
        self.obs_buf = self.compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )

            external_forces_and_torques = torch.cat(
                (
                    self.push_forces[:, 0, :],
                    self.push_torques[:, 0, :],
                ),
                dim=-1,
            )

            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.base_euler_zyx * self.obs_scales.quat,
                    self.obs_buf,
                    self.projected_gravity,
                    self.last_actions[:, :, 0],
                    self.last_actions[:, :, 1],
                    self.dof_acc * self.obs_scales.dof_acc,
                    heights,
                    self.torques * self.obs_scales.torque,
                    (self.base_mass - self.raw_base_mass).view(self.num_envs, 1),
                    self.base_com,
                    self.friction_coef.view(self.num_envs, 1),
                    self.restitution_coef.view(self.num_envs, 1),
                    external_forces_and_torques * self.priv_obs_scales.external_wrench,
                ),
                dim=-1,
            )

        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

        self.obs_history = torch.cat(
            (
                self.obs_history[:, self.num_obs:],
                self.obs_buf,
            ),
            dim=-1,
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:7] = noise_scales.quat * noise_level * self.obs_scales.quat

        noise_vec[7:12] = 0.0

        noise_vec[12:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18:26] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        noise_vec[26:34] = 0.0
        noise_vec[34:38] = 0.0

        return noise_vec

    def check_termination(self):
        contact_fail = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 10.0,
            dim=1,
        )

        roll_fail = torch.abs(self.base_euler_zyx[:, 0]) > 0.45
        pitch_fail = torch.abs(self.base_euler_zyx[:, 1]) > 0.35

        fail_buf = contact_fail | roll_fail | pitch_fail

        self.fail_buf = self.fail_buf * fail_buf.long() + fail_buf.long()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        self.edge_reset_buf[:] = False
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf |= self.base_position[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1

        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.edge_reset_buf
        )

    def _reward_step_contact(self):
        contact = self._feet_contact()
        return (contact == self.desired_stance).float().mean(dim=1)

    '''
    让轮子实际离地高度 ≈ 目标抬起高度，但只在"应该抬起"的时候才要求匹配
    '''
    def _reward_step_swing_height(self):
        wheel_radius = self.cfg.asset.wheel_radius
        step_height = self.commands[:, 6].unsqueeze(1)

        wheel_clearance = torch.clamp(
            self.feet_pos[:, :, 2] - wheel_radius,
            min=0.0,
            max=0.3,
        )

        target_clearance = self.desired_swing.float() * step_height

        height_error = torch.square(wheel_clearance - target_clearance)
        return torch.exp(-height_error / 0.0025).mean(dim=1)

    '''
    控制轮子在摆动期（抬起阶段）的升降速度
    '''
    def _reward_step_swing_velocity(self):
        stance_ratio = self.cfg.commands.stance_ratio

        # 摆动期进度 [0, 1]
        swing_progress = torch.clamp(
            (self.leg_phase - stance_ratio) / (1.0 - stance_ratio),
            min=0.0,
            max=1.0,
        )

        # 三角波速度曲线：从 +v_max 到 -v_max
        max_swing_vel = 0.25  # m/s
        target_vz = max_swing_vel * (1.0 - 2.0 * swing_progress)

        vz_error = torch.square(self.feet_vel[:, :, 2] - target_vz)
        reward = torch.exp(-vz_error / 0.04) * self.desired_swing.float()

        active_count = self.desired_swing.float().sum(dim=1).clamp(min=1.0)
        return reward.sum(dim=1) / active_count


    def _reward_no_double_air(self):
        contact = self._feet_contact()
        both_air = torch.logical_not(contact[:, 0]) & torch.logical_not(contact[:, 1])
        return both_air.float()

    def _reward_wrong_double_contact(self):
        contact = self._feet_contact()
        both_contact = contact[:, 0] & contact[:, 1]
        desired_double_stance = self.desired_stance[:, 0] & self.desired_stance[:, 1]
        wrong_double_contact = both_contact & torch.logical_not(desired_double_stance)
        return wrong_double_contact.float()

    def _reward_lateral_vel(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_wheel_vel(self):
        return torch.mean(
            torch.square(self.dof_vel[:, self.wheel_indices]),
            dim=1,
        )

    def _reward_torques(self):
        return torch.mean(torch.square(self.torques), dim=1)

    '''
    左轮接触地面时，希望它在机体左侧合理位置；右轮接触地面时，希望它在机体右侧合理位置。摆动腿不接触地面时不参与奖励。
    目的：防止两条腿落地时越靠越近
    '''
    def _reward_stance_wheel_y_position(self):
        contact = self._feet_contact()
        stance_mask = contact.float()

        target_half_width = self.cfg.asset.track_width * 0.5

        left_y = self.wheel_pos_left_local[:, 1]
        right_y = self.wheel_pos_right_local[:, 1]

        left_error = torch.square(left_y - target_half_width)
        right_error = torch.square(right_y + target_half_width)

        reward = torch.stack(
            (
                torch.exp(-left_error / 0.0016),
                torch.exp(-right_error / 0.0016),
            ),
            dim=1,
        )

        active_count = stance_mask.sum(dim=1).clamp(min=1.0)
        return (reward * stance_mask).sum(dim=1) / active_count

    '''
    原地位置漂移惩罚，保证机器人在原地踏步
    '''
    def _reward_base_xy_drift(self):
        xy_error = self.root_states[:, :2] - self.env_origins[:, :2]
        return torch.sum(torch.square(xy_error), dim=1)

