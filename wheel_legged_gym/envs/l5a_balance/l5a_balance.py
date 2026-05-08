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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym import torch_utils
import torch
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs.base.legged_robot import LeggedRobot
from wheel_legged_gym.utils.terrain import Terrain
from wheel_legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
    get_euler_zyx_tensor,
    quat_from_euler_zyx,
)
from wheel_legged_gym.utils.helpers import class_to_dict
from .l5a_balance_config import L5A_BALANCE_Cfg
import math

class L5A_BALANCE(LeggedRobot):
    def __init__(self, cfg: L5A_BALANCE_Cfg, sim_params, physics_engine, sim_device, headless):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg # 获取训练环境配置参数
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def step(self, actions):
        """
        应用动作，进行模拟，并调用 self.post_physics_step()。

        该方法首先对输入的动作进行裁剪，以确保其在允许的范围内。然后，它渲染当前帧，调用 pre_physics_step() 方法进行模拟前的准备工作，
        并在模拟过程中多次调用 leg_post_physics_step() 方法。在模拟结束后，它调用 post_physics_step() 方法进行模拟后的处理工作，
        并返回裁剪后的观测、奖励、完成标志和其他信息。

        Args:
            actions (torch.Tensor): 形状为 (num_envs, num_actions_per_env) 的动作张量
        """
        # 裁剪动作以确保其在允许的范围内
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 渲染当前帧
        self.render()

        # 在模拟前进行准备工作,实际准备的是奖励函数,包含了即时奖励和奖励积分
        self.pre_physics_step()

        # 进行多次模拟步骤
        for _ in range(self.cfg.control.decimation):
            # 在模拟后进行处理工作
            # self.leg_post_physics_step()
            # 更新环境步数缓冲区
            self.envs_steps_buf += 1
            # 更新动作FIFO缓冲区,fifo维度(num_envs,max_delays,num_actions),unsqueeze(1)->在维度1增加一个大小为1的维度,也就是(num_envs,1,num_actions),注意这里是把新动作插到了前面
            self.action_fifo = torch.cat((self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1) # action_fifo[:,:-1,:]->现有动作队列,[所有环境,从开始到倒数第二个元素,所有动作维度]
            # 计算扭矩                                                                                     torch.cat(tensors,dim=1)->沿指定维度连接张量序列,前提是其他维度要一致
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :] # torch.arange()-生成一维整数序列(0,1,2,3...,num_envs-1)
            ).view(self.torques.shape)
            # 设置执行器的扭矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # 将力矩设置到gym中
            # 如果启用了机器人推动功能，则随机推动机器人
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            # 进行模拟
            self.gym.simulate(self.sim) # 仿真一步
            # 如果设备是CPU，则获取模拟结果
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            # 更新自由度状态张量
            self.gym.refresh_dof_state_tensor(self.sim) # 更新自由度状态张量
            '''
            踏步
            '''
            self.gym.refresh_dof_force_tensor(self.sim)##add
            # 计算自由度速度
            self.compute_dof_vel()

        # 在模拟后进行处理工作
        self.post_physics_step()

        # 返回裁剪后的观测、裁剪后的状态（如果有）、奖励、完成标志和其他信息
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
        )

    def compute_dof_vel(self):
        # 计算关节位置差分，并处理角度边界（例如从359度到1度）
        # 使用remainder（取余）来确保角度差分在[-π, π]范围内
        diff = torch.remainder(self.dof_pos - self.last_dof_pos + self.pi, 2 * self.pi) - self.pi

        # 计算关节角速度（位置差分除以时间步长）
        self.dof_pos_dot = diff / self.sim_params.dt

        # 如果配置文件指定使用位置差分作为速度
        # 这通常用于处理速度测量不准确的情况
        if self.cfg.env.dof_vel_use_pos_diff:
            self.dof_vel = self.dof_pos_dot

        # 更新上一时刻的关节位置
        self.last_dof_pos[:] = self.dof_pos[:]

    def post_physics_step(self):
        """
        检查终止条件，计算观测值和奖励。
        调用 self._post_physics_step_callback() 进行通用计算。
        如果需要，调用 self._draw_debug_vis()。

        该方法在物理模拟步骤之后执行以下操作：
        1. 刷新 actor 根状态张量、网络接触力张量和刚体状态张量。
        2. 更新 episode 长度缓冲区和通用步骤计数器。
        3. 准备各种物理量，如基座的四元数、线速度、角速度、加速度等。
        4. 调用 self._post_physics_step_callback() 进行通用计算。
        5. 检查终止条件，计算奖励，重置环境，计算观测值。
        6. 更新上一次的动作、基座位置、自由度速度和根速度。
        7. 如果启用了调试可视化，则绘制调试可视化。
        """
        # 刷新 actor 根状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 刷新网络接触力张量
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 刷新刚体状态张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 更新 episode 长度缓冲区
        self.episode_length_buf += 1
        # 更新通用步骤计数器,该变量从不重置,持续累加直到训练结束
        self.common_step_counter += 1

        #jump
        self.progress_buf += 1


        # 准备基座的四元数
        self.base_quat[:] = self.root_states[:, 3:7]

        # 计算基座的线速度
        # self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        # # 将基座线速度从世界坐标系转换到基座坐标系
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # 将基座角速度从世界坐标系转换到基座坐标系
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 计算加速度
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt

        # 将重力向量从世界坐标系转换到基座坐标系
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 获取基座的欧拉角
        self.base_euler_zyx = get_euler_zyx_tensor(self.base_quat)

        # without yaw
        base_euler_zyx_local = self.base_euler_zyx.clone()
        base_euler_zyx_local[:, 2] = 0

        self.base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)
        self.base_lin_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 7:10]) # head坐标系中的线速度
        self.base_ang_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 10:13]) # head坐标系中的角速度
        # 使用轮子速度作为机器人线速度,使用轮子差速作为机器人角速度
        self.wheel_lin_vel = torch.sum(self.dof_vel[:, self.wheel_indices[[0, 1]]], dim=1) * self.cfg.asset.wheel_radius * 0.5
        self.wheel_ang_vel = (
            (self.dof_vel[:, self.wheel_indices[1]] - self.dof_vel[:, self.wheel_indices[0]]) * self.cfg.asset.wheel_radius / self.cfg.asset.track_width
        )

        # 计算基座坐标系下两个轮的位置和姿态
        self.wheel_pos_left_local = quat_rotate_inverse(self.base_quat, (self.rigid_body_pos[:, self.wheel_link_indices[0], :]-self.rigid_body_pos[:, self.base_link_indices[0], :]))
        self.wheel_pos_right_local = quat_rotate_inverse(self.base_quat, (self.rigid_body_pos[:, self.wheel_link_indices[1], :]-self.rigid_body_pos[:, self.base_link_indices[0], :]))
        self.wheel_euler_left = get_euler_zyx_tensor(self.rigid_body_quat[:, self.wheel_indices[0], :])
        self.wheel_euler_left_local = quat_rotate_inverse(self.base_quat, self.wheel_euler_left)
        self.wheel_euler_right = get_euler_zyx_tensor(self.rigid_body_quat[:, self.wheel_indices[1], :])
        self.wheel_euler_right_local = quat_rotate_inverse(self.base_quat, self.wheel_euler_right)
        self.forward_kinematics() # 计算运动学信息

        '''
        踏步
        '''
        self.update_feet_state()

        # 调用回调函数进行通用计算,重新采样命令和地形高度信息
        self._post_physics_step_callback()

        # 控制命令重采样
        # jump_edge = (self.jump_cmd > 0.5) & (self.last_jump_cmd < 0.5)
        # self.jump_request |= jump_edge
        # self.commands[:, 5] = self.jump_request
        # self.commands[:, 5] = 0


        # 检查终止条件
        self.check_termination()
        # 计算奖励
        self.compute_reward()
        #jump state update
        self.update_stage()
        # print("stand:", self.stage_buf[:, 0].sum().item(),",sit:", self.stage_buf[:, 1].sum().item(),",jump:", self.stage_buf[:, 2].sum().item()
        #       ,",air:", self.stage_buf[:, 3].sum().item(),",land:", self.stage_buf[:, 4].sum().item())
        # print(self.base_height)
        # 获取需要重置的环境 ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 重置指定的环境
        self.reset_idx(env_ids)
        # 计算观测值
        self.compute_observations()  # 在某些情况下，可能需要进行模拟步骤以刷新某些观测值（例如身体位置）

        # 更新上一次的动作
        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        # 更新上一次的基座位置
        self.last_base_position[:] = self.base_position[:]
        # 更新上一次的joint速度
        self.last_dof_vel[:] = self.dof_vel[:]
        # 更新上一次的base速度_inWorld
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # 更新上一次的高度
        self.base_height_last[:] = self.base_height

        self.last_jump_cmd = self.jump_cmd.clone()
        # reset 的 env，强制清零
        if len(env_ids) > 0:
            self.last_jump_cmd[env_ids] = 0.0

        # 如果启用了调试可视化，则绘制调试可视化
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        # 检查环境是否需要重置
        reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10.0,
            dim=1,
        )
        # 如果接触力超过10N，则认为是失败

        fail_buf = torch.logical_or(
            torch.abs(self.base_euler_zyx[:, 1]) > 0.2, torch.abs(self.base_euler_zyx[:, 0]) > 0.2
        )  # base_euler_zyx (roll pitch yaw)
        # 检查横滚或俯仰角是否超过阈值

        self.fail_buf *= fail_buf  # 如果已失败，则保持失败状态
        self.fail_buf += fail_buf  # 如果是第一次失败，则标记为失败
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # 检查是否达到最大步数
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            # 如果机器人在x轴方向超出最大x坐标，则重置
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            # 如果机器人在x轴方向超出最小x坐标，则重置
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            # 如果机器人在y轴方向超出最大y坐标，则重置
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
            # 如果机器人在y轴方向超出最小y坐标，则重置

        ##jump
        sit_timeout   = 5.0
        sit_stuck = (
            (self.stage_buf[:, 1] == 1) &
            (self.stage_time_buf > sit_timeout) &
            (self.base_lin_vel[:, 2] > -0.01)
        )
        land_timeout = 3.0

        jump_timeout  = 3.0
        stage_timeout = (
            (sit_stuck)   |
            ((self.stage_buf[:, 2] == 1) & (self.stage_time_buf > jump_timeout))  |
            ((self.stage_buf[:, 4] == 1) & (self.stage_time_buf > land_timeout))
        )
   
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt) | self.time_out_buf | self.edge_reset_buf | stage_timeout | reset_buf
        )
        # 将达到失败次数、超时或超出边界的情况都标记为重置
        # fail_buf：失败标记
        # time_out_buf：超时标记
        # edge_reset_buf：超出边界标记

    def forward_kinematics(self):
        self.wheel_pos_left_local_x = self.wheel_pos_left_local[:, 0]
        self.wheel_pos_left_local_y = self.wheel_pos_left_local[:, 1]
        self.wheel_pos_left_local_z = self.wheel_pos_left_local[:, 2]
        self.wheel_pos_right_local_x = self.wheel_pos_right_local[:, 0]
        self.wheel_pos_right_local_y = self.wheel_pos_right_local[:, 1]
        self.wheel_pos_right_local_z = self.wheel_pos_right_local[:, 2]
        self.wheel_euler_left_local_x = self.wheel_euler_left_local[:, 0]
        self.wheel_euler_left_local_z = self.wheel_euler_left_local[:, 2]
        self.wheel_euler_right_local_x = self.wheel_euler_right_local[:, 0]
        self.wheel_euler_right_local_z = self.wheel_euler_right_local[:, 2]
        # print("wheel_x: ", self.wheel_pos_left_local_x, self.wheel_pos_right_local_x)
        # print("wheel_y: ", self.wheel_pos_left_local_y, self.wheel_pos_right_local_y)
        # print("wheel_ind", self.wheel_indices)
        # print("end_x", self.end_x)

    def reset_idx(self, env_ids):
        """Reset some environments.
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
        if self.cfg.terrain.curriculum: # 启用地形课程学习
            self._update_terrain_curriculum(env_ids) # 调整地形复杂度与命令
            if self.cfg.commands.curriculum: # 根据超时环境调整命令生成难度
                time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten() # 找到所有超时的环境,超时环境不一定表示失败,而是可能表示机器人已经掌握了当前难度水平,需要更大挑战
                self.update_command_curriculum(time_out_env_ids) # 调整命令
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0): # 每个回合更新一次
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids) # 重置关节状态
        self._reset_root_states(env_ids) # 重置本体位置/姿态
        self._resample_commands(env_ids) # 为每个环境生成新任务指令
        # reset buffers
        self.last_actions[env_ids] = 0.0 # 运动历史清零
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0 # 回合计数器重置,统计当前回合已进行的步数,回合有终止或失败条件,然后重置时该变量刷新重置
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0 # 环境从创建以来累计的总步数,用于课程学习,固定步数周期做一次调整等地方
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]

        self.measured_heights = self._get_heights()
        self.base_height_last[env_ids] = torch.mean(self.root_states[env_ids, 2].unsqueeze(1) - self.measured_heights[env_ids], dim=1,)# base高度
        
        self.obs_history[env_ids] = 0 
        obs_buf = self.compute_proprioception_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length) # 用当前观测值重置历史观测
        # fill extras
        self.extras["episode"] = {} # 创建一个空字典用于存储当前回合的统计信息
        for key in self.episode_sums.keys(): # 遍历所有奖励累加器
            self.extras["episode"]["rew_" + key] = ( # 计算标准化平均奖励
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0 # 计算env_ids环境下key奖励的平均奖励
        # log additional curriculum info
        if self.cfg.terrain.curriculum: # 地形课程信息记录
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum: # 命令课程信息记录
            self.extras["episode"]["a_flat_max_command_x"] = torch.mean( # 记录在平地上允许的最大线速度命令
                self.command_ranges["lin_vel_x"][self.flat_idx, 1].float()
            )
        # 记录各种非平地形的最大线速度命令(x方向)
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum: # 当两种课程同时启用时的地形分类命令难度
            self.extras["episode"]["a_smooth_slope_max_command_x"] = torch.mean( # 平滑斜坡地形
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
            self.extras["episode"]["a_rough_slope_max_command_x"] = torch.mean( # 粗糙斜坡地形
                self.command_ranges["lin_vel_x"][self.rough_slope_idx, 1].float()
            )
            self.extras["episode"]["a_stair_up_max_command_x"] = torch.mean( # 上楼梯地形
                self.command_ranges["lin_vel_x"][self.stair_up_idx, 1].float()
            )
            self.extras["episode"]["a_stair_down_max_command_x"] = torch.mean( # 下楼梯地形
                self.command_ranges["lin_vel_x"][self.stair_down_idx, 1].float()
            )
            self.extras["episode"]["a_discrete_max_command_x"] = torch.mean( # 离散障碍地形
                self.command_ranges["lin_vel_x"][self.discrete_idx, 1].float()
            )
        # send timeout info to the algorithm,发送超时信息给强化学习算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7] # 本体姿态四元数更新
        self.base_euler_zyx = get_euler_zyx_tensor(self.base_quat) # 欧拉角姿态计算
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids]) # 重力在基座坐标系的投影

        # without yaw
        base_euler_zyx_local = self.base_euler_zyx.clone() # 局部姿态处理,忽略yaw角度
        base_euler_zyx_local[:, 2] = 0
        self.base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)

        self.rwd_linVelTrackEnhancedPrev[env_ids] = 0 # 误差积分重置
        self.rwd_angVelTrackEnhancedPrev[env_ids] = 0
        #每次reset环境时重置last_contact_forces
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.last_contact_forces = self.contact_forces
        #jump
        self.land_time_buf[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.jump_request[env_ids] = False
        # self.has_jumped[env_ids] = False
        # self.next_jump_time[env_ids] = (torch.rand(len(env_ids), device=self.device) * (3.0 - 0.4) + 0.4)
        # self.commands[env_ids,5] = 0.0
        self.last_jump_cmd[env_ids] = 0.0
        # 清空所有 stage
        self.stage_buf[env_ids, :] = 0.0
        # 强制设为 stand
        self.stage_buf[env_ids, 0] = 1.0
        # 可选：清空阶段时间，防止残留
        self.stage_time_buf[env_ids] = 0.0
        self.cmd_time_buf[env_ids] = -1.0


        # 清除之前命令
        if self.cfg.commands.jump_train:
            self.commands[env_ids, 5] = 0.0
            # 采样跳跃高度命令
            l, r = self.command_ranges["jump_height"]
            jump_heights = torch.rand(len(env_ids), 1, device=self.device) * (r - l) + l
            mask = (torch.rand(len(env_ids), 1, device=self.device) < self.cfg.commands.threshold)
            jump_heights[mask] = 0.0
            self.commands[env_ids, 5] = jump_heights.squeeze(1)
        self.jump = self.commands[:, 5] != 0
        # calculate commands
        self.commands_stages[env_ids, 0:3] = 0
        # masks0 = (self.cmd_time_buf[env_ids] == 0).type(torch.float32)
        masks0 = ((self.cmd_time_buf[env_ids] < 0)*(self.jump[env_ids])).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf[env_ids]*self.dt < self.cmd_time_buf[env_ids] + 0.6)*(self.jump[env_ids]).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)*(self.jump[env_ids])
        self.commands_stages[env_ids, 0] = masks0
        self.commands_stages[env_ids, 1] = masks1
        self.commands_stages[env_ids, 2] = masks2

        # self.commands[env_ids, 5] = 0.0

        

        

    def compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        # 初始化奖励缓冲区为零
        self.rew_buf[:] = 0.0
        # 遍历所有奖励函数
        for i in range(len(self.reward_functions)):
            # 获取奖励函数的名称
            name = self.reward_names[i]
            # 调用奖励函数并乘以相应的权重,reward_functions返回的是(num_envs,)维度
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # print("rew_be_name", name, rew)
            # 裁剪奖励值，防止奖励值过大或过小, 单个奖励的最大值为1
            rew = torch.clip( # 剪裁到正负0.01
                rew,
                -self.cfg.rewards.clip_single_reward * self.dt,
                self.cfg.rewards.clip_single_reward * self.dt,
            )
            # print("rew_af_name", name, rew)
            # 将裁剪后的奖励值加到奖励缓冲区中,rew_buf是每次迭代的总奖励
            self.rew_buf += rew
            # 将裁剪后的奖励值加到该奖励项的总累计奖励中
            self.episode_sums[name] += rew
        # 如果配置中设置了只使用正奖励，则将奖励缓冲区中的值裁剪为非负数
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        # 如果配置中设置了终止奖励，则将终止奖励加到奖励缓冲区中
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # 如果是复杂地形,那么使用自定义原点
            self.root_states[env_ids] = self.base_init_state # 位置/姿态/角速度/线速度
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        """随机推动机器人"""
        # 获取需要被推动的环境ID
        # 根据推动间隔时间和仿真时间步长，确定哪些环境需要被推动
        env_ids = (
            (self.envs_steps_buf % int(self.cfg.domain_rand.push_interval_s / self.sim_params.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return

        # 计算最大推力
        # 根据机器人质量和最大速度变化计算最大推力
        # F = ma = m * (Δv/Δt)
        max_push_force = self.base_mass.mean().item() * self.cfg.domain_rand.max_push_vel_xy / self.sim_params.dt
        max_push_torque = self.base_mass.mean().item() * self.cfg.domain_rand.max_push_ang_vel / self.sim_params.dt

        # 重置外力张量
        self.push_forces[:] = 0
        self.push_torques[:] = 0

        # 生成随机推力
        # 在[-max_push_force, max_push_force]范围内随机生成三维推力
        push_forces = torch_rand_float(-max_push_force, max_push_force, (self.num_envs, 3), device=self.device)
        push_torques = torch_rand_float(-max_push_torque, max_push_torque, (self.num_envs, 3), device=self.device)

        # 将推力从世界坐标系转换到机器人局部坐标系
        self.push_forces[env_ids, 0, :] = quat_rotate(self.base_quat[env_ids], push_forces[env_ids]) # 推力作用在env_ids环境的0号刚体(base)上
        self.push_torques[env_ids, 0, :] = quat_rotate(self.base_quat[env_ids], push_torques[env_ids])

        # 减小竖直方向(z轴)的推力
        self.push_forces[env_ids, 0, 2] *= 0.5

        # 将外力施加到仿真环境中
        # 使用Isaac Gym的API应用外力和扭矩
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.push_forces),
            gymtorch.unwrap_tensor(self.push_torques),
            gymapi.ENV_SPACE,
        )

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # # 打印初始质量信息（调试用）
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")

        # 随机化基座质量
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                # 获取配置中设定的额外质量范围
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                # 为每个环境生成随机的额外质量
                self.base_add_mass = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (max_add_mass - min_add_mass)
                    + min_add_mass
                )
                self.raw_base_mass = props[0].mass
                # 计算基座的总质量（原始质量+额外质量）
                self.base_mass = props[0].mass + self.base_add_mass
            # 为当前环境添加额外质量
            props[0].mass += self.base_add_mass[env_id]
        else:
            # 如果不进行随机化，则使用原始质量
            self.base_mass[:] = props[0].mass

        # 随机化基座重心位置
        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                # 获取配置中设定的重心偏移范围
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                # 在x方向随机生成重心偏移
                self.base_com[:, 0] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_x * 2)
                    - com_x
                )
                # 在y方向随机生成重心偏移
                self.base_com[:, 1] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_y * 2)
                    - com_y
                )
                # 在z方向随机生成重心偏移
                self.base_com[:, 2] = (
                    torch.rand(
                        self.num_envs,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=False,
                    )
                    * (com_z * 2)
                    - com_z
                )
            # 为当前环境设置重心偏移
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]

        # 随机化转动惯量
        if self.cfg.domain_rand.randomize_inertia:
            for i in range(len(props)):
                # 获取配置中设定的转动惯量缩放范围
                low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
                # 随机生成转动惯量缩放系数
                inertia_scale = np.random.uniform(low_bound, high_bound)
                # 应用缩放系数到质量和转动惯量
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale

        return props

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 检查是否需要重新采样命令
        # 如果当前环境的episode长度是重新采样时间的整数倍，则需要重新采样命令
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        # 对需要重新采样命令的环境重新采样命令
        self._resample_commands(env_ids)
        # 如果启用了heading命令，则根据目标和当前方向计算角速度命令
        if self.cfg.commands.heading_command:
            # 计算当前方向的前向向量
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算当前方向的偏航角
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 根据目标偏航角和当前偏航角计算角速度命令
            self.commands[:, 1] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 4] - heading), -1.0, 1.0)

        # 如果启用了测量地形高度，则计算并存储测量的地形高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        # 计算并存储机器人的平均高度,去掉地形高度的影响,squeeze删除维度,unsqueeze增加维度
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        
        self.commands[env_ids, 0] = torch_rand_float( # 线速度重采样
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        
        # randint high must > low
        self.commands[env_ids, 3] = ( # 模式重采样,0-四轮模式,1-两轮模式
            torch.randint(
                self.command_ranges["mode"][0],
                self.command_ranges["mode"][1] + 1,
                (len(env_ids), 1),
                device=self.device,
            )
            .float()
            .squeeze(1)
        )

        self.wheel_mode = self.commands[:, 3].unsqueeze(-1) # 将模式信息单独存储,并增加一个维度

        # 1. 减少使用 torch.tensor 构造新张量，从 Python 列表直接构造是低效的。
        command_ranges_height = torch.as_tensor(self.command_ranges["height"], device=self.device) # 高度重采样
        # 分别提取分组范围
        low = command_ranges_height[0:2]
        high = command_ranges_height[2:4]

        # 2. 直接通过 torch.where 获取 height_ranges
        height_ranges = torch.where( # 如果是0,那么选择low,如果是1,那么选择high
            self.commands[env_ids, 3].unsqueeze(-1) == 0,
            low,  # 对应 self.command_ranges["height"][0:2]
            high,  # 对应 self.command_ranges["height"][2:4]
        )

        # 3. 提取索引和计算逻辑，将重复索引操作提取出来
        height_min, height_max = height_ranges.T  # [N, 0] -> height_min; [N, 1] -> height_max

        # 随机生成新范围内的命令，并在一行联合运算
        self.commands[env_ids, 2] = (height_max - height_min) * torch.rand( # 高度命令重采样
            len(env_ids), device=self.device
        ) + height_min

        if self.cfg.commands.heading_command: # 如果有朝向命令,那么在第五个命令处随机化一个朝向值
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 1] = torch_rand_float( # 否则直接在第二个命令处随机化一个角速度值
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        ##jump
        progress = torch.clamp(self.common_step_counter / torch.tensor(1e5, device=self.device), 0.0, 1.0)
        p_jump = 0.4 + progress * (0.85 - 0.4)
        jump_mask = torch.rand(len(env_ids), device=self.device) < p_jump
        self.jump_cmd[env_ids] = jump_mask.float()
        self.last_jump_cmd[env_ids] = 0.0
        self.jump_request[env_ids] = False
        # 清空所有 stage
        self.stage_buf[env_ids, :] = 0.0
        # 强制设为 stand
        self.stage_buf[env_ids, 0] = 1.0
        # 可选：清空阶段时间，防止残留
        self.stage_time_buf[env_ids] = 0.0
        # print("P:",p_jump.float())
        # self.commands[env_ids, 5] = (torch.rand(len(env_ids), device=self.device) < 0.5).float()
        self.cmd_time_buf[env_ids] = -1.0

        if self.cfg.commands.jump_train:
            # 清除之前命令
            self.commands[env_ids, 5] = 0.0
            # 采样跳跃高度命令
            l, r = self.command_ranges["jump_height"]
            jump_heights = torch.rand(len(env_ids), 1, device=self.device) * (r - l) + l
            mask = (torch.rand(len(env_ids), 1, device=self.device) < self.cfg.commands.threshold)
            jump_heights[mask] = 0.0
            self.commands[env_ids, 5] = jump_heights.squeeze(1)
        self.jump = self.commands[:, 5] != 0
        
        # calculate commands
        self.commands_stages[env_ids, 0:3] = 0
        # masks0 = (self.cmd_time_buf[env_ids] == 0).type(torch.float32)
        masks0 = ((self.cmd_time_buf[env_ids] < 0)*(self.jump[env_ids])).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf[env_ids]*self.dt < self.cmd_time_buf[env_ids] + 0.6)*(self.jump[env_ids]).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)*(self.jump[env_ids])
        self.commands_stages[env_ids, 0] = masks0
        self.commands_stages[env_ids, 1] = masks1
        self.commands_stages[env_ids, 2] = masks2
        # self.commands[env_ids, 5] = 0.0

        '''
        踏步
        '''
        self.commands[env_ids, 6] = torch_rand_float( # 踏步高度重采样
            self.command_ranges["step_rise_height"][0],
            self.command_ranges["step_rise_height"][1],
            (len(env_ids), 1),
            device=self.device,
            ).squeeze(1)
       
        

    def compute_proprioception_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)

        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                # self.base_lin_vel * self.obs_scales.lin_vel, # 3, 机器人base线速度
                self.base_ang_vel * self.obs_scales.ang_vel, # 3 ,机器人base角速度(在base坐标系)
                self.base_quat_local * self.obs_scales.quat, # 4 ,机器人姿态四元数
                self.commands[:, :6] * self.commands_scale,  # 4 , 外界命令+1    +1
                self.dof_pos[:, :3] * self.obs_scales.dof_pos,  # 3 ,机器人关节位置,左边髋膝关节
                self.dof_pos[:, 4:7] * self.obs_scales.dof_pos,  # 3 ,机器人关节位置,右边髋膝关节
                self.dof_vel * self.obs_scales.dof_vel,  # 8 , 8个关节速度
                self.actions,  # 8 ,8个关节输出(上一时刻),
                # self.commands_stages, #3
                sin_phase,
                cos_phase
            ),
            dim=-1,
        )
        return obs_buf

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = self.compute_proprioception_observations()

        if self.cfg.env.num_privileged_obs is not None: # 使用特权观测
            heights = ( # 机器人相对于地形的高度观测值,是机器人感知地形变化的关键信息
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            external_forces_and_torques = torch.cat((self.push_forces[:, 0, :], self.push_torques[:, 0, :]), dim=-1) # 观测0号刚体(base)受到的力和力矩

            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,  # 3,基座线速度
                    self.base_euler_zyx * self.obs_scales.quat,  # 3,基座欧拉角
                    self.obs_buf, # action网络观测值
                    self.projected_gravity,  # 3,重力投影
                    self.last_actions[:, :, 0],  # 8,上上动作
                    self.last_actions[:, :, 1],  # 8,上一动作
                    self.dof_acc * self.obs_scales.dof_acc,  # 8,关节加速度,速度差分来的
                    # (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 8
                    heights,  # 7*11,地形信息
                    self.torques * self.obs_scales.torque,  # 8,力矩信息
                    (self.base_mass - self.raw_base_mass).view(self.num_envs, 1),  # 1,base质量
                    self.base_com,  # 3,base质心
                    # self.default_dof_pos - self.raw_default_dof_pos,  # 8
                    self.friction_coef.view(self.num_envs, 1),  # 1,摩擦系数
                    self.restitution_coef.view(self.num_envs, 1),  # 1,弹性系数
                    external_forces_and_torques * self.priv_obs_scales.external_wrench,  # 6,基座外力(矩)
                    # self.stage_buf.float(),##jump stage 5
                ),
                dim=-1,
            )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec # rand_link生成[0,1)之间的随机数

        self.obs_history = torch.cat((self.obs_history[:, self.num_obs :], self.obs_buf), dim=-1)

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # Extract joint position and wheel velocity references
        self.joint_pos_ref = actions[:, self.joint_indices] * self.cfg.control.action_scale_pos # 获取关节位置参考
        self.wheel_vel_ref = actions[:, self.wheel_indices] * self.cfg.control.action_scale_vel # 获取轮子速度参考

        # Compute joint and wheel torques
        torque_joint = self.p_gains[:, self.joint_indices] * ( # 计算关节力矩
            self.joint_pos_ref + self.default_dof_pos[:, self.joint_indices] - self.dof_pos[:, self.joint_indices]
        ) + self.d_gains[:, self.joint_indices] * (-self.dof_vel[:, self.joint_indices])

        torque_wheel = self.d_gains[:, self.wheel_indices] * (self.wheel_vel_ref - self.dof_vel[:, self.wheel_indices]) # 轮子力矩

        # Combine torques while preserving the original order
        torques = torch.zeros_like(actions)  # Initialize with the same shape as actions
        torques[:, self.joint_indices] = torque_joint
        torques[:, self.wheel_indices] = torque_wheel

        return torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits) # 剪裁后输出力矩

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # noise_vec[0:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel  # lin_vel
        # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # ang_vel
        # noise_vec[6:10] = noise_scales.quat * noise_level * self.obs_scales.quat  # quat
        # noise_vec[10:14] = 0.0  # commands
        # noise_vec[14:16] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        # noise_vec[16:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        # noise_vec[18:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dof_vel_all
        # noise_vec[24:30] = 0.0  # previous actions
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # ang_vel
        noise_vec[3:7] = noise_scales.quat * noise_level * self.obs_scales.quat  # quat
        noise_vec[7:11] = 0.0  # commands
        noise_vec[11:14] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        noise_vec[14:17] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos 
        noise_vec[17:25] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # dof_vel_all
        noise_vec[25:33] = 0.0  # previous actions
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # 演员(机器人)的根状态(位置/旋转/速度),存储了13个值:[x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz]
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # 所有关节(自由度)的状态(位置和速度),对于每个关节,存储了2个值:[position,velocity]
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim) # 所有刚体的净接触力,存储每个刚体的三维接触力:[Fx,Fy,Fz]
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # 所有刚体在世界坐标系中的位置,维度:[num_envs * num_bodies, 13]

        self.gym.refresh_dof_state_tensor(self.sim) # 刷新自由度状态张量,确保数据最新,以上四个是指向对应物理量的指针,刷新数据后保证指针指向的是最新的数据
        self.gym.refresh_actor_root_state_tensor(self.sim) # 刷新演员根状态张量
        self.gym.refresh_net_contact_force_tensor(self.sim) # 刷新刚体净接触力张量
        self.gym.refresh_rigid_body_state_tensor(self.sim) # 刷新刚体状态张量

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # 机器人根状态张量转换为PyTorch张量,actor_root_state是指向GPU数据的原始指针,wrap_tensor是将指针指向数据包装成PyTorch张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # 机器人自由度状态张量转换为PyTorch张量
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] # 关节位置,维度(num_envs,num_dof),view是引用,因此关节位置和速度会自动刷新,将张量按照期望维度排列,[...,0]是提取该维度信息
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1] # 关节速度,维度(num_envs,num_dof),提取后维度减1
        self.dof_acc = torch.zeros_like(self.dof_vel) # 初始化自由度加速度为0
        self.base_quat = self.root_states[:, 3:7] # 获取base姿态
        self.base_euler_zyx = get_euler_zyx_tensor(self.base_quat)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor) # 获取刚体状态
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3] # 获取刚体位置
        self.rigid_body_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 3:7] # 获取刚体姿态

        # without yaw
        base_euler_zyx_local = self.base_euler_zyx.clone()
        base_euler_zyx_local[:, 2] = 0
        self.base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view( # 获取机器人各刚体接触力,维度(num_envs,num_bodies),每个index存放[Fx,Fy,Fz]信息
            self.num_envs, -1, 3
        )  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0 # 初始化一个公共步数计数器
        self.extras = {} # 初始化一个空字典用于存放额外信息
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # 存储噪声缩放向量
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat( # 创建重力向量
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1)) # 创建前向向量
        self.torques = torch.zeros( # 初始化力矩张量
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques_scale = torch.ones( # 初始化力矩缩放因子
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros( # 初始化p增益系数
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros( # 初始化d增益系数
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros( # 初始化动作张量
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros( # 存放前两个周期的actions,用于平滑连续命令之间的差异
            self.num_envs,
            self.num_actions,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.base_position = self.root_states[:, :3] # 存储所有环境的位置
        # print(self.base_position)
        self.last_base_position = self.base_position.clone() # 上一个时刻所有环境的位置
        self.last_dof_pos = torch.zeros_like(self.dof_pos) # 上一个时刻所有关节的位置
        self.last_dof_vel = torch.zeros_like(self.dof_vel) # 上一个时刻所有关节的速度
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13]) # 上一个时刻所有环境的速度
        self.commands = torch.zeros( # 初始化命令张量,用于存储每个环境的控制指令,额外一列可能用于存储命令的持续时间或其他附加信息
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor( # 初始化命令缩放因子,用于标准化命令值
            [
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.height_measurements,
                self.obs_scales.mode,
                self.obs_scales.jump,
                self.obs_scales.step_rise_height,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros( # 初始化上一个接触状态信息
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.last_contact_forces = torch.zeros(
            self.num_envs,
            self.num_bodies,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # 基座线速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 基座角速度
        self.base_lin_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 7:10]) # head坐标系中的线速度
        self.base_ang_vel_head = quat_rotate_inverse(self.base_quat_local, self.root_states[:, 10:13]) # head坐标系中的角速度
        self.push_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False) # 推动力
        self.push_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, requires_grad=False) # 推动力矩
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 重力投影
        self.action_delay_idx = torch.zeros( # 动作延时
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        delay_max = np.int64(np.round(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt) + 1) # 最大延时
        self.action_fifo = torch.zeros( # 演员fifo,维度(num_envs,delay_max,num_actions)
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.terrain.measure_heights: # 测量高度的位置
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) # base高度
        self.base_height_last = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) # base高度
        self.base_height_local = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.joint_indices = torch.tensor(list(self.cfg.asset.joint_indices), device=self.device) # 关节索引
        self.wheel_indices = torch.tensor(list(self.cfg.asset.wheel_indices), device=self.device) # 轮子索引
        self.base_link_indices = torch.tensor(list(self.cfg.asset.base_link_indices), device=self.device) # base_link索引
        self.joint_link_indices = torch.tensor(list(self.cfg.asset.joint_link_indices), device=self.device)  # joint_link索引
        self.wheel_link_indices = torch.tensor(list(self.cfg.asset.wheel_link_indices), device=self.device)   # wheel_link索引

        self.wheel_lin_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_ang_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.raw_default_dof_pos = torch.zeros(  # 初始化默认关节位置,单个环境
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_dof_pos = torch.zeros( # 存储所有环境的默认关节位置
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):  # 遍历所有关节,设置默认位置和控制器增益
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.raw_default_dof_pos[i] = angle
            self.default_dof_pos[:, i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.0
                self.d_gains[:, i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        if self.cfg.domain_rand.randomize_Kp: # 随机化Kp
            (
                p_gains_scale_min,
                p_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(
                p_gains_scale_min,
                p_gains_scale_max,
                self.p_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_Kd: # 随机化Kd
            (
                d_gains_scale_min,
                d_gains_scale_max,
            ) = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                d_gains_scale_min,
                d_gains_scale_max,
                self.d_gains.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_motor_torque: # 随机化电机力矩
            (
                torque_scale_min,
                torque_scale_max,
            ) = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                torque_scale_min,
                torque_scale_max,
                self.torques_scale.shape,
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_default_dof_pos: # 随机化默认位置
            self.default_dof_pos += torch_rand_float(
                self.cfg.domain_rand.randomize_default_dof_pos_range[0],
                self.cfg.domain_rand.randomize_default_dof_pos_range[1],
                (self.num_envs, self.num_dof),
                device=self.device,
            )
        if self.cfg.domain_rand.randomize_action_delay: # 随机化动作延时
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()

        self.rwd_linVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_linVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        self.rwd_angVelTrackEnhancedPrev = torch.zeros(self.num_envs, device=self.device)
        ##jump
        self.num_stages = len(self.cfg.asset.stage_names)
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages),dtype=torch.float32, device=self.device)
        self.land_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # self.start_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # self.jump_command_time = torch.rand(self.num_envs, device=self.device) * 2.0#range:0-3s

        # self.next_jump_time = torch.rand(self.num_envs, device=self.device) * (2 - 0.5) + 0.5 #(t_max - t_min) + t_min
        # self.has_jumped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


        self.last_jump_cmd = torch.zeros(self.num_envs, device=self.device)
        self.jump_cmd = torch.zeros(self.num_envs, device=self.device)
        self.jump_request = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.default_dof_pos_jump = torch.zeros(self.num_envs,self.num_dof,dtype=torch.float,device=self.device,requires_grad=False,)
        for i in range(self.num_dofs):  # 遍历所有关节,设置默认位置和控制器增益
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos_jump[:, i] = angle

        self.default_dof_pos_sit = torch.zeros(self.num_envs,self.num_dof,dtype=torch.float,device=self.device,requires_grad=False,)
        for i in range(self.num_dofs):  # 遍历所有关节,设置默认位置和控制器增益
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles_sit[name]
            self.default_dof_pos_sit[:, i] = angle

        self.stage_time_buf = torch.zeros(self.num_envs, device=self.device)
        self.cmd_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.commands_stages = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        
        # jump
        self.jump = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        '''
        踏步
        '''
        self._init_foot()



    def _prepare_reward_function(self):
        """准备一个奖励函数列表，这些函数将被调用以计算总奖励。
        查找self._reward_<REWARD_NAME>，其中<REWARD_NAME>是cfg中所有非零奖励权重的名称。
        """
        # 移除零权重的奖励项，并将非零权重乘以时间步长dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)  # 移除权重为0的奖励项
            else:
                self.reward_scales[key] *= self.dt  # 将奖励权重乘以时间步长

        # 准备奖励函数列表和名称列表
        self.reward_functions = []  # 存储奖励函数
        self.reward_names = []  # 存储奖励名称
        for name, scale in self.reward_scales.items():
            if name == "termination":  # 跳过终止奖励
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))  # 将奖励函数添加到列表中

        # 初始化每个环境的奖励累计值
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,  # 环境数量
                dtype=torch.float,  # 数据类型为浮点数
                device=self.device,  # 使用指定设备
                requires_grad=False,  # 不需要梯度
            )
            for name in self.reward_scales.keys()  # 为每个环境下每个奖励项创建累计值
        }

    def _get_heights(self, env_ids=None):
        """采样每个机器人周围指定点的地形高度。
        这些点根据机器人基座的位置进行偏移，并根据基座的偏航角进行旋转。
            Args:
            env_ids (List[int], optional): 需要返回高度的环境子集。默认为None。
            Raises:
            NameError: 当地形网格类型为'none'时抛出异常。
            Returns:
            torch.Tensor: 包含采样高度的张量。
        """
        # 如果地形是平面，返回全零张量
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points, # 11*7
                device=self.device,
                requires_grad=False,
            )
        # 如果没有地形，抛出异常
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 根据是否指定环境ID来处理采样点
        if env_ids:
            # 对指定环境的采样点进行旋转和平移
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), # 给每个环境复制77份四元数姿态
                self.height_points[env_ids], # 高度点,此时只有xy有值,z是0,维度(num_envs,num_height_points,3)
            ) + (self.root_states[env_ids, :3]).unsqueeze(1) # 维度变为(num_envs,1,3),维度相同即可相加扩展广播,会将root_states的对应维度扩展(复制)num_height_points份
        else:
            # 对所有环境的采样点进行旋转和平移
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]
            ).unsqueeze(1)

        # 添加地形边界大小,假设世界坐标系是0,但是在地形坐标系中,边界起点才是0,原来世界坐标系的原点就是border_size了
        points += self.terrain.cfg.border_size
        # 将点坐标转换为地形网格的索引
        points = (points / self.terrain.cfg.horizontal_scale).long()
        # 提取x和y坐标
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        # 裁剪坐标以确保在有效范围内
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # 获取相邻三个点的高度
        heights1 = self.height_samples[px, py]  # 当前点的高度
        heights2 = self.height_samples[px + 1, py]  # x方向下一个点的高度
        heights3 = self.height_samples[px, py + 1]  # y方向下一个点的高度
        # 取三个高度中的最小值
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 返回最终的高度值，并应用垂直缩放
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        # 获取机器人资产文件的路径
        asset_path = self.cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        # 获取资产文件所在的目录
        asset_root = os.path.dirname(asset_path)
        # 获取资产文件的文件名
        asset_file = os.path.basename(asset_path)

        # 创建资产选项对象
        asset_options = gymapi.AssetOptions()
        # 设置默认的自由度驱动模式
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # 设置是否折叠固定关节
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # 设置是否用胶囊体替换圆柱体
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # 设置是否翻转视觉附件
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # 设置是否固定基座链接
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # 设置资产的密度
        asset_options.density = self.cfg.asset.density
        # 设置资产的角阻尼
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 设置资产的线阻尼
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 设置资产的最大角速度
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 设置资产的最大线速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 设置资产的电枢
        asset_options.armature = self.cfg.asset.armature
        # 设置资产的厚度
        asset_options.thickness = self.cfg.asset.thickness
        # 设置是否禁用重力
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 加载机器人资产
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # 获取资产的自由度数量
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        # 获取资产的刚体数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        # 获取资产的自由度属性
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # 获取资产的刚体形状属性
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # 保存资产的刚体名称
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # print("body_names:", body_names)
        # 保存资产的自由度名称
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        # print("dof_names:", self.dof_names)
        # 获取刚体数量
        self.num_bodies = len(body_names)
        # 获取自由度数量
        self.num_dofs = len(self.dof_names)
        # 获取脚部名称
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 获取需要惩罚接触的名称
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        # 获取需要终止接触的名称
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 初始化基座状态
        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # 设置初始姿态
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 获取环境原点
        self._get_env_origins()
        # 设置环境的下限和上限
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        # 初始化演员句柄列表
        self.actor_handles = []
        # 初始化环境列表
        self.envs = []
        # 初始化摩擦系数
        self.friction_coef = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化恢复系数
        self.restitution_coef = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化基座质量
        self.base_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化基座质心
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # 计算位置
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # 处理刚体形状属性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # 设置资产的刚体形状属性
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 创建演员
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            # 处理自由度属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            # 设置演员的自由度属性
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            # 获取演员的刚体属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            # 处理刚体属性
            body_props = self._process_rigid_body_props(body_props, i)
            # 设置演员的刚体属性
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            # 添加环境到列表
            self.envs.append(env_handle)
            # 添加演员句柄到列表
            self.actor_handles.append(actor_handle)

        # 初始化脚部索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        # 初始化需要惩罚接触的索引
        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        # 初始化需要终止接触的索引
        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.priv_obs_scales = self.cfg.normalization.priv_obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel() # 误差
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        # self.rwd_linVelTrackEnhancedPrev = self.tracking_integral_lin_vel() # 误差积分
        self.rwd_linVelTrackEnhancedPrev = self._reward_tracking_lin_vel_enhance() # 误差积分
        self.rwd_angVelTrackEnhancedPrev = self._reward_tracking_ang_vel_enhance()

    def tracking_integral_lin_vel(self):
        # 计算期望线性速度与实际线性速度之间的误差
        lin_vel_error = self.commands[:, 0] - self.base_lin_vel_head[:, 0]
        # lin_vel_error = torch.square(self.commands[:, 0] - self.wheel_lin_vel)
        ans = self.cfg.rewards.enhance_factor * self.rwd_linVelTrackEnhancedPrev + lin_vel_error
        # # 使用指数函数计算奖励，奖励值随着误差的减小而增加
        return ans  # Ensure the reward is non-negative
        # return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1
    #jump
    def update_stage(self):
        ##stage_buf: ["stand", "sit", "jump", "air", "land"]
        #当前为land且着陆时间大于0.2s：进入stand
        from4_to0 = torch.logical_and(self.stage_buf[:, 4] == 1.0, self.progress_buf*self.dt > self.land_time_buf + 0.2).type(torch.float32)
        
        id_jump_finished = from4_to0.bool()
        self.cmd_time_buf[id_jump_finished] = -1.0

        self.stage_buf[:, 4] = (1.0 - from4_to0)*self.stage_buf[:, 4]#重置land
        self.stage_buf[:, 0] = from4_to0 + (1.0 - from4_to0)*self.stage_buf[:, 0]#更新为stand
        foot_contact_forces = self.contact_forces[:, self.feet_indices, :]      #获取脚部接触力
        foot_contact = (torch.norm(foot_contact_forces, dim=2) > 0.6).type(torch.float)#维度:num_envs*num_feet,获取脚部是否触底（以0.6N为边界）
        #当前为air且至少一条腿触地：进入land
        from3_to4 = torch.logical_and(self.stage_buf[:, 3] == 1.0,foot_contact.mean(dim=-1) > 0.0).type(torch.float32)
        self.stage_buf[:, 3] = (1.0 - from3_to4)*self.stage_buf[:, 3]#重置air
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4)*self.stage_buf[:, 4]#更新为land
        #当前为jump且两条腿都离地且机器人高度大于0.78m：进入air
        from2_to3 = torch.logical_and(self.stage_buf[:, 2] == 1.0, torch.logical_and(foot_contact.mean(dim=-1) < 0.1,self.base_height >= 0.6)).type(torch.float32)
        # from2_to3 = torch.logical_and(self.stage_buf[:, 2] == 1.0, foot_contact.mean(dim=-1) < 0.1).type(torch.float32)
        self.stage_buf[:, 2] = (1.0 - from2_to3)*self.stage_buf[:, 2]#重置jump
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3)*self.stage_buf[:, 3]#更新为air
        #当前为sit且两条腿都着地且机器人高度小于0.6m：进入jump
        from1_to2 = torch.logical_and(self.stage_buf[:, 1] == 1.0, torch.logical_and(self.base_height <= 0.55, foot_contact.mean(dim=-1) == 1.0)).type(torch.float32)
        self.stage_buf[:, 1] = (1.0 - from1_to2)*self.stage_buf[:, 1]#重置sit
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2)*self.stage_buf[:, 2]#更新为jump
        #最早起跳时间限制:不允许一reset就立刻下蹲,这是为了：*打破对reset初始条件的过拟合*给系统一个稳定起始段
        #高度限制:防止机器人还没站稳/没站直,就直接进入下蹲逻辑
        #当前为stand且两条腿都着地且满足起跳时间限制且高度大于0.6m：进入sit;加入控制跳跃为1时
        # from0_to1 = torch.logical_and(self.jump_request,self.stage_buf[:, 0] == 1.0).type(torch.float32)
        # from0_to1 = torch.logical_and(self.jump_request,torch.logical_and(self.stage_buf[:, 0] == 1.0,self.stage_time_buf > 0.5)).float()
        # from0_to1 = torch.logical_and(self.base_height >= 0.6,torch.logical_and(self.stage_buf[:, 0] == 1.0,self.stage_time_buf > 0.5)).float()
        from0_to1 = torch.logical_and(self.stage_time_buf > 0.3,torch.logical_and(self.base_height >= 0.55,torch.logical_and(self.stage_buf[:, 0] == 1.0,self.jump))).float()
        # self.commands[from0_to1.bool(),5] = 1
        self.stage_buf[:, 0] = (1.0 - from0_to1)*self.stage_buf[:, 0]#重置stand
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1)*self.stage_buf[:, 1]#更新为sit
        #进入sit后重置land_time_buf
        id_enter_sit = from0_to1.bool()
        self.land_time_buf[id_enter_sit] = 0.0

        # update land_time_buf 
        land_masks = torch.logical_and(self.land_time_buf == 0, self.stage_buf[:, 4] == 1).type(torch.float32)
        self.land_time_buf[:] = land_masks*(self.progress_buf*self.dt) + (1.0 - land_masks)*self.land_time_buf

        cmd_masks = torch.logical_and(self.cmd_time_buf < 0, self.stage_buf[:, 1] == 1).type(torch.float32)
        self.cmd_time_buf[:] = cmd_masks*(self.progress_buf*self.dt) + (1.0 - cmd_masks)*self.cmd_time_buf

        # ============================
        # 强制约束：jump_request == False → 回到 stand
        # ============================
        # no_jump = ~self.jump_request.bool()
        # if no_jump.any():
        #     # 清空所有 stage
        #     self.stage_buf[no_jump, :] = 0.0
        #     # 强制设为 stand
        #     self.stage_buf[no_jump, 0] = 1.0
        #     # 可选：清空阶段时间，防止残留
        #     self.stage_time_buf[no_jump] = 0.0
        # 统计 stage 时间
        # 如果 stage 没变，累计时间；如果变了，清零
        current_stage = torch.argmax(self.stage_buf, dim=1)
        # print(current_stage)
        if not hasattr(self, "last_stage"):
            self.last_stage = current_stage.clone()
        same_stage = current_stage == self.last_stage
        self.stage_time_buf = torch.where(
            same_stage,
            self.stage_time_buf + self.dt,
            torch.zeros_like(self.stage_time_buf)
        )
        self.last_stage = current_stage.clone()

        # calculate commands
        self.commands_stages[:, 0:3] = 0
        # masks0 = ((self.cmd_time_buf == 0)).type(torch.float32)#还没下命令
        masks0 = ((self.cmd_time_buf < 0)*(self.jump)).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf*self.dt < self.cmd_time_buf + 0.6)*(self.jump).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)*(self.jump)
        self.commands_stages[:, 0] = masks0
        self.commands_stages[:, 1] = masks1
        self.commands_stages[:, 2] = masks2
        


    # ------------ reward functions----------------
    ################## 速度控制 ##################
    def _reward_tracking_lin_vel(self):
        """
        计算增强的线性速度跟踪奖励。

        该方法通过计算期望线性速度与实际线性速度之间的误差，并使用指数函数来计算奖励。
        奖励值随着误差的减小而增加，从而鼓励机器人更好地跟踪期望的线性速度。

        Returns:
            torch.Tensor: 增强的线性速度跟踪奖励。
        """
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) 
        ans = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        return ans

    def _reward_tracking_lin_vel_enhance(self):
        # 计算期望线性速度与实际线性速度之间的误差
        # lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel_head[:, 0])
        # lin_vel_error = self.commands[:, 0] - self.base_lin_vel[:, 0]
        # ans = self.rwd_linVelTrackEnhancedPrev + self.cfg.rewards.enhance_factor * lin_vel_error
        # ans = torch.exp(-torch.square(ans) / self.cfg.rewards.tracking_sigma / 10)
        # # 使用指数函数计算奖励，奖励值随着误差的减小而增加
        # return ans

        # 计算期望线性速度与实际线性速度之间的误差
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        # lin_vel_error = torch.square(self.commands[:, 0] - self.wheel_lin_vel)
        # ans = self.rwd_linVelTrackEnhancedPrev + self.cfg.rewards.enhance_factor * lin_vel_error
        # # 使用指数函数计算奖励，奖励值随着误差的减小而增加
        # return ans.clip(0, 1)  # Ensure the reward is non-negative
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        # ang_vel_error = torch.square(self.commands[:, 1] - self.wheel_ang_vel)
        ans = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        return ans

    def _reward_tracking_ang_vel_enhance(self):
        # Tracking of angular velocity commands (x axes)
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        # ang_vel_error = torch.square(self.commands[:, 1] - self.wheel_ang_vel)
        ans = self.rwd_angVelTrackEnhancedPrev + self.cfg.rewards.enhance_factor * ang_vel_error
        return ans.clip(0, 1)  # Ensure the reward is non-negative

    def _reward_tracking_lin_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return lin_vel_error
        return delta_phi

    def _reward_tracking_ang_vel_pbrs(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi

################## 跳跃阶段控制 ##################
    ##height
    def _reward_height(self):
        l, r = self.command_ranges["jump_height"]
        jump_height_scale = (self.commands[:, 5] - l) / (r - l) + 0.1
        upward_vel = torch.clamp(self.base_lin_vel[:, 2], min=0.0) 
        reward  = self.stage_buf[:, 0]*(-torch.abs(self.base_height - 0.651))
        reward += self.stage_buf[:, 1]*(-torch.abs(self.base_height - 0.45))#0.58
        # reward += self.stage_buf[:, 2]*upward_vel
        # # stage 3：高度命中
        # height_err = self.base_height - self.commands[:, 5]
        # reward += self.stage_buf[:, 3] * torch.exp(-height_err**2 / 0.15)
        # reward += self.stage_buf[:, 2]*(self.base_height <= 1.0*self.commands[:, 5])*(self.base_height)*upward_vel
        # reward += self.stage_buf[:, 3]*(self.base_height <= 1.0*self.commands[:, 5])*(self.base_height)
        reward += self.stage_buf[:, 2]*(self.base_height <= 0.8)*(self.base_height)*upward_vel
        reward += self.stage_buf[:, 3]*(self.base_height <= 0.8)*(self.base_height)
        reward += self.stage_buf[:, 4]*(torch.exp(-torch.square(self.base_height - 0.651)/0.05))
        return reward

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[0,1,4,5]]), dim=1)

    ##lin_vel_z
    def _reward_lin_vel_z_0(self):
        # 惩罚 z 速度
        return torch.square(self.base_lin_vel[:, 2])* self.stage_buf[:, 0]
    
    def _reward_lin_vel_z_1(self):
        # 只惩罚向下速度，且超过 0.05
        downward_vel = torch.clamp(-self.base_lin_vel[:, 2] - 0.1, min=0.0)
        return torch.square(downward_vel) * self.stage_buf[:, 1]


    def _reward_lin_vel_z_2(self):
        # 奖励向上的 z 速度
        upward_vel = torch.clamp(self.base_lin_vel[:, 2], min=0.0)
        # # 惩罚向下的 z 速度
        downward_vel = torch.clamp(-self.base_lin_vel[:, 2], min=0.0)
        return  upward_vel * (self.stage_buf[:, 2] + self.stage_buf[:, 3])# - 10 * downward_vel * self.stage_buf[:, 2]

    def _reward_lin_vel_z_4(self):
        # 惩罚向下的 z 速度
        downward_vel = torch.clamp(-self.base_lin_vel[:, 2], min=0.0)
        #向上速度太大也惩罚
        upward_vel = torch.clamp(self.base_lin_vel[:, 2]-0.1, min=0.0)
        return (torch.square(downward_vel) * 10 + torch.square(upward_vel)) * self.stage_buf[:, 4]

    def _reward_style(self):
        # 全身误差
        err_default = torch.square(self.dof_pos[:, self.joint_indices] - self.default_dof_pos_jump[:, self.joint_indices]).mean(dim=1)
        err_sit = torch.square(self.dof_pos[:, self.joint_indices] - self.default_dof_pos_sit[:, self.joint_indices]).mean(dim=1)
        reward = (
            self.stage_buf[:, 0] * (torch.exp(-err_default/0.05)) +   # stand
            self.stage_buf[:, 1] * (torch.exp(-err_sit/0.05))     +   # sit
            self.stage_buf[:, 4] * (torch.exp(-err_default/0.05))     # land
        )
        return reward
    
    def _reward_dof_vel_wheel(self):
        vel2 = torch.square(self.dof_vel[:, self.wheel_indices]).mean(dim=1)
        high_enough = self.base_height > 0.7
        airborne = self.stage_buf[:, 3] == 1.0
        mask = high_enough & airborne
        # 轮速越接近 0，reward 越接近 1
        return torch.exp(-0.005 * vel2) * mask.float()
    
    def _reward_torques(self):
        return -torch.square(self.torques).mean(dim=-1)

    
    # def _reward_contact(self):
    #     # 接触力 [N, 2, 3]
    #     wheel_forces = self.contact_forces[:, self.feet_indices, :]
    #     wheel_force_mag = torch.norm(wheel_forces, dim=2)   # [N, 2]
    #     # 用一个很小的阈值判断“是否接触”
    #     contact_thresh = 1.0
    #     contact = wheel_force_mag > contact_thresh          # [N, 2] bool
    #     left_contact  = contact[:, 0]
    #     right_contact = contact[:, 1]
    #     # ===== 阶段判断 =====
    #     should_contact = (
    #         self.stage_buf[:, 0] +
    #         self.stage_buf[:, 1] +
    #         self.stage_buf[:, 4]
    #     ) > 0.5
    #     should_not_contact = (self.stage_buf[:, 3]) > 0.5
    #     reward = torch.zeros(self.num_envs, device=self.device)
    #     # ===== 1 该接触：必须双轮同时接触 =====
    #     both_contact = left_contact & right_contact
    #     reward += should_contact * both_contact.float()
    #     # ===== 2 不该接触：必须双轮都不接触 =====
    #     both_air = (~left_contact) & (~right_contact)
    #     reward += should_not_contact * both_air.float()
    #     return reward





    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        计算保持基座平坦方向的奖励。使用基座欧拉角和投影重力向量来惩罚与期望基座方向的偏差。
        """
        ans = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]) * 50, dim=1))
        return ans

    def _reward_leg_end_x_diff(self):
        ans = torch.exp(-torch.abs(self.wheel_pos_left_local_x - self.wheel_pos_right_local_x) / 0.01)
        # print("4", torch.square(self.wheel_pos_left_local_x - self.wheel_pos_right_local_x))
        return ans
    

################## 动作柔顺 ##################

    def _reward_dof_vel(self):
        height_error = torch.abs(self.base_height - self.commands[:, 2])
        w = torch.exp(-100.0 * height_error) 
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, self.joint_indices]), dim=1) * (1.1 - w)
        # return torch.sum(torch.square(self.dof_vel[:, self.joint_indices]), dim=1)
    

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc[:, self.joint_indices]), dim=1)
        
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:, :, 0] - self.actions), dim=1)

    def _reward_action_smooth(self):
        """
        鼓励机器人动作的平滑性,通过惩罚连续动作之间的大差异。
        这对于实现流畅的运动和减少机械应力很重要。
        """
        ans = torch.sum(
            torch.square(self.actions[:, self.joint_indices] + self.last_actions[:, self.joint_indices, 1] - 2 * self.last_actions[:, self.joint_indices, 0]), dim=1
        )
        return ans
################## 关节限制 ##################
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        dof_pos_subset = self.dof_pos[:, self.joint_indices]
        lower_limits = self.dof_pos_limits[self.joint_indices, 0]
        upper_limits = self.dof_pos_limits[self.joint_indices, 1]
        lower_penalty = -(dof_pos_subset - lower_limits).clip(max=0.0)  # 下限惩罚
        upper_penalty = (dof_pos_subset - upper_limits).clip(min=0.0)  # 上限惩罚
        penalties = lower_penalty + upper_penalty
        return torch.sum(penalties, dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(
                min=0.0, max=1.0
            ),
            dim=1,
        )
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0, max=1.0),
            dim=1,
        )
    
################## 碰撞惩罚 ##################
    def _reward_collision(self):
        # Penalize collisions on selected bodies based on squared forces
        forces = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        )
        penalty = torch.square(forces)
        return penalty.sum(dim=1)
    
    ####################################### 新增 #################################

    '''
    奖励脚步在空中停留较长时间的步态
    '''
    def _reward_feet_air_time(self):
        # 奖励长时间的步态
        # 需要过滤接触信息，因为PhysX在网格上的接触报告不可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # 仅在首次接触地面时奖励
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 # 命令为零时不奖励
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    '''
    为每个存活的环境提供恒定的正奖励
    '''
    def _reward_alive(self):
        # Reward for staying alive
        # return 1.0
        return torch.ones(self.num_envs, device=self.device)

    '''
    惩罚足部在非接触状态下的摆动高度与目标高度(0.08m)的偏差，鼓励足部在摆动时保持合适的高度

    记得切换为高度命令
    '''
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.# 判断是否接触地面
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact# 计算摆动脚的高度误差
        return torch.sum(pos_error, dim=(1))
    
    '''
    奖励机器人足部接触状态与期望的支撑相位匹配的程度，当足部在支撑相位时接触地面或在摆动相位时离地则获得奖励
    '''
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
        

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        