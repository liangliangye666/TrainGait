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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask:

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym() # 获取Isaac Gym底层API功能,后续所有操作都通过self.gym调用,可认为是一把"万能钥匙"或"总控台",用于观察和改造"self.sim"这个物理仿真世界

        self.sim_params = sim_params # 存储基础参数
        self.physics_engine = physics_engine # 物理引擎
        self.sim_device = sim_device # 仿真设备
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device) # 解析仿真设备
        self.headless = headless # 无头模式

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline: # 确定环境使用设备
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id # 默认图形设备与模拟设备一致
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs # 并行环境数量
        self.num_obs = cfg.env.num_observations # 每个环境观测值数量
        self.num_privileged_obs = cfg.env.num_privileged_obs # 特权观测维度
        self.num_actions = cfg.env.num_actions # 动作空间维度
        self.obs_history_length = cfg.env.obs_history_length # 历史观测堆叠长度

        # optimization flags for pytorch JIT,禁用表示不预热模型,避免前几次延迟(可能几百ms)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros( # 初始化观测值缓存,维度(num_envs,num_obs),存储在gpu上
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.obs_history = torch.zeros( # 初始化历史观测缓存,维度(num_envs,num_obs*obs_history_length),存储在gpu上,样式:([0,0,0,0,0,0],[0,0,0,0,0,0])->两个环境,3个观测*2个历史长度
            self.num_envs,
            self.num_obs * self.obs_history_length,
            device=self.device,
            dtype=torch.float,
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float) # 初始化奖励缓存,维度(num_envs,),样式:([0,0,0,0])->四个环境
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long) # 初始化重置缓存,维度(num_envs,)
        self.fail_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) # 初始化失败缓存,维度(num_envs,)
        self.episode_length_buf = torch.zeros( # 初始化当前回合运行步数缓存,每个环境可能有多个回合,当摔倒或达成目标时可能会开启新的回合,维度(num_envs,)
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.envs_steps_buf = torch.zeros( # 初始化环境总运行步数缓存,永不重置,维度(num_envs,),用于课程学习
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros( # 初始化超时缓存,维度(num_envs,)
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.edge_reset_buf = torch.zeros( # 初始化边缘超限重置缓存,维度(num_envs,),用于环境重置
            self.num_envs, device=self.device, dtype=torch.bool
        )
        if self.num_privileged_obs is not None: # 初始化特权观测缓存,维度(num_envs,num_privileged_obs)
            self.privileged_obs_buf = torch.zeros(
                self.num_envs,
                self.num_privileged_obs,
                device=self.device,
                dtype=torch.float,
            )
        else:
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {} # 获取额外信息

        # create envs, sim and viewer
        self.create_sim() # 创建仿真,类似于购买地皮,并根据地皮的地址完成规划和建造,并为地皮创建机器人,做好仿真前的各种准备
        self.gym.prepare_sim(self.sim) # 在gym环境准备仿真,所有环境属性配置完成,可以开始仿真

        # todo: read from config
        self.enable_viewer_sync = True # 使能同步观测标志位
        self.viewer = None # 不指定观测标志位

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False: # 无头模式
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

    def get_observations(self): # 返回当前所有环境的观测和额外信息,作用:供算法/网络获取当前状态
        return (
            self.obs_buf,
            self.obs_history,
        )

    def get_privileged_observations(self): # 获取特权观测
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self): # 重置所有环境/返回初始观测和额外信息,作用:在回合结束或初始化时调用,确保每个环境都回到初始状态
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device)) # 重置所有环境信息,torch.arange返回一个0到self.num_envs-1的张量,并且存储在self.device中
        obs, privileged_obs, _, _, _, _ = self.step( # 用全零参数来获取初始观测值
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def step(self, actions): # 对所有环境执行动作,返回新观测/奖励/done标志和额外信息,作用:与环境交互的核心接口,算法通过它采集数据
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
