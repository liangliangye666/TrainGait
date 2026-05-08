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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from wheel_legged_gym.rsl_rl.algorithms import PPO
from wheel_legged_gym.rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    ActorCriticSequence,
)
from wheel_legged_gym.rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # 可以使用字典形式访问,是因为已经提前转化为了数据字典
        self.cfg = train_cfg["runner"] # 配置PPO算法的runner信息
        self.alg_cfg = train_cfg["algorithm"] # 配置PPO算法的algorithm信息
        self.policy_cfg = train_cfg["policy"] # 配置PPO算法的policy信息
        self.device = device # 配置训练设备
        self.env = env # 仿真环境对象,包含机器人类型,自由度,奖励,噪声,命令设置等
        if self.env.num_privileged_obs is not None: # 配置评论家网络使用特权观测/普通观测的通道数量
            num_critic_obs = self.env.num_privileged_obs # 如果有特权观测,评论家使用特权观测
        else:
            num_critic_obs = self.env.num_obs # 否则使用普通观测
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic,使用序列演员评论家算法
        if self.cfg["policy_class_name"] == "ActorCriticSequence": # 使用序列演员评论家算法,观测值加入潜在维度
            num_critic_obs += self.policy_cfg["latent_dim"]
        actor_critic: ActorCritic = actor_critic_class( # 实例化演员评论家架构,将仿真环境的演员观测值数量/评论家观测值数量/动作数量和策略设置传入GPU
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg) # 实例化PPO算法
        self.num_steps_per_env = self.cfg["num_steps_per_env"] # 每一个环境迭代次数
        self.save_interval = self.cfg["save_interval"] # 每100次迭代检查一次是否保存模型

        # init storage and model,经验回放存储系统,用于创建与管理训练过程中收集的经验数据
        self.alg.init_storage(
            self.env.num_envs, # 并行环境数量
            self.num_steps_per_env, # 每个环境收集的步数
            [self.env.num_obs], # actor网络输入形状
            [num_critic_obs], # critic网络输入形状
            [self.env.obs_history_length * self.env.num_obs], # 历史观测序列形状
            [self.env.num_actions], #动作空间形状
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset() # 重置所有环境

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer,初始化TensorBoard写入器(用于可视化训练过程)
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len: # 随机初始化回合长度(课程学习)
            self.env.episode_length_buf = torch.randint_like( # 生成整数的范围[0,max_episode_length]
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # 获取初始观测
        obs, obs_history = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # 数据转移到设备
        obs, obs_history, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            critic_obs.to(self.device),
        )
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example),设置策略网络为训练模式
        # 初始化性能跟踪器
        ep_infos = [] # 存储回合信息
        rewbuffer = deque(maxlen=100) # 最近100个回合的奖励缓冲区
        lenbuffer = deque(maxlen=100) # 最近100个回合的长度缓冲区
        cur_reward_sum = torch.zeros( # 当前回合每个环境的累计奖励值
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros( # 当前回合每个环境的运行时间(步数)
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        # 计算总迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter): # 记录迭代开始时间
            start = time.time()
            # Rollout,经验收集阶段
            with torch.inference_mode(): # 禁用梯度计算,加速数据收集
                for i in range(self.num_steps_per_env): # 每个环境收集固定步数
                    actions = self.alg.act(obs, obs_history, critic_obs) # 根据策略生成动作,并且记录当前动作的价值,当前动作的概率的对数,当前动作的高斯分布,以及观测/历史观测/评论家观测
                    obs, privileged_obs, rewards, dones, infos, obs_history = ( # 环境执行动作到下一状态,返回当前状态的各种观测值和奖励/完成标志位等
                        self.env.step(actions)
                    )
                    critic_obs = privileged_obs if privileged_obs is not None else obs # 更新critic输入
                    obs, obs_history, critic_obs, rewards, dones = ( # 将最新数据转移到计算设备
                        obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos, obs) # 处理环境步(存储经验),其中rewards是各奖励函数的分奖励,dones是回合完成的环境,infos是回合信息,obs是观测值

                    if self.log_dir is not None: # 日志记录
                        # Book keeping,收集回合信息
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards # 更新当前回合统计,各分奖励在当前回合的各自累计值
                        cur_episode_length += 1 # 各环境的当前回合长度
                        new_ids = (dones > 0).nonzero(as_tuple=False) # 处理完成的回合
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0 # 已完成的回合奖励重置
                        cur_episode_length[new_ids] = 0 # 已完成的回合长度重置

                stop = time.time() # 记录数据收集时间
                collection_time = stop - start

                # Learning step,策略学习阶段
                start = stop # 重置计时器
                if self.cfg["policy_class_name"] == "ActorCriticSequence": # 序列模型特殊处理
                    critic_obs__ = torch.cat( # 拼接critic输入,用一批训练数据去更新一次评论家网络,因此只获取最后一次观测值
                        (critic_obs, self.alg.actor_critic.encode(obs_history)), dim=-1
                    )
                else:
                    critic_obs__ = critic_obs
                self.alg.compute_returns(critic_obs__) # 计算广义优势估计(GAE)
            # 策略网络更新
            mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss = ( # 执行PPO更新,返回值分别是:值函数损失均值,策略替代损失均值,KL散度均值,额外损失均值
                self.alg.update()
            )
            stop = time.time()
            learn_time = stop - start
            # 记录训练指标
            if self.log_dir is not None:
                self.log(locals()) # 记录所有局部变量
            if it % self.save_interval == 0: # 定期保存模型
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear() # 清空回合信息
            # 更新当前迭代计数
        self.current_learning_iteration = num_learning_iterations
        self.save(  # 保存最终模型
            os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations))
        )

    def log(self, locs, width=80, pad=35): # locs-字典参数,包含当前训练迭代的所有关键数据
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs # 累计总时间步数
        self.tot_time += locs["collection_time"] + locs["learn_time"] # 累计总训练时间
        iteration_time = locs["collection_time"] + locs["learn_time"] # 计算当前迭代的总耗时

        ep_string = f"" # 初始化空字符串,用于存储回合信息的格式化输出
        if locs["ep_infos"]: # 检查是否有回合信息
            for key in locs["ep_infos"][0]: # 遍历回合信息字典中所有键
                infotensor = torch.tensor([], device=self.device) # 初始化空张量,用于收集所有环境的该指标数据
                for ep_info in locs["ep_infos"]: # 遍历每个环境的回合信息
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor): # 如果指标值不是张量,转换为张量
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0: # 如果指标是标量(0维张量),增加一个维度
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device))) # 将当前环境的指标值拼接到总张量中
                value = torch.mean(infotensor) # 计算所有reset的环境的各奖励指标的平均值
                self.writer.add_scalar("Episode/" + key, value, locs["it"]) # 将指标记录到TensorBoard,使用"Episode/"前缀,当前迭代作为步数
                ep_string += f"""{f'Mean {key}:':>{pad}} {value:.4f}\n""" # 将指标添加到控制台输出字符串,使用右对齐格式
        mean_std = self.alg.actor_critic.std.mean() # 计算策略噪声的标准差平均值
        fps = int( # 计算每秒帧数
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar( # 记录价值函数损失
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"]) # 记录编码器损失(latent状态神经网络损失)
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"] # 记录代理损失(策略梯度损失)
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"]) # 记录当前学习率
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"]) # 记录策略噪声标准差
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"]) # 记录KL散度
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"]) # 记录总FPS
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"] # 记录数据收集时间
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"]) # 记录策略学习时间
        if len(locs["rewbuffer"]) > 0: # 检查奖励缓冲区是否有数据
            self.writer.add_scalar( # 记录平均奖励
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar( # 记录平均回合长度
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m " # 构建控制台输出标题

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
