#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from wheel_legged_gym.rsl_rl.modules import ActorCritic
from wheel_legged_gym.rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        extra_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        kl_decay=0,
        device="cpu",
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.kl_decay = max(kl_decay, 0)
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.actor.parameters()},
                {"params": self.actor_critic.critic.parameters()},
                {"params": self.actor_critic.std},
            ],
            lr=learning_rate,
        )
        self.extra_optimizer = None
        if self.actor_critic.is_sequence:
            self.extra_optimizer = optim.Adam(
                [
                    {"params": self.actor_critic.encoder.parameters()},
                ],
                lr=extra_learning_rate,
            )
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, critic_obs):
        if self.actor_critic.is_recurrent: # 处理循环网络状态
            self.transition.hidden_states = self.actor_critic.get_hidden_states() # 获取当前隐藏状态(用于RNN/LSTM/GRU等循环网络)
        # Compute the actions and values, 计算动作和值函数
        if self.actor_critic.is_sequence:
            self.transition.actions = self.actor_critic.act(obs, obs_history).detach() # 序列模型的动作计算,符合高斯分布
            latent = self.actor_critic.get_latent() # 计算潜在状态latent,网络输入为obs_history,网络输出为latent,潜在状态编码历史信息,作为状态估计器使用
            critic_obs = torch.cat((critic_obs, latent), dim=-1) # 扩展critic观测
        else:
            self.transition.actions = self.actor_critic.act(obs).detach() # 非序列模型的动作计算
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach() # 价值评估,得到所有环境当前状态的价值函数,用于优势函数计算,维度(num_envs,1)
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob( # 计算当前动作概率的对数
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach() # 记录动作概率分布的均值
        self.transition.action_sigma = self.actor_critic.action_std.detach() # 记录动作概率分布的标准差,通过这两个参数就可以得到动作的高斯分布
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.clone() # 保存当前状态快照(在环境执行前)
        self.transition.observation_history = obs_history.clone() # 保存历史观测
        self.transition.critic_observations = critic_obs.clone() # 保存评论家观测输入
        return self.transition.actions # 返回动作

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone() # 将环境返回的奖励存储到当前的转移对象中
        self.transition.dones = dones # 将回合结束标志存储到当前转移对象
        # Bootstrapping on time outs
        if "time_outs" in infos: # 超时的奖励也要加上,让奖励更加准确
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition) # 把所有信息存放到storage中
        self.transition.clear() # 清空传送块
        self.actor_critic.reset(dones) # 已完成的环境重置,由于使用的普通全连接多层感知机(MLP),不具备记忆功能,因此不会处理时间序列依赖,所以reset函数为空不影响

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach() # 对当前状态进行价值评估,作为最后一个状态的价值
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        if self.kl_decay != 0: # 动态调整期望的KL散度阈值
            self.desired_kl = max(self.desired_kl - self.kl_decay, 0.001) # 最小阈值0.001,防止过度约束
        num_updates = 0 # 用于跟踪多个epoch的平均训练指标,初始化一些损失和计数器
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        if self.actor_critic.is_recurrent: # 如果是循环网络(RNN,LSTM,GRU),需要处理序列数据,保持时间步之间的状态连续性
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator( # 标准mini-batch生成,数据是从storage中取得的
                self.num_mini_batches, self.num_learning_epochs
            )
        for (
            obs_batch, # 当前观测状态(batch_size,obs_dim)
            obs_history_batch, # 历史观测序列(batch_size,seq_len,obs_dim)
            critic_obs_batch, # Critic观测输入(batch_size,critic_obs_dim)
            actions_batch, # 执行的动作(batch_size,action_dim)
            target_values_batch, # 目标价值(batch_size,),这就是采集的经验中存储的历次动作的价值,从这往下的所有参数都是用旧策略得到的参数保存下来的
            advantages_batch, # 优势函数(batch_size,)
            returns_batch, # 回报(batch_size,)
            old_actions_log_prob_batch, # 旧策略的动作概率(batch_size,),        旧策略参数是从经验收集阶段保存的,用于重要性采样,旧策略在当前整个更新过程中是不变的,所有策略参数始终保持恒定
            old_mu_batch, # 旧策略的动作均值(batch_size,action_dim)
            old_sigma_batch, # 旧策略的动作标准差(batch_size,action_dim)
            hid_states_batch, # 隐藏状态(RNN)(num_layers,batch_size,hidden_size)
            masks_batch, # 序列掩码(batch_size,seq_len)
        ) in generator:
            if self.actor_critic.is_sequence:
                self.actor_critic.act( # 根据输入的当前观测和历史观测序列,得到actor网络在当前状态下输出动作的高斯分布,act函数会返回这个高斯分布的一个随机采样,作为下一步要执行的动作
                    obs_batch, # 当前观测                   这里没有用任何变量接收返回值,显然只是为了更新当前动作网络参数,比如下面的新策略的动作概率
                    obs_history_batch, # 历史观测序列
                    masks=masks_batch, # 序列掩码
                    hidden_states=hid_states_batch[0], # RNN初始隐藏状态
                )
            else:
                self.actor_critic.act(
                    obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
                )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob( # 计算旧策略产生的动作在新策略下的概率的对数
                actions_batch
            )
            value_batch = self.actor_critic.evaluate( # critic网络进行价值评估更新
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean # 当前actor网络,在当前状态输入下的均值
            sigma_batch = self.actor_critic.action_std # 当前actor网络,在当前状态输入下的标准差
            entropy_batch = self.actor_critic.entropy # 策略熵,衡量随机性

            # KL
            with torch.inference_mode():
                kl = torch.sum( # KL散度计算,衡量新旧策略(高斯分布)之间的差异
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl) # KL散度的均值

                if self.desired_kl is not None and self.schedule == "adaptive":  # 自适应学习率调整
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp( # 重要性采样比率
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio # 标准策略梯度损失
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp( # 剪裁代理损失,限制策略更新的最大幅度
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() # 取原始代理损失和剪裁代理损失的最小值(取了负号),物理意义:优势>0时,限制策略过度提高好动作的概率
                                                                            # 通过限制ratio的范围(1-e,1+e)实现的,                优势<0时,限制策略过度降低坏动作的概率
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + ( # 计算剪裁后的价值估计
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2) # 计算原始价值损失函数
                value_losses_clipped = (value_clipped - returns_batch).pow(2) # 计算剪裁价值损失函数
                value_loss = torch.max(value_losses, value_losses_clipped).mean() # 取最大损失函数,我们希望损失越小越好,损失越小说明评论家网络估计的越准确
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean() # 若不剪裁,使用标准MSE

            loss = ( # 将代理损失,价值损失与策略熵加权损失函数
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播,反向传播时,PyTorch计算图自动计算各自参数的梯度,代理损失与策略熵只与演员网络参数theta有关,价值损失只与评论家网络参数phi有关
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) # 梯度剪裁
            self.optimizer.step() # 参数更新,同时更新theta和
            # 统计信息记录
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            num_updates += 1

        num_updates_extra = 0 # 初始化辅助训练统计,辅助训练的更新次数
        mean_extra_loss = 0 # 辅助任务的累计损失
        if self.extra_optimizer is not None: # 模型包含辅助训练任务,通常用于:表示学习/自监督训练/多任务学习
            generator = self.storage.encoder_mini_batch_generator( # 自监督数据生成器
                self.num_mini_batches, self.num_learning_epochs
            )
            for next_obs_batch, critic_obs_batch, obs_history_batch in generator:
                if self.actor_critic.is_sequence: # 如果是序列演员评论家
                    latent_batch = self.actor_critic.encode(obs_history_batch) # 编码历史观测
                    vel_est_loss = ( # 速度估计损失,学习编码器的前3个维度预测速度
                        (latent_batch[:, :3] - critic_obs_batch[:, :3]).pow(2).mean()
                    )
                    if self.actor_critic.latent_dim > 3: # 观测去噪损失,学习编码器的剩余维度重建观测
                        obs_denoise_loss = ( # 后面各个维度的损失
                            (
                                latent_batch[:, 3 : self.actor_critic.latent_dim]
                                - critic_obs_batch[:, 3 : self.actor_critic.latent_dim]
                            )
                            .pow(2)
                            .mean()
                        )
                        extra_loss = vel_est_loss + obs_denoise_loss
                    else:
                        extra_loss = vel_est_loss

                self.extra_optimizer.zero_grad() # 梯度清零
                extra_loss.backward() # 反向传播
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.1) # 梯度剪裁
                self.extra_optimizer.step() # 参数更新,仅更新编码器参数

                mean_extra_loss += extra_loss.item() # 统计记录
                num_updates_extra += 1

        mean_value_loss /= num_updates # 计算主任务平均价值损失
        mean_surrogate_loss /= num_updates # 计算主任务平均代理损失
        mean_kl /= num_updates # 计算主任务平均KL
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra # 计算额外任务平均损失
        self.storage.clear() # 清空缓存

        return (mean_value_loss, mean_surrogate_loss, mean_kl, mean_extra_loss)
