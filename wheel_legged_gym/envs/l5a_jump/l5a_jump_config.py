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

from wheel_legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class L5A_JUMP_Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096 * 4 # 4096,设置并行训练的环境数量,在仿真中会同时运行此数量的独立实例
        num_actions = 8 # 每个环境中机器人的动作空间维度,通常对应于机器人的关节数量
        num_observations = 3 + 4 + 4 + 3 + 3 + num_actions*2 + 1 +3 # 定义状态观测向量的维度为27,即每个环境的状态观测是一个包含27个特征值的向量
        num_privileged_obs = 3 + 3 + num_observations + 3 + num_actions*4 + 7 * 11 + 3 + 1 * 3 + 6 + 5# 特权观测,评论家网络输入+jump_stages(5)
        obs_history_length = 5  # number of observations stacked together,状态观测历史堆叠的长度
        obs_history_dec = 1 # 状态历史堆叠的衰减参数,当前时刻权重最高,这里设置为1说明没有衰减
        env_spacing = 3.0  # not used with heightfields/trimeshes,环境中示例之间的间隔为3.0米
        send_timeouts = True  # send time out information to the algorithm,当环境中的某个体达到终止条件(如超时)时,会向训练算法发送超时信息
        episode_length_s = 20  # episode length in seconds,每个回合的最大时长为20s,超过这个时间,回合会被终止
        dof_vel_use_pos_diff = True # True:关节速度通过位置差分来计算,False:直接使用仿真器提供的关节速度
        fail_to_terminal_time_s = 0.5 # 机器人失败(如摔倒)到环境终止的时间为0.5s,避免过早终止导致学习不稳定
        # enable_auto_jump = True   # 训练用


    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane" # 地形网格类型:"plane-平坦地形","trimesh-三角形网格","heightfield-高度场"
        horizontal_scale = 0.1  # [m],水平缩放比例,表示每个网格单元代表10cm x 10cm
        vertical_scale = 0.005  # [m],垂直缩放比例,高度图中每个单位值的实际高度
        border_size = 25  # [m],边界大小,地形周围的边界区域大小,防止机器人移动到地形边缘外
        curriculum = True
        static_friction = 0.5 # 静摩擦系数
        dynamic_friction = 0.5 # 动摩擦系数
        restitution = 0.5 # 弹性系数
        # rough terrain only:
        measure_heights = True
        measured_points_x = [ # 地形测量点
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = False  # select a unique terrain type and pass all arguments,是否使用单一地形
        terrain_kwargs = None  # Dict of arguments for selected terrain,特定地形参数
        max_init_terrain_level = 5  # starting curriculum state,初始最大地形难度级别
        terrain_length = 8.0 # 地形片段的长度(m)
        terrain_width = 8.0 # 地形片段的宽度(m)
        num_rows = 10  # number of terrain rows (levels),地形网格的行数(难度级别)
        num_cols = 20  # number of terrain cols (types),地形网格的列数(类型)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.2, 0.1] # 地形类型比例分布
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces,坡度阈值,当坡度超过此阈值时,地形会被修正为垂直表面

    class init_state(LeggedRobotCfg.init_state):

        pos = [0.0, 0.0, 0.651]  # [0.0, 0.0, 0.63]  # x,y,z [m]  #0.515,base初始位置
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat],base初始姿态
        default_joint_angles = {  # target angles when action = 0.0,各关节初始角度
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": 0.261799,
            "left_knee_joint": -0.508606,
            "left_wheel_joint": 0.0,
            "right_hip_roll_joint": 0.0, 
            "right_hip_pitch_joint": 0.261799,   
            "right_knee_joint": -0.508606,  
            "right_wheel_joint": 0.0,
        }
        # default_joint_angles_sit = {##0.4991
        #     "left_hip_roll_joint": 0.0,
        #     "left_hip_pitch_joint": 0.64577,
        #     "left_knee_joint": -1.10898,
        #     "left_wheel_joint": 0.0,
        #     "right_hip_roll_joint": 0.0, 
        #     "right_hip_pitch_joint": 0.64577,   
        #     "right_knee_joint": -1.10898,  
        #     "right_wheel_joint": 0.0,
        # }
        default_joint_angles_sit = {##0.4792
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": 0.69813,
            "left_knee_joint": -1.18141,
            "left_wheel_joint": 0.0,
            "right_hip_roll_joint": 0.0, 
            "right_hip_pitch_joint": 0.69813,   
            "right_knee_joint": -1.18141,  
            "right_wheel_joint": 0.0,
        }



    class commands(LeggedRobotCfg.commands):
        curriculum = False  # True,是否使用课程学习
        basic_max_curriculum = 1.5 # 不同阶段的课程学习的难度值
        advanced_max_curriculum = 2 # 高级阶段的最大课程难度值
        curriculum_threshold = 0.7 # 课程学习的阈值
        num_commands = 6 # 命令的数量
        resampling_time = 6.0  # time before command are changed[s],重新采样命令的时间间隔
        heading_command = False  # if true: compute ang vel command from heading error,true:根据朝向误差计算角速度命令,false:轮差速计算角速度
        threshold = 0.15 #0.2
        jump_train = True

        class ranges: # 定义了每个命令可以取值的范围
            lin_vel_x = [-1.0, 1.0]  # min max [m/s],线速度命令范围
            ang_vel_yaw = [-1.0, 1.0] # 角速度命令范围
            height = [0.40, 0.47, 0.641, 0.661]  # 高度范围上下浮动3cm
            mode = [1, 1]  # 0为四轮模式，1为两轮
            heading = [-3.14, 3.14] # 朝向范围
            jump_height = [1.0,1.0]

    class control(LeggedRobotCfg.control):
        # 位置动作的缩放系数
        action_scale_pos = 0.5
        # 速度动作的缩放系数
        action_scale_vel = 10.0

        # PD控制器参数:
        # 各关节的刚度系数
        stiffness = {"hip_roll": 20.0, "hip_pitch": 20.0, "knee": 30.0, "wheel": 0}  # [N*m/rad]
        # 各关节的阻尼系数
        damping = {"hip_roll": 2, "hip_pitch": 2, "knee": 3, "wheel": 2.0}  # [N*m*s/rad]

        # 抽取率：每个策略时间步长内的控制动作更新次数
        decimation = 2

    class asset(LeggedRobotCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/l5a/urdf/l5aurdf20260209.urdf" # 机器人urdf路径
        name = "l5a" # 机器人名称
        foot_name = "wheel" # 足部名称
        joint_indices = [0, 1, 2, 4, 5, 6] # 关节索引
        wheel_indices = [3, 7] # 轮子索引
        base_link_indices = [0]
        joint_link_indices = [1, 2, 3, 5, 6, 7]
        wheel_link_indices = [4, 8]
        wheel_radius = 0.127
        track_width = 0.1185*2
        penalize_contacts_on = ["right_hip", "right_knee", "base", "left_hip", "left_knee"] # 惩罚区域
        terminate_after_contacts_on = ["right_hip", "right_knee", "base", "left_hip", "left_knee"] # 终止区域
        # terminate_after_contacts_on = ["base", "right_knee", "left_knee"] # 终止区域
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter,控制机器人自身各部分之间是否能够发生碰撞检测
        flip_visual_attachments = False # 设置是否反转视觉附件
        stage_names = ["stand", "sit", "jump", "air", "land"]

    class domain_rand(LeggedRobotCfg.domain_rand):

        randomize_friction = True # 摩擦力随机化
        friction_range = [0.1, 2.0]

        randomize_restitution = True # 弹性系数随机化
        restitution_range = [0.0, 1.0]

        randomize_base_mass = True # 质量随机化
        added_mass_range = [-2.0, 2.0]

        randomize_inertia = True # 惯量随机化
        randomize_inertia_range = [0.9, 1.1]

        randomize_base_com = True # 质心随机化
        rand_com_vec = [0.01, 0.01, 0.01]

        push_robots = True # 推动机器人
        push_interval_s = 3
        max_push_vel_xy = 0.5  # 最大推动线速度
        max_push_ang_vel = 0.4  # 最大推动角速度

        load_robots = False  # 0602加的

        randomize_Kp = True # Kp,Kd随机化
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]

        randomize_motor_torque = True # 电机力矩随机化
        randomize_motor_torque_range = [0.9, 1.1]

        randomize_default_dof_pos = True # 关节位置随机化
        randomize_default_dof_pos_range = [-0.05, 0.05]

        randomize_action_delay = True # 动作延时随机化
        delay_ms_range = [10, 30]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales: # 观测值缩放因子,将观测值缩放到统一维度,先给零观测值,把这些变量先读出来,然后缩放到正负1范围内
            lin_vel = 2.0
            ang_vel = 0.25  # 0.25

            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025

            quat = 1.0
            height_measurements = 12.0  # 40.0   #5.0

            torque = 0.05
            mode = 1
            jump = 1

        class priv_obs_scales:
            external_wrench = 1.0

        clip_observations = 100.0 # 观测值剪裁
        clip_actions = 100.0 # 动作剪裁

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.4

        class noise_scales(LeggedRobotCfg.noise.noise_scales): # 噪声缩放因子
            dof_pos = 0.05  # 关节位置噪声
            dof_vel = 0.05  # 关节速度噪声
            lin_vel = 0.1  # 线速度噪声
            ang_vel = 0.1  # 角速度噪声
            quat = 0.03  # 四元数噪声
            height_measurements = 0.1  # 高度测量噪声

    class rewards:
        class scales: # 奖励缩放因子

            # 速度跟踪
            tracking_lin_vel = 1
            # tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1
            
            # 跳跃阶段控制
            height = 3.0
            hip_pos = -2
            lin_vel_z_0 = -1.0
            lin_vel_z_1 = -1.0
            lin_vel_z_2 = 2.0

            contact = 1.5#4.0

            orientation = 1.0 # 基座重力投影方向
            leg_end_x_diff = 1.0 # 两条腿不要叉开
            style = 1.0
            torques = 0.0001

            # 机器人姿态柔顺性
            # ang_vel_xy = -0.05

            # # 机器人关节柔顺性
            # dof_vel = -0.05#0.1
            # # dof_vel_upper = -0.005
            dof_vel_wheel = 1.0
            # dof_acc = -2.5e-7
            # torques = -2e-7 #-0.0001
            action_rate = -0.02
            # action_smooth = -0.005

            # 机器人关节限制
            dof_pos_limits = -5
            dof_vel_limits = -5
            torque_limits = -5
            
            # 碰撞惩罚
            collision = -2.0

        # 是否只保留正奖励（true 时负的总奖励修剪为 0，防止训练中提早终止的问题）
        only_positive_rewards = False

        enhance_factor = 0.01  # 增强奖励的系数

        # 单个奖励的修剪上限值
        clip_single_reward = 5

        # 跟踪奖励的高斯分布参数 σ，跟踪奖励计算公式：exp(-error^2/sigma)
        tracking_sigma = 0.25

        # 关节位置的软限制，作为关节的百分比限制（超过此限制值的关节位置将受惩罚）
        soft_dof_pos_limit = 0.97

        # 关节速度的软限制（超过此速度限制将受到惩罚）
        soft_dof_vel_limit = 0.95

        # 扭矩的软限制（超过此扭矩限制将受到惩罚）
        soft_torque_limit = 0.95

        # 接触力
        min_wheel_contact_force = 20.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 模拟时间步长 [秒]
        substeps = 1  # 每个时间步的子步数
        gravity = [0.0, 0.0, -9.81]  # 重力加速度 [m/s^2]
        up_axis = 1  # 选择“上轴”：0 表示 y 轴，1 表示 z 轴

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 20  # 物理引擎线程数
            solver_type = 1  # 求解器类型：0 表示 PGS（逐次松弛法），1 表示 TGS（无约束梯度下降法）
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01  # 碰撞检测距离偏移量 [米]
            rest_offset = 0.0  # 静止状态偏移量 [米]
            bounce_threshold_velocity = 0.5  # 弹跳速度阈值 [米/秒]
            max_depenetration_velocity = 1.0  # 最大分离速度 [米/秒]
            max_gpu_contact_pairs = 2**23  # GPU 支持的最大碰撞对数，默认值 2**24，适用于 8000 环境及以上
            default_buffer_size_multiplier = 5  # 默认缓冲区大小的乘数
            contact_collection = 2  # 碰撞数据收集模式：0 表示不收集，1 表示收集最后一个子步的数据，2 表示收集所有子步的数据（默认值为 2）


class L5A_JUMP_CfgPPO(LeggedRobotCfgPPO):
    # PPO算法配置
    seed = 1  # 随机种子,确保实验可重复性,相同的种子会产生相同的随机行为
    runner_class_name = "OnPolicyRunner"  # 运行器类名,每次迭代都使用最新生成的数据进行训练

    class policy(LeggedRobotCfgPPO.policy): # 继承基类策略配置,覆盖特定参数
        init_noise_std = 0.5 # 策略网络输出动作的初始噪声标准差,用于探索,噪声值越高探索性越强,训练初期增加探索,后期逐渐衰减
        actor_hidden_dims = [512, 256, 128] # 3层神经网络,策略网络,用于生成动作
        critic_hidden_dims = [768, 256, 128] # 3层神经网络,价值网络,评估状态价值
        activation = "elu" # 激活函数,ELU比ReLU更平滑,缓解梯度消失问题

        # only for ActorCriticSequence
        num_encoder_obs = L5A_JUMP_Cfg.env.obs_history_length * L5A_JUMP_Cfg.env.num_observations # 编码器输入维度=历史观测长度*单步观测维度
        latent_dim = 3  # at least 3 to estimate base linear velocity,基座速度空间
        encoder_hidden_dims = [256, 128, 64] # 编码器网络层数和数量
        # encoder_hidden_dims = [128, 64]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        kl_decay = (LeggedRobotCfgPPO.algorithm.desired_kl - 0.002) / LeggedRobotCfgPPO.runner.max_iterations # KL散度,用于限制策略更新幅度

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticSequence"  # could be ActorCritic, ActorCriticSequence,使用序列处理的actor-critic
        algorithm_class_name ="PPO" # 明确使用PPO算法
        num_steps_per_env = 48  # per iteration,每个环境每次迭代收集48步数据
        max_iterations = 20000  # number of policy updates,最大迭代次数

        # logging
        experiment_name = "l5a_jump"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None  # updated from load_run and chkpt
