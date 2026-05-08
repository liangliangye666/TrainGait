from wheel_legged_gym.envs.l5a_balance.l5a_balance_config import (
    L5A_BALANCE_Cfg,
    L5A_BALANCE_CfgPPO,
)


class L5A_STEP_Cfg(L5A_BALANCE_Cfg):
    class env(L5A_BALANCE_Cfg.env):
        num_envs = 4096
        num_actions = 8

        # actor 观测维度：
        # base_ang_vel 3
        # base_quat_local 4
        # command_obs 5: lin_x, yaw, height, mode, step_height
        # joint_pos 6
        # dof_vel 8
        # actions 8
        # sin/cos phase 2
        # desired_swing mask 2
        # total = 38
        num_observations = 38

        # critic 特权观测维度：
        # base_lin_vel 3
        # base_euler 3
        # obs_buf 38
        # projected_gravity 3
        # last_actions0 8
        # last_actions1 8
        # dof_acc 8
        # heights 77
        # torques 8
        # base_mass_delta 1
        # base_com 3
        # friction 1
        # restitution 1
        # external_wrench 6
        # total = 168
        num_privileged_obs = 168

        obs_history_length = 5
        episode_length_s = 20
        fail_to_terminal_time_s = 0.4
        dof_vel_use_pos_diff = True

    class terrain(L5A_BALANCE_Cfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = True
        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.0

    class commands(L5A_BALANCE_Cfg.commands):
        curriculum = False
        num_commands = 7
        resampling_time = 4.0
        heading_command = False
        jump_train = False

        # 步态参数
        gait_period = 0.8
        stance_ratio = 0.55
        contact_force_threshold = 1.0

        class ranges(L5A_BALANCE_Cfg.commands.ranges):
            # 原地交替抬腿，先不要让它走
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

            # 只用两轮模式，所以高度范围用正常两轮站立高度
            height = [0.641, 0.661]
            mode = [1, 1]

            # 不训练跳跃
            jump_height = [0.0, 0.0]

            # 抬轮高度，建议先 6~10cm
            step_rise_height = [0.06, 0.10]

    class control(L5A_BALANCE_Cfg.control):
        action_scale_pos = 0.25
        action_scale_vel = 0.5
        stiffness = {
            "hip_roll": 40.0,
            "hip_pitch": 40.0,
            "knee": 80.0,
            "wheel": 0.0,
        }
        damping = {
            "hip_roll": 2.0,
            "hip_pitch": 2.0,
            "knee": 2.0,
            "wheel": 1.5,
        }
        decimation = 2

    class domain_rand(L5A_BALANCE_Cfg.domain_rand):
        # 第一阶段建议先降低随机化难度，先让动作学出来
        randomize_friction = True
        friction_range = [0.6, 1.2]

        randomize_restitution = True
        restitution_range = [0.0, 0.1]

        randomize_base_mass = True
        added_mass_range = [0.0, 0.0]

        randomize_inertia = False
        randomize_base_com = False

        push_robots = False
        push_interval_s = 3
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        randomize_Kp = False
        randomize_Kd = False
        randomize_motor_torque = False
        randomize_default_dof_pos = False
        randomize_action_delay = False
        delay_ms_range = [0, 10]

    class normalization(L5A_BALANCE_Cfg.normalization):
        class obs_scales(L5A_BALANCE_Cfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            quat = 1.0
            height_measurements = 12.0
            torque = 0.05
            mode = 1.0
            jump = 1.0
            step_rise_height = 10.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(L5A_BALANCE_Cfg.noise):
        # 第一阶段建议关噪声，动作稳定后再打开
        add_noise = False
        noise_level = 0.4

        class noise_scales(L5A_BALANCE_Cfg.noise.noise_scales):
            dof_pos = 0.03
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            quat = 0.02
            height_measurements = 0.05

    class rewards(L5A_BALANCE_Cfg.rewards):
        class scales:
            # 基础生存
            alive = 0.2

            # 速度保持：原地抬腿，不要乱跑，不要乱转
            tracking_lin_vel = 0.6
            tracking_ang_vel = 0.6
            lateral_vel = -3.0      # 原先是-0.8
            base_xy_drift = -2.0    # 原先是-0.8


            # 姿态和高度
            base_height = 0.8
            orientation = 1.2
            lin_vel_z = -0.8
            ang_vel_xy = -0.08
            leg_end_x_diff = 0.5    # 原先是0.3

            # 核心步态奖励
            step_contact = 3.0
            step_swing_height = 3.0
            step_swing_velocity = 0.3
            no_double_air = -3.0
            wrong_double_contact = -0.8
            stance_wheel_y_position = 0.8


            # 防止靠轮子乱滚蒙混过关
            wheel_vel = -0.002

            # 动作平滑和能耗
            dof_vel = -0.05         # 原先是-0.02
            dof_acc = -2.5e-7
            action_rate = -0.08     # 原先是-0.02
            action_smooth = -0.005
            torques = -2.0e-5

            # 安全约束
            collision = -2.0
            dof_pos_limits = -0.5
            dof_vel_limits = -0.05
            torque_limits = -0.05
            contact_no_vel = -0.05

        only_positive_rewards = False
        clip_single_reward = 5
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.97
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        min_wheel_contact_force = 20.0


class L5A_STEP_CfgPPO(L5A_BALANCE_CfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(L5A_BALANCE_CfgPPO.policy):
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        activation = "elu"

        num_encoder_obs = L5A_STEP_Cfg.env.obs_history_length * L5A_STEP_Cfg.env.num_observations
        latent_dim = 3
        encoder_hidden_dims = [256, 128, 64]

    class algorithm(L5A_BALANCE_CfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        desired_kl = 0.005

    class runner(L5A_BALANCE_CfgPPO.runner):
        policy_class_name = "ActorCriticSequence"
        algorithm_class_name = "PPO"
        num_steps_per_env = 48
        max_iterations = 10000

        experiment_name = "l5a_step"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
