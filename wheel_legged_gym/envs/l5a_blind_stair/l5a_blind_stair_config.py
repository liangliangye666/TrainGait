from wheel_legged_gym.envs.l5a_step.l5a_step_config import (
    L5A_STEP_Cfg,
    L5A_STEP_CfgPPO,
)


class L5A_BLIND_STAIR_Cfg(L5A_STEP_Cfg):
    class env(L5A_STEP_Cfg.env):
        num_envs = 4096
        episode_length_s = 20
        fail_to_terminal_time_s = 0.4

    class terrain(L5A_STEP_Cfg.terrain):
        # Use a generated heightfield/trimesh instead of modifying existing tasks.
        mesh_type = "trimesh"
        curriculum = False
        selected = False
        measure_heights = True

        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 1
        num_cols = 1
        border_size = 1.0
        horizontal_scale = 0.025
        vertical_scale = 0.005
        slope_treshold = 0.75

        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.0

        # Four-direction pit-like stair terrain.
        blind_stair_step_height = 0.08
        blind_stair_step_width = 0.25
        blind_stair_num_steps = 10
        blind_stair_pit_half_size = 0.65

    class commands(L5A_STEP_Cfg.commands):
        curriculum = False
        num_commands = 7
        resampling_time = 20.0
        heading_command = False
        jump_train = False

        # Contact-triggered gait parameters.
        contact_force_threshold = 30.0
        contact_window_size = 3
        swing_duration = 0.60
        contralateral_delay = 0.30

        feedforward_init_weight = 1.0
        feedforward_anneal_steps = 5000
        feedforward_hip_amplitude = 0.25
        feedforward_knee_amplitude = -0.50

        class ranges(L5A_STEP_Cfg.commands.ranges):
            # Robot should first roll along its randomized initial heading.
            lin_vel_x = [0.45, 0.75]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.641, 0.661]
            mode = [1, 1]
            jump_height = [0.0, 0.0]
            step_rise_height = [0.08, 0.08]

    class init_state(L5A_STEP_Cfg.init_state):
        # Start in the low central area of the stair pit.
        pos = [0.0, 0.0, 0.70]

    class domain_rand(L5A_STEP_Cfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.2]
        randomize_restitution = True
        restitution_range = [0.0, 0.2]
        randomize_base_mass = True
        added_mass_range = [-0.3, 0.8]
        randomize_inertia = False
        randomize_base_com = True
        rand_com_vec = [0.02, 0.02, 0.02]
        randomize_Kp = True
        randomize_Kd = True
        randomize_motor_torque = True
        randomize_default_dof_pos = True
        randomize_action_delay = True
        delay_ms_range = [0, 20]
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.3

    class rewards(L5A_STEP_Cfg.rewards):
        class scales(L5A_STEP_Cfg.rewards.scales):
            alive = 0.2
            tracking_lin_vel = 0.8
            tracking_ang_vel = 0.4
            base_height = 0.6
            orientation = 1.2
            lin_vel_z = -0.6
            ang_vel_xy = -0.05
            lateral_vel = -0.6

            # CTBC-style contact-triggered climbing terms.
            contact_trigger_swing = 3.0
            contact_trigger_clearance = 2.5
            contact_trigger_sequence = 1.5
            climb_progress = 2.0
            stair_height_progress = 3.0
            wheel_zero_velocity_in_swing = 0.4

            no_double_air = -2.0
            wrong_double_contact = -0.5
            stance_wheel_y_position = 0.6
            wheel_vel = -0.0005
            dof_vel = -0.02
            dof_acc = -2.5e-7
            action_rate = -0.04
            action_smooth = -0.005
            torques = -1.0e-5
            collision = -2.0
            dof_pos_limits = -0.5
            dof_vel_limits = -0.05
            torque_limits = -0.05

        only_positive_rewards = False
        clip_single_reward = 5
        tracking_sigma = 0.25
        min_wheel_contact_force = 20.0


class L5A_BLIND_STAIR_CfgPPO(L5A_STEP_CfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(L5A_STEP_CfgPPO.policy):
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        activation = "elu"

        num_encoder_obs = L5A_BLIND_STAIR_Cfg.env.obs_history_length * L5A_BLIND_STAIR_Cfg.env.num_observations
        latent_dim = 3
        encoder_hidden_dims = [256, 128, 64]

    class algorithm(L5A_STEP_CfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        desired_kl = 0.005

    class runner(L5A_STEP_CfgPPO.runner):
        policy_class_name = "ActorCriticSequence"
        algorithm_class_name = "PPO"
        num_steps_per_env = 48
        max_iterations = 10000
        experiment_name = "l5a_blind_stair"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
