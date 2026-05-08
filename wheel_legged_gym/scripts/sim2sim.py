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
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import math
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from wheel_legged_gym.envs import Y1A_2WHEEL_VMCCfg
import torch


class cmd:
    vel_x = 0.5
    vel_yaw = 0
    mode = 1
    height = 0.45 * (1 - mode) + 0.53 * mode
    heading = 0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def forward_kinematics(cfg, theta1, theta2):
    end_x = cfg.asset.l2 * np.cos(math.pi / 2 - theta1 + theta2) - cfg.asset.l1 * np.sin(theta1) - cfg.asset.offset_x
    end_y = cfg.asset.offset_z - cfg.asset.l1 * np.cos(theta1) - cfg.asset.l2 * np.sin(math.pi / 2 - theta1 + theta2)
    L0 = np.sqrt(end_x**2 + end_y**2)
    theta0 = -np.arctan2(end_x, end_y)
    return L0, theta0


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    # print("q: ", q[:3])
    dq = data.qvel.astype(np.double)
    # print("dq: ", dq[:6])
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)

    # ddq = data.qacc.astype(np.double)
    # print("ddq: ", ddq[:3])

    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def initialize_qpos(model, data):
    if cmd.mode == 0:  # 4wheel
        data.qpos = model.key_qpos[0]
        # data.qpos = np.array([0, 0, 0.5297, 1, 0, 0, 0, 0.6109, -0.0873, 0, 0, 0.6109, -0.0873, 0, 0], dtype=np.double)
    elif cmd.mode == 1:  # 2wheel
        data.qpos = model.key_qpos[1]
        # data.qpos = np.array([0, 0, 0.477, 0.707, 0.0, 0.0, 0.707, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.double)


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)

    # # 获取关节名称和对应的索引
    # for joint_id in range(model.njnt):
    #     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    #     qpos_index = model.jnt_qposadr[joint_id]
    #     print(f"Joint {joint_id}: {joint_name}, qpos index: {qpos_index}")

    # # 遍历所有 actuators，获取类型为 'motor' 的
    # motor_names = []
    # for actuator_id in range(model.nu):
    #     actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    #     actuator_type = model.actuator_gaintype[actuator_id]  # 获取 actuator 类型
    #     if actuator_type == 0:  # 类型 0 表示 'motor'
    #         motor_names.append(actuator_name)
    #
    # print("Motors in order:", motor_names)

    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    # Set the initial state
    initialize_qpos(model, data)

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 0

    obs_history_length = cfg.env.obs_history_length

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        # right_hip_joint right_knee_joint rf_wheel_joint rb_wheel_joint
        # left_hip_joint left_knee_joint lf_wheel_joint lb_wheel_joint
        q = q[-cfg.env.num_actions :]
        dq = dq[-cfg.env.num_actions :]

        # 1000hz -> 100hz
        # if count_lowlevel % cfg.sim_config.decimation == 0:

        obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
        # eu_ang = quaternion_to_euler_array(quat)
        # eu_ang[eu_ang > math.pi] -= 2 * math.pi

        theta1 = q[[0, 4]]  # hip
        theta2 = q[[1, 5]]  # knee
        theta1_dot = dq[[0, 4]]
        theta2_dot = dq[[1, 5]]

        # 使用正向运动学计算腿长L0和腿的倾角theta0
        L0, theta0 = forward_kinematics(cfg, theta1, theta2)
        print("theta0: ", theta0, L0)

        # 计算dt时间后的腿长和腿倾角
        L0_temp, theta0_temp = forward_kinematics(
            cfg, theta1 + theta1_dot * cfg.sim_config.dt, theta2 + theta2_dot * cfg.sim_config.dt
        )
        # 计算腿长的变化率
        L0_dot = (L0_temp - L0) / cfg.sim_config.dt
        # 计算腿倾角的变化率
        theta0_dot = (theta0_temp - theta0) / cfg.sim_config.dt

        obs[0, 0:4] = quat * cfg.normalization.obs_scales.quat
        obs[0, 4:7] = omega * cfg.normalization.obs_scales.ang_vel

        obs[0, 7] = cmd.vel_x * cfg.normalization.obs_scales.lin_vel
        obs[0, 8] = cmd.vel_yaw * cfg.normalization.obs_scales.ang_vel
        obs[0, 9] = cmd.height * cfg.normalization.obs_scales.height_measurements
        obs[0, 10] = cmd.mode * cfg.normalization.obs_scales.mode

        obs[0, 11:13] = theta0 * cfg.normalization.obs_scales.dof_pos
        obs[0, 13:15] = theta0_dot * cfg.normalization.obs_scales.dof_vel
        obs[0, 15:17] = L0 * cfg.normalization.obs_scales.l0
        obs[0, 17:19] = L0_dot * cfg.normalization.obs_scales.l0_dot

        obs[0, 19:21] = q[:2] * cfg.normalization.obs_scales.dof_pos
        obs[0, 21:23] = q[4:6] * cfg.normalization.obs_scales.dof_pos
        obs[0, 23:31] = dq * cfg.normalization.obs_scales.dof_vel
        obs[0, 31:39] = action

        obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

        policy_input = np.zeros([1, cfg.env.num_observations + 3], dtype=np.float32)

        policy_input = np.hstack((obs, v.reshape(1, -1) * cfg.normalization.obs_scales.lin_vel))

        # left_hip_joint left_knee_joint lb_wheel_joint lf_wheel_joint
        # right_hip_joint right_knee_joint rb_wheel_joint rf_wheel_joint
        action[:] = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
        action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

        # print("action:", action)

        target_q[[0, 1, 4, 5]] = action[[0, 1, 4, 5]] * cfg.control.action_scale_pos
        target_dq[[2, 3, 6, 7]] = action[[2, 3, 6, 7]] * cfg.control.action_scale_vel

        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau[[2, 6]] = (1 - cmd.mode) * tau[[2, 6]]
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        # print("tau:", tau)

        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.")
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(Y1A_2WHEEL_VMCCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/y1a/mjcf/y1a-terrain.xml"
            else:
                mujoco_model_path = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/y1a/mjcf/y1a.xml"
            sim_duration = 10.0
            dt = 0.005
            decimation = 10

        class robot_config:
            kps = np.array([400, 400, 0, 0, 600, 600, 0, 0], dtype=np.double)
            kds = np.array([40, 40, 20, 20, 40, 40, 20, 20], dtype=np.double)
            # tau_limit = np.array([300, 300, 60, 60, 300, 300, 60, 60], dtype=np.double)
            tau_limit = np.array([745, 880, 460, 460, 745, 880, 460, 460], dtype=np.double)
            # tau_limit = 800.0 * np.ones(8, dtype=np.double)  # 力矩限制

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
