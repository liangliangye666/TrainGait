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


import math  # 导入Python标准数学库,可提供基础数学函数(如三角函数/指数/对数等),用于机器人运动计算/坐标变换等
import numpy as np  # 导入NumPy科学计算库,np是别名,用于高效的多维数组操作(如位姿数据/传感器读数矩阵等),提供线性代数/统计函数支持机器人状态处理
import mujoco  # 导入MuJoCo物理引擎核心接口,解析机器人模型XML文件,执行物理仿真计算,获取/设置关节状态/传感器数据
import mujoco_viewer  # 导入MuJoCo可视化工具,实时渲染仿真环境,提供3D交互式视图窗口,调试机器人动作与物理响应
from tqdm import tqdm # 进度条可视化工具,适合监控长时间运行的任务
from collections import deque # 导入双端队列数据结构,高效存储历史状态数据,用于强化学习的经验回放机制
from scipy.spatial.transform import Rotation as R # 导入SciPy旋转变换工具,处理3D旋转问题(四元数/欧拉角/旋转矩阵转换)
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR # 导入自定义路径常量
from wheel_legged_gym.envs import L5A_STEP_Cfg # 导入机器人环境配置类作为父类
import torch # 导入PyTorch深度学习框架,构建神经网络策略(如Actor-Critic架构),自动求导训练强化学习模型,GPU加速仿真数据处理(如状态特征提取)


class cmd:
    vel_x = -0.0
    vel_yaw = 0
    mode = 1
    height = 0.45 * (1 - mode) + 0.651 * mode
    step_rise_height = 0.1


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


def get_euler_zyx_tensor(quat):

    x, y, z, w = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (math.pi / 2), math.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Stack results
    return np.array([roll, pitch, yaw])

def quat_from_euler_zyx(euler_zyx):
    roll,pitch,yaw = euler_zyx[0],euler_zyx[1],euler_zyx[2]
    
    cr = math.cos(roll/2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])

def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    # print("q: ", q[:3]), 打印x,y,z
    dq = data.qvel.astype(np.double)
    # print("dq: ", dq[:6]), 打印vx,vy,vz,wx,wy,wz
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double) # 读取四元数姿态信息,并且重新排列,保存为double类型
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame,基座坐标系中的速度
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double) # 基座坐标系中力方向[gx, gy, gz]

    # ddq = data.qacc.astype(np.double)
    # print("ddq: ", ddq[:3])
    # q-117维,dq-16维,quat-4维,v-3维,omega-3维,gvec-3维
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q,default_q, q, kp, target_dq, dq, kd): # target_q:目标位置增量,default_q:关节默认位置/中立位置
    """Calculates torques from position commands"""
    return (target_q + default_q - q) * kp + (target_dq - dq) * kd


def initialize_qpos(model, data):
    data.qpos = model.key_qpos

def construct_policy_input(q, dq, base_quat_local, v, omega, action, step_idx, cfg):
    """
    构造 policy 输入的 38 维观测向量，与 L5A_STEP.compute_proprioception_observations() 一致

    Args:
        q: 8 维关节位置（MuJoCo 顺序）
        dq: 8 维关节速度
        base_quat_local: 4 维基座四元数（yaw=0）
        omega: 3 维基座角速度
        action: 8 维上一时刻动作
        step_idx: 当前仿真步数
        cfg: Sim2simCfg 对象

    Returns:
        np.ndarray: shape (1, 38) 的观测值
    """
    obs = np.zeros(cfg.env.num_observations + 3, dtype=np.float32)

    # [0:3]  base_ang_vel
    obs[0:3] = omega * cfg.normalization.obs_scales.ang_vel

    # [3:7]  base_quat_local
    obs[3:7] = base_quat_local * cfg.normalization.obs_scales.quat

    # [7:12] command_obs: [lin_vel_x, ang_vel_yaw, height, mode, step_rise_height]
    obs[7] = cmd.vel_x * cfg.normalization.obs_scales.lin_vel
    obs[8] = cmd.vel_yaw * cfg.normalization.obs_scales.ang_vel
    obs[9] = cmd.height * cfg.normalization.obs_scales.height_measurements
    obs[10] = cmd.mode * cfg.normalization.obs_scales.mode
    obs[11] = cmd.step_rise_height * cfg.normalization.obs_scales.step_rise_height

    # [12:18] 关节位置 (joint_indices=[0,1,2,4,5,6] → 不含轮子)
    obs[12:15] = q[:3] * cfg.normalization.obs_scales.dof_pos      # left_hip, left_pitch, left_knee
    obs[15:18] = q[4:7] * cfg.normalization.obs_scales.dof_pos     # right_hip, right_pitch, right_knee

    # [18:26] 所有 8 个 DOF 速度
    obs[18:26] = dq * cfg.normalization.obs_scales.dof_vel

    # [26:34] 上一时刻动作
    obs[26:34] = action

    # [34] sin_phase, [35] cos_phase
    period = cfg.commands.gait_period          # 0.8
    phase = (step_idx * cfg.sim_config.dt) % period / period
    obs[34] = np.sin(2 * np.pi * phase)
    obs[35] = np.cos(2 * np.pi * phase)

    # [36:38] desired_swing (左腿/右腿是否应该抬起)
    phase_left = phase
    phase_right = (phase + 0.5) % 1.0
    stance_ratio = cfg.commands.stance_ratio   # 0.55
    obs[36] = float(phase_left >= stance_ratio)     # 左腿 swing
    obs[37] = float(phase_right >= stance_ratio)    # 右腿 swing

    # obs[38]=v[0]
    # obs[39]=v[1]
    # obs[40]=v[2]

    obs[38:41] = v * cfg.normalization.obs_scales.lin_vel


    # clip
    obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

    return obs.reshape(1, -1)


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

    default_q = data.qpos[7:].copy()
    print(default_q)

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 0

    for i in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data) # 获取17+16+4+3+3+3=46维的观测量

        # without yaw
        base_euler_zyx = get_euler_zyx_tensor(quat)
        base_euler_zyx_local = base_euler_zyx
        base_euler_zyx_local[2] = 0
        base_quat_local = quat_from_euler_zyx(base_euler_zyx_local)

        # right_hip_joint right_knee_joint rf_wheel_joint rb_wheel_joint
        # left_hip_joint left_knee_joint lf_wheel_joint lb_wheel_joint
        q = q[-cfg.env.num_actions :] # q和dq都取后面6个关节相关的值
        dq = dq[-cfg.env.num_actions :]

        # 1000hz -> 100hz
        # if count_lowlevel % cfg.sim_config.decimation == 0:

        '''
        构建观测值
        '''
        obs = construct_policy_input(q, dq, base_quat_local, v, omega, action, i, cfg)


        policy_input = np.zeros([1, cfg.env.num_observations + 3], dtype=np.float32)

        policy_input = obs
        # print("policy_input: ", policy_input)
        # left_hip_joint left_knee_joint fl_wheel_joint rl_wheel_joint
        # right_hip_joint right_knee_joint fr_wheel_joint rr_wheel_joint

        action[:] = policy(torch.tensor(policy_input, dtype=torch.float32))[0].detach().numpy()
        action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
        # print("action:", action)

        target_q[[0, 1, 2, 4, 5, 6]] = action[[0, 1, 2, 4, 5, 6]] * cfg.control.action_scale_pos
        target_dq[[3, 7]] = action[[3, 7]] * cfg.control.action_scale_vel

        # Generate PD control
        tau = pd_control(target_q,default_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)  # Calc torques
        # print("wheel_vel: " , q[[2, 5]])
        # tau[[3, 7]] = (1 - cmd.mode) * tau[[3, 7]]
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        # print("tau:", tau)

        data.ctrl = tau
        # print("tau: ", tau)

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.") # 创建命令行参数解析器
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.") # 指定参数类型为字符串,此参数必须提供,帮助信息说明参数用途
    parser.add_argument("--terrain", action="store_true", help="terrain or plane") # 如果命令行中包含此参数,则值为True,否则为False,帮助信息说明地形
    args = parser.parse_args() # 解析命令行传入的参数,并将结果存储在args对象中

    class Sim2simCfg(L5A_STEP_Cfg): # 定义仿真配置类,继承自基础配置

        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/l2c/mjcf/y1a-terrain.xml"
            else:
                mujoco_model_path = f"{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/l5a/xml/l5aurdf20260209.xml"
            print(mujoco_model_path)
            sim_duration = 200.0 # 单次仿真持续时间为10s
            dt = 0.005 # 仿真步长为0.005s
            decimation = 2 # 设置控制频率与仿真频率的比值为10,即每隔10步执行一次控制策略

        class robot_config:
            # kps = np.array([20,20, 30, 0, 20,20, 30, 0], dtype=np.double)
            # kds = np.array([2,2, 3, 2, 2,2, 3, 2], dtype=np.double)
            kps = np.array([40, 40, 80, 0, 40, 40, 80, 0], dtype=np.double)
            kds = np.array([2, 2, 2, 1.5, 2, 2, 2, 1.5], dtype=np.double)
            # tau_limit = np.array([300, 300, 60, 60, 300, 300, 60, 60], dtype=np.double)
            tau_limit = np.array([745,745, 460, 400, 745,745, 460, 400], dtype=np.double)
            # tau_limit = 800.0 * np.ones(8, dtype=np.double)  # 力矩限制

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
