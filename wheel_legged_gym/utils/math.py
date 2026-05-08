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

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import *
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    """
    将向量 vec 绕着四元数 quat 的 yaw 轴旋转

    参数:
    - quat: 四元数张量，形状为 (N, 4)，其中 N 是四元数的数量
    - vec: 向量张量，形状为 (N, 3)，其中 N 是向量的数量

    返回:
    - 旋转后的向量张量，形状为 (N, 3)
    """
    # 克隆并重塑四元数张量，使其形状为 (N, 4)
    quat_yaw = quat.clone().view(-1, 4) # view(N,M)->将张量重塑为N,M形式
    # 将四元数的 x 和 y 分量设置为 0，只保留 z 和 w 分量
    quat_yaw[:, :2] = 0.0
    # 归一化四元数
    quat_yaw = normalize(quat_yaw)
    # 应用四元数旋转到向量
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    """
    将角度值限制在 [-π, π] 范围内，确保角度值不会超出这个范围

    参数:
    - angles: 角度张量，形状为 (N,)，其中 N 是角度的数量

    返回:
    - 处理后的角度张量，形状为 (N,)
    """
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    """
    在指定范围内生成平方根浮点数张量

    参数:
    - lower: 张量元素的最小值
    - upper: 张量元素的最大值
    - shape: 张量的形状
    - device: 张量存储的设备

    返回:
    - 包含生成浮点数的张量
    """
    r = 2 * torch.rand(*shape, device=device) - 1  # 在 -1 和 1 之间生成随机浮点数
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))  # 对于负数，取其相反数的平方根
    r = (r + 1.0) / 2.0  # 将 [-1, 1] 范围缩放到 [0, 1]
    return (upper - lower) * r + lower  # 根据指定范围，缩放并加上最小值


# 计算两个七维位姿之间的相对位姿
def compute_relative_pose(pose1, pose2):
    """
    计算两个七维位姿之间的相对位姿。
    计算pose2相对于pose1的相对位姿。

    参数:
    - pose1: tuple, (position1, quaternion1)
    - pose2: tuple, (position2, quaternion2)

    输出:
    - relative_position: 3D position difference
    - relative_quaternion: quaternion representing the relative rotation
    """
    # 提取位置和四元数
    p1, q1 = pose1
    p2, q2 = pose2

    # 1. 计算位置差
    position_diff = p2 - p1

    # 2. 计算相对旋转四元数
    q1_inv = quat_conjugate(q1)  # 计算q1的逆
    relative_rotation = quat_mul(q1_inv, q2)

    # 3. 可选：归一化四元数
    relative_rotation = quat_unit(relative_rotation)

    return position_diff, relative_rotation


def get_euler_zyx_tensor(quat):

    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (np.pi / 2), torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Stack results
    euler_zyx = torch.stack((roll, pitch, yaw), dim=1)
    return euler_zyx

def quat_from_euler_zyx(euler_zyx):
    roll,pitch,yaw = euler_zyx[:,0],euler_zyx[:,1],euler_zyx[:,2]
    
    cr = torch.cos(roll/2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((x, y, z, w), dim=-1)
    
    