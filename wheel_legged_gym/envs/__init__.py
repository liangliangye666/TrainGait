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

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
# from .wheel_legged.wheel_legged_config import WheelLeggedCfg, WheelLeggedCfgPPO
# from .wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
# from .wheel_legged_vmc.wheel_legged_vmc_config import (
#     WheelLeggedVMCCfg,
#     WheelLeggedVMCCfgPPO,
# )
# from .wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
#     WheelLeggedVMCFlatCfg,
#     WheelLeggedVMCFlatCfgPPO,
# )

# # from .y1a_2wheel_vmc.y1a_2wheel_vmc import Y1A_2WHEEL_VMC
# # from .y1a_2wheel_vmc.y1a_2wheel_vmc_config import (
# #     Y1A_2WHEEL_VMCCfg,
# #     Y1A_2WHEEL_VMCCfgPPO,
# # )

# from .l2c_2wheel.l2c_2wheel import L2C_2WHEEL
# from .l2c_2wheel.l2c_2wheel_config import (
#     L2C_2WHEEL_Cfg,
#     L2C_2WHEEL_CfgPPO,
# )

# # from .y4a_2wheel.y4a_2wheel_stand_up import Y4A_2WHEEL
# # from .y4a_2wheel.y4a_2wheel_config_stand_up import (Y4A_2WHEEL_Cfg, Y4A_2WHEEL_CfgPPO,)
# from .y4a_2wheel.y4a_2wheel_jump1 import Y4A_2WHEEL
# from .y4a_2wheel.y4a_2wheel_config_jump1 import (Y4A_2WHEEL_Cfg, Y4A_2WHEEL_CfgPPO,)

from .l5a_jump.l5a_jump import L5A_JUMP
from .l5a_jump.l5a_jump_config import (L5A_JUMP_Cfg, L5A_JUMP_CfgPPO,)
from .l5a_balance.l5a_balance import L5A_BALANCE
from .l5a_balance.l5a_balance_config import (L5A_BALANCE_Cfg, L5A_BALANCE_CfgPPO,)
from .l5a_step.l5a_step import L5A_STEP
from .l5a_step.l5a_step_config import (
    L5A_STEP_Cfg,
    L5A_STEP_CfgPPO,
)


import os

from wheel_legged_gym.utils.task_registry import task_registry

# task_registry.register(
#     "wheel_legged",
#     LeggedRobot,
#     WheelLeggedCfg(),
#     WheelLeggedCfgPPO(),
# )
# task_registry.register(
#     "wheel_legged_vmc",
#     LeggedRobotVMC,
#     WheelLeggedVMCCfg(),
#     WheelLeggedVMCCfgPPO(),
# )
# task_registry.register(
#     "wheel_legged_vmc_flat",
#     LeggedRobotVMC,
#     WheelLeggedVMCFlatCfg(),
#     WheelLeggedVMCFlatCfgPPO(),
# )
# # task_registry.register(
# #     "y1a_2wheel_vmc",
# #     Y1A_2WHEEL_VMC,
# #     Y1A_2WHEEL_VMCCfg(),
# #     Y1A_2WHEEL_VMCCfgPPO(),
# # )
# task_registry.register(
#     "l2c_2wheel",
#     L2C_2WHEEL,
#     L2C_2WHEEL_Cfg(),
#     L2C_2WHEEL_CfgPPO(),
# )
# task_registry.register(
#     "y4a_2wheel",
#     Y4A_2WHEEL,
#     Y4A_2WHEEL_Cfg(),
#     Y4A_2WHEEL_CfgPPO(),
#     )
task_registry.register(
    "l5a_jump",
    L5A_JUMP,
    L5A_JUMP_Cfg(),
    L5A_JUMP_CfgPPO(),
    )
task_registry.register(
    "l5a_balance",
    L5A_BALANCE,
    L5A_BALANCE_Cfg(),
    L5A_BALANCE_CfgPPO(),
    )
task_registry.register(
    "l5a_step",
    L5A_STEP,
    L5A_STEP_Cfg(),
    L5A_STEP_CfgPPO(),
)
