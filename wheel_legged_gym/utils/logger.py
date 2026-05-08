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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 4
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="real")
        if log["est_lin_vel_x"]:
            a.plot(time, log["est_lin_vel_x"], linestyle="--", label="est")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()

        # plot base vel yaw
        a = axs[0, 1]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="real")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw")
        a.legend()

        # plot base height
        a = axs[0, 2]
        if log["base_height"]:
            a.plot(time, log["base_height"], label="real")
        if log["command_height"]:
            a.plot(time, log["command_height"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base height [m]", title="Base Height")
        a.legend()

        # plot joint targets and real positions
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="real")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], linestyle="--", label="target")
        a.set(xlabel="time [s]", ylabel="Position [rad]", title="Joint Position")
        a.legend()

        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="real")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        a.legend()

        a = axs[1, 2]
        if log["wheel_vel"]:
            a.plot(time, log["wheel_vel"], label="real")
        if log["wheel_vel_target"]:
            a.plot(time, log["wheel_vel_target"], linestyle="--", label="target")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Wheel Velocity")
        a.legend()

        a = axs[2, 0]
        if log["command_mode"]:
            a.plot(time, log["command_mode"], label="commanded")
        a.set(xlabel="time [s]", ylabel="robot mode", title="Robot Mode")
        a.legend()

        # plot joint torque
        a = axs[2, 1]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()

        # plot dof torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        a.legend()

        a = axs[3, 0]
        if log["base_roll_obs"]:
            a.plot(time, log["base_roll_obs"], label="roll real")
        if log["base_roll_est"]:
            a.plot(time, log["base_roll_est"], linestyle="--", label="roll est")
        a.set(xlabel="time [s]", ylabel="euler roll [rad]", title="base euler roll")
        a.legend()

        a = axs[3, 1]
        if log["base_pitch_obs"]:
            a.plot(time, log["base_pitch_obs"], label="pitch real")
        if log["base_pitch_est"]:
            a.plot(time, log["base_pitch_est"], linestyle="--", label="pitch est")
        a.set(xlabel="time [s]", ylabel="euler pitch [rad]", title="base euler pitch")
        a.legend()

        a = axs[3, 2]
        if log["base_yaw_obs"]:
            a.plot(time, log["base_yaw_obs"], label="yaw real")
        if log["base_yaw_est"]:
            a.plot(time, log["base_yaw_est"], linestyle="--", label="yaw est")
        a.set(xlabel="time [s]", ylabel="euler yaw [rad]", title="base euler yaw")
        a.legend()

        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
