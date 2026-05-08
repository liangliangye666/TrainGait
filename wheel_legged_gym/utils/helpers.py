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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {
            "name": "--task", # 任务名称
            "type": str,
            "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume", # 指示是否从检查点恢复训练
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name", # 区分不同系列的实验
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name", # 指定单次训练运行的标识符
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run", # 自动加载最新运行
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint", # 选择特定训练迭代次数的模型
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless", # 禁用图形界面减少资源占用
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod", # 启用Horovod框架进行多GPU并行训练,显著加速大规模训练任务
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device", # 默认使用第一个CUDA GPU,灵活支持GPU/多GPU配置
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs", # 控制并行环境实例数量,值越大训练越快,但需更多显存
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed", # 设置随机数生成器种子,确保实验可复现性
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations", # 最大迭代次数,设置训练停止条件,防止无限期训练
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {"name": "--exptid", "type": str, "default": "", "help": "exptid"}, # 自定义实验标识符,可用于第三方系统集成
    ]
    # parse arguments,解析变量
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    # if hasattr(actor_critic, "memory_a"):
    #     # assumes LSTM: TODO add GRU
    #     exporter = PolicyExporterLSTM(actor_critic)
    #     exporter.export(path)
    # else:
    #     os.makedirs(path, exist_ok=True)
    #     path = os.path.join(path, "policy_1.pt")
    #     model = copy.deepcopy(actor_critic.actor).to("cpu")
    #     traced_script_module = torch.jit.script(model)
    #     traced_script_module.save(path)
    os.makedirs(path, exist_ok=True)

    # Export actor model
    actor_path = os.path.join(path, "actor.pt")
    actor_model = copy.deepcopy(actor_critic.actor).cpu()
    torch.save(actor_model, actor_path)

    # Export encoder if sequence model
    if getattr(actor_critic, "is_sequence", False):
        encoder_path = os.path.join(path, "encoder.pt")
        encoder_model = copy.deepcopy(actor_critic.encoder).cpu()
        torch.save(encoder_model, encoder_path)

    # Export memory model if LSTM
    if hasattr(actor_critic, "memory_a"):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        # Export scripted model
        policy_path = os.path.join(path, "policy_1.pt")
        traced_script_module = torch.jit.script(actor_model)
        traced_script_module.save(policy_path)

        encoder_path = os.path.join(path, "estimator.pt")
        traced_script_module = torch.jit.script(encoder_model)
        traced_script_module.save(encoder_path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(
            f"hidden_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )
        self.register_buffer(
            f"cell_state",
            torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size),
        )

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
