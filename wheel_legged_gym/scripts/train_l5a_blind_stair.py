import isaacgym

from wheel_legged_gym.envs import *
from wheel_legged_gym.envs.l5a_blind_stair.l5a_blind_stair import L5A_BLIND_STAIR
from wheel_legged_gym.envs.l5a_blind_stair.l5a_blind_stair_config import (
    L5A_BLIND_STAIR_Cfg,
    L5A_BLIND_STAIR_CfgPPO,
)
from wheel_legged_gym.utils import get_args, task_registry


def register_blind_stair_task():
    task_registry.register(
        "l5a_blind_stair",
        L5A_BLIND_STAIR,
        L5A_BLIND_STAIR_Cfg(),
        L5A_BLIND_STAIR_CfgPPO(),
    )


def train(args):
    register_blind_stair_task()
    args.task = "l5a_blind_stair"
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
