import yaml
import argparse
import numpy as np
import os
import sys
import inspect
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs
from rlkit.envs.wrappers import NormalizedBoxActEnv, ProxyEnv

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import OptionReparamTanhMultivariateGaussianPolicy
from rlkit.data_management.option_replay_buffer import OptionEnvReplayBuffer
from rlkit.torch.algorithms.option.option_sac import OptionSAC
from rlkit.torch.algorithms.option.torch_option_rl_algorithm import (
    TorchOptionRLAlgorithm,
)
from rlkit.samplers import OptionPathSampler


def experiment(variant):
    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    algorithm_name = variant["algorithm_name"]
    option_dim = variant["option_dim"]

    print("\n\nEnv: {}: {}".format(env_specs["env_creator"], env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space_n))
    print("Act Space: {}\n\n".format(env.action_space_n))

    obs_space_n = env.observation_space_n
    act_space_n = env.action_space_n

    policy_mapping_dict = dict(
        zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
    )

    policy_trainer_n = {}
    policy_n = {}

    for agent_id in env.agent_id:
        policy_id = policy_mapping_dict.get(agent_id)
        if policy_id not in policy_trainer_n:
            print(f"Create {policy_id} for {agent_id} ...")
            obs_space = obs_space_n[agent_id]
            act_space = act_space_n[agent_id]
            assert isinstance(obs_space, gym.spaces.Box)
            assert isinstance(act_space, gym.spaces.Box)
            assert len(obs_space.shape) == 1
            assert len(act_space.shape) == 1

            obs_dim = obs_space_n[agent_id].shape[0]
            action_dim = act_space_n[agent_id].shape[0]

            net_size = variant["net_size"]
            num_hidden = variant["num_hidden_layers"]
            vf = FlattenMlp(
                hidden_sizes=num_hidden * [net_size],
                input_size=obs_dim,
                output_size=option_dim,
            )
            policy = OptionReparamTanhMultivariateGaussianPolicy(
                hidden_sizes=[num_hidden * [net_size], num_hidden * [net_size]],
                obs_dim=obs_dim,
                action_dim=action_dim,
                option_dim=option_dim,
            )

            trainer = OptionSAC(
                policy=policy,
                vf=vf,
                batch_size=variant["rl_alg_params"]["batch_size"],
                **variant["option_sac_params"],
            )
            policy_trainer_n[policy_id] = trainer
            policy_n[policy_id] = policy
        else:
            print(f"Use existing {policy_id} for {agent_id} ...")

    env_wrapper = ProxyEnv  # Identical wrapper
    for act_space in act_space_n.values():
        if isinstance(act_space, gym.spaces.Box):
            env_wrapper = NormalizedBoxActEnv
            break

    env = env_wrapper(env)
    training_env = get_envs(env_specs, env_wrapper)
    training_env.seed(env_specs["training_env_seed"])

    replay_buffer = OptionEnvReplayBuffer(
        variant["rl_alg_params"]["replay_buffer_size"],
        env,
        option_dim,
        random_seed=np.random.randint(10000),
    )

    algorithm = TorchOptionRLAlgorithm(
        trainer=trainer,
        trainer_name=algorithm_name,
        env=env,
        training_env=training_env,
        eval_sampler_func=OptionPathSampler,
        replay_buffer=replay_buffer,
        exploration_policy=policy,
        **variant["rl_alg_params"],
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    # make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    if (exp_specs["using_gpus"] > 0) and (torch.cuda.is_available()):
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)
