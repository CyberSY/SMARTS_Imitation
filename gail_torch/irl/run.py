import os
import torch
import pickle
import gym
import yaml
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from gail_torch.utils import Memory, make_env
from gail_torch.policy import PPOPolicy, DiscretePPOPolicy, Discriminator
from gail_torch.sampler import Sampler


def do_train(config):
    """
    init the env, agent and train the agents
    """
    env = make_env(config["env_specs"]["env_name"])
    print("=============================")
    print("=1 env {} is right ...".format(config["env_specs"]["env_name"]))
    print("=============================")

    expert_memory = Memory(
        config["exp_params"]["memory_size"], action_space=env.action_space
    )
    with open(config["basic"]["expert_path"], "rb") as f:
        expert_data_dict = pickle.load(f)
    expert_memory.load_expert(expert_data_dict)
    print("=2 Loading dataset with {} samples ...".format(len(expert_memory)))
    print("=============================")

    if config["others"]["use_tensorboard_log"]:
        log_dir = os.path.join(
            config["basic"]["log_dir"],
            config["basic"]["exp_name"],
            datetime.now().strftime("%Y.%m.%d.%H.%M.%S"),
        )
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    discriminator = Discriminator(
        observation_space=env.observation_space,
        action_space=env.action_space,
        writer=writer,
        expert_memory=expert_memory,
        state_only=config["exp_params"]["state_only"],
        device=config["basic"]["device"],
    ).to(config["basic"]["device"])

    if isinstance(env.action_space, gym.spaces.Discrete):
        print("=2 using discrete ppo policy ...")
        policy = DiscretePPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            discriminator=discriminator,
            writer=writer,
            device=config["basic"]["device"],
        ).to(config["basic"]["device"])
    elif isinstance(env.action_space, gym.spaces.Box):
        print("=2 using continuous ppo policy ...")
        policy = PPOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            discriminator=discriminator,
            activation="tanh",
            writer=writer,
            device=config["basic"]["device"],
        ).to(config["basic"]["device"])
    else:
        raise NotImplementedError
    print("=============================")

    model_save_dir = os.path.join(
        config["basic"]["model_save_dir"], config["basic"]["exp_name"]
    )
    if config["others"]["model_save_freq"] and not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    memory = Memory(config["exp_params"]["memory_size"], action_space=env.action_space)

    print("=3 starting iterations ...")
    print("=============================")
    sampler = Sampler(
        env,
        policy,
        memory,
        config["basic"]["device"],
        writer,
        num_threads=config["exp_params"]["num_threads"],
    )

    for update_cnt in range(config["exp_params"]["num_updates"]):

        if (
            config["others"]["model_save_freq"] > 0
            and update_cnt > 0
            and update_cnt % config["others"]["model_save_freq"] == 0
        ):
            torch.save(
                (policy, discriminator),
                os.path.join(model_save_dir, f"model_{update_cnt}.pkl"),
            )
            print("=model saved at episode {}".format(update_cnt))

        memory, _ = sampler.collect_samples(config["exp_params"]["update_timestep"])
        policy.update(memory)
        discriminator.update(memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GAIfO for single agent environments"
    )
    parser.add_argument(
        "--exp_config",
        "-e",
        help="path to the experiment configuration yaml file",
        default="./config/gail.yaml",
    )
    parser.add_argument(
        "--use_tensorboard_log",
        "-t",
        action="store_true",
    )
    args = parser.parse_args()

    with open(args.exp_config, "rb") as f:
        config = yaml.load(f)

    config["others"]["use_tensorboard_log"] = args.use_tensorboard_log
    if args.use_tensorboard_log:
        print("\nUSING TENSORBOARD LOG\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING GPU\n")
    else:
        device = torch.device("cpu")
    config["basic"]["device"] = device

    if config["others"]["model_save_freq"] and not os.path.exists(
        config["basic"]["model_save_dir"]
    ):
        os.makedirs(config["basic"]["model_save_dir"])
    if config["others"]["use_tensorboard_log"] and not os.path.exists(
        config["basic"]["tb_log_dir"]
    ):
        os.makedirs(config["basic"]["tb_log_dir"])

    print(config, "\n")

    do_train(config)
