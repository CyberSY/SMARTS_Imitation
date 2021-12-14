import abc

from rlkit.env_creators import MujocoEnv, ParticleEnv, MpeEnv, GymEnv, SmartsEnv, MASmartsEnv
from rlkit.envs.wrappers import ProxyEnv
from rlkit.envs.vecenvs import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
]


def get_env(env_specs):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """
    env_creator = env_specs["env_creator"]

    if env_creator == "mujoco":
        env_class = MujocoEnv
    elif env_creator == "particle":
        env_class = ParticleEnv
    elif env_creator == "mpe":
        env_class = MpeEnv
    elif env_creator == "gym":
        env_class = GymEnv
    elif env_creator == "smarts":
        env_class = SmartsEnv
    elif env_creator == "masmarts":
        env_class = MASmartsEnv
    else:
        raise NotImplementedError

    env = env_class(**env_specs)

    return env


def get_envs(
    env_specs,
    env_wrapper=None,
    vehicle_ids_list=None,
    env_num=1,
    wait_num=None,
    auto_reset=False,
    seed=None,
    **kwargs,
):
    """
    env_specs:
        env_name: 'halfcheetah'
        env_kwargs: {} # kwargs to pass to the env constructor call
    """

    if env_wrapper is None:
        env_wrapper = ProxyEnv

    env_creator = env_specs["env_creator"]

    if env_creator == "mujoco":
        env_class = MujocoEnv
    elif env_creator == "particle":
        env_class = ParticleEnv
    elif env_creator == "mpe":
        env_class = MpeEnv
    elif env_creator == "gym":
        env_class = GymEnv
    elif env_creator == "smarts":
        env_class = SmartsEnv
    elif env_creator == "masmarts":
        env_class = MASmartsEnv
    else:
        raise NotImplementedError

    if env_num == 1:
        print("\n WARNING: Single environment detected, wrap to DummyVectorEnv.")
        envs = DummyVectorEnv(
            [
                lambda i=i: env_wrapper(env_class(vehicle_ids=vehicle_ids_list[i], **env_specs))
                for i in range(env_num)
            ],
            auto_reset=auto_reset,
            **kwargs,
        )

    else:
        envs = SubprocVectorEnv(
            [
                lambda i=i: env_wrapper(env_class(vehicle_ids=vehicle_ids_list[i], **env_specs))
                for i in range(env_num)
            ],
            wait_num=wait_num,
            auto_reset=auto_reset,
            **kwargs,
        )

    envs.seed(seed)
    return envs


class EnvFactory(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __get__(self, task_params):
        """
        Implements returning and environment corresponding to given task params
        """
        pass

    @abc.abstractmethod
    def get_task_identifier(self, task_params):
        """
        Returns a hashable description of task params so it can be used
        as dictionary keys etc.
        """
        pass

    def task_params_to_obs_task_params(self, task_params):
        """
        Sometimes this may be needed. For example if we are training a
        multitask RL algorithm and want to give it the task params as
        part of the state.
        """
        raise NotImplementedError()
