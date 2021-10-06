import gym
import smarts_imitation

from rlkit.env_creators.base_env import BaseEnv


class SmartsEnv(BaseEnv):
    """A wrapper for gym Mujoco environments to fit in multi-agent apis."""

    def __init__(self, **configs):
        super().__init__(**configs)

        # create underlying smarts simulator
        scenario_name = configs["scenario_name"]
        env_kwargs = configs["env_kwargs"]
        if scenario_name == "interaction_dataset":
            self._env = gym.make("SMARTS-Imitation-v0", **env_kwargs)
        elif scenario_name == "ngsim":
            self._env = gym.make("SMARTS-Imitation-v1", **env_kwargs)
        else:
            raise NotImplementedError

        self._default_agent_name = "agent_0"
        self.agent_ids = [self._default_agent_name]
        self.n_agents = len(self.agent_ids)
        self.observation_space_n = {
            self._default_agent_name: self._env.observation_space
        }
        self.action_space_n = {self._default_agent_name: self._env.action_space}

    def seed(self, seed):
        return self._env.seed(seed)

    def reset(self):
        return {self._default_agent_name: self._env.reset()}

    def step(self, action_n):
        action = action_n[self._default_agent_name]
        next_obs, rew, done, info = self._env.step(action)
        next_obs_n = {self._default_agent_name: next_obs}
        rew_n = {self._default_agent_name: rew}
        done_n = {self._default_agent_name: done}
        info_n = {self._default_agent_name: info}
        return next_obs_n, rew_n, done_n, info_n

    def render(self, **kwargs):
        return self._env.render(**kwargs)
