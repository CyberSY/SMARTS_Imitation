import numpy as np
import gym

from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario
from envision.client import Client as Envision

from smarts_imitation.utils import agent


class SMARTSImitation(gym.Env):
    def __init__(self, scenarios):
        super(SMARTSImitation, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.agent_spec = agent.get_agent_spec(self.obs_stacked_size)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,))
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
            # envision=Envision(),
        )

    def seed(self, seed):
        np.random.seed(seed)

    def _convert_obs(self, observations):
        observations[self.agent_id] = self.agent_spec.observation_adapter(
            observations[self.agent_id]
        )
        ego_state = []
        other_info = []
        for feat in observations[self.agent_id]:
            if feat in ["ego_pos", "speed", "heading"]:
                ego_state.append(observations[self.agent_id][feat])
            else:
                other_info.append(observations[self.agent_id][feat])
        ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
        other_info = np.concatenate(other_info, axis=1).reshape(-1)
        full_obs = np.concatenate((ego_state, other_info))
        return full_obs

    def step(self, action):
        observations, rewards, dones, infos = self.smarts.step(
            {self.agent_id: self.agent_spec.action_adapter(action)}
        )
        full_obs = self._convert_obs(observations)
        return (
            full_obs,
            rewards[self.agent_id],
            dones[self.agent_id],
            {},
        )

    def reset(self):
        if self.agent_itr >= len(self.agent_ids):
            self._next_scenario()
        self.agent_id = self.agent_ids[self.agent_itr]
        agent_mission = self.agent_missions[self.agent_id]
        self.scenario.set_ego_missions({self.agent_id: agent_mission})
        self.smarts.switch_ego_agent({self.agent_id: self.agent_spec.interface})
        observations = self.smarts.reset(self.scenario)
        full_obs = self._convert_obs(observations)
        self.agent_itr += 1
        return full_obs

    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.agent_missions = self.scenario.discover_missions_of_traffic_histories()
        self.agent_ids = list(self.agent_missions.keys())
        np.random.shuffle(self.agent_ids)
        self.agent_itr = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()
