import numpy as np
import gym
from dataclasses import replace

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider

from smarts_imitation.utils import agent


class MASMARTSImitation(gym.Env):
    def __init__(self, scenarios, action_range):
        super(MASMARTSImitation, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.agent_spec = agent.get_agent_spec(self.obs_stacked_size)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float64
        )

        assert (
            action_range.shape == (2, 2) and (action_range[1] >= action_range[0]).all()
        ), action_range
        self._action_range = action_range  # np.array([[low], [high]])

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=None,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def _convert_obs(self, raw_observations):
        full_obs = {}
        for vehicle_id in self.vehicle_ids:
            observation = self.agent_spec.observation_adapter(
                raw_observations[vehicle_id]
            )
            ego_state = []
            other_info = []
            for feat in observation:
                if feat in ["ego_pos", "speed", "heading"]:
                    ego_state.append(observation[feat])
                else:
                    other_info.append(observation[feat])
            ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
            other_info = np.concatenate(other_info, axis=1).reshape(-1)
            vihicle_full_obs = np.concatenate((ego_state, other_info))
            full_obs[vehicle_id] = vihicle_full_obs
        return full_obs

    def step(self, action):
        for vihicle_id in self.vihicle_ids:
            vihicle_action = action[vihicle_id]
            vihicle_action = np.clip(vihicle_action, -1, 1)
            # Transform the normalized action back to the original range
            # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
            # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
            vihicle_action = (self._action_range[1] - self._action_range[0]) * (
                action + 1
            ) / 2 + self._action_range[0]
            action[vihicle_id] = vihicle_action

        raw_observations, rewards, dones, _ = self.smarts.step(action)
        full_obs = self._convert_obs(raw_observations)

        info = {}
#        info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
#        info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0

        return (
            full_obs,
            rewards[self.vehicle_id],
            dones[self.vehicle_id],
            info,
        )

    def reset(self):
        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        
        vihicle_interfaces = {}
        for vehicle_id in self.vehicle_ids:
            vihicle_interfaces[vehicle_id] = self.agent_spec.interface
            if(not history_start_time or history_start_time > self.vehicle_missions[vehicle_id].start_time):
                history_start_time = self.vehicle_missions[vehicle_id].start_time

        traffic_history_provider.start_time = history_start_time

#       modified_mission = replace(vehicle_mission, start_time=0.0)
        ego_missions = {}
        for vehicle_id in self.vehicle_ids:
            ego_missions[vehicle_id] = replace(self.vehicle_missions[vehicle_id],self.vehicle_missions[vehicle_id].start_time
                                                - history_start_time)
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(vihicle_interfaces)

        observations = self.smarts.reset(self.scenario)
        full_obs = self._convert_obs(observations)
        return full_obs

    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_ids = list(self.vehicle_missions.keys())
        np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()

