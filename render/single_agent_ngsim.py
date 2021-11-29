import numpy as np
import gym
import sys
import pickle
from dataclasses import replace

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider

from smarts_imitation.utils import agent
from envision.client import Client as Envision

from dataclasses import replace

sys.path.append(r"/NAS2020/Workspaces/DRLGroup/syzhang/SMARTS_Imitation/ILSwiss")
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.sac.sac_alpha import (
    SoftActorCritic,
) 
from rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import MLPDisc

import joblib

class SMARTSImitation(gym.Env):
    def __init__(self, scenarios, action_range):
        super(SMARTSImitation, self).__init__()
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

        envision_client = Envision(
                endpoint=None,
                sim_name="NGSIM_TEST",
                output_dir=None,
                headless=None,
            )
        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def _convert_obs(self, raw_observations):
        observation = self.agent_spec.observation_adapter(
            raw_observations[self.vehicle_id]
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
        full_obs = np.concatenate((ego_state, other_info))
        return full_obs

    def step(self, action):
        action = np.clip(action, -1, 1)
        # Transform the normalized action back to the original range
        # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
        # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
        action = (self._action_range[1] - self._action_range[0]) * (
            action + 1
        ) / 2 + self._action_range[0]

        raw_observations, rewards, dones, _ = self.smarts.step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)}
        )
        full_obs = self._convert_obs(raw_observations)

        info = {}
        info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
        info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0

        return (
            full_obs,
            rewards[self.vehicle_id],
            dones[self.vehicle_id],
            info,
        )

    def reset(self):
        if self.vehicle_itr >= len(self.vehicle_ids):
            self._next_scenario()

        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts.switch_ego_agents({self.vehicle_id: self.agent_spec.interface})

        observations = self.smarts.reset(self.scenario)
        full_obs = self._convert_obs(observations)
        self.vehicle_itr += 1
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

env = SMARTSImitation(scenarios = ["./ngsim"],action_range=np.array([[-2.5, -0.2],[2.5, 0.2]]))
obs_space = env.observation_space
act_space = env.action_space

variant = {  "disc_num_blocks": 2,
    "disc_hid_dim": 128,
    "disc_hid_act": "tanh",
    "disc_use_bn": False,
    "disc_clamp_magnitude": 10.0,
    "policy_net_size": 256,
    "policy_num_hidden_layers": 2}
obs_dim = obs_space.shape[0]
action_dim = act_space.shape[0]
net_size = variant["policy_net_size"]
num_hidden = variant["policy_num_hidden_layers"]
policy = ReparamTanhMultivariateGaussianPolicy(
    hidden_sizes=num_hidden * [net_size],
    obs_dim=obs_dim,
    action_dim=action_dim,
)

log_path = "../logs/gail-smarts-ngsim--terminal--gp-4.0--rs-2.0--trajnum--1/gail_smarts_ngsim--terminal--gp-4.0--rs-2.0--trajnum--1_2021_10_11_11_27_19_0000--s-723894/best.pkl"
# log_path = "../logs/magail-smarts-ngsim--terminal--gp-4.0--rs-2.0--trajnum--1/magail_smarts_ngsim--terminal--gp-4.0--rs-2.0--trajnum--1_2021_11_18_07_08_07_0000--s-723894/best.pkl"
with open(log_path, "rb") as f:
        param = joblib.load(f)
policy.load_state_dict(param["policy_0"]["policy"].state_dict())
observations = env.reset()
dones = {}
print(observations)


for step in range(1000):
    agent_actions = policy.get_actions(np.array([observations]))[0]
    if(dones):
        break
    observations, rew, dones, _ = env.step(agent_actions)

env.destroy()