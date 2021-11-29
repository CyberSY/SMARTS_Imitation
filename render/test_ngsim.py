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

class MASMARTSImitation(gym.Env):
    def __init__(self, scenarios, action_range, agent_number):
        super(MASMARTSImitation, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.n_agents = agent_number
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
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

    def change_agents_n(self, agent_number):
        self.n_agents = agent_number
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]

    def _convert_obs(self, raw_observations):
        full_obs_n = {}
        for agent_id in raw_observations.keys():
            # if agent_id not in raw_observations.keys():
            #     continue
            observation = self.agent_spec.observation_adapter(
                raw_observations[agent_id]
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
            full_obs_n[agent_id] = full_obs
        return full_obs_n

    def step(self, action):
        action_n = {}
        for agent_id in action.keys():
            # if agent_id not in action.keys():
            #     continue
            if(self.dones[agent_id]):
                continue
            agent_action = action[agent_id]
            agent_action = np.clip(agent_action, -1, 1)
            # Transform the normalized action back to the original range
            # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
            # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
            agent_action = (self._action_range[1] - self._action_range[0]) * (
                agent_action + 1
            ) / 2 + self._action_range[0]
            action_n[agent_id] = self.agent_spec.action_adapter(agent_action)
        raw_observations, rewards, dones, _ = self.smarts.step(action_n)
        self.dones = dones
        full_obs = self._convert_obs(raw_observations)
        info = {}
        for agent_id in full_obs.keys():
            # if agent_id not in full_obs.keys():
            #     continue
            info[agent_id] = {}
            info[agent_id]["reached_goal"] = raw_observations[agent_id].events.reached_goal
            info[agent_id]["collision"] = len(raw_observations[agent_id].events.collisions) > 0
            
        # info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
        # info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0

        return (
            full_obs,
            rewards,
            dones,
            info,
        )

    def reset(self):
        # self.change_agents_n(agent_number)
        # sample = self.scenario.traffic_history.random_overlapping_sample(
        #         self.veh_start_times, agent_number
        #     )
        if self.vehicle_itr + self.n_agents >= (self.n_agents_max - 1):
            self._next_scenario()
        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        self.vehicle_id = self.vehicle_ids[self.vehicle_itr:self.vehicle_itr + self.n_agents]
        # for veh_id in sample:
        #     self.vehicle_id.append(veh_id)
        # self.agentid_to_vehid.clear()
        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]
        
        agent_interfaces = {}
        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            agent_interfaces[agent_id] = self.agent_spec.interface
            if(history_start_time > self.vehicle_missions[vehicle].start_time):
                history_start_time = self.vehicle_missions[vehicle].start_time

        traffic_history_provider.start_time = history_start_time
#       modified_mission = replace(vehicle_mission, start_time=0.0)
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            ego_missions[agent_id] = replace(self.vehicle_missions[vehicle], start_time=self.vehicle_missions[vehicle].start_time - history_start_time)
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        observations = self.smarts.reset(self.scenario)
        full_obs = self._convert_obs(observations)
        self.dones = {}
        for agent_id in full_obs.keys():
            self.dones[agent_id] = False
        self.vehicle_itr += self.n_agents
        return full_obs

    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {
            v_id: mission.start_time for v_id, mission in self.vehicle_missions.items()
        }
        vlist = []
        for vehicle_id,start_time in self.veh_start_times.items():
            vlist.append((vehicle_id,start_time))
        dtype = [('id',int),('start_time',float)]
        vlist = np.array(vlist,dtype = dtype)
        vlist = np.sort(vlist,order = 'start_time')
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f'{vlist[id][0]}'
        # np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0
        self.n_agents_max = len(self.vehicle_ids)

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()
env = MASMARTSImitation(scenarios = ["./ngsim"],action_range=np.array([[-2.5, -0.2],[2.5, 0.2]]),agent_number = 5)
obs_space_n = env.observation_space
act_space_n = env.action_space

policy_mapping_dict = dict(
    zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
)

policy_trainer_n = {}
policy_n = {}
disc_model_n = {}
variant = {  "disc_num_blocks": 2,
    "disc_hid_dim": 128,
    "disc_hid_act": "tanh",
    "disc_use_bn": False,
    "disc_clamp_magnitude": 10.0,
    "policy_net_size": 256,
    "policy_num_hidden_layers": 2}
variant["adv_irl_params"] = {"state_only":True}
for agent_id in env.agent_ids:
    policy_id = policy_mapping_dict.get(agent_id)
    if policy_id not in policy_trainer_n:
        print(f"Create {policy_id} for {agent_id} ...")
        obs_space = obs_space_n
        act_space = act_space_n
        assert isinstance(obs_space, gym.spaces.Box)
        assert isinstance(act_space, gym.spaces.Box)
        assert len(obs_space.shape) == 1
        assert len(act_space.shape) == 1

        obs_dim = obs_space.shape[0]
        action_dim = act_space.shape[0]

        # build the policy models
        net_size = variant["policy_net_size"]
        num_hidden = variant["policy_num_hidden_layers"]
        qf1 = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        qf2 = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        vf = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim,
            output_size=1,
        )
        policy = ReparamTanhMultivariateGaussianPolicy(
            hidden_sizes=num_hidden * [net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        # build the discriminator model
        disc_model = MLPDisc(
            obs_dim + action_dim
            if not variant["adv_irl_params"]["state_only"]
            else 2 * obs_dim,
            num_layer_blocks=variant["disc_num_blocks"],
            hid_dim=variant["disc_hid_dim"],
            hid_act=variant["disc_hid_act"],
            use_bn=variant["disc_use_bn"],
            clamp_magnitude=variant["disc_clamp_magnitude"],
        )

        # set up the algorithm
        trainer = SoftActorCritic(
            policy=policy, qf1=qf1, qf2=qf2, vf=vf, action_space=env.action_space,
        )

        policy_trainer_n[policy_id] = trainer
        policy_n[policy_id] = policy
        disc_model_n[policy_id] = disc_model
    else:
        print(f"Use existing {policy_id} for {agent_id} ...")

log_path = "../logs/gail-smarts-ngsim--terminal--gp-4.0--rs-2.0--trajnum--1/gail_smarts_ngsim--terminal--gp-4.0--rs-2.0--trajnum--1_2021_10_11_11_27_19_0000--s-723894/best.pkl"
# log_path = "../logs/magail-smarts-ngsim--terminal--gp-4.0--rs-2.0--trajnum--1/magail_smarts_ngsim--terminal--gp-4.0--rs-2.0--trajnum--1_2021_11_18_07_08_07_0000--s-723894/best.pkl"
with open(log_path, "rb") as f:
        param = joblib.load(f)
policy_n["policy_0"].load_state_dict(param["policy_0"]["policy"].state_dict())
observations = env.reset()
dones = {}
print(observations)
for agent_id in observations.keys():
    dones[agent_id] = False

d = 0
for step in range(2000):
    agent_actions = {}
    for agent_id,observation in observations.items():
        if(dones[agent_id]):
            d += 1
            continue
        agent_actions[agent_id] = policy_n["policy_0"].get_actions(np.array([observation]))[0]
    if(d == 5):
        break
    observations, rew, dones, _ = env.step(agent_actions)

env.destroy()