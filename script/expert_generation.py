from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from envision.client import Client as Envision
from smarts.core.scenario import Scenario
import numpy as np
import pickle
import argparse

from smarts_imitation.utils import adapter
from smarts_imitation.utils import agent


def main(scenarios):
    """ Collect expert observations.

    Each input scenario is associated with some trajectory files. These trajectories
    will be replayed on SMARTS and observations of each vehicle will be collected and
    stored in a dict.

    Args:
        scenarios: A string of the path to scenarios to be processed.

    Returns:
        A dict in the form of {"observation": [...], "next_observation": [...], "done": [...]}.
    """

    observation_stack_size = 1

    agent_spec = agent.get_agent_spec(observation_stack_size)
    observation_adapter = adapter.get_observation_adapter(observation_stack_size)

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
        # envision=Envision(),
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenarios],
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    expert_obs = []
    expert_obs_next = []
    expert_terminals = []
    cars_obs = {}
    cars_obs_next = {}
    cars_terminals = {}

    prev_vehicles = set()
    done_vehicles = set()
    while True:
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles

        if len(current_vehicles) == 0:
            break

        smarts.attach_sensors_to_vehicles(
            agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        obs, _, _, dones = smarts.observe_from(smarts.vehicle_index.social_vehicle_ids())

        for v in done_vehicles:
            cars_terminals[f"Agent-{v}"][-1] = True
            print(f"Agent-{v} Ended")

        for k in obs.keys():
            obs[k] = observation_adapter(obs[k])

        cars = obs.keys()

        for car in cars:
            full_obs = []
            ego_state = []
            other_info = []
            for feat in obs[car]:
                if feat in ["ego_pos", "speed", "heading"]:
                    ego_state.append(obs[car][feat])
                else:
                    other_info.append(obs[car][feat])
            ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
            other_info = np.concatenate(other_info, axis=1).reshape(-1)
            full_obs = np.concatenate((ego_state, other_info))

            if cars_obs.__contains__(car):
                cars_obs[car].append(full_obs)
                cars_terminals[car].append(dones[car])
            else:
                cars_obs[car] = [full_obs]
                cars_terminals[car] = [dones[car]]

    for car in cars_obs:
        cars_obs[car] = np.array(cars_obs[car])
        cars_obs_next[car] = cars_obs[car][1:, :]
        cars_obs[car] = cars_obs[car][:-1, :]
        cars_terminals[car] = np.array(cars_terminals[car][:-1])
        expert_obs.append(cars_obs[car])
        expert_obs_next.append(cars_obs_next[car])
        expert_terminals.append(cars_terminals[car])

    with open("expert.pkl", "wb") as f:
        pickle.dump({"observation": expert_obs, "next_observation": expert_obs_next, "done": expert_terminals}, f)

    smarts.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        help="A list of scenarios. Each element can be either the scenario to run "
        "(see scenarios/ for some samples you can use) OR a directory of scenarios "
        "to sample from.",
        type=str,
        default="/NAS2020/Workspaces/DRLGroup/zbzhu/SMARTS_Imitation/smarts-imitation/interaction_dataset/scenarios/interaction_dataset_merging",
    )
    args = parser.parse_args()
    main(
        scenarios=args.scenarios,
    )
