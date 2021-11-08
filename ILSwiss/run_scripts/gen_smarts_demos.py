import yaml
import argparse
import os
import sys
import pickle
import inspect
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)


import numpy as np

from rlkit.data_management.path_builder import PathBuilder

from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.utils.math import radians_to_vec
from smarts_imitation.utils import adapter, agent


def observation_transform(raw_observations, observation_adapter):
    observations = {}
    for vehicle in raw_observations.keys():
        observations[vehicle] = observation_adapter(raw_observations[vehicle])
        ego_state = []
        other_info = []
        for feat in observations[vehicle]:
            if feat in ["ego_pos", "speed", "heading"]:
                ego_state.append(observations[vehicle][feat])
            else:
                other_info.append(observations[vehicle][feat])
        ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
        other_info = np.concatenate(other_info, axis=1).reshape(-1)
        observations[vehicle] = np.concatenate((ego_state, other_info))
    return observations


def calculate_actions(raw_observations, raw_next_observations, dt=0.1):
    actions = {}
    for car in raw_observations.keys():
        if car not in raw_next_observations.keys():
            continue
        car_next_state = raw_next_observations[car].ego_vehicle_state
        acceleration = car_next_state.linear_acceleration[:2].dot(
            radians_to_vec(car_next_state.heading)
        )
        angular_velocity = car_next_state.yaw_rate
        actions[car] = np.array([acceleration, angular_velocity])
    return actions


def sample_demos(
    scenarios,
    wrap_absorbing=False,
):
    observation_stack_size = 1
    agent_spec = agent.get_agent_spec(observation_stack_size)
    observation_adapter = adapter.get_observation_adapter(observation_stack_size)

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenarios],
        list([]),
    )

    prev_vehicles = set()
    done_vehicles = set()

    path_builders = {}
    demo_paths = []

    """ Reset environment. """
    smarts.reset(next(scenarios_iterator))
    smarts.step({})
    smarts.attach_sensors_to_vehicles(
        agent_spec, smarts.vehicle_index.social_vehicle_ids()
    )
    raw_observations, _, _, dones = smarts.observe_from(
        smarts.vehicle_index.social_vehicle_ids()
    )
    observations = observation_transform(raw_observations, observation_adapter)

    while True:
        """Step in the environment."""
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles

        if len(current_vehicles) == 0:
            break

        smarts.attach_sensors_to_vehicles(
            agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        raw_next_observations, _, _, dones = smarts.observe_from(
            smarts.vehicle_index.social_vehicle_ids()
        )
        next_observations = observation_transform(
            raw_next_observations, observation_adapter
        )
        actions = calculate_actions(raw_observations, raw_next_observations)

        """ Handle terminated vehicles. """
        for vehicle in done_vehicles:
            cur_path_builder = path_builders["Agent-" + vehicle]
            cur_path_builder["agent_0"]["terminals"][-1] = True
            demo_paths.append(cur_path_builder)
            print(f"Agent-{vehicle} Ended")

        """ Store data in the corresponding path builder. """
        vehicles = next_observations.keys()

        for vehicle in vehicles:
            if vehicle in observations:
                if vehicle not in path_builders:
                    path_builders[vehicle] = PathBuilder(["agent_0"])

                path_builders[vehicle]["agent_0"].add_all(
                    observations=observations[vehicle],
                    actions=actions[vehicle],
                    rewards=np.array([0.0]),
                    next_observations=next_observations[vehicle],
                    terminals=np.array([False]),
                )

        raw_observations = raw_next_observations
        observations = next_observations

    return demo_paths


def experiment(specs):

    save_path = "./demos"

    # obtain demo paths
    demo_paths = sample_demos(
        specs["env_specs"]["scenario_path"],
        wrap_absorbing=False,
    )

    with open(
        Path(save_path).joinpath(
            "smarts_{}.pkl".format(exp_specs["env_specs"]["scenario_name"]),
        ),
        "wb",
    ) as f:
        pickle.dump(demo_paths, f)

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

    experiment(exp_specs)
