import pickle
import argparse
import numpy as np
from pathlib import Path

from smarts.core.scenario import Scenario


def main(scenario_path, test_ratio):
    scenario_iterator = Scenario.scenario_variations(
        [scenario_path],
        list([]),
    )
    scenario = next(scenario_iterator)
    vehicle_missions = scenario.discover_missions_of_traffic_histories()

    veh_start_times = {
        v_id: mission.start_time for v_id, mission in vehicle_missions.items()
    }
    vlist = []
    for vehicle_id,start_time in veh_start_times.items():
        vlist.append((vehicle_id,start_time))
    dtype = [('id',int),('start_time',float)]
    vlist = np.array(vlist,dtype = dtype)
    vlist = np.sort(vlist,order = 'start_time')
    vehicle_ids = list(vehicle_missions.keys())
    for id in range(len(vehicle_ids)):
        vehicle_ids[id] = f'{vlist[id][0]}'

    test_vehicle_ids = vehicle_ids[:int(len(vehicle_ids) * test_ratio)]
    train_vehicle_ids = vehicle_ids[int(len(vehicle_ids) * test_ratio):]

    save_dir = Path(scenario_path)
    with open(save_dir / "multiagent_train_ids.pkl", "wb") as f:
        print(f"Train Vehicle Num: {len(train_vehicle_ids)}")
        pickle.dump(train_vehicle_ids, f)

    with open(save_dir / "multiagent_test_ids.pkl", "wb") as f:
        print(f"Test Vehicle Num: {len(test_vehicle_ids)}")
        pickle.dump(test_vehicle_ids, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario_path",
        help="A list of scenarios. Each element can be either the scenario to run "
        "(see scenarios/ for some samples you can use) OR a directory of scenarios "
        "to sample from.",
        type=str,
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    main(
        scenario_path=args.scenario_path,
        test_ratio=args.test_ratio,
    )
