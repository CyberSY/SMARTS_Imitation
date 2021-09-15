import argparse
import csv
import json
import math

from collections import defaultdict

import numpy as np


"""Meta information of the track files, please see:
https://www.opendatanetwork.com/dataset/data.transportation.gov/8ect-6jqj
"""


def cal_heading(prev_pos, cur_pos, next_pos):
    p = np.array(prev_pos[:2])
    c = np.array(cur_pos[:2])
    n = np.array(next_pos[:2])

    v1 = (c - p) / np.linalg.norm(c - p)
    v2 = (n - c) / np.linalg.norm(n - c)

    if any(np.isnan(v1)) and any(np.isnan(v2)):
        return 0
    elif any(np.isnan(v1)):
        average = v2
    elif any(np.isnan(v2)):
        average = v1
    else:
        average = v1 + v2

    r = math.atan2(average[1], average[0]) % (2 * math.pi)
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NGSIM-dataset-converter")
    parser.add_argument(
        "input",
        help="Original dataset in csv format",
        type=str,
    )

    parser.add_argument(
        "output",
        help="History file in JSON format",
        type=str,
    )

    args = parser.parse_args()
    traffic_history = defaultdict(dict)
    with open(args.input, newline="") as csvfile:
        cur_vehicle_id = ""
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            state = {
                "vehicle_id": row["Vehicle ID"],
                "vehicle_type": row["Vehicle Class"],
                "position": [float(row["Local X"]), float(row["Local Y"]), 0],
                "speed": round(float(row["Vehicle Velocity"]), 3),
                "vehicle_length": float(row["Vehicle Length"]),
                "vehicle_width": float(row["Vehicle Width"]),
            }
            timestamp = round(int(row["Frame ID"]) / 1000, 3)
            traffic_history[timestamp][state["vehicle_id"]] = state

    for t in traffic_history:
        for v in traffic_history[t]:
            p = traffic_history.get(t - 0.001, {}).get(v)
            n = traffic_history.get(t + 0.001, {}).get(v)
            c = traffic_history[t][v]

            if not p:
                p = c
            if not n:
                n = c

            heading = cal_heading(p["position"], c["position"], n["position"])
            c["heading"] = heading - math.pi / 2

    with open(args.output, "w") as f:
        f.write(json.dumps(traffic_history))
