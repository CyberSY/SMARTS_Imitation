# Imitation Learning with SMARTS Platform

This repository contains a [SMARTS](https://github.com/huawei-noah/SMARTS.git)-based imitation learning environment and implementations of various imitation learning algorithms on this environment. Currently the environment supports the MERGEING scenario in the [INTERACTION](https://interaction-dataset.com/details-and-format) dataset and [NGSIM i-80](https://www.fhwa.dot.gov/publications/research/operations/06137/) dataset.

<!-- Since we do not have access to the exact action (throttle, brakes, steering, ...) excuted by human driver from recorded trajectories in the Interaction dataset, we modify the original GAIL to its state-only variant, [GAIfO](https://arxiv.org/pdf/1807.06158). -->

This repository currently supports training imitation learning agents with GAIL, GAIfO (a state-only variant of GAIL) and BC. Multi-agent imitation learning algorithms such as MAGAIL will be available soon. This repository uses a modified version of [ILSwiss](https://github.com/Ericonaldo/ILSwiss.git) as the learning framework.

<!-- Installation:

1. Install SMARTS simulation platform.
2. Install the SMARTS-based imitation learning environment:
```bash
pip install -e ./smarts-imitation
```

Expert data generation:

```bash
python script/expert_generation.py --scenario smarts-imitation/interaction_dataset/scenarios/interaction_dataset_merging
```

Run imitation learning:

```bash
cd gail_torch
python irl/run.py -e config/gaifo.yaml
```
 -->
