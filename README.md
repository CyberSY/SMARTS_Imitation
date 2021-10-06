# Imitation Learning with SMARTS Platform

This repository contains a [SMARTS](https://github.com/huawei-noah/SMARTS.git)-based imitation learning environment and a minimal implementation of [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf)(GAIL). Currently the environment supports the MERGEING scenario in the [INTERACTION](https://interaction-dataset.com/details-and-format) dataset and NGSIM i-80.

Since we do not have access to the exact action (throttle, brakes, steering, ...) excuted by human driver from recorded trajectories in the Interaction dataset, we modify the original GAIL to its state-only variant, [GAIfO](https://arxiv.org/pdf/1807.06158).

Installation:

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
