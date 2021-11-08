# Imitation Learning with SMARTS Platform

This repository contains a [SMARTS](https://github.com/huawei-noah/SMARTS.git)-based imitation learning environment and implementations of various imitation learning algorithms on this environment. Currently the environment supports the MERGEING scenario in the [INTERACTION](https://interaction-dataset.com/details-and-format) dataset and [NGSIM i-80](https://www.fhwa.dot.gov/publications/research/operations/06137/) dataset.

<!-- Since we do not have access to the exact action (throttle, brakes, steering, ...) excuted by human driver from recorded trajectories in the Interaction dataset, we modify the original GAIL to its state-only variant, [GAIfO](https://arxiv.org/pdf/1807.06158). -->

This repository currently supports training imitation learning agents with GAIL, GAIfO (a state-only variant of GAIL) and BC. Multi-agent imitation learning algorithms such as MAGAIL will be available soon. This repository uses a modified version of [ILSwiss](https://github.com/Ericonaldo/ILSwiss.git) as the learning framework.

Installation:

1. Install SMARTS simulation platform in the `./SMARTS` folder.

2. Build the NGSIM scenario with

   ```bash
   scl scenario build --clean smarts_imitation/ngsim
   ```

3. Install the *smarts_imitation* environment with

   ```bash
   pip install -e ./smarts_imitation
   ```

4. Setup the ILSwiss and run the following command under `./ILSwiss` folder. Make sure there is `smarts_ngsim.pkl` in `./ILSwiss/demos`, otherwise you may generate the expert demo yourself with `script/expert_generation.py`.

   ```bash
   python run_experiment.py -e exp_specs/gail/gailfo_smarts_ngsim.yaml
   ```

   or

   ```bash
   python run_experiment.py -e exp_specs/gail/gail_smarts_ngsim.yaml
   ```
