import os
import numpy as np

from gym.envs.registration import register

register(
    id="SMARTS-Imitation-v0",
    entry_point="smarts_imitation.envs:SMARTSImitation",
    kwargs=dict(
        scenarios=[
            os.path.join(
                os.path.dirname(__file__),
                "../interaction_dataset/scenarios/interaction_dataset_merging",
            )
        ],
        action_range=np.array(
            [
                [-2, -0.1],
                [2, 0.1],
            ]
        ),
    ),
)

register(
    id="SMARTS-Imitation-v1",
    entry_point="smarts_imitation.envs:SMARTSImitation",
    kwargs=dict(
        scenarios=[
            os.path.join(
                os.path.dirname(__file__),
                "../ngsim",
            )
        ],
        action_range=np.array(
            [
                [-2.5, -0.2],
                [2.5, 0.2],
            ]
        ),
    ),
)

register(
    id="SMARTS-Imitation-v2",
    entry_point="smarts_imitation.envs:MASMARTSImitation",
    kwargs=dict(
        scenarios=[
            os.path.join(
                os.path.dirname(__file__),
                "../ngsim",
            )
        ],
        action_range=np.array(
            [
                [-2.5, -0.2],
                [2.5, 0.2],
            ]
        ),
    ),
)

