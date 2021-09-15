from gym.envs.registration import register
import os

register(
    id="SMARTS-Imitation-v0",
    entry_point="smarts_imitation.envs:SMARTSImitation",
    kwargs=dict(
        scenarios=[
            os.path.join(
                os.path.dirname(__file__),
                "../interaction_dataset/scenarios/interaction_dataset_merging",
            )
        ]
    ),
)
