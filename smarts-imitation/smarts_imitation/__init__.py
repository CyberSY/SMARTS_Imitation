from gym.envs.registration import register

register(
    id="SMARTS-Imitation-v0",
    entry_point="smarts_imitation.envs:SMARTSImitation",
    kwargs=dict(
        scenarios=["interaction_dataset/scenarios/interaction_dataset_merging"]
    ),
)
