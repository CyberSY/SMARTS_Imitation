import numpy as np
import gym

from smarts_imitation.envs import MASMARTSImitation

env = MASMARTSImitation(scenarios = ["smarts-imitation/ngsim"],action_range=np.array([[-2.5, -0.2],[2.5, 0.2]]),agent_number = 1)

obs = env.reset()

env.destroy()