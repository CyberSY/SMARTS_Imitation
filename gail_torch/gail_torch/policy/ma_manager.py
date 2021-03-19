import torch
from torch import nn

from gail_torch.policy import BasePolicy
from gail_torch.utils import MultiAgentMemory


class MultiAgentManager(nn.Module):
    def __init__(self, policies: BasePolicy, device=torch.device("cpu")):
        super(MultiAgentManager, self).__init__()
        self.agent_num = len(policies)
        self.policies = policies
        self.device = device
        self._cnt = 0
        for i in range(self.agent_num):
            self.policies[i].set_agent_id(i)

    def set_device(self, device):
        for i in range(self.agent_num):
            self.policies[i].set_device(device)

    def get_action(self, obs_n):
        act_n = []
        for agent_id, obs in enumerate(obs_n):
            act, _ = self.policies[agent_id].get_action(obs)
            act_n.append(act)
        return torch.stack(act_n), {}

    def update(self, memory: MultiAgentMemory):
        for policy in self.policies:
            policy.update(memory)
        self._cnt += 1
