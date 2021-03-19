import gym
import torch
import torch.nn as nn

from gail_torch.utils import normal_log_density


class discrete_actor(nn.Module):
    def __init__(self, state_dim, num_actions, num_units, softmax_output=True):
        super(discrete_actor, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(state_dim, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, num_actions)
        self.softmax_output = softmax_output

        self.train()

    def get_log_prob(self, x, actions):
        if self.softmax_output:
            action_prob = self.forward(x)
        else:
            action_prob = torch.softmax(self.forward(x), dim=1)
        return torch.log(
            action_prob.gather(1, torch.where(actions > 0)[1].unsqueeze(1))
        )

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        if self.softmax_output:
            action_prob = torch.softmax(action_logits, dim=1)
            return action_prob
        else:
            return action_logits


class discrete_critic(nn.Module):
    def __init__(self, num_inputs, num_units):
        super(discrete_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, 1)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits


class discrete_qnet(nn.Module):
    def __init__(self, num_inputs, num_actions, num_units):
        super(discrete_qnet, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, num_actions)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits


# TODO(zbzhu): change discriminator_net to unified mlp class
class discriminator_net(nn.Module):
    def __init__(self, num_inputs, num_units):
        super(discriminator_net, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, 1)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))
        prob = torch.sigmoid(self.fc_out(x))
        return prob


class continuous_actor(nn.Module):
    def __init__(self, num_inputs, action_dim, num_units, log_std=0):
        super(continuous_actor, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, action_dim)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        self.train()

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_mean = self.fc_out(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std


class continuous_critic(nn.Module):
    def __init__(self, num_inputs, num_units):
        super(continuous_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.fc_in = nn.Linear(num_inputs, num_units)
        self.fc1 = nn.Linear(num_units, num_units)
        self.fc_out = nn.Linear(num_units, 1)

        self.train()

    def forward(self, x):
        x = self.LReLU(self.fc_in(x))
        x = self.LReLU(self.fc1(x))
        action_logits = self.fc_out(x)
        return action_logits
