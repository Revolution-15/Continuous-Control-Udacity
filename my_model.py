import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1a_units=128, fc2a_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1a_units)
        self.fc2 = nn.Linear(fc1a_units, fc2a_units)
        self.fc3 = nn.Linear(fc2a_units, action_size)

        # self.bn1a = nn.BatchNorm1d(fc1a_units)
        # self.bn2a = nn.BatchNorm1d(fc2a_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # print(state.shape)
        # if (state.dim()==1):
        #     state = torch.unsqueeze(state,0)
            # print(state.shape)
        # print(self.fc1(state).shape)
        # x = F.relu(self.bn1a(self.fc1(state)))
        # x = F.relu(self.bn2a(self.fc2(x)))

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1c_units=128, fc2c_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fc1c_units)
        self.fc2 = nn.Linear(fc1c_units+action_size, fc2c_units)
        self.fc3 = nn.Linear(fc2c_units, 1)

        # self.bn1c = nn.BatchNorm1d(fc1c_units)
        # self.bn2c = nn.BatchNorm1d(fc2c_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # print(state.shape)
        # xs = F.relu(self.bn1c(self.fcs1(state)))
        # x = torch.cat((xs, action), dim=1)
        # x = F.relu(self.bn2c(self.fc2(x)))

        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)
