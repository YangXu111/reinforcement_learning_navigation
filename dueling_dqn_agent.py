import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent, ReplayMemory, device
from torch import optim, nn
import torch.nn.functional as F
from constants import CONSTANTS as C


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        print('Dueling Network')
        self.fc1 = nn.Linear(input_size, C['hidden_layer_size'])
        # self.fc2 = nn.Linear(C['hidden_layer_size'], C['hidden_layer_size'])
        # self.fc3 = nn.Linear(C['hidden_layer_size'], C['hidden_layer_size'])
        self.fc_state = nn.Linear(C['hidden_layer_size'], C['hidden_layer_size'])
        self.output_state = nn.Linear(C['hidden_layer_size'], 1)
        self.fc_adv = nn.Linear(C['hidden_layer_size'], C['hidden_layer_size'])
        self.output_adv = nn.Linear(C['hidden_layer_size'], output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        state_val = F.relu(self.fc_state(x))
        state_val = F.relu(self.output_state(state_val))
        adv = F.relu(self.fc_adv(x))
        adv = F.relu(self.output_adv(adv))
        return state_val + adv - torch.mean(adv)


class DuelingDQNAgent(DQNAgent):
    def _initiate_network(self, state_size, action_size):
        self.target_network = Network(state_size, action_size).to(device)
        self.network = Network(state_size, action_size).to(device)

