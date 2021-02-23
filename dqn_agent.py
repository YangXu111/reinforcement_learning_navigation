import random
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import transforms, models
from collections import namedtuple, OrderedDict
from constants import CONSTANTS as C

torch.manual_seed(0)
random.seed(C['random_seed'])
# torch.set_deterministic(True)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print('The current device is %s' % device)


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, C['hidden_layer_size'])
        self.fc2 = nn.Linear(C['hidden_layer_size'], C['hidden_layer_size'])
        self.fc3 = nn.Linear(C['hidden_layer_size'], C['hidden_layer_size'])
        self.output = nn.Linear(C['hidden_layer_size'], output_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.output(x)


class CnnNetwork(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # input shape (1, 3, 84, 84)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, bias=False)
        # define a pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(20**2 * 256, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 4)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TransferLearningNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        tailnet = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 500)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(500, action_size))
        ]))

        self.model.classifier = tailnet

    def forward(self, state):
        return self.model(state)


class DQNAgent:
    def __init__(self, state_size, action_size, use_cnn_network=False):
        self.use_cnn_network = use_cnn_network
        self._initiate_network(state_size, action_size, use_cnn_network)
        self.action_size = action_size
        self.counter = 0
        self.replay_memory = self._initiate_memory()
        self.optimizer = optim.SGD(self._param_for_optimizer(), lr=C['learning_rate'])
        self.criterion = nn.MSELoss()

        self.transform = transforms.Compose([transforms.Resize(224),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def _initiate_network(self, state_size, action_size, use_cnn_network):
        if not use_cnn_network:
            self.target_network = Network(state_size, action_size).to(device)
            self.network = Network(state_size, action_size).to(device)
        else:
            # self.target_network = TransferLearningNetwork(action_size).to(device)
            # self.network = TransferLearningNetwork(action_size).to(device)
            self.target_network = CnnNetwork(action_size).to(device)
            self.network = CnnNetwork(action_size).to(device)

    def _initiate_memory(self):
        return ReplayMemory()

    def _param_for_optimizer(self):
        # if self.use_cnn_network:
        #     return self.network.model.classifier.parameters()
        return self.network.parameters()

    def preprocessing(self, state):
        result = torch.tensor(state).to(device).transpose(0, 3).float().squeeze().unsqueeze(0)
        # if self.use_cnn_network:
        #     return self.transform(result)
        return result

    def action(self, state, **kwargs):
        """
        Produce the action given the current state and increase the counter
        :param state: state of the system
        :param epsilon: for epsilon policy
        :param update_interval: the interval for temporal difference learning
        :param sample_size: the number of samples
        :return: action by epsilon policy
        """
        update_interval = C['update_interval']
        sample_size = C['sample_size']
        epsilon = kwargs.get('epsilon', 0.1)
        beta = kwargs.get('beta', None)

        if random.random() > epsilon:
            self.network.eval()
            action = torch.argmax(self.network(state)).cpu().data.numpy().item()  # fixed type to int
            self.network.train()
        else:
            action = random.randint(0, self.action_size - 1)
        self.counter = (self.counter+1) % update_interval
        if self.counter == 0 and self.replay_memory.len() >= sample_size:
            self.learn(beta=beta)
        return action

    def learn(self, **kwargs):
        """
        Sample the replay memory and update the network weights
        :gamma: discount rate
        :return: None
        """
        gamma = C['gamma']
        states, actions, rewards, next_states, dones = self.replay_memory.sample()

        # print(states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())
        target_q = rewards + gamma * self.target_network(next_states).detach().max(dim=1)[0].unsqueeze(1) * (1-dones)
        current_q = self.network(states).gather(1, actions)
        # print(target_q.size(), current_q.size())
        self.optimizer.zero_grad()
        loss = self.criterion(target_q, current_q)
        loss.backward()
        del target_q, current_q, loss

        self.optimizer.step()
        self.update_target_network()

    def update_target_network(self, tau=C['tau']):
        """
        Update the slow network weights by copying the fast network weights
        :tau: slow network param = (1 - tau)*slow network param + tau*fast network param
        :return: None
        """
        for slow_param, fast_param in zip(self.target_network.parameters(), self.network.parameters()):
            slow_param.data.copy_((1 - tau) * slow_param.data + tau * fast_param.data)


Experience = namedtuple('experience', 'state, action, reward, next_state, done')


class ReplayMemory:
    def __init__(self):
        self.memory = []
        self.cur_index = -1

    def add(self, state, action, reward, next_state, done, memory_size=C['memory_size']):
        """
        Put tuple (state, action, reward, next_state, done) into the experience memory
        :return: None
        """
        # print(state.shape, next_state.shape)
        expr = Experience(state, action, reward, next_state, done)
        self.cur_index = (self.cur_index + 1) % memory_size
        if self.cur_index == len(self.memory):
            self.memory.append(expr)
        else:
            self.memory[self.cur_index] = expr

    def sample(self, sample_size=C['sample_size']):
        """
        Sample the memory and return sample_size tuples of (state, action, reward, next_state, done)
        :param sample_size: the number of tuples in the sample
        :return: a tuple of vertically stacked (state, action, reward, next_state, done)
        """
        experiences = random.sample(self.memory, sample_size)
        return self._format_samples(experiences)

    def _format_samples(self, experiences):
        states = torch.vstack([e.state for e in experiences])  # fixed float()
        actions = torch.tensor([e.action for e in experiences]).unsqueeze(1).to(device)
        rewards = torch.tensor([e.reward for e in experiences]).unsqueeze(1).to(device)
        next_states = torch.vstack([e.next_state for e in experiences])  # fixed float()
        dones = torch.tensor([e.done for e in experiences]).long().unsqueeze(1).to(device)  # fixed long()

        return states, actions, rewards, next_states, dones

    def len(self):
        return len(self.memory)
