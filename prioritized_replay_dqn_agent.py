import numpy as np
import torch
from icecream import ic
import matplotlib.pyplot as plt
from collections import namedtuple
from dqn_agent import DQNAgent, ReplayMemory, device
from constants import PRIORITIZED_REPLAY_CONSTANTS as C

Experience = namedtuple('experience', 'state, action, reward, next_state, done')


class PrioritizedReplayDQNAgent(DQNAgent):
    def _initiate_memory(self):
        return PrioritizedReplayMemory()

    def learn(self, **kwargs):
        """
        Sample the replay memory and update the network weights
        :gamma: discount rate
        :return: None
        """
        gamma, alpha, beta = C['gamma'], C['alpha'], kwargs['beta']
        states, actions, rewards, next_states, dones, indices, probs = self.replay_memory.sample()
        # print(states.shape)
        self.optimizer.zero_grad()
        # ic(states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())
        with torch.no_grad():
            weights = (C['memory_size']*probs) ** (-beta)
            weights = weights / max(weights)
            target_q = rewards + gamma * self.target_network(next_states).detach().max(dim=1)[0].unsqueeze(1) * (1-dones)
        current_q = self.network(states).gather(1, actions)
        delta = target_q - current_q
        loss = torch.mean(weights * delta**2)
        # loss = self.criterion(target_q, current_q)
        loss.backward()

        self.optimizer.step()
        self.update_target_network()

        priorities = (((torch.abs(delta.data) + C['small_const']).detach()) ** alpha).cpu().numpy().squeeze().astype(float)
        self.replay_memory.update_priority(indices, priorities, weights.cpu())


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self):
        super().__init__()
        self.sumtree = SumTree()

    def add(self, state, action, reward, next_state, done, memory_size=C['memory_size']):
        """
        Put tuple (state, action, reward, next_state, done) into the experience memory
        :return: None
        """
        super().add(state, action, reward, next_state, done, memory_size)
        self.sumtree.append(self.cur_index)

    def sample(self, sample_size=C['sample_size']):
        """
        Sample the memory and return sample_size tuples of (state, action, reward, next_state, done)
        :param sample_size: the number of tuples in the sample
        :return: a tuple of vertically stacked (state, action, reward, next_state, done)
        """
        indices, probs = self.sumtree.sample(sample_size)
        # ic(indices, probs)
        experiences = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = self._format_samples(experiences)
        return states, actions, rewards, next_states, dones, indices, probs

    def update_priority(self, indices, priorities, weights):
        self.sumtree.update_val(indices, priorities, weights)


class SumTree:
    def __init__(self):
        self.lst = []
        self.leaf_start_index = -1
        self.sumtree2memory = {}
        self.memory2sumtree = {}

    def append(self, memory_index):
        if len(self.lst) == 0:
            self.lst.append(1)
            self.leaf_start_index = 0
            self.sumtree2memory = {0: memory_index}
            self.memory2sumtree = {memory_index: 0}
            return

        val = self.find_max()
        if memory_index not in self.memory2sumtree:
            left_sumtree_index = len(self.lst)
            right_sumtree_index = len(self.lst) + 1
            parent_index = (left_sumtree_index-1) // 2
            self.lst.append(self.lst[parent_index])
            self.lst.append(val)
            left_memory_index = self.sumtree2memory[parent_index]
            del self.sumtree2memory[parent_index]
            self.sumtree2memory.update({
                left_sumtree_index: left_memory_index,
                right_sumtree_index: memory_index
            })
            self.memory2sumtree.update({
                left_memory_index: left_sumtree_index,
                memory_index: right_sumtree_index
            })
            self._update_ancestors(parent_index)
            self.leaf_start_index = parent_index + 1
        else:
            replace_index = self.memory2sumtree[memory_index]
            self.lst[replace_index] = val
            self._update_ancestors((replace_index-1) // 2)

    def _update_ancestors(self, parent_index):
        i = parent_index
        while i >= 0:
            self.lst[i] = self.lst[2*i + 1] + self.lst[2*i + 2]
            i = (i-1) // 2

    def sample(self, sample_size):
        bins = np.linspace(0, self.lst[0], sample_size + 1)[:-1]
        stratified_vals = (bins + np.random.rand(1, sample_size) * self.lst[0] / sample_size).squeeze()
        result = np.array([self._val2_memory_index(val) for val in stratified_vals])
        # ic(self.lst, result)
        sample_sumtree_indices = result[:, 0].astype(int).squeeze()
        sample_memory_indices = result[:, 1].astype(int).squeeze()
        sample_probs = torch.tensor(result[:, 2]).float().to(device).float().unsqueeze(dim=1)
        # if len(self.lst) - self.leaf_start_index == C['memory_size']:
        #     self._plot(sample_sumtree_indices)
        # ic(sample_probs)
        return sample_memory_indices, sample_probs

    def _plot(self, selected_indices):
        plt.figure(figsize=(20, 4))
        data = np.asarray(self.lst)
        plt.plot(data[self.leaf_start_index:], 'bo')
        plt.plot(selected_indices-self.leaf_start_index, data[selected_indices], 'mo')
        plt.show()

    def _val2_memory_index(self, val):
        i = 0
        while 2*i + 2 <= len(self.lst) - 1:
            if val <= self.lst[2*i + 1]:
                i = 2*i + 1
            else:
                val -= self.lst[2*i + 1]
                i = 2*i + 2
        return [i, self.sumtree2memory[i], self.lst[i] / self.lst[0]]

    def update_val(self, memory_indices, vals, weights):
        sumtree_indices = np.asarray([self.memory2sumtree[i] for i in memory_indices])
        # indices = sumtree_indices - self.leaf_start_index
        # if len(self.lst) - self.leaf_start_index == C['memory_size']:
        #     plt.figure(figsize=(20, 4))
        #     data = np.asarray(self.lst)
        #     plt.plot(indices, data[sumtree_indices], 'bo')
        #     # print(data[sumtree_indices])
        #     for i in range(len(vals)):
        #         self.lst[sumtree_indices[i]] = vals[i]
        #     data = np.asarray(self.lst)
        #     plt.plot(indices, data[sumtree_indices], 'mo')
        #     # print(data[sumtree_indices])
        #
        #     plt.figure(figsize=(20, 4))
        #     plt.bar(indices, weights)
        #     plt.show()
        # else:
        #     for i in range(len(vals)):
        #         self.lst[sumtree_indices[i]] = vals[i]
        for i in range(len(vals)):
            self.lst[sumtree_indices[i]] = vals[i]

    def find_max(self):
        result = torch.max(torch.tensor(self.lst[self.leaf_start_index:]).to(device)).cpu().item()
        # ic(result)
        return result
