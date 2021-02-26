import torch
import matplotlib.pyplot as plt
from dqn_agent import Network, DQNAgent, ReplayMemory, device
from constants import CONSTANTS as C

class DoubleDQNAgent(DQNAgent):
    def learn(self, **kwargs):
        """
        Sample the replay memory and update the network weights
        :gamma: discount rate
        :return: None
        """
        gamma = C['gamma']
        states, actions, rewards, next_states, dones = self.replay_memory.sample()

        self.optimizer.zero_grad()
        # print(states.size(), actions.size(), rewards.size(), next_states.size(), dones.size())
        with torch.no_grad():
            max_actions = torch.argmax(self.network(next_states), 1).unsqueeze(1)
            target_q = rewards + gamma * self.target_network(next_states).gather(1, max_actions) * (1 - dones)
        current_q = self.network(states).gather(1, actions)
        # print(target_q.size(), current_q.size())
        loss = self.criterion(target_q, current_q)
        loss.backward()

        prev_delta = (target_q - current_q.detach()).cpu().numpy().squeeze().astype(float)
        self.optimizer.step()
        self.update_target_network()

        # Plot to check Q value delta before and after network update
        # if self.replay_memory.len() == C['memory_size']:
        #     with torch.no_grad():
        #         max_actions = torch.argmax(self.network(next_states), 1).unsqueeze(1)
        #         target_q = rewards + gamma * self.target_network(next_states).gather(1, max_actions) * (1 - dones)
        #         current_q = self.network(states).gather(1, actions)
        #     post_delta = (target_q - current_q).cpu().numpy().squeeze().astype(float)
        #     plt.figure(figsize=(20, 4))
        #     plt.plot(prev_delta, 'bo')
        #     plt.plot(post_delta, 'mo')
        #     plt.show()



