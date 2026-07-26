import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn import DQN


class Agent:

    def __init__(self, state_size, action_size, config):

        self.device = config.DEVICE

        self.gamma = config.GAMMA

        self.batch_size = config.BATCH_SIZE

        self.epsilon = config.EPSILON_START

        self.epsilon_min = config.EPSILON_END

        self.epsilon_decay = config.EPSILON_DECAY

        self.policy_net = DQN(state_size, action_size).to(self.device)

        self.target_net = DQN(state_size, action_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.LEARNING_RATE
        )

        self.loss_fn = nn.MSELoss()

    def select_action(self, state, env):

        if random.random() < self.epsilon:
            return env.action_space.sample()

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def train_step(self, memory):

        if len(memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(np.array(states)).to(self.device)

        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        rewards = torch.FloatTensor(rewards).to(self.device)

        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze()

        with torch.no_grad():

            next_q = self.target_net(next_states).max(1)[0]

        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def update_target(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )