from utils.plot import plot_rewards
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer

os.makedirs("checkpoints", exist_ok=True)

env = gym.make(ENV_NAME)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size).to(DEVICE)
target_net = DQN(state_size, action_size).to(DEVICE)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

memory = ReplayBuffer(BUFFER_SIZE)

epsilon = EPSILON_START
best_reward = 0


def select_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()

    state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        return policy_net(state).argmax().item()


def optimize():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states = torch.FloatTensor(np.array(states)).to(DEVICE)
    actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
    rewards = torch.FloatTensor(rewards).to(DEVICE)
    next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
    dones = torch.FloatTensor(dones).to(DEVICE)

    current_q = policy_net(states).gather(1, actions).squeeze()

    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]

    target_q = rewards + GAMMA * max_next_q * (1 - dones)

    loss = criterion(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


rewards_history = []

from tqdm import tqdm

for episode in tqdm(range(EPISODES), desc="Training"):

    state, _ = env.reset()
    total_reward = 0

    for _ in range(MAX_STEPS):

        action = select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        optimize()

        if done:
            break

    rewards_history.append(total_reward)

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy_net.state_dict(), MODEL_PATH)

    

env.close()
plot_rewards(rewards_history)