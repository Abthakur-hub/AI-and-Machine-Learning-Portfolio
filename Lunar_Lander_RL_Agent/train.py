import os
from tqdm import tqdm
import gymnasium as gym
import torch

import config
from models.agent import Agent
from utils.replay_buffer import ReplayBuffer
from utils.plot import plot_rewards

os.makedirs("checkpoints", exist_ok=True)

env = gym.make(config.ENV_NAME)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = Agent(state_size, action_size, config)
memory = ReplayBuffer(config.BUFFER_SIZE)

best_reward = float("-inf")
reward_history = []

for episode in tqdm(range(config.EPISODES), desc="Training"):

    state, _ = env.reset()

    total_reward = 0

    for _ in range(config.MAX_STEPS):

        action = agent.select_action(state, env)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.train_step(memory)

        state = next_state

        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)

    agent.decay_epsilon()

    if episode % config.TARGET_UPDATE == 0:
        agent.update_target()

    if total_reward > best_reward:
        best_reward = total_reward

        torch.save(
            agent.policy_net.state_dict(),
            config.MODEL_PATH
        )

plot_rewards(reward_history)

env.close()

print(f"\nBest Reward : {best_reward:.2f}")

print("Training Complete!")