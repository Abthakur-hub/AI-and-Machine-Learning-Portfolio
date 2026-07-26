import gymnasium as gym
import torch
import numpy as np

import config
from models.dqn import DQN

NUM_EPISODES = 10

env = gym.make(config.ENV_NAME)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = DQN(state_size, action_size).to(config.DEVICE)

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=config.DEVICE
    )
)

model.eval()

scores = []

for episode in range(NUM_EPISODES):

    state, _ = env.reset()

    done = False
    total_reward = 0

    while not done:

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        done = terminated or truncated

    scores.append(total_reward)

    print(f"Episode {episode + 1}: {total_reward:.2f}")

print("\n" + "=" * 40)
print(f"Average Reward : {np.mean(scores):.2f}")
print(f"Best Reward    : {np.max(scores):.2f}")
print(f"Worst Reward   : {np.min(scores):.2f}")
print("=" * 40)

env.close()