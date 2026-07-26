import gymnasium as gym
import torch

from config import *
from models.dqn import DQN

env = gym.make(ENV_NAME, render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = DQN(state_size, action_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

state, _ = env.reset()

done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action = model(state_tensor).argmax().item()

    state, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    done = terminated or truncated

print(f"Total Reward: {total_reward}")

env.close()