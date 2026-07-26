import os

import gymnasium as gym
import torch

import config
from models.dqn import DQN

os.makedirs("videos", exist_ok=True)

env = gym.wrappers.RecordVideo(
    gym.make(
        config.ENV_NAME,
        render_mode="rgb_array"
    ),
    video_folder="videos",
    episode_trigger=lambda x: True
)

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

state, _ = env.reset()

done = False

while not done:

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        action = model(state_tensor).argmax().item()

    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

env.close()

print("Video saved successfully!")