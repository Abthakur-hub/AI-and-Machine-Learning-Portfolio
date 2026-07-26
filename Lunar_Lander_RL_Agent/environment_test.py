import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

state, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()