import torch

# Environment
ENV_NAME = "CartPole-v1"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 1e-3
GAMMA = 0.99

# Replay Buffer
BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Epsilon-Greedy
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Target Network
TARGET_UPDATE = 10

# Training
EPISODES = 500
MAX_STEPS = 500

# Model Saving
MODEL_PATH = "checkpoints/best_model.pth"