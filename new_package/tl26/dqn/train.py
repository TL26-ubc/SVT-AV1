import random

import numpy as np

from .agent import DQNAgent

input_shape = (10, 10)  # Example 10x10 input
num_actions = 4  # Example number of actions
agent = DQNAgent(input_shape, num_actions)

# Example training loop
for episode in range(100):
    state = np.random.rand(10, 10).astype(np.float32)  # Random example state
    done = False
    while not done:
        action = agent.select_action(state, epsilon=0.1)
        next_state = np.random.rand(10, 10).astype(np.float32)  # Random example next state
        reward = random.random()
        done = random.random() < 0.1  # Randomly decide to end episode
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        agent.train()