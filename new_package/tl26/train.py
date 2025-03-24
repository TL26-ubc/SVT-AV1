import random

import numpy as np
import torch
from tl26.dqn.agent import DQNAgent
from tl26.feedback import Frame_feedback

input_shape = (10, 10)  # Example 10x10 input
num_actions = 64  # Example number of actions
agent = DQNAgent(input_shape, num_actions)


# TODO: Implement reward function
def reward_function(state: Frame_feedback, done):
    # Example reward function
    reward = np.random.random()
    return reward


# this function will run periodically
def train(state: Frame_feedback):
    state_tensor = torch.tensor(state.to_float_list(), dtype=torch.float32)
    print(state_tensor)
    return agent.sample(state_tensor)
