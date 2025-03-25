import random

import numpy as np
import torch
from tl26.dqn.agent import DQNAgent
from tl26.feedback import Frame_feedback

input_shape = 18  # 17 features as state
num_actions = 64 * 2 # [-63,63] action space 
agent = DQNAgent(input_shape, num_actions)


# TODO: Implement reward function
def reward_function(state: Frame_feedback, done):
    # Example reward function
    reward = np.random.random()
    return reward


# this function will run periodically
def sample(state: torch.Tensor):
    print(state)
    return agent.sample(state)
