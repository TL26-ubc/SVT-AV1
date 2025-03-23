import random

import numpy as np
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
def train(state: Frame_feedback, next_state: Frame_feedback, done):
    agent.replay_buffer.push(state, 0, reward_function(), next_state, done)
    agent.forward()
    action = agent.select_action(state, 0.1)
    return action  # should be delta Q value
