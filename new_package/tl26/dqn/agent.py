import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import DQN

# Q network take state as input and output Q values for each action

# Q learning: Q(state, action) = reward + gamma * max(Q(next_state, next_action) - Q(state, action))
# DQN algorithm: Q(state, action) = reward + gamma * max(Q(next_state, next_action))
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_shape, num_actions, lr=1e-3, gamma=0.99, batch_size=32, buffer_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        # self.replay_buffer = ReplayBuffer(buffer_size)
        self.model = DQN(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    # sample a action based on the env
    def sample(self, state):
        """
        Forward pass to select an action based on the current state
        
        Parameters:
        - state: Current state observation
        
        Returns:
        - action: Selected action (quantization parameter offset)
        - q_values: Raw Q-values for debugging (optional)
        """
        # Convert state to tensor and add batch & channel dimensions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Get Q-values from model
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Select action with highest Q-value
        action = q_values.argmax(dim=1).item()
        
        return action, q_values.cpu().numpy()

    # update the model based on the env
    def step(self, states, actions, rewards, next_states, dones):
        """
        Backward pass to update the model based on experience
        
        Parameters:
        - states: Batch of current states
        - actions: Batch of actions taken
        - rewards: Batch of rewards received
        - next_states: Batch of next states
        - dones: Batch of done flags indicating if episode ended
        
        Returns:
        - loss: Training loss value
        """
        # Convert inputs to tensors and move to device
        states = torch.FloatTensor(states).unsqueeze(1).to(self.device)  # Add channel dimension
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        
        # Compute target Q values using the Bellman equation
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
            