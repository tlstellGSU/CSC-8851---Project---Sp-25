import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class MeanFieldQNetwork(nn.Module):
    """Neural network for Mean Field Q-Learning."""
    def __init__(self, state_dim, action_dim):
        super(MeanFieldQNetwork, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 32)

        self.final = nn.Linear(32, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.final(x)

class MeanFieldQLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.95):
        self.q_network = MeanFieldQNetwork(state_dim, action_dim)
        self.target_q_network = MeanFieldQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = []
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, 3)
            
        else:
            state_tensor = torch.FloatTensor(state).view(1, -1)
            q_values = self.q_network(state_tensor)
            action_idx = torch.argmax(q_values).item()
        
        return action_idx  # or: return self.angle_range[action_idx] if you want angle

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, mean_actions = zip(*batch)

        #print(f"DEBUG: Raw states before conversion: {np.array(states).shape}")
        states = torch.FloatTensor(np.array(states))
        states = states.view(states.shape[0], -1)  # Reshape after conversion
        #print(f"DEBUG: Reshaped states shape: {states.shape}")  # Add this print
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))  # Convert first
        next_states = next_states.view(next_states.shape[0], -1)  # Reshape after conversion
        #print(f"DEBUG: Reshaped next_states shape: {next_states.shape}")  # Debug print
        mean_actions = torch.FloatTensor(mean_actions)

        #print(f"DEBUG: States shape: {states.shape}, Actions shape: {actions.shape}")

        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_q_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * next_q_values.detach()

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_experience(self, state, action, reward, next_state, mean_action):
        self.replay_buffer.append((state, action, reward, next_state, mean_action))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

