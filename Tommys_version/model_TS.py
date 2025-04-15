import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# -------------------------------
# Define the Neural Network Class
# -------------------------------

class MeanFieldQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MeanFieldQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + 1, 512)  # <- Add 1 for mean action
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 32)
        self.final = nn.Linear(32, action_dim)

    def forward(self, x):  # x is full input: state + mean_action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.final(x)


# -------------------------------
# Define the Fish Agent Class
# -------------------------------

class Fish:
    def __init__(self, position, velocity, max_speed=4.0, max_force=0.1, perception_radius=15):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2)
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception_radius = perception_radius

    def update(self):
        # Update velocity and limit it to max_speed.
        self.velocity += self.acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # Update position.
        self.position += self.velocity

        # Reset acceleration.
        self.acceleration *= 0

    def apply_force(self, force):
        self.acceleration += force

    def apply_behavior(self, fishes):
        separation = self.separate(fishes)
        alignment  = self.align(fishes)
        cohesion   = self.cohere(fishes)
        
        # Weight each behavior: these weights can be tuned.
        separation_weight = 1.5
        alignment_weight  = 1.0
        cohesion_weight   = 1.0
        
        self.apply_force(separation * separation_weight)
        self.apply_force(alignment * alignment_weight)
        self.apply_force(cohesion * cohesion_weight)

    def separate(self, fishes):
        desired_separation = self.perception_radius * 0.5  # adjust as needed
        steer = np.zeros(2)
        total = 0

        for other in fishes:
            if other is self:
                continue
            diff = self.position - other.position
            distance = np.linalg.norm(diff)
            if 0 < distance < desired_separation:
                steer += diff / distance  # weighted by inverse distance
                total += 1
        if total > 0:
            steer /= total
            if np.linalg.norm(steer) > 0:
                steer = (steer / np.linalg.norm(steer)) * self.max_speed - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = (steer / np.linalg.norm(steer)) * self.max_force
        return steer

    def align(self, fishes):
        perception = self.perception_radius
        avg_velocity = np.zeros(2)
        total = 0

        for other in fishes:
            if other is self:
                continue
            distance = np.linalg.norm(self.position - other.position)
            if distance < perception:
                avg_velocity += other.velocity
                total += 1
        if total > 0:
            avg_velocity /= total
            if np.linalg.norm(avg_velocity) > 0:
                avg_velocity = (avg_velocity / np.linalg.norm(avg_velocity)) * self.max_speed
            steer = avg_velocity - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = (steer / np.linalg.norm(steer)) * self.max_force
            return steer
        return np.zeros(2)

    def cohere(self, fishes):
        perception = self.perception_radius
        center_of_mass = np.zeros(2)
        total = 0

        for other in fishes:
            if other is self:
                continue
            distance = np.linalg.norm(self.position - other.position)
            if distance < perception:
                center_of_mass += other.position
                total += 1
        if total > 0:
            center_of_mass /= total
            desired = center_of_mass - self.position
            if np.linalg.norm(desired) > 0:
                desired = (desired / np.linalg.norm(desired)) * self.max_speed
            steer = desired - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = (steer / np.linalg.norm(steer)) * self.max_force
            return steer
        return np.zeros(2)

# -------------------------------
# Define the Environment Class
# -------------------------------

class FishSchoolEnv:
    def __init__(self, 
                 num_fish=100, 
                 grid_size=60, 
                 velocity=3, 
                 omega_max=np.pi/3, 
                 dt=1, 
                 perception_range=15,
                 obs_grid_size=16,
                 num_actions=5):  # NEW
        self.num_fish = num_fish
        self.grid_size = grid_size
        self.velocity = velocity
        self.omega_max = omega_max
        self.dt = dt
        self.perception_range = perception_range
        self.obs_grid_size = obs_grid_size
        self.num_actions = num_actions  # NEW

        self.positions = np.random.rand(num_fish, 2) * grid_size
        self.orientations = np.random.uniform(0, 2 * np.pi, num_fish)

    def get_state(self, fish_index):
        focal_pos = self.positions[fish_index]
        focal_orientation = self.orientations[fish_index]
        observation = np.zeros((2, self.obs_grid_size, self.obs_grid_size))
        
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]

        for neighbor in neighbors:
            rel_pos = self.positions[neighbor] - focal_pos
            dx, dy = rel_pos
            rotated_x = dx * np.cos(-focal_orientation) - dy * np.sin(-focal_orientation)
            rotated_y = dx * np.sin(-focal_orientation) + dy * np.cos(-focal_orientation)

            grid_x = int((rotated_x / self.perception_range) * (self.obs_grid_size // 2)) + (self.obs_grid_size // 2)
            grid_y = int((rotated_y / self.perception_range) * (self.obs_grid_size // 2)) + (self.obs_grid_size // 2)

            if 0 <= grid_x < self.obs_grid_size and 0 <= grid_y < self.obs_grid_size:
                observation[0, grid_x, grid_y] = 1
                rel_orientation = (self.orientations[neighbor] - focal_orientation) / np.pi
                observation[1, grid_x, grid_y] = rel_orientation

        return observation  # Shape: (2, 16, 16)

    def get_mean_action(self, fish_index, actions):
        focal_pos = self.positions[fish_index]
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]
        return np.mean(actions[neighbors]) if len(neighbors) > 0 else 0

    def action_to_steering(self, action_index):
        """Maps discrete action index to a continuous steering change in [-omega_max, omega_max]."""
        return np.interp(action_index, [0, self.num_actions - 1], [-self.omega_max, self.omega_max])

    def step(self, action_indices):
        """Update fish positions using discrete actions (converted to continuous steering changes)."""
        # Convert discrete actions to steering angles
        steering_changes = np.array([self.action_to_steering(a) for a in action_indices])

        self.orientations += steering_changes * self.dt
        dx = self.velocity * np.cos(self.orientations) * self.dt
        dy = self.velocity * np.sin(self.orientations) * self.dt

        self.positions[:, 0] = (self.positions[:, 0] + dx) % self.grid_size
        self.positions[:, 1] = (self.positions[:, 1] + dy) % self.grid_size

    def get_reward(self, fish_index):
        focal_pos = self.positions[fish_index]
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]

        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            return -1.0

        # Alignment
        alignments = np.cos(self.orientations[neighbors] - self.orientations[fish_index])
        alignment_score = np.mean(alignments)

        # Cohesion (inverse of distance to center of mass)
        center_of_mass = np.mean(self.positions[neighbors], axis=0)
        cohesion_score = -np.linalg.norm(center_of_mass - self.positions[fish_index])

        reward = 0.5 * alignment_score + 0.5 * cohesion_score
        if num_neighbors > 10:
            reward -= 1.0
        return reward

    def render(self):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        scatter = ax.scatter(self.positions[:, 0], self.positions[:, 1], c='blue', marker='o')

        def update(frame):
            actions = np.random.randint(0, self.num_actions, size=self.num_fish)
            self.step(actions)
            scatter.set_offsets(self.positions)

        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        plt.show()

# -------------------------------
# Define the ReplayBuffer Class
# -------------------------------

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, mean_action, action, reward, next_state, next_mean_action, done):
        self.buffer.append((state, mean_action, action, reward, next_state, next_mean_action, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, mean_action, action, reward, next_state, next_mean_action, done = map(np.array, zip(*batch))
        return state, mean_action, action, reward, next_state, next_mean_action, done
    
    def __len__(self):
        return len(self.buffer)