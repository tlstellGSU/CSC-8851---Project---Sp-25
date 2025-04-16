import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import torch.nn.functional as F

# -------------------------------
# Define the Neural Network Class
# -------------------------------

class MultiAgentNet(nn.Module):
    def __init__(self, spatial_channels=2, orientation_dim=3, action_reward_dim=2, 
                 num_actions=5, dropout_p=0.3, obs_size=16):
        """
        :param spatial_channels: Number of input channels for spatial observations.
        :param orientation_dim: Dimension for orientation/behavior input.
        :param action_reward_dim: Dimension for last action + reward input.
        :param num_actions: Number of possible actions (M).
        :param dropout_p: Dropout probability.
        :param obs_size: Spatial observation width/height (default: 16).
        """
        super(MultiAgentNet, self).__init__()
        self.num_outputs = 2 * num_actions + 1  # As per your design.
        self.dropout_p = dropout_p

        # ---- Branch A: Spatial Input ----
        # Input shape: [B, spatial_channels, obs_size, obs_size]
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(spatial_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Use a dummy input to compute the flattened feature size.
        dummy_input = torch.zeros(1, spatial_channels, obs_size, obs_size)
        dummy_output = self.spatial_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)
        print("Dynamically computed flattened_size for spatial_fc:", flattened_size)
        # Create spatial_fc using the computed size.
        self.spatial_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),  # This should be Linear(8192, 256) if flattened_size is 8192.
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        # (For debugging, you can print the weight shape:)
        # print("Spatial fc Linear weight shape:", self.spatial_fc[1].weight.shape)

        # ---- Branch B: Orientation / Behavior Input ----
        self.orientation_fc = nn.Sequential(
            nn.Linear(orientation_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        # ---- Branch C: Last Action + Reward Input ----
        self.action_reward_fc = nn.Sequential(
            nn.Linear(action_reward_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        # ---- Final Fully Connected Layers ----
        self.fc1 = nn.Sequential(
            nn.Linear(256 + 32 + 32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.output_layer = nn.Linear(64, self.num_outputs)

    def forward(self, spatial_input, orientation_input, action_reward_input):
        # Branch A
        x_spatial = self.spatial_conv(spatial_input)
        x_spatial = self.spatial_fc(x_spatial)  # Now uses correct input size.
        # Branch B
        x_orient = self.orientation_fc(orientation_input)
        # Branch C
        x_action = self.action_reward_fc(action_reward_input)
        # Concatenate
        x = torch.cat([x_spatial, x_orient, x_action], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.output_layer(x)
        return F.softmax(out, dim=1)


# -------------------------------
# Define the Fish Agent Class
# -------------------------------

class Fish:
    def __init__(self, position, orientation, max_speed=4.0, omega_max=np.pi/4, M=9, 
                 
                 #perception_radius=np.random.normal(15,2)
                 perception_radius=15
                 
                 ):
        """
        :param position: Initial (x, y) position
        :param orientation: Initial orientation θ in radians
        :param max_speed: Constant speed
        :param omega_max: Max angular change in radians
        :param M: Number of discrete actions (must be odd)
        :param perception_radius: For future behavior rules
        """
        assert M % 2 == 1, "M must be an odd number"
        
        self.position = np.array(position, dtype=float)
        self.orientation = float(orientation)
        self.max_speed = max_speed
        self.omega_max = omega_max
        self.M = M
        self.perception_radius = perception_radius

    def get_discrete_actions(self):
        """
        Returns the list of M discrete angular changes between [-omega_max, omega_max]
        """
        return np.linspace(-self.omega_max, self.omega_max, self.M)

    def update(self, action_index):
        """
        Updates the fish's position and orientation using first-order kinematics
        :param action_index: Integer index in [0, M-1] representing chosen angular adjustment
        """
        omega = self.get_discrete_actions()[action_index]

        # Update orientation
        self.orientation += omega

        # Normalize orientation to [-pi, pi] for consistency (optional but helpful)
        self.orientation = (self.orientation + np.pi) % (2 * np.pi) - np.pi

        # Update position
        dx = self.max_speed * np.cos(self.orientation)
        dy = self.max_speed * np.sin(self.orientation)
        self.position += np.array([dx, dy])

    def get_state(self):
        """
        Returns the current state as a tuple: (x, y, theta)
        """
        return (self.position[0], self.position[1], self.orientation)

    def compute_alignment_metric(self, fishes):
        """
        Computes the alignment metric ϕ:
          ϕ = (1/N) * | sum (normalized velocity vector of each fish) |
        Uses all fishes (neighbors including self). Since all fish move at constant speed,
        the normalized velocity vector for a fish is [cos(theta), sin(theta)].
        
        :param fishes: List of all Fish instances.
        :return: Scalar value representing the alignment.
        """
        N = len(fishes)
        sum_norm_vel = np.zeros(2)
        for fish in fishes:
            # Compute normalized velocity vector from orientation.
            norm_vel = np.array([np.cos(fish.orientation), np.sin(fish.orientation)])
            sum_norm_vel += norm_vel
        phi = np.linalg.norm(sum_norm_vel) / N
        return phi

    def compute_cohesion_metric(self, fishes):
        """
        Computes the cohesion metric ξ:
          ξ = (1/N) * (number of fishes within perception radius)
        Here, each fish counts as a neighbor if its Euclidean distance from self is less 
        than or equal to the perception_radius. (Includes self.)
        
        :param fishes: List of all Fish instances.
        :return: Scalar value representing the cohesion.
        """
        N_total = len(fishes)
        count_neighbors = 0
        for fish in fishes:
            # Use Euclidean distance.
            distance = np.linalg.norm(self.position - fish.position)
            if distance <= self.perception_radius:
                count_neighbors += 1
        xi = count_neighbors / N_total
        return xi

    def compute_density_metric(self, fishes):
        """
        Computes the density metric, ANNR:
          D_actual = average distance to all neighbors (excluding self) within the perception radius.
          D_expected = 0.5 / sqrt(total_agents / A), where A = π * (perception_radius)^2.
          ANNR = D_actual / D_expected.
          
        If there are no other fishes within perception, D_actual is set equal to perception_radius.
        
        :param fishes: List of all Fish instances.
        :return: Scalar value representing the density metric.
        """
        total_agents = len(fishes)
        distances = []
        for fish in fishes:
            # Exclude self to avoid zero distance.
            if fish is self:
                continue
            distance = np.linalg.norm(self.position - fish.position)
            if distance <= self.perception_radius:
                distances.append(distance)
        if len(distances) > 0:
            D_actual = np.mean(distances)
        else:
            # If no neighbors, use perception_radius as a default.
            D_actual = self.perception_radius
        
        A = np.pi * (self.perception_radius ** 2)
        D_expected = 0.5 / np.sqrt(total_agents / A)
        annr = D_actual / D_expected
        return annr

    def compute_behavior_metrics(self, fishes):
        """
        Computes the three behavioral metrics and returns them as a feature vector.
        
        :param fishes: List of all Fish instances.
        :return: A numpy array [alignment, cohesion, density]
                 These three scalars can then be fed to branch B of the network.
        """
        alignment = self.compute_alignment_metric(fishes)
        cohesion = self.compute_cohesion_metric(fishes)
        density = self.compute_density_metric(fishes)
        return np.array([alignment, cohesion, density])
    

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
                 num_actions=5):
        self.num_fish = num_fish
        self.grid_size = grid_size
        self.velocity = velocity
        self.omega_max = omega_max
        self.dt = dt
        self.perception_range = perception_range
        self.obs_grid_size = obs_grid_size
        self.num_actions = num_actions

        self.positions = np.random.rand(num_fish, 2) * grid_size
        self.orientations = np.random.uniform(0, 2 * np.pi, num_fish)

    def get_state(self, fish_index):
        focal_pos = self.positions[fish_index]
        focal_orientation = self.orientations[fish_index]
        observation = np.zeros((2, self.obs_grid_size, self.obs_grid_size))
        
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        # Exclude self from neighbor marking in the grid
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

        return observation  # Shape: (2, obs_grid_size, obs_grid_size)

    def get_behavior_metrics(self, fish_index):
        """
        Computes three behavior metrics for branch B input:
          Alignment (phi):
            phi = 1/(N_neighbors) * norm( sum_{i in neighbors} [cos(theta_i), sin(theta_i)] )
          Cohesion (xi):
            xi = (# perceived neighbors) / (total number of fish)
          Density (ANNR):
            D_actual = average distance to all neighbors (excluding self; default = perception_range if none)
            D_expected = 0.5 / sqrt(total_agents / (pi * perception_range^2))
            annr = D_actual / D_expected
        Neighbors here are all fish (including self for alignment).
        """
        focal_pos = self.positions[fish_index]
        focal_orientation = self.orientations[fish_index]
        
        # For metrics, consider neighbors within perception range.
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        # For alignment and cohesion, include fish whose distance is <= perception_range.
        neighbors_idx = np.where(distances <= self.perception_range)[0]
        N_neighbors = len(neighbors_idx)
        
        # Alignment: sum normalized velocity vectors (derived from orientations)
        sum_norm_vel = np.zeros(2)
        for idx in neighbors_idx:
            theta = self.orientations[idx]
            norm_vec = np.array([np.cos(theta), np.sin(theta)])
            sum_norm_vel += norm_vec
        phi = np.linalg.norm(sum_norm_vel) / N_neighbors if N_neighbors > 0 else 0.0

        # Cohesion: fraction of fish perceived (count neighbors divided by total fish)
        xi = N_neighbors / self.num_fish

        # Density: compute average distance to neighbors (excluding self)
        other_idx = np.setdiff1d(neighbors_idx, np.array([fish_index]))
        if len(other_idx) > 0:
            D_actual = np.mean(distances[other_idx])
        else:
            D_actual = self.perception_range

        A = np.pi * (self.perception_range ** 2)
        D_expected = 0.5 / np.sqrt(self.num_fish / A)
        annr = D_actual / D_expected

        return np.array([phi, xi, annr])  # 3-dimensional feature vector

    def get_mean_action(self, fish_index, actions):
        """
        Previously, this method returned the average action of the neighbors.
        Now, we update it to return the behavior metrics vector for the fish.
        """
        return self.get_behavior_metrics(fish_index)

    def action_to_steering(self, action_index):
        """Maps discrete action index to a continuous steering change in [-omega_max, omega_max]."""
        return np.interp(action_index, [0, self.num_actions - 1], [-self.omega_max, self.omega_max])

    def step(self, action_indices):
        """Update fish positions using discrete actions (converted to continuous steering changes)."""
        steering_changes = np.array([self.action_to_steering(a) for a in action_indices])
        self.orientations += steering_changes * self.dt
        dx = self.velocity * np.cos(self.orientations) * self.dt
        dy = self.velocity * np.sin(self.orientations) * self.dt

        # Wrap around grid boundaries
        self.positions[:, 0] = (self.positions[:, 0] + dx) % self.grid_size
        self.positions[:, 1] = (self.positions[:, 1] + dy) % self.grid_size

    def get_reward(self, fish_index):
        """
        Computes reward:
           R = 0.1 * (# perceived neighbors) - 0.1 * (# collisions)
        A collision is defined as when the distance between two fish is less than a threshold.
        """
        focal_pos = self.positions[fish_index]
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        # Perceived neighbors: exclude self (distance > 0) and within perception range.
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]
        num_neighbors = len(neighbors)

        # Collision threshold: set to 1.0 (this can be adjusted if needed)
        collision_threshold = 1.0
        collisions = np.where((distances < collision_threshold) & (distances > 0))[0]
        num_collisions = len(collisions)

        reward = 0.1 * num_neighbors - 0.1 * num_collisions
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


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, behavior, action, reward, next_state, next_behavior, done):
        self.buffer.append((state, behavior, action, reward, next_state, next_behavior, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, behavior, action, reward, next_state, next_behavior, done = map(
            lambda x: list(x), zip(*batch)
        )
        return state, behavior, action, reward, next_state, next_behavior, done
    
    def __len__(self):
        return len(self.buffer)
    

