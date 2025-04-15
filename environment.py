import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FishSchoolEnv:
    # def __init__(self, num_fish=100, grid_size=60, velocity=3, omega_max=np.pi/3, dt=1, perception_range=10):
    #     self.num_fish = num_fish
    #     self.grid_size = grid_size
    #     self.velocity = velocity
    #     self.omega_max = omega_max
    #     self.dt = dt
    #     self.perception_range = perception_range

    #     # Initialize fish positions and orientations
    #     self.positions = np.random.rand(num_fish, 2) * grid_size
    #     self.orientations = np.random.uniform(0, 2*np.pi, num_fish)

       # class FishSchoolEnv:
    def __init__(self, 
                 num_fish=100, 
                 grid_size=60, 
                 velocity=3, 
                 omega_max=np.pi/3, 
                 dt=1, 
                 #perception_range=np.random.normal(15, 2), 
                 perception_range=15,
                 obs_grid_size=16):
        self.num_fish = num_fish
        self.grid_size = grid_size
        self.velocity = velocity
        self.omega_max = omega_max
        self.dt = dt
        self.perception_range = perception_range
        self.obs_grid_size = obs_grid_size  # Size of local observation grid (16x16)

        # Initialize fish positions and orientations
        self.positions = np.random.rand(num_fish, 2) * grid_size
        self.orientations = np.random.uniform(0, 2 * np.pi, num_fish)

    def get_state(self, fish_index):
        """
        Generate an image-like observation for the given fish.
        - 2D Grid with 2 Channels:
            - Channel 1: Binary positions of nearby fish.
            - Channel 2: Relative orientations of neighbors.
        """
        focal_pos = self.positions[fish_index]
        focal_orientation = self.orientations[fish_index]

        # Initialize 2-channel observation grid
        observation = np.zeros((2, self.obs_grid_size, self.obs_grid_size))  # Shape: (2, 16, 16)
        
        # Get neighboring fish within perception range
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]

        for neighbor in neighbors:
            neighbor_pos = self.positions[neighbor] - focal_pos  # Relative position
            neighbor_orientation = self.orientations[neighbor]

            # Rotate relative position to align with the focal fish's orientation
            dx, dy = neighbor_pos
            rotated_x = dx * np.cos(-focal_orientation) - dy * np.sin(-focal_orientation)
            rotated_y = dx * np.sin(-focal_orientation) + dy * np.cos(-focal_orientation)

            # Convert rotated position into grid coordinates
            grid_x = int((rotated_x / self.perception_range) * (self.obs_grid_size // 2)) + (self.obs_grid_size // 2)
            grid_y = int((rotated_y / self.perception_range) * (self.obs_grid_size // 2)) + (self.obs_grid_size // 2)

            # Ensure grid_x and grid_y are within bounds
            if 0 <= grid_x < self.obs_grid_size and 0 <= grid_y < self.obs_grid_size:
                observation[0, grid_x, grid_y] = 1  # Mark presence in binary grid
                observation[1, grid_x, grid_y] = (neighbor_orientation - focal_orientation) / np.pi  # Relative orientation

        return observation  # Returns a (2, 16, 16) array

    def get_mean_action(self, fish_index, actions):
        """Compute mean action of neighboring fish."""
        focal_pos = self.positions[fish_index]
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]

        return np.mean(actions[neighbors]) if len(neighbors) > 0 else 0

    def step(self, actions):
        """Update fish positions using their actions."""
        mean_actions = np.array([self.get_mean_action(i, actions) for i in range(self.num_fish)])
        self.orientations += mean_actions * self.omega_max * self.dt

        dx = self.velocity * np.cos(self.orientations) * self.dt
        dy = self.velocity * np.sin(self.orientations) * self.dt

        self.positions[:, 0] = (self.positions[:, 0] + dx) % self.grid_size
        self.positions[:, 1] = (self.positions[:, 1] + dy) % self.grid_size

    # def get_state(self, fish_index):
    #     """Get local observation for Mean Field Q-Learning."""
    #     focal_pos = self.positions[fish_index]
    #     distances = np.linalg.norm(self.positions - focal_pos, axis=1)
    #     neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]

    #     if len(neighbors) > 0:
    #         mean_neighbor_pos = np.mean(self.positions[neighbors], axis=0) - focal_pos
    #         mean_neighbor_orientation = np.mean(self.orientations[neighbors])
    #     else:
    #         mean_neighbor_pos = np.array([0, 0])
    #         mean_neighbor_orientation = 0

    #     return np.concatenate([mean_neighbor_pos / self.grid_size, [mean_neighbor_orientation / np.pi]])
    
    def get_reward(self, fish_index):
        """
        Compute the reward for a given fish.
        +Reward for having more neighbors (cohesion).
        -Penalty for collisions (overcrowding).  
        """
        focal_pos = self.positions[fish_index]
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
    
    # Count the number of neighbors within the perception range
        num_neighbors = np.sum((distances < self.perception_range) & (distances > 0))  # Exclude self

    # Define reward function
        alignment = np.mean(np.cos(self.orientations[num_neighbors] - self.orientations[fish_index]))
        cohesion = np.linalg.norm(np.mean(self.positions[num_neighbors], axis=0) - self.positions[fish_index])
        reward = 0.5 * alignment + 0.5 * cohesion

        #reward = num_neighbors * 0.5  # More neighbors, higher reward
        if num_neighbors > 10:  # If too many, penalize (collision-like behavior)
            reward -= 1.0

        return reward
    
    def render(self):
        """Render the fish positions dynamically using Matplotlib animation."""
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        scatter = ax.scatter(self.positions[:, 0], self.positions[:, 1], c='blue', marker='o')

        def update(frame):
            actions = np.zeros(self.num_fish)  # Use trained agent's actions here in future
            self.step(actions)  # Move fish
            scatter.set_offsets(self.positions)  # Update fish positions

        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        plt.show()


