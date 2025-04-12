import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FishSchoolEnv:
    def __init__(self, num_fish=100, grid_size=60, velocity=3, omega_max=np.pi/3, dt=1, perception_range=5):
        """
        Initialize the fish schooling environment with MARL components.
        """
        self.num_fish = num_fish
        self.grid_size = grid_size
        self.velocity = velocity
        self.omega_max = omega_max
        self.dt = dt  # Time step
        self.perception_range = 10 # How far a fish can "see"

        # Initialize fish positions and orientations randomly
        self.positions = np.random.rand(num_fish, 2) * grid_size  # (x, y)
        self.orientations = np.random.uniform(0, 2*np.pi, num_fish)  # Random orientations

    def get_state(self, fish_index):
        """
        Get the local observation (state) of a specific fish.
        """
        focal_pos = self.positions[fish_index]
        focal_orientation = self.orientations[fish_index]

        # Find neighboring fish within the perception range
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        neighbors = np.where((distances < self.perception_range) & (distances > 0))[0]  # Exclude self
        
        # Get relative positions and orientations of neighbors
        neighbor_positions = self.positions[neighbors] - focal_pos  # Relative positions
        neighbor_orientations = self.orientations[neighbors] - focal_orientation  # Relative angles

        # Normalize the state (limit perception range)
        state = {
            "relative_positions": neighbor_positions / self.perception_range,  # Scale between -1 and 1
            "relative_orientations": neighbor_orientations / np.pi  # Scale between -1 and 1
        }
        return state

    def step(self, actions):
        """
        Update the fish positions based on actions.
        """
        self.orientations += actions * self.omega_max * self.dt  # Update orientations

        # Compute new positions using unicycle model
        dx = self.velocity * np.cos(self.orientations) * self.dt
        dy = self.velocity * np.sin(self.orientations) * self.dt

        # Update positions with periodic boundary conditions
        self.positions[:, 0] = (self.positions[:, 0] + dx) % self.grid_size
        self.positions[:, 1] = (self.positions[:, 1] + dy) % self.grid_size

    def get_reward(self, fish_index):
        """
        Compute the reward for a given fish.
        +Reward for having more neighbors (cohesion).
        -Penalty for collisions (overcrowding).
        """
        focal_pos = self.positions[fish_index]
        distances = np.linalg.norm(self.positions - focal_pos, axis=1)
        num_neighbors = np.sum((distances < self.perception_range) & (distances > 0))  # Exclude self

        # Define reward function
        reward = num_neighbors * 0.1  # More neighbors, higher reward
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
            actions = np.random.uniform(-1, 1, self.num_fish)  # Random actions
            self.step(actions)  # Move fish
            scatter.set_offsets(self.positions)  # Update fish positions

        ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
        plt.show()

# Run the environment
if __name__ == "__main__":
    env = FishSchoolEnv(num_fish=50)  # Initialize environment
    
    for i in range(10):  # Run 10 time steps
        for fish_idx in range(env.num_fish):
            state = env.get_state(fish_idx)
            reward = env.get_reward(fish_idx)
            print(f"Fish {fish_idx}: Reward={reward:.2f}, State={state}")
        
        actions = np.random.uniform(-1, 1, env.num_fish)  # Random actions
        env.step(actions)
    
    env.render()
