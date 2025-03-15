# import numpy as np
# import matplotlib.pyplot as plt

# class FishSchoolEnv:
#     def __init__(self, num_fish=100, grid_size=60, velocity=3, omega_max=np.pi/3, dt=1):
#         """
#         Initialize the fish schooling environment.
        
#         :param num_fish: Number of fish agents
#         :param grid_size: The size of the 2D grid (Body Lengths)
#         :param velocity: Constant linear velocity (BL/s)
#         :param omega_max: Maximum turning angle (radians)
#         :param dt: Time step for updates
#         """
#         self.num_fish = num_fish
#         self.grid_size = grid_size
#         self.velocity = velocity
#         self.omega_max = omega_max
#         self.dt = dt  # Time step

#         # Initialize fish positions randomly within the grid
#         self.positions = np.random.rand(num_fish, 2) * grid_size  # (x, y)
#         self.orientations = np.random.uniform(0, 2*np.pi, num_fish)  # Random orientations

#     def step(self, actions):
#         """
#         Update the fish positions based on the given actions (angular velocities).
        
#         :param actions: Array of angular velocity changes for each fish
#         """
#         # Update orientations
#         self.orientations += actions * self.omega_max * self.dt
        
#         # Compute new positions using the unicycle model
#         dx = self.velocity * np.cos(self.orientations) * self.dt
#         dy = self.velocity * np.sin(self.orientations) * self.dt

#         # Update positions with periodic boundary conditions (wrapping around)
#         self.positions[:, 0] = (self.positions[:, 0] + dx) % self.grid_size
#         self.positions[:, 1] = (self.positions[:, 1] + dy) % self.grid_size

#     def render(self):
#         """Render the fish positions using Matplotlib."""
#         plt.figure(figsize=(6,6))
#         plt.xlim(0, self.grid_size)
#         plt.ylim(0, self.grid_size)
#         plt.scatter(self.positions[:, 0], self.positions[:, 1], c='blue', marker='o')
#         plt.title("Fish Schooling Simulation")
#         plt.show()

# # Example Usage
# if __name__ == "__main__":
#     env = FishSchoolEnv(num_fish=50)  # Initialize environment
#     for _ in range(100):  # Run 100 time steps
#         actions = np.random.uniform(-1, 1, env.num_fish)  # Random angular velocity changes
#         env.step(actions)
#         env.render()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FishSchoolEnv:
    def __init__(self, num_fish=100, grid_size=60, velocity=3, omega_max=np.pi/3, dt=1):
        """
        Initialize the fish schooling environment.
        
        :param num_fish: Number of fish agents
        :param grid_size: The size of the 2D grid (Body Lengths)
        :param velocity: Constant linear velocity (BL/s)
        :param omega_max: Maximum turning angle (radians)
        :param dt: Time step for updates
        """
        self.num_fish = num_fish
        self.grid_size = grid_size
        self.velocity = velocity
        self.omega_max = omega_max
        self.dt = dt  # Time step

        # Initialize fish positions randomly within the grid
        self.positions = np.random.rand(num_fish, 2) * grid_size  # (x, y)
        self.orientations = np.random.uniform(0, 2*np.pi, num_fish)  # Random orientations

    def step(self, actions):
        """
        Update the fish positions based on the given actions (angular velocities).
        
        :param actions: Array of angular velocity changes for each fish
        """
        # Update orientations
        self.orientations += actions * self.omega_max * self.dt
        
        # Compute new positions using the unicycle model
        dx = self.velocity * np.cos(self.orientations) * self.dt
        dy = self.velocity * np.sin(self.orientations) * self.dt

        # Update positions with periodic boundary conditions (wrapping around)
        self.positions[:, 0] = (self.positions[:, 0] + dx) % self.grid_size
        self.positions[:, 1] = (self.positions[:, 1] + dy) % self.grid_size

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
    env.render()
