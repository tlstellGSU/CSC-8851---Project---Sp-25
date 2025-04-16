import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import numpy as np
import torch

from model_TS import FishSchoolEnv
from model_TS import MultiAgentNet  # Make sure your model class is in q_network.py

# Parameters (make sure these match your training setup)
NUM_FISH = 50
GRID_SIZE = 60
OBS_GRID_SIZE = 16
VELOCITY = 3
PERCEPTION_RANGE = 15
NUM_ACTIONS = 5

# Load trained Q-network
q_network = MultiAgentNet(
    spatial_channels=2, 
    orientation_dim=4, 
    action_reward_dim=2, 
    num_actions=NUM_ACTIONS, 
    obs_size=OBS_GRID_SIZE
)

q_network.load_state_dict(torch.load("mean_field_q_network.pth"))
q_network.eval()

# Initialize environment
env = FishSchoolEnv(num_fish=NUM_FISH, 
                    grid_size=GRID_SIZE, 
                    velocity=VELOCITY, 
                    perception_range=PERCEPTION_RANGE, 
                    obs_grid_size=OBS_GRID_SIZE)

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_title(f"Fish Schooling Simulation - Frame {frame}")

    actions = []

    for i in range(env.num_fish):
        # Get the state of fish i
        observation = env.get_state(i)
        print(f"State for fish {i}: {observation.shape}")  # Debugging line
        
        # Split the observation into spatial and orientation components
        spatial_obs = observation[0]  # Shape: (obs_grid_size, obs_grid_size)
        orientation_obs = observation[1]  # Shape: (obs_grid_size, obs_grid_size)

        # Convert each part into tensors
        spatial_tensor = torch.FloatTensor(spatial_obs).unsqueeze(0)  # shape (1, 1, obs_grid_size, obs_grid_size)
        orientation_tensor = torch.FloatTensor(orientation_obs).unsqueeze(0)  # shape (1, 1, obs_grid_size, obs_grid_size)

        # Concatenate spatial and orientation tensors along the channel dimension
        input_tensor = torch.cat([spatial_tensor, orientation_tensor], dim=1)  # Shape: (1, 2, obs_grid_size, obs_grid_size)

        # You may need to change how you pass action_reward if it's not part of the state
        action_reward_tensor = torch.zeros(1, 2)  # Action-reward input can be handled as zeros or based on your model

        # Pass each tensor as an input to the network
        with torch.no_grad():
            q_vals = q_network(input_tensor, action_reward_tensor)

        # Choose action based on max Q-value
        action = q_vals.argmax().item()

        actions.append(action)

    # Update environment with the chosen actions
    env.step(actions)

    # Plot positions of fish in the environment
    ax.scatter(env.positions[:, 0], env.positions[:, 1], c='blue', marker='o')

# Create animation
global anim
anim = animation.FuncAnimation(fig, update, frames=200, interval=50)

# Display animation in notebook
html_anim = HTML(anim.to_jshtml())
display(html_anim)
