import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import numpy as np
import torch

from model_TS import FishSchoolEnv
from model_TS import MeanFieldQNetwork  # Make sure your model class is in q_network.py

# Parameters (make sure these match your training setup)
NUM_FISH = 50
GRID_SIZE = 60
OBS_GRID_SIZE = 16
VELOCITY = 3
PERCEPTION_RANGE = 15
NUM_ACTIONS = 5

# Load trained Q-network
q_network = MeanFieldQNetwork(state_dim=512, action_dim=NUM_ACTIONS)
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
        obs = env.get_state(i)
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)  # (1, state_dim)
        mean_act = env.get_mean_action(i, np.zeros(env.num_fish))  # You can replace with real actions if available
        mean_act_tensor = torch.FloatTensor([[mean_act]])  # (1, 1)

        input_tensor = torch.cat([obs_tensor, mean_act_tensor], dim=1)  # (1, state_dim + 1)
        with torch.no_grad():
            q_vals = q_network(input_tensor)
        action = q_vals.argmax().item()

        actions.append(action)

    env.step(actions)
    ax.scatter(env.positions[:, 0], env.positions[:, 1], c='blue', marker='o')

    
# Create animation
global anim
anim = animation.FuncAnimation(fig, update, frames=200, interval=50)

# Display animation in notebook
html_anim = HTML(anim.to_jshtml())
display(html_anim)
