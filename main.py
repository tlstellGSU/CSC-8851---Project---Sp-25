import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from environment import FishSchoolEnv
from mean_field_q import MeanFieldQLearningAgent
import numpy as np

# Load trained agent
env = FishSchoolEnv(num_fish=10)
agent = MeanFieldQLearningAgent(state_dim=16, action_dim=3)

# Run simulation with trained agent
for step in range(500):  # Run for 500 steps
    states = [env.get_state(i) for i in range(env.num_fish)]
    actions = np.array([agent.select_action(s) for s in states])
    env.step(actions)

env.render()  # Display fish schooling behavior

# ✅ Ensure animation is stored persistently
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_title(f"Fish Schooling Simulation - Frame {frame}")

    # Get new positions
    states = [env.get_state(i) for i in range(env.num_fish)]
    actions = np.array([agent.select_action(s) for s in states])
    env.step(actions)

    # Assuming env.positions contains fish coordinates
    ax.scatter(env.positions[:, 0], env.positions[:, 1], c='blue', marker='o')

# ✅ Store animation in a variable to prevent deletion
global anim  
anim = animation.FuncAnimation(fig, update, frames=200, interval=50)

# ✅ Display the animation properly
html_anim = HTML(anim.to_jshtml())
display(html_anim)