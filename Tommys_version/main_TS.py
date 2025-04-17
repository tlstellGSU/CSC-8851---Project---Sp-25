import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model_TS import FishSchoolEnv, MultiAgentNet

# -------------------------------
# 1. Load trained Q‑network
# -------------------------------
NUM_ACTIONS = 137
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = MultiAgentNet(num_actions=NUM_ACTIONS).to(device)
q_network.load_state_dict(torch.load("mean_field_q_network.pth", map_location=device))
q_network.eval()

# -------------------------------
# 2. Instantiate environment
# -------------------------------
env = FishSchoolEnv(
    num_fish=200,
    grid_size=60,
    velocity=3,
    perception_range=15,
    obs_grid_size=16,
    num_actions=NUM_ACTIONS
)

# -------------------------------
# 3. Set up matplotlib figure
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, env.grid_size)
ax.set_ylim(0, env.grid_size)
ax.set_aspect('equal')

# Initial positions and directions
xs = env.positions[:, 0]
ys = env.positions[:, 1]
# Arrow components: unit vectors (we'll scale by a constant for visibility)
Us = np.cos(env.orientations)
Vs = np.sin(env.orientations)

# Create the quiver plot: arrows centered at (x,y), pointing (U,V)
quiv = ax.quiver(xs, ys, Us, Vs,
                 angles='xy', scale_units='xy', scale=1.5,
                 width=0.005, headwidth=3, headlength=5)

# -------------------------------
# 4. Animation update function
# -------------------------------
def update(frame_num):
    actions = np.zeros(env.num_fish, dtype=int)

    # 4A. Choose actions via Q-network
    for i in range(env.num_fish):
        obs = env.get_state(i)                                # (2,16,16)
        behav = env.get_mean_action(i, actions)               # (3,)
        branchB = np.pad(behav, (0, 1), mode='constant')      # (4,)
        branchC = np.zeros(2, dtype=float)                    

        s_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        bB_t = torch.FloatTensor(branchB).unsqueeze(0).to(device)
        bC_t = torch.FloatTensor(branchC).unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals = q_network(s_t, bB_t, bC_t)
            actions[i] = q_vals.argmax(dim=1).item()

    # 4B. Step environment
    env.step(actions)

    # 4C. Update arrow positions and directions
    xs = env.positions[:, 0]
    ys = env.positions[:, 1]
    Us = np.cos(env.orientations)
    Vs = np.sin(env.orientations)

    quiv.set_offsets(np.stack([xs, ys], axis=1))
    quiv.set_UVC(Us, Vs)

    return quiv,

# -------------------------------
# 5. Run the animation
# -------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=200,
    interval=100,
    blit=True
)

plt.tight_layout()
plt.show()
