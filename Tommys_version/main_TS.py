import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

from model_TS import FishSchoolEnv, MultiAgentNet

# -------------------------------
# 1. Load trained Q‑network
# -------------------------------
NUM_ACTIONS = 137   # or 137, whatever you trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = MultiAgentNet(num_actions=NUM_ACTIONS).to(device)
q_network.load_state_dict(torch.load("mean_field_q_network.pth", map_location=device))
q_network.eval()

# -------------------------------
# 2. Instantiate environment
# -------------------------------
env = FishSchoolEnv(
    num_fish=50,
    grid_size=60,
    velocity=3,
    perception_range=3,
    #perception_range=np.random.normal(15, 2),
    obs_grid_size=16,
    num_actions=NUM_ACTIONS
)

# -------------------------------
# 3. Prepare trails
# -------------------------------
trail_length = 3  # how many past positions to show
# one deque per fish, storing (x, y)
trails = [deque(maxlen=trail_length) for _ in range(env.num_fish)]
# initialize with the starting positions
for i in range(env.num_fish):
    x0, y0 = env.positions[i]
    for _ in range(trail_length):
        trails[i].append((x0, y0))

# -------------------------------
# 4. Set up matplotlib figure
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, env.grid_size)
ax.set_ylim(0, env.grid_size)
ax.set_aspect('equal')

# initial quiver (arrows)
xs = env.positions[:, 0]
ys = env.positions[:, 1]
Us = np.cos(env.orientations)
Vs = np.sin(env.orientations)
quiv = ax.quiver(xs, ys, Us, Vs,
                 angles='xy', scale_units='xy', scale=1.5,
                 width=0.005, headwidth=3, headlength=5)

# one Line2D per fish for its trail
lines = []
for _ in range(env.num_fish):
    line, = ax.plot([], [], lw=1, alpha=0.6)
    lines.append(line)

# -------------------------------
# 5. Animation update function
# -------------------------------
def update(frame_num):
    # 5A. Choose actions for all fish
    actions = np.zeros(env.num_fish, dtype=int)
    for i in range(env.num_fish):
        obs = env.get_state(i)                               # (2,16,16)
        behav = env.get_mean_action(i, actions)              # (3,)
        branchB = np.pad(behav, (0,1), mode='constant')      # (4,)
        branchC = np.zeros(2, dtype=float)

        s_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        bB_t = torch.FloatTensor(branchB).unsqueeze(0).to(device)
        bC_t = torch.FloatTensor(branchC).unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals = q_network(s_t, bB_t, bC_t)
            actions[i] = q_vals.argmax(dim=1).item()

    # 5B. Step the environment
    env.step(actions)

    # 5C. Update trails & line plots
    for i, line in enumerate(lines):
        x_i, y_i = env.positions[i]
        
        # Only add to trail if movement is not too large
        if trails[i]:
            prev_x, prev_y = trails[i][-1]
            dist = np.hypot(x_i - prev_x, y_i - prev_y)
            if dist <= env.grid_size / 2:
                trails[i].append((x_i, y_i))
            else:
                trails[i].clear()
                trails[i].append((x_i, y_i))
        else:
            trails[i].append((x_i, y_i))

        xs_trail, ys_trail = zip(*trails[i])
        line.set_data(xs_trail, ys_trail)
    # 5D. Update quiver arrows
    xs = env.positions[:, 0]
    ys = env.positions[:, 1]
    Us = np.cos(env.orientations)
    Vs = np.sin(env.orientations)
    quiv.set_offsets(np.stack([xs, ys], axis=1))
    quiv.set_UVC(Us, Vs)

    return lines + [quiv]

# -------------------------------
# 6. Run the animation
# -------------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    frames=3000,
    interval=1000,
    blit=True
)

plt.tight_layout()
plt.show()
