import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from model_TS import FishSchoolEnv
from model_TS import MultiAgentNet  # Your pre-defined network with three branches.
from model_TS import ReplayBuffer  # Your replay buffer

NUM_ACTIONS = 5
ACTION_DIM = NUM_ACTIONS

env = FishSchoolEnv(num_fish=50, grid_size=60, velocity=3, perception_range=15, obs_grid_size=16, num_actions=NUM_ACTIONS)

# Instantiate networks.
q_network = MultiAgentNet()  # Uses your defined three-branch architecture.
target_network = MultiAgentNet()

# Fix the branch-A linear layer (spatial_fc) so that it has the proper input size.
# The spatial observation is of shape (2, 16, 16) => flattened to 512.
q_network.spatial_fc[1] = nn.Linear(512, 256)
target_network.spatial_fc[1] = nn.Linear(512, 256)

target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(capacity=10000)

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
num_episodes = 500
batch_size = 64
gamma = 0.99

for episode in range(num_episodes):
    # Lists for storing the three components of state and next state.
    spatial_states = []
    branchB_states = []  # Behavior metrics (padded to size 4)
    branchC_states = []  # Dummy values (size 2)
    actions = np.zeros(env.num_fish, dtype=int)
    spatial_next_states = []
    branchB_next_states = []
    branchC_next_states = []

    episode_loss = 0

    # Choose actions for each fish.
    for fish_index in range(env.num_fish):
        obs = env.get_state(fish_index)  # shape: (2, 16, 16)
        behav_feat = env.get_mean_action(fish_index, actions)  # returns a 3-element vector
        # Pad behavior metrics to length 4 for branch B.
        branchB_input = np.pad(behav_feat, (0, 1), mode='constant')   # shape: (4,)
        # For branch C, supply a dummy vector (size 2).
        branchC_input = np.zeros(2, dtype=float)

        # Build input tensors for the Q-network.
        spatial_tensor = torch.FloatTensor(obs).unsqueeze(0)         # (1, 2, 16, 16)
        branchB_tensor = torch.FloatTensor(branchB_input).unsqueeze(0) # (1, 4)
        branchC_tensor = torch.FloatTensor(branchC_input).unsqueeze(0) # (1, 2)

        # Epsilon-greedy policy.
        if np.random.rand() < epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            with torch.no_grad():
                q_vals = q_network(spatial_tensor, branchB_tensor, branchC_tensor)
                action = q_vals.argmax().item()

        actions[fish_index] = action
        spatial_states.append(obs)
        branchB_states.append(branchB_input)
        branchC_states.append(branchC_input)

    # Apply all fish actions at once.
    env.step(actions)

    # Store transitions for each fish.
    for fish_index in range(env.num_fish):
        next_obs = env.get_state(fish_index)
        next_behav_feat = env.get_mean_action(fish_index, actions)
        next_branchB_input = np.pad(next_behav_feat, (0, 1), mode='constant')
        next_branchC_input = np.zeros(2, dtype=float)
        reward = env.get_reward(fish_index)

        spatial_next_states.append(next_obs)
        branchB_next_states.append(next_branchB_input)
        branchC_next_states.append(next_branchC_input)

        replay_buffer.push(
            (spatial_states[fish_index], branchB_states[fish_index], branchC_states[fish_index]),   # state tuple
            branchB_states[fish_index],   # mean_action (branch B features for current state)
            actions[fish_index],
            reward,
            (spatial_next_states[fish_index], branchB_next_states[fish_index], branchC_next_states[fish_index]),  # next_state tuple
            branchB_next_states[fish_index],   # next_mean_action (branch B for next state)
            done=1
        )

    # Update Q-network from replay buffer.
    if len(replay_buffer) > batch_size:
        # The replay buffer now returns a tuple for state and next_state in the form:
        # (spatial, branchB, branchC), action, reward, (spatial, branchB, branchC), done.
        states, mean_actions, actions_batch, rewards_batch, next_states, next_mean_actions, dones_batch = replay_buffer.sample(batch_size)
        
        # Unpack the state components.
        spatial_batch = torch.FloatTensor(np.array([s[0] for s in states]))  # (batch, 2, 16, 16)
        branchB_batch = torch.FloatTensor(np.array([s[1] for s in states]))   # (batch, 4)
        branchC_batch = torch.FloatTensor(np.array([s[2] for s in states]))   # (batch, 2)

        next_spatial_batch = torch.FloatTensor(np.array([s[0] for s in next_states]))
        next_branchB_batch = torch.FloatTensor(np.array([s[1] for s in next_states]))
        next_branchC_batch = torch.FloatTensor(np.array([s[2] for s in next_states]))

        action_batch = torch.LongTensor(actions_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(rewards_batch)
        done_batch = torch.FloatTensor(dones_batch)

        # Q(s,a) from current network.
        q_values = q_network(spatial_batch, branchB_batch, branchC_batch)
        q_value = q_values.gather(1, action_batch).squeeze(1)

        # Target Q-values from target network.
        with torch.no_grad():
            next_q_values = target_network(next_spatial_batch, next_branchB_batch, next_branchC_batch)
            next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward_batch + gamma * next_q_value * (1 - done_batch)

        loss = F.mse_loss(q_value, expected_q_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()

    # Sync target network every 10 episodes.
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode+1} complete - loss: {episode_loss:.3f}")

torch.save(q_network.state_dict(), "mean_field_q_network.pth")
