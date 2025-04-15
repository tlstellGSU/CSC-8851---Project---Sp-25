import numpy as np
import torch
from model_TS import FishSchoolEnv
from model_TS import MeanFieldQNetwork  # assume you defined it elsewhere
from model_TS import ReplayBuffer  # your replay buffer
import torch.nn.functional as F

NUM_ACTIONS = 5
STATE_DIM = 512  # 2x16x16 = 512
ACTION_DIM = NUM_ACTIONS

env = FishSchoolEnv(num_fish=50, grid_size=60, velocity=3, perception_range=15, obs_grid_size=16, num_actions=NUM_ACTIONS)


q_network = MeanFieldQNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
target_network = MeanFieldQNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
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
    states = []
    actions = np.zeros(env.num_fish, dtype=int)
    mean_actions = np.zeros(env.num_fish)
    rewards = np.zeros(env.num_fish)
    next_states = []

    episode_loss = 0

    for fish_index in range(env.num_fish):
        obs = env.get_state(fish_index)
        mean_act = env.get_mean_action(fish_index, actions)
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)  # Shape: (1, 512)
        mean_act_tensor = torch.FloatTensor([mean_act])

        if np.random.rand() < epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            with torch.no_grad():
                input_tensor = torch.cat([obs_tensor, mean_act_tensor.unsqueeze(1)], dim=1)
                q_vals = q_network(input_tensor)                
                action = q_vals.argmax().item()

        actions[fish_index] = action
        states.append(obs)
        mean_actions[fish_index] = mean_act

    # Apply all fish actions at once
    env.step(actions)

    for fish_index in range(env.num_fish):
        next_obs = env.get_state(fish_index)
        next_mean_act = env.get_mean_action(fish_index, actions)
        reward = env.get_reward(fish_index)

        replay_buffer.push(
            states[fish_index],
            mean_actions[fish_index],
            actions[fish_index],
            reward,
            next_obs,
            next_mean_act,
            done=1  # always 1 for now
        )

    # Update Q-network from replay buffer
    if len(replay_buffer) > batch_size:
        states, mean_actions, actions, rewards, next_states, next_mean_actions, dones = replay_buffer.sample(batch_size)


        state_batch = torch.FloatTensor(np.array([s.flatten() for s in states]))        
        mean_action_batch = torch.FloatTensor(mean_actions).unsqueeze(1)
        action_batch = torch.LongTensor(actions).unsqueeze(1)
        reward_batch = torch.FloatTensor(rewards)
        next_state_batch = torch.FloatTensor(np.array([s.flatten() for s in next_states]))
        next_mean_action_batch = torch.FloatTensor(next_mean_actions).unsqueeze(1)
        done_batch = torch.FloatTensor(dones)


        # Q(s,a)
        input_tensor = torch.cat([state_batch, mean_action_batch], dim=1)
        q_values = q_network(input_tensor)
        q_value = q_values.gather(1, action_batch).squeeze(1)

        # max Q'(s', a')
        with torch.no_grad():
            next_input_tensor = torch.cat([next_state_batch, next_mean_action_batch], dim=1)
            next_q_values = target_network(next_input_tensor)
            next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward_batch + gamma * next_q_value * (1 - done_batch)

        loss = F.mse_loss(q_value, expected_q_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()

    # Sync target network every few episodes (optional)
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode+1} complete - loss: {episode_loss:.3f}")

torch.save(q_network.state_dict(), "mean_field_q_network.pth")