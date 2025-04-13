from environment import FishSchoolEnv
from mean_field_q import MeanFieldQLearningAgent
import numpy as np

# Initialize environment and agent
env = FishSchoolEnv(num_fish=50)
agent = MeanFieldQLearningAgent(state_dim=16, action_dim=180)  # 3 Actions (left, straight, right)

# Training loop
for episode in range(1000):
    states = [env.get_state(i) for i in range(env.num_fish)]
    actions = np.array([agent.select_action(s) for s in states])

    env.step(actions)

    next_states = [env.get_state(i) for i in range(env.num_fish)]
    rewards = [env.get_reward(i) for i in range(env.num_fish)]
    mean_actions = np.array([env.get_mean_action(i, actions) for i in range(env.num_fish)])

    for i in range(env.num_fish):
        #print(f"DEBUG: State {i} shape: {np.array(env.get_state(i)).shape}")
        agent.store_experience(states[i], actions[i], rewards[i], next_states[i], mean_actions[i])

    agent.update()

    agent.epsilon = max(0.05, agent.epsilon * 0.995)

    if episode % 50 == 0:
        print(f"Episode {episode} - Average Reward: {np.mean(rewards):.2f}")

print("Training complete!")


