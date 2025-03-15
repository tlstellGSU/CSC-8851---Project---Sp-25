from environment import FishSchoolEnv
from mean_field_q import MeanFieldQLearningAgent
import numpy as np

# Load trained agent
env = FishSchoolEnv(num_fish=50)
agent = MeanFieldQLearningAgent(state_dim=3, action_dim=3)

# Run simulation with trained agent
for step in range(500):  # Run for 500 steps
    states = [env.get_state(i) for i in range(env.num_fish)]
    actions = np.array([agent.select_action(s) for s in states])
    env.step(actions)

env.render()  # Display fish schooling behavior

