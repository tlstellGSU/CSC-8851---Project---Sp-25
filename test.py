from environment import FishSchoolEnv
import numpy as np

# Initialize environment
env = FishSchoolEnv(num_fish=50)  # Create environment with 50 fish

# Select a fish to test
fish_id = 0  # Test on the first fish

# Get observation
obs = env.get_state(fish_id)

# Print observation details
print("Observation shape:", obs.shape)  # Should print (2, 16, 16)
print("\nChannel 1 (Binary Positions):")
print(obs[0])  # Show presence of neighbors

print("\nChannel 2 (Relative Orientations):")
print(obs[1])  # Show relative orientations
