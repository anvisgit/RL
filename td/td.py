import gym
import numpy as np
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
episodes = 5000
env = gym.make("FrozenLake-v1", is_slippery=False)
num_states = env.observation_space.n
V = np.zeros(num_states)
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:# Random policy
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * V[next_state]
        td_error = td_target - V[state]
        V[state] += alpha * td_error
        state = next_state
print("Learned State Values\n")

for s in range(num_states):
    print(f"State {s:2d}: {V[s]:.4f}")

env.close()
