import gymnasium as gym
import numpy as np

# Environment: FrozenLake (deterministic, 4x4 grid)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)  # is_slippery=False makes it deterministic
n_states = env.observation_space.n   # 16
n_actions = env.action_space.n       # 4

# Step 1: Collect random experience to learn the model
n_episodes_collect = 1000
max_steps = 100

# Model storage: count transitions and rewards
transitions = np.zeros((n_states, n_actions, n_states))  # count s' from (s,a)
rewards = np.zeros((n_states, n_actions))                # sum of rewards from (s,a)
counts = np.zeros((n_states, n_actions))                 # how many times (s,a) seen

np.random.seed(42)
env.reset(seed=42)

for _ in range(n_episodes_collect):
    state, _ = env.reset()
    for _ in range(max_steps):
        action = env.action_space.sample()  # random action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update counts
        transitions[state, action, next_state] += 1
        rewards[state, action] += reward
        counts[state, action] += 1
        
        state = next_state
        if done:
            break

# Estimate model
# Avoid division by zero
eps = 1e-6
P = transitions / (counts[..., np.newaxis] + eps)  # P(s'|s,a)
R = rewards / (counts + eps)                       # Expected R(s,a)

# Step 2: Value Iteration on the learned model (planning)
gamma = 0.99
theta = 1e-6
V = np.zeros(n_states)

while True:
    delta = 0
    for s in range(n_states):
        v_old = V[s]
        # Bellman optimality update using model
        q_sa = np.zeros(n_actions)
        for a in range(n_actions):
            # Expected value: R + gamma * sum P(s'|s,a) * V(s')
            q_sa[a] = R[s, a] + gamma * np.sum(P[s, a] * V)
        V[s] = np.max(q_sa)
        delta = max(delta, abs(v_old - V[s]))
    if delta < theta:
        break

# Extract optimal policy from V
policy = np.zeros(n_states, dtype=int)
for s in range(n_states):
    q_sa = np.zeros(n_actions)
    for a in range(n_actions):
        q_sa[a] = R[s, a] + gamma * np.sum(P[s, a] * V)
    policy[s] = np.argmax(q_sa)

# Step 3: Test the learned policy
def test_policy(n_episodes=100):
    total_reward = 0
    success = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        if state == 15:  # Goal state
            success += 1
    return total_reward / n_episodes, success / n_episodes

avg_reward, success_rate = test_policy()
print(f"Average reward: {avg_reward:.3f}")
print(f"Success rate (reaching goal): {success_rate:.2%}")