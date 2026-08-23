import numpy as np
import random
import matplotlib.pyplot as plt

# environment

num_states = 5
goal_state = 4

# Actions:
# 0 = left
# 1 = right
num_actions = 2

# Q-table: rows = states, columns = actions
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha         = 0.1        # learning rate
gamma         = 0.95       # discount factor
epsilon       = 1.0        # exploration probability
epsilon_decay = 0.995
epsilon_min   = 0.05

episodes = 500
max_steps = 20

rewards_per_episode = []


# Environment step function

def step(state, action):
    """
    Given current state and action, return:
    next_state, reward, done
    """

    if action == 0:  # move left
        next_state = max(0, state - 1)
    else:            # move right
        next_state = min(goal_state, state + 1)

    if next_state == goal_state:
        reward = 1
        done = True
    else:
        reward = 0
        done = False

    return next_state, reward, done


# -----------------------------
# Epsilon-greedy policy
# -----------------------------

def choose_action(state, epsilon):
    """
    Choose an action using epsilon-greedy policy.
    """

    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(Q[state])


# -----------------------------
# Training loop: SARSA
# -----------------------------

for episode in range(episodes):
    state = 0
    total_reward = 0

    # SARSA chooses the first action before entering the loop
    action = choose_action(state, epsilon)

    for step_idx in range(max_steps):

        next_state, reward, done = step(state, action)

        # Choose next action using the current policy
        next_action = choose_action(next_state, epsilon)

        # SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action

        total_reward += reward

        if done:
            break

    # Reduce exploration over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_per_episode.append(total_reward)


# -----------------------------
# Results
# -----------------------------

print("Learned Q-table:")
print(Q)

print("\nBest action for each state:")
for state in range(num_states):
    best_action = np.argmax(Q[state])

    if state == goal_state:
        print(f"State {state}: GOAL")
    elif best_action == 0:
        print(f"State {state}: move LEFT")
    else:
        print(f"State {state}: move RIGHT")


# -----------------------------
# Test learned policy
# -----------------------------

print("\nTesting learned policy:")

state = 0
path = [state]

for _ in range(max_steps):
    action = np.argmax(Q[state])
    next_state, reward, done = step(state, action)

    path.append(next_state)
    state = next_state

    if done:
        break

print("Path followed by agent:")
print(path)


# -----------------------------
# Plot rewards
# -----------------------------

plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SARSA training progress")
plt.show()