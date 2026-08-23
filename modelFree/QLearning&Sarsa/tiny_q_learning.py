# The environment is:
# S0 -- S1 -- S2 -- S3 -- GOAL
# 0 = left
# 1 = right

# It gets reward +1 only when it reaches the goal.

import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Small environment
# -----------------------------

# States:
# 0, 1, 2, 3, 4
# State 4 is the goal
num_states = 9
goal_state = 8

# Actions:
# 0 = left
# 1 = right
num_actions = 2

# Q-table: rows = states, columns = actions
Q = np.zeros((num_states, num_actions))

# -----------------------------
# Hyperparameters
# -----------------------------

alpha = 0.5        # learning rate
gamma = 0.95       # discount factor
epsilon = 1.0      # exploration probability
epsilon_decay = 0.999
epsilon_min = 0.05

episodes = 4000
max_steps = 20

rewards_per_episode = []

n_goal_reach = 0


# -----------------------------
# Environment step function
# -----------------------------

def step(state, action):
    """
    Given current state and action, return:
    next_state, reward, done
    """
    global n_goal_reach
    
    if action == 0:  # move left
        next_state = max(0, state - 1)
    else:            # move right
        next_state = min(goal_state, state + 1)

    if next_state == goal_state:
        n_goal_reach += 1
        reward = 1
        done = True
    else:
        reward = 0
        done = False

    return next_state, reward, done


# -----------------------------
# Training loop
# -----------------------------

for episode in range(episodes):
    state = 0
    total_reward = 0

    for step_idx in range(max_steps):

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-learning update rule
        best_next_action_value = np.max(Q[next_state])

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * best_next_action_value - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break
        
        # print(Q)

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
plt.title("Q-learning training progress")
plt.show()