# If you know only the environment, there is not one single value function yet.
# If you start knowing just the environment, the usual method is value iteration.

# You need to decide what you mean by “value”:

# Vπ(s) = value of state s under a specific policy π
# V*(s) = best possible value of state s under the optimal policy

# So:

# environment + policy  -> compute Vπ(s)
# environment only      -> compute V*(s) using value iteration


# The Bellman optimality equation is:

# V*(s) = max_a [ r + γ V*(s') ]

# Meaning:

# For each state:
#     try every possible action
#     look at the reward and next state
#     keep the action that gives the highest value


import numpy as np

gamma = 0.9
theta = 1e-6

goal_state = (1, 2)

states = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

actions = ["up", "down", "left", "right"]

model = {
    # From (0,0)
    ((0, 0), "up"):    ((0, 0), 0),
    ((0, 0), "down"):  ((1, 0), 0),
    ((0, 0), "left"):  ((0, 0), 0),
    ((0, 0), "right"): ((0, 1), 0),

    # From (0,1)
    ((0, 1), "up"):    ((0, 1), 0),
    ((0, 1), "down"):  ((1, 1), 0),
    ((0, 1), "left"):  ((0, 0), 0),
    ((0, 1), "right"): ((0, 2), 0),

    # From (0,2)
    ((0, 2), "up"):    ((0, 2), 0),
    ((0, 2), "down"):  ((1, 2), 1),
    ((0, 2), "left"):  ((0, 1), 0),
    ((0, 2), "right"): ((0, 2), 0),

    # From (1,0)
    ((1, 0), "up"):    ((0, 0), 0),
    ((1, 0), "down"):  ((1, 0), 0),
    ((1, 0), "left"):  ((1, 0), 0),
    ((1, 0), "right"): ((1, 1), 0),

    # From (1,1)
    ((1, 1), "up"):    ((0, 1), 0),
    ((1, 1), "down"):  ((1, 1), 0),
    ((1, 1), "left"):  ((1, 0), 0),
    ((1, 1), "right"): ((1, 2), 1),
}

# Initialize value of every state to 0
V = {state: 0.0 for state in states}

# -----------------------------
# Value Iteration
# -----------------------------

while True:
    delta = 0

    for state in states:
        if state == goal_state:
            continue

        old_value = V[state]

        action_values = []

        for action in actions:
            next_state, reward = model[(state, action)]

            value = reward + gamma * V[next_state]
            action_values.append(value)

        # Bellman optimality update
        V[state] = max(action_values)

        delta = max(delta, abs(old_value - V[state]))

    if delta < theta:
        break

print("Optimal value function V*(s):")
for state in states:
    print(f"V*({state}) = {V[state]:.3f}")


# -----------------------------
# Extract optimal policy
# -----------------------------

policy = {}

for state in states:
    if state == goal_state:
        policy[state] = "GOAL"
        continue

    best_action = None
    best_value = -float("inf")

    for action in actions:
        next_state, reward = model[(state, action)]
        value = reward + gamma * V[next_state]

        if value > best_value:
            best_value = value
            best_action = action

    policy[state] = best_action

print("\nOptimal policy:")
for state in states:
    print(f"π*({state}) = {policy[state]}")