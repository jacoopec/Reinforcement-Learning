import numpy as np

# With a stochastic policy, the agent does not always choose one fixed action.
# So from state (0, 0), the agent chooses:

# right with probability 70%
# down  with probability 20%
# up    with probability 5%
# left  with probability 5%
# Then the value function becomes a weighted average over actions:

# Vπ(s) = Σ π(a|s) [r + γ Vπ(s')]

# Where:
# π(a|s) = probability of choosing action a in state s
# The values are lower than with the deterministic optimal policy because the agent sometimes makes bad or useless moves, like going left, up, or staying against a wall.

# So the difference is:

# Deterministic policy:
# V(s) = r + γ V(s')

# Stochastic policy:
# V(s) = weighted average over all possible actions


gamma = 0.9
goal_state = (1, 2)

states = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

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

# Stochastic policy:
# state -> action probabilities
policy = {
    (0, 0): {
        "right": 0.7,
        "down": 0.2,
        "up": 0.05,
        "left": 0.05
    },

    (0, 1): {
        "right": 0.6,
        "down": 0.3,
        "left": 0.05,
        "up": 0.05
    },

    (0, 2): {
        "down": 0.8,
        "left": 0.1,
        "up": 0.05,
        "right": 0.05
    },

    (1, 0): {
        "right": 0.7,
        "up": 0.1,
        "left": 0.1,
        "down": 0.1
    },

    (1, 1): {
        "right": 0.8,
        "up": 0.1,
        "left": 0.05,
        "down": 0.05
    },
}

# Initialize state values
V = {state: 0.0 for state in states}

num_iterations = 100

for _ in range(num_iterations):
    new_V = V.copy()

    for state in states:
        if state == goal_state:
            new_V[state] = 0.0
            continue

        value = 0.0

        for action, action_probability in policy[state].items():
            next_state, reward = model[(state, action)]

            value += action_probability * (reward + gamma * V[next_state])

        new_V[state] = value

    V = new_V

print("State-value function Vπ(s) with stochastic policy:")
for state, value in V.items():
    print(f"V({state}) = {value:.3f}")