# If you already have the optimal value function V*, you can compute the optimal policy by choosing, in every state, the action that maximizes:

# r + γ V*(next_state)
# The key idea is:
# π*(s) = argmax_a [r + γ V*(s')]

# The important point is: having only V* is not enough. You also need the environment model, because to choose an action you must know:

# if I take action a from state s,
# which next state do I reach,
# and what reward do I receive?

# So the full recipe is:

# V* + environment model -> optimal policy

# For a deterministic model:

# π*(s) = argmax_a [r + γ V*(s')]

# For a stochastic model:

# π*(s) = argmax_a Σ p(s', r | s, a) [r + γ V*(s')]

gamma = 0.9

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

# Suppose this is already known:
# optimal value function V*
V_star = {
    (0, 0): 0.81,
    (0, 1): 0.90,
    (0, 2): 1.00,
    (1, 0): 0.90,
    (1, 1): 1.00,
    (1, 2): 0.00,
}

# -----------------------------
# Compute optimal policy from V*
# -----------------------------

optimal_policy = {}

tolerance = 1e-9

for state in states:
    if state == goal_state:
        optimal_policy[state] = ["GOAL"]
        continue

    action_values = {}

    for action in actions:
        next_state, reward = model[(state, action)]
        action_value = reward + gamma * V_star[next_state]
        action_values[action] = action_value

    best_value = max(action_values.values())

    best_actions = [
        action
        for action, value in action_values.items()
        if abs(value - best_value) < tolerance
    ]

    optimal_policy[state] = best_actions

print("All optimal actions from V*:")

for state, actions in optimal_policy.items():
    print(f"π*({state}) = {actions}")