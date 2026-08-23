import numpy as np

gamma = 0.9

goal_state = (1, 2)


# To apply that state-value function to your 2D model, I also need a policy.
# My model:
# (state, action) -> (next_state, reward)
# But the value function is:
# V π(s)
# So it means:
# the value of state s when following policy π
# Since your My is deterministic, the expectation is simple: there is only one possible next state.

model = {
    ((0, 0), "up"):    ((0, 0), 0),
    ((0, 0), "down"):  ((1, 0), 0),
    ((0, 0), "left"):  ((0, 0), 0),
    ((0, 0), "right"): ((0, 1), 0),

    ((0, 1), "up"):    ((0, 1), 0),
    ((0, 1), "down"):  ((1, 1), 0),
    ((0, 1), "left"):  ((0, 0), 0),
    ((0, 1), "right"): ((0, 2), 0),

    ((0, 2), "up"):    ((0, 2), 0),
    ((0, 2), "down"):  ((1, 2), 1),
    ((0, 2), "left"):  ((0, 1), 0),
    ((0, 2), "right"): ((0, 2), 0),

    ((1, 0), "up"):    ((0, 0), 0),
    ((1, 0), "down"):  ((1, 0), 0),
    ((1, 0), "left"):  ((1, 0), 0),
    ((1, 0), "right"): ((1, 1), 0),

    ((1, 1), "up"):    ((0, 1), 0),
    ((1, 1), "down"):  ((1, 1), 0),
    ((1, 1), "left"):  ((1, 0), 0),
    ((1, 1), "right"): ((1, 2), 1),
}

policy = {
    (0, 0): "right",
    (0, 1): "right",
    (0, 2): "down",
    (1, 0): "right",
    (1, 1): "right",
}

states = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
]

V = {state: 0.0 for state in states}

num_iterations = 100

for _ in range(num_iterations):
    new_V = V.copy()

    for state in states:
        if state == goal_state:
            new_V[state] = 0.0
            continue

        action = policy[state]
        next_state, reward = model[(state, action)]

        # Bellman equation for deterministic policy and deterministic model
        new_V[state] = reward + gamma * V[next_state]

    V = new_V

print("State-value function Vπ(s):")
for state, value in V.items():
    print(f"V({state}) = {value:.3f}")