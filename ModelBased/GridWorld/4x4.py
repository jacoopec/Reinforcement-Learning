import numpy as np

# ----------------------------
# Gridworld definition
# ----------------------------
GRID_SIZE = 5
GAMMA = 1.0
REWARD = -1.0
THETA = 1e-10

TERMINALS = {(0, 0), (4, 4)}

ACTIONS = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}

POLICY_PROB = 1.0 / len(ACTIONS)  # π(a|s)=0.25


def is_terminal(state):
    return state in TERMINALS

def next_state(state, action):
    """
    Deterministic transition.
    If the move goes off-grid, the agent stays in the same state.
    """
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return state

    return (nr, nc)

def policy_evaluation():
    """
    Iterative policy evaluation for v_pi(s).
    """
    V = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    while True:
        delta = 0.0
        new_V = V.copy()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                s = (r, c)

                if is_terminal(s):
                    new_V[r, c] = 0.0
                    continue

                v = 0.0
                for a in ACTIONS:
                    s_next = next_state(s, a)
                    v += POLICY_PROB * (REWARD + GAMMA * V[s_next])

                new_V[r, c] = v
                delta = max(delta, abs(V[r, c] - new_V[r, c]))

        V = new_V

        if delta < THETA:
            break

    return V


def compute_q_from_v(V):
    """
    Compute q_pi(s,a) from the converged value function V.
    q_pi(s,a) = sum_{s',r} p(s',r|s,a)[r + gamma * V(s')]
    Since transitions are deterministic here:
    q_pi(s,a) = r + gamma * V(s')
    """
    Q = {}

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = (r, c)
            Q[s] = {}

            if is_terminal(s):
                for a in ACTIONS:
                    Q[s][a] = 0.0
                continue

            for a in ACTIONS:
                s_next = next_state(s, a)
                Q[s][a] = REWARD + GAMMA * V[s_next]

    return Q

def print_values(V):
    print("State-value function V(s):")
    for row in V:
        print(" ".join(f"{x:6.1f}" for x in row))

def print_q_for_state(Q, state):
    print(f"\nAction-values q_pi(s,a) for state {state}:")
    for action, value in Q[state].items():
        print(f"  {action:>5}: {value:.2f}")


if __name__ == "__main__":
    V = policy_evaluation()
    # print_values(V)
    Q = compute_q_from_v(V)


    # # Example: the highlighted state in the image is row=2, col=1 (0-based indexing)
    highlighted_state = (2, 2)
    print_q_for_state(Q, highlighted_state)