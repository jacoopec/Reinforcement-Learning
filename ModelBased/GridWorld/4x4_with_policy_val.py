import numpy as np

GRID_SIZE = 4
GAMMA = 1.0
REWARD = -1.0
THETA = 1e-10

TERMINALS = {(0, 0), (3, 3)}
ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}


def is_terminal(state):
    return state in TERMINALS


def next_state(state, action):
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = ACTION_DELTA[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return state  # bump into wall, stay in place

    return (nr, nc)


def get_all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def policy_evaluation(policy):
    V = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    while True:
        delta = 0.0
        new_V = V.copy()

        for s in get_all_states():
            r, c = s

            if is_terminal(s):
                new_V[r, c] = 0.0
                continue

            a = policy[s]
            s_next = next_state(s, a)
            new_V[r, c] = REWARD + GAMMA * V[s_next]

            delta = max(delta, abs(new_V[r, c] - V[r, c]))

        V = new_V

        if delta < THETA:
            break

    return V


def best_action_and_value(state, V):
    best_a = None
    best_q = float("-inf")

    for a in ACTIONS:
        s_next = next_state(state, a)
        q = REWARD + GAMMA * V[s_next]

        if q > best_q:
            best_q = q
            best_a = a

    return best_a, best_q


def policy_iteration():
    # initial arbitrary policy
    policy = {}
    for s in get_all_states():
        if not is_terminal(s):
            policy[s] = "up"

    while True:
        V = policy_evaluation(policy)
        policy_stable = True

        for s in get_all_states():
            if is_terminal(s):
                continue

            old_action    = policy[s]
            new_action, _ = best_action_and_value(s, V)
            policy[s]     = new_action

            if new_action != old_action:
                policy_stable = False

        if policy_stable:
            return policy, V


def print_values(V):
    print("Optimal Value Function:")
    for row in V:
        print(" ".join(f"{x:6.1f}" for x in row))


def print_policy(policy):
    print("\nOptimal Policy:")
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if is_terminal(s):
                row.append(" T ")
            else:
                row.append(f" {ARROWS[policy[s]]} ")
        print(" ".join(row))


if __name__ == "__main__":
    policy, V = policy_iteration()
    print_values(V)
    print_policy(policy)