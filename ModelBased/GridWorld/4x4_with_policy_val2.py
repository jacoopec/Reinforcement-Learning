import numpy as np
import random

GRID_SIZE = 4
GAMMA = 0.8
REWARD = -1.0
THETA = 0.5

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

# Set a seed so the random initial policy is reproducible.
# Remove or change this line if you want a different random start each run.
random.seed(42)


def is_terminal(state):
    return state in TERMINALS


def all_states():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def next_state(state, action):
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = ACTION_DELTA[action]
    nr, nc = r + dr, c + dc

    # If off-grid, stay in same state
    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return state

    return (nr, nc)


def random_policy():
    policy = {}
    for s in all_states():
        if not is_terminal(s):
            policy[s] = random.choice(ACTIONS)
    return policy


def print_policy(policy, title="Policy"):
    print(f"\n{title}:")
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            s = (r, c)
            if is_terminal(s):
                row.append(" T ")
            else:
                row.append(f" {ARROWS[policy[s]]} ")
        print(" ".join(row))


def print_values(V, title="Value Function"):
    print(f"\n{title}:")
    for r in range(GRID_SIZE):
        print(" ".join(f"{V[r, c]:6.1f}" for c in range(GRID_SIZE)))


def policy_evaluation(policy, verbose=True):
    V = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)
    sweep = 0

    while True:
        delta = 0.0
        new_V = V.copy()

        for s in all_states():
            r, c = s

            if is_terminal(s):
                new_V[r, c] = 0.0
                continue

            a = policy[s]
            s_next = next_state(s, a)
            new_V[r, c] = REWARD + GAMMA * V[s_next]

            delta = max(delta, abs(new_V[r, c] - V[r, c]))

        V = new_V
        sweep += 1

        if verbose:
            print(f"  Evaluation sweep {sweep:2d}, delta = {delta:.10f}")

        if delta < THETA:
            break

    return V


def q_value(state, action, V):
    s_next = next_state(state, action)
    return REWARD + GAMMA * V[s_next]


def best_action(state, V):
    q_values = {a: q_value(state, a, V) for a in ACTIONS}
    best_q = max(q_values.values())

    # Random tie-break among equally good actions
    best_actions = [a for a, q in q_values.items() if np.isclose(q, best_q)]
    return random.choice(best_actions), q_values


def policy_improvement(policy, V):
    policy_stable = True
    changes = []

    for s in all_states():
        if is_terminal(s):
            continue

        old_action = policy[s]
        new_action, q_values = best_action(s, V)
        policy[s] = new_action

        if old_action != new_action:
            policy_stable = False
            changes.append((s, old_action, new_action, q_values))

    return policy_stable, changes


def policy_iteration():
    policy = random_policy()
    iteration = 0

    print_policy(policy, title="Initial Random Policy")

    while True:
        iteration += 1
        print(f"\n{'=' * 50}")
        print(f"POLICY ITERATION STEP {iteration}")
        print(f"{'=' * 50}")

        # 1) Policy Evaluation
        print("\n[1] Policy Evaluation")
        V = policy_evaluation(policy, verbose=True)
        print_values(V, title=f"V after evaluation at step {iteration}")

        # 2) Policy Improvement
        print("\n[2] Policy Improvement")
        policy_stable, changes = policy_improvement(policy, V)

        if changes:
            print("  Changed states:")
            for s, old_a, new_a, q_vals in changes:
                q_text = ", ".join(f"{a}:{q_vals[a]:.1f}" for a in ACTIONS)
                print(
                    f"    state {s}: {old_a} -> {new_a}    "
                    f"[{q_text}]"
                )
        else:
            print("  No action changed.")

        print_policy(policy, title=f"Policy after improvement at step {iteration}")

        if policy_stable:
            print("\nPolicy is stable. Stopping.")
            return policy, V


if __name__ == "__main__":
    final_policy, final_V = policy_iteration()

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print_values(final_V, title="Final Value Function")
    print_policy(final_policy, title="Final Policy")