import numpy as np
import matplotlib.pyplot as plt

# 0 = wall, 1 = free cell
# Hand-transcribed from the maze image
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1],
], dtype=int)

start = (1, 0)   # (row, col)
goal = (5, 7)    # (row, col)

actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

action_list = list(actions.keys())

gamma = 0.95
step_reward = -1.0
goal_reward = 20.0
theta = 1e-8


def in_bounds(r, c):
    return 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1]


def is_terminal(state):
    return state == goal


def next_state(state, action):
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc

    if not in_bounds(nr, nc) or maze[nr, nc] == 0:
        return state  # invalid move => stay in place

    return (nr, nc)


def reward(state, action, nxt):
    if nxt == goal:
        return goal_reward
    return step_reward


def all_states():
    states = []
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if maze[r, c] == 1:
                states.append((r, c))
    return states


states = all_states()


def initialize_policy():
    """
    Pick a valid initial action for each non-terminal state.
    """
    policy = {}
    for s in states:
        if is_terminal(s):
            policy[s] = "G"
            continue

        for a in action_list:
            s2 = next_state(s, a)
            if s2 != s or s2 == goal:
                policy[s] = a
                break
        else:
            policy[s] = "U"  # fallback
    return policy


def policy_evaluation(policy):
    """
    Iterative policy evaluation:
    compute V^pi for the current policy.
    """
    V = np.zeros_like(maze, dtype=float)

    while True:
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if is_terminal(s):
                new_V[s] = 0.0
                continue

            a = policy[s]
            s2 = next_state(s, a)
            r = reward(s, a, s2)
            new_V[s] = r + gamma * V[s2]

            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if delta < theta:
            break

    return V


def policy_improvement(V, policy):
    """
    Improve policy greedily using the current value function.
    """
    stable = True

    for s in states:
        if is_terminal(s):
            continue

        old_action = policy[s]

        best_action = None
        best_value = -np.inf

        for a in action_list:
            s2 = next_state(s, a)
            r = reward(s, a, s2)
            q = r + gamma * V[s2]

            if q > best_value:
                best_value = q
                best_action = a

        policy[s] = best_action

        if best_action != old_action:
            stable = False

    return policy, stable


def policy_iteration():
    policy = initialize_policy()
    iteration = 0

    while True:
        iteration += 1
        V = policy_evaluation(policy)
        policy, stable = policy_improvement(V, policy)

        print(f"Policy iteration step {iteration} complete")

        if stable:
            print("Policy is stable. Converged.")
            break

    return V, policy


def follow_policy(policy, max_steps=100):
    path = [start]
    state = start

    for _ in range(max_steps):
        if is_terminal(state):
            break

        action = policy[state]
        state = next_state(state, action)
        path.append(state)

        # safety in case of loops
        if len(path) >= 2 and path[-1] == path[-2]:
            break

    return path


def print_policy(policy):
    arrow_map = {"U": "↑", "D": "↓", "L": "←", "R": "→", "G": "G"}

    print("\nOptimal Policy:")
    for r in range(maze.shape[0]):
        row = []
        for c in range(maze.shape[1]):
            s = (r, c)

            if maze[r, c] == 0:
                row.append("#")
            elif s == start:
                row.append("S")
            elif s == goal:
                row.append("G")
            else:
                row.append(arrow_map[policy[s]])
        print(" ".join(row))


def plot_maze_with_path(maze, path, start, goal, V=None, policy=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw maze: walls black, free cells white
    ax.imshow(1 - maze, cmap="gray", origin="upper")

    # Grid
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # Values
    if V is not None:
        for r in range(maze.shape[0]):
            for c in range(maze.shape[1]):
                if maze[r, c] == 1 and (r, c) not in [start, goal]:
                    ax.text(c, r - 0.15, f"{V[r, c]:.1f}", ha="center", va="center", fontsize=8)

    # Policy arrows
    if policy is not None:
        arrow_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
        for s, a in policy.items():
            if s in [start, goal]:
                continue
            r, c = s
            ax.text(c, r + 0.22, arrow_map[a], ha="center", va="center", fontsize=10)

    # Path
    if path:
        ys = [r for r, c in path]
        xs = [c for r, c in path]
        ax.plot(xs, ys, linewidth=3, marker="o")

    # Start / Goal
    ax.text(start[1], start[0], "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(goal[1], goal[0], "G", ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_title("Maze solved with Policy Iteration")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    '''
    It solves the maze in two repeating steps:

    Policy evaluation
    It computes how good the current policy is.
    Policy improvement
    It replaces each action with the best action according to the current values.

    This repeats until the policy stops changing.

    Difference from your value-iteration version
    Value iteration directly updates values with a max over actions every step.
    Policy iteration keeps a current policy, evaluates it, then improves it.

    So policy iteration is more like:

    “Assume I follow these arrows.”
    “How good is that?”
    “Can I improve the arrows?”'''
    
    V, policy = policy_iteration()
    path = follow_policy(policy)

    print("\nMaze array:")
    print(maze)

    print("\nValue Function:")
    print(np.round(V, 2))

    print_policy(policy)

    print("\nPath from start to goal:")
    print(path)

    plot_maze_with_path(maze, path, start, goal, V=V, policy=policy)