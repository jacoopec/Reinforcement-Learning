import numpy as np
import matplotlib.pyplot as plt

# 0 = wall, 1 = free cell
# Hand-transcribed from the maze image.
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1],
], dtype=int)

start = (1, 0)   # row, col
goal = (5, 7)    # row, col

actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

gamma = 0.95
step_reward = -1.0
goal_reward = 20.0
theta = 1e-8


def in_bounds(r, c):
    return 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1]


def is_free(state):
    r, c = state
    return maze[r, c] == 1


def is_terminal(state):
    return state == goal


def next_state(state, action):
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc

    if not in_bounds(nr, nc) or maze[nr, nc] == 0:
        return state  # hit wall => stay put

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


def value_iteration():
    V = np.zeros_like(maze, dtype=float)

    while True:
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if is_terminal(s):
                continue

            values = []
            for a in actions:
                s2 = next_state(s, a)
                r = reward(s, a, s2)
                values.append(r + gamma * V[s2])

            best = max(values)
            new_V[s] = best
            delta = max(delta, abs(best - V[s]))

        V = new_V
        if delta < theta:
            break

    return V


def extract_policy(V):
    policy = {}

    for s in states:
        if is_terminal(s):
            policy[s] = "G"
            continue

        best_action = None
        best_value = -np.inf

        for a in actions:
            s2 = next_state(s, a)
            r = reward(s, a, s2)
            q = r + gamma * V[s2]
            if q > best_value:
                best_value = q
                best_action = a

        policy[s] = best_action

    return policy


def follow_policy(policy, max_steps=100):
    path = [start]
    state = start

    for _ in range(max_steps):
        if is_terminal(state):
            break
        action = policy[state]
        state = next_state(state, action)
        path.append(state)

    return path


def plot_maze_with_path(maze, path, start, goal, V=None, policy=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # base maze
    ax.imshow(1 - maze, cmap="gray", origin="upper")

    # grid lines
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # optional value labels
    if V is not None:
        for r in range(maze.shape[0]):
            for c in range(maze.shape[1]):
                if maze[r, c] == 1 and (r, c) not in [start, goal]:
                    ax.text(c, r, f"{V[r, c]:.1f}", ha="center", va="center", fontsize=8)

    # optional policy arrows
    if policy is not None:
        arrow_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
        for s, a in policy.items():
            if s == goal or s == start:
                continue
            r, c = s
            ax.text(c, r + 0.28, arrow_map[a], ha="center", va="center", fontsize=10)

    # path
    if path:
        ys = [r for r, c in path]
        xs = [c for r, c in path]
        ax.plot(xs, ys, linewidth=3, marker="o")

    # start/goal markers
    ax.text(start[1], start[0], "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(goal[1], goal[0], "G", ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_title("Maze solved with Value Iteration")
    plt.tight_layout()
    plt.show()


def print_policy(policy):
    arrow_map = {"U": "↑", "D": "↓", "L": "←", "R": "→", "G": "G"}
    print("Policy:")
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


if __name__ == "__main__":
    V = value_iteration()
    policy = extract_policy(V)
    path = follow_policy(policy)

    print("Maze array:")
    print(maze)
    print("\nPath:")
    print(path)
    print()
    print_policy(policy)

    plot_maze_with_path(maze, path, start, goal, V=V, policy=policy)