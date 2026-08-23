"""
State-transition graph animation (nodes=cells, edges=legal moves) with a dot moving node-to-node.

- Builds the transition graph from a GridWorld maze (free cells only).
- Trains a tiny Q-learning agent.
- Extracts the greedy path from START to GOAL.
- Draws the graph and animates a dot along the path.

Speed control:
  STEP_MS = 250  # milliseconds per node-to-node move

Run:
  python state_graph_move.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# =========================
# Speed parameter (edit me)
# =========================
STEP_MS = 250  # ms per transition (e.g., 50 fast, 250 normal, 800 slow)

# 0 = free, 1 = wall
GRID = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)

START = (0, 0)
GOAL = (5, 5)

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N,E,S,W


def step(grid: np.ndarray, state: tuple[int, int], a_idx: int):
    """Deterministic transition. Illegal move -> stay. Reward = -1 per step."""
    H, W = grid.shape
    r, c = state
    dr, dc = ACTIONS[a_idx]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == 1:
        nr, nc = r, c
    next_state = (nr, nc)
    reward = -1
    done = next_state == GOAL
    return next_state, reward, done


def q_learning(
    grid: np.ndarray,
    episodes: int = 800,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_min: float = 0.05,
    eps_decay: float = 0.995,
    seed: int = 0,
):
    H, W = grid.shape
    Q = np.zeros((H, W, 4), dtype=float)
    rng = np.random.default_rng(seed)
    eps = eps_start

    for _ in range(episodes):
        s = START
        for _ in range(max_steps):
            r, c = s
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                a = int(np.argmax(Q[r, c]))
            s2, rew, done = step(grid, s, a)
            r2, c2 = s2
            Q[r, c, a] += alpha * (rew + gamma * np.max(Q[r2, c2]) - Q[r, c, a])
            s = s2
            if done:
                break
        eps = max(eps_min, eps * eps_decay)

    return Q


def greedy_rollout(grid: np.ndarray, Q: np.ndarray, start: tuple[int, int], max_len: int = 200):
    path = [start]
    s = start
    for _ in range(max_len):
        r, c = s
        a = int(np.argmax(Q[r, c]))
        s2, _, done = step(grid, s, a)
        path.append(s2)
        s = s2
        if done:
            break
    return path


def build_transition_graph(grid: np.ndarray):
    """Return (nodes, edges_undirected) for free cells. Edges connect adjacent free cells."""
    H, W = grid.shape
    nodes = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == 0]
    node_set = set(nodes)

    edges = set()
    for (r, c) in nodes:
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_set:
                # undirected edge as sorted pair
                a = (r, c)
                b = (nr, nc)
                edges.add(tuple(sorted([a, b])))

    return nodes, sorted(edges)


def pos(cell: tuple[int, int]):
    """2D position for plotting: x=col, y=-row (so it looks like a grid)."""
    r, c = cell
    return (c, -r)


def main():
    # Learn a policy and get a path
    Q = q_learning(GRID)
    path = greedy_rollout(GRID, Q, START)

    # Build graph
    nodes, edges = build_transition_graph(GRID)
    positions = {n: pos(n) for n in nodes}

    # Precompute edge line segments for fast drawing
    edge_xs = []
    edge_ys = []
    for a, b in edges:
        xa, ya = positions[a]
        xb, yb = positions[b]
        edge_xs.append([xa, xb])
        edge_ys.append([ya, yb])

    # Node arrays
    node_x = np.array([positions[n][0] for n in nodes])
    node_y = np.array([positions[n][1] for n in nodes])

    # Setup plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("State-transition graph (dot follows greedy path)")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Draw edges
    for xs, ys in zip(edge_xs, edge_ys):
        ax.plot(xs, ys, lw=1)

    # Draw nodes
    ax.scatter(node_x, node_y, s=120)  # default color

    # Emphasize start/goal
    sx, sy = positions[START]
    gx, gy = positions[GOAL]
    ax.scatter([sx], [sy], s=220, marker="o")
    ax.scatter([gx], [gy], s=220, marker="o")
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=12)
    ax.text(gx, gy, "G", ha="center", va="center", fontsize=12)

    # Moving dot + trail
    (dot,) = ax.plot([], [], marker="o", markersize=10)
    (trail,) = ax.plot([], [], lw=2)

    trail_x, trail_y = [], []

    # Nice bounds
    pad = 0.8
    ax.set_xlim(node_x.min() - pad, node_x.max() + pad)
    ax.set_ylim(node_y.min() - pad, node_y.max() + pad)

    def init():
        dot.set_data([], [])
        trail.set_data([], [])
        return dot, trail

    def update(i):
        cell = path[i]
        x, y = positions[cell]

        trail_x.append(x)
        trail_y.append(y)

        dot.set_data([x], [y])
        trail.set_data(trail_x, trail_y)

        ax.set_title(f"State-transition graph (step {i+1}/{len(path)})")
        return dot, trail

    animation.FuncAnimation(
        fig,
        update,
        frames=len(path),
        init_func=init,
        interval=STEP_MS,  # <-- speed control
        blit=True,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
