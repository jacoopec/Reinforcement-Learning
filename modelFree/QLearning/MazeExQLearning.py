import numpy as np
import random

# ----------------------------
# 1) Define the grid (YOU fill)
# ----------------------------
# Use:
#   'S' = start
#   'G' = goal
#   '#' = wall
#   '.' = free cell
#
# Example map (replace with YOUR maze):
GRID = [
    "......",
    "S##.#.",
    "..##..",
    "#..#.#",
    ".#.#..",
    "....#G",
]

# ----------------------------
# 2) Environment
# ----------------------------
ACTIONS = ["N", "E", "S", "W"]
A2D = {
    0: (-1, 0),  # N
    1: (0, +1),  # E
    2: (+1, 0),  # S
    3: (0, -1),  # W
}

STEP_REWARD = -1  # as in your slide

def parse_grid(grid):
    rows, cols = len(grid), len(grid[0])
    start = goal = None
    walls = set()
    for r in range(rows):
        if len(grid[r]) != cols:
            raise ValueError("All rows must have same length.")
        for c, ch in enumerate(grid[r]):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
            elif ch == "#":
                walls.add((r, c))
    if start is None or goal is None:
        raise ValueError("Grid must contain exactly one S and one G.")
    return rows, cols, start, goal, walls

ROWS, COLS, START, GOAL, WALLS = parse_grid(GRID)

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def step(state, action_idx):
    """Deterministic move: bump into walls/bounds -> stay."""
    r, c = state
    dr, dc = A2D[action_idx]
    nr, nc = r + dr, c + dc

    if (not in_bounds(nr, nc)) or ((nr, nc) in WALLS):
        nr, nc = r, c  # blocked

    next_state = (nr, nc)
    reward = STEP_REWARD
    done = (next_state == GOAL)
    return next_state, reward, done

# ----------------------------
# 3) Q-learning
# ----------------------------
Q = np.zeros((ROWS, COLS, len(ACTIONS)), dtype=float)

def epsilon_greedy_action(state, epsilon):
    r, c = state
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    return int(np.argmax(Q[r, c, :]))

def train(
    episodes=5000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.999,
    max_steps_per_episode=500
):
    epsilon = epsilon_start

    for ep in range(episodes):
        state = START

        for _ in range(max_steps_per_episode):
            a = epsilon_greedy_action(state, epsilon)
            next_state, reward, done = step(state, a)

            r, c = state
            nr, nc = next_state

            # Q-learning update:
            # Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_a' Q(s',a') - Q(s,a))
            td_target = reward + (0.0 if done else gamma * np.max(Q[nr, nc, :]))
            td_error = td_target - Q[r, c, a]
            Q[r, c, a] += alpha * td_error

            state = next_state
            if done:
                break

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

def extract_greedy_policy_path(max_steps=500):
    """Follow greedy policy from START to GOAL after training."""
    state = START
    path = [state]
    for _ in range(max_steps):
        if state == GOAL:
            break
        a = int(np.argmax(Q[state[0], state[1], :]))
        state, _, _ = step(state, a)
        path.append(state)
    return path

def print_policy():
    """Print arrows for best action in each free cell."""
    arrow = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    out = []
    for r in range(ROWS):
        row_chars = []
        for c in range(COLS):
            if (r, c) in WALLS:
                row_chars.append("#")
            elif (r, c) == START:
                row_chars.append("S")
            elif (r, c) == GOAL:
                row_chars.append("G")
            else:
                best_a = int(np.argmax(Q[r, c, :]))
                row_chars.append(arrow[best_a])
        out.append("".join(row_chars))
    print("\nLearned greedy policy:")
    print("\n".join(out))

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    train(episodes=8000, alpha=0.1, gamma=0.99)

    print_policy()
    path = extract_greedy_policy_path()
    print("\nGreedy path length:", len(path) - 1)
    print("Path:", path)