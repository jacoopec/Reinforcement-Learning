"""
Simple maze solver using Value Iteration.

Legend
------
# = wall
S = start
G = goal
. = free cell

This script uses a small grid maze similar to the example discussed:
- deterministic actions: up, down, left, right
- hitting a wall keeps the agent in place
- reward = -1 per step
- reward = +10 when entering the goal
"""

from math import inf

# ---------------------------------------------------------------------
# Maze definition
# ---------------------------------------------------------------------

MAZE = [
    "########",
    "#......#",
    "S.##.#.#",
    "#..##..#",
    "##..#.##",
    "#.#.#..#",
    "....#..G",
    "########",
]

ROWS = len(MAZE)
COLS = len(MAZE[0])

ACTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

STEP_REWARD = -1.0
GOAL_REWARD = 10.0
GAMMA = 0.95
THETA = 1e-6  # convergence threshold


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def find_cell(ch):
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == ch:
                return (r, c)
    raise ValueError(f"Cell '{ch}' not found in maze.")


START = find_cell("S")
GOAL = find_cell("G")


def is_wall(state):
    r, c = state
    return MAZE[r][c] == "#"


def is_terminal(state):
    return state == GOAL


def valid_states():
    states = []
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] != "#":
                states.append((r, c))
    return states


STATES = valid_states()


def next_state(state, action):
    """Deterministic transition. If action hits wall, stay in place."""
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS:
        return state

    if MAZE[nr][nc] == "#":
        return state

    return (nr, nc)


def reward(state, action, nxt):
    """Reward for transition."""
    if nxt == GOAL:
        return GOAL_REWARD
    return STEP_REWARD


# ---------------------------------------------------------------------
# Value Iteration
# ---------------------------------------------------------------------

def value_iteration():
    V = {s: 0.0 for s in STATES}

    while True:
        delta = 0.0

        for s in STATES:
            if is_terminal(s):
                continue

            old_v = V[s]

            action_values = []
            for a in ACTIONS:
                s2 = next_state(s, a)
                r = reward(s, a, s2)
                q = r + GAMMA * V[s2]
                action_values.append(q)

            V[s] = max(action_values)
            delta = max(delta, abs(old_v - V[s]))

        if delta < THETA:
            break

    return V


def extract_policy(V):
    policy = {}

    for s in STATES:
        if is_terminal(s):
            policy[s] = "G"
            continue

        best_action = None
        best_value = -inf

        for a in ACTIONS:
            s2 = next_state(s, a)
            r = reward(s, a, s2)
            q = r + GAMMA * V[s2]

            if q > best_value:
                best_value = q
                best_action = a

        policy[s] = best_action

    return policy


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def print_values(V):
    print("State values:")
    for r in range(ROWS):
        row_out = []
        for c in range(COLS):
            cell = MAZE[r][c]
            s = (r, c)

            if cell == "#":
                row_out.append("#####".rjust(8))
            elif s == START:
                row_out.append(f"{V[s]:7.2f}S")
            elif s == GOAL:
                row_out.append(f"{V[s]:7.2f}G")
            else:
                row_out.append(f"{V[s]:8.2f}")
        print(" ".join(row_out))
    print()


def print_policy(policy):
    arrow = {"U": "↑", "D": "↓", "L": "←", "R": "→", "G": "G"}

    print("Optimal policy:")
    for r in range(ROWS):
        row_out = []
        for c in range(COLS):
            cell = MAZE[r][c]
            s = (r, c)

            if cell == "#":
                row_out.append("#")
            elif s == START:
                row_out.append("S")
            elif s == GOAL:
                row_out.append("G")
            else:
                row_out.append(arrow[policy[s]])
        print(" ".join(row_out))
    print()


def simulate(policy, max_steps=100):
    state = START
    path = [state]

    for _ in range(max_steps):
        if is_terminal(state):
            break
        action = policy[state]
        state = next_state(state, action)
        path.append(state)

    return path


def print_path(path):
    path_set = set(path[1:-1])  # keep S and G as labels
    print("Path followed from S to G:")
    for r in range(ROWS):
        row_out = []
        for c in range(COLS):
            s = (r, c)
            cell = MAZE[r][c]

            if cell == "#":
                row_out.append("#")
            elif s == START:
                row_out.append("S")
            elif s == GOAL:
                row_out.append("G")
            elif s in path_set:
                row_out.append("*")
            else:
                row_out.append(".")
        print(" ".join(row_out))
    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    V = value_iteration()
    policy = extract_policy(V)
    path = simulate(policy)

    print_values(V)
    print_policy(policy)
    print_path(path)

    print("Path coordinates:")
    print(path)