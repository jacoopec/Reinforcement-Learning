import random
from collections import defaultdict

# ----------------------------
# Gridworld definition
# ----------------------------
ROWS, COLS = 3, 4
WALL = (1, 1)

TERMINALS = {
    (0, 3): 1.0,
    (1, 3): -1.0
}

LIVING_REWARD = -0.04

ACTIONS = ['U', 'D', 'L', 'R']

# Action effects
MOVE = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1),
}

# For stochastic movement:
# intended = 0.8, perpendicular left/right = 0.1 each
LEFT_OF = {'U': 'L', 'D': 'R', 'L': 'D', 'R': 'U'}
RIGHT_OF = {'U': 'R', 'D': 'L', 'L': 'U', 'R': 'D'}

GAMMA = 0.99
ALPHA = 0.1
EPSILON = 0.1
EPISODES = 50000
MAX_STEPS = 100


def in_bounds(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS


def is_wall(state):
    return state == WALL


def is_terminal(state):
    return state in TERMINALS


def valid_state(state):
    return in_bounds(state) and not is_wall(state)


def next_position(state, action):
    """Move in given action, but stay if hits wall/boundary."""
    if is_terminal(state):
        return state

    r, c = state
    dr, dc = MOVE[action]
    ns = (r + dr, c + dc)

    if not valid_state(ns):
        return state
    return ns


def stochastic_step(state, action):
    """
    Execute intended action with stochasticity:
      intended: 0.8
      left slip: 0.1
      right slip: 0.1
    """
    if is_terminal(state):
        return state, 0.0

    rnd = random.random()
    if rnd < 0.8:
        actual = action
    elif rnd < 0.9:
        actual = LEFT_OF[action]
    else:
        actual = RIGHT_OF[action]

    ns = next_position(state, actual)

    if ns in TERMINALS:
        reward = TERMINALS[ns]
    else:
        reward = LIVING_REWARD

    return ns, reward


def all_states():
    states = []
    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c)
            if not is_wall(s):
                states.append(s)
    return states


def non_terminal_states():
    return [s for s in all_states() if not is_terminal(s)]


def random_start_state():
    candidates = non_terminal_states()
    return random.choice(candidates)


# ----------------------------
# Q-learning
# ----------------------------
Q = defaultdict(float)

for episode in range(EPISODES):
    s = random_start_state()

    for step in range(MAX_STEPS):
        if is_terminal(s):
            break

        # epsilon-greedy action selection
        if random.random() < EPSILON:
            a = random.choice(ACTIONS)
        else:
            qvals = [Q[(s, act)] for act in ACTIONS]
            max_q = max(qvals)
            best_actions = [act for act in ACTIONS if Q[(s, act)] == max_q]
            a = random.choice(best_actions)

        ns, reward = stochastic_step(s, a)

        if is_terminal(ns):
            target = reward
        else:
            target = reward + GAMMA * max(Q[(ns, act)] for act in ACTIONS)

        Q[(s, a)] += ALPHA * (target - Q[(s, a)])
        s = ns


# ----------------------------
# Extract V(s) and policy
# ----------------------------
V = {}
policy = {}

for s in all_states():
    if is_wall(s):
        continue
    if is_terminal(s):
        V[s] = TERMINALS[s]
        policy[s] = 'T'
    else:
        qvals = {a: Q[(s, a)] for a in ACTIONS}
        best_a = max(qvals, key=qvals.get)
        V[s] = qvals[best_a]
        policy[s] = best_a


# ----------------------------
# Print results
# ----------------------------
arrow = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'T': 'T'}

print("Learned State Values:\n")
for r in range(ROWS):
    row_vals = []
    for c in range(COLS):
        s = (r, c)
        if s == WALL:
            row_vals.append(" WALL ".center(10))
        else:
            row_vals.append(f"{V[s]: .2f}".center(10))
    print(" | ".join(row_vals))
print()

print("Learned Policy:\n")
for r in range(ROWS):
    row_pol = []
    for c in range(COLS):
        s = (r, c)
        if s == WALL:
            row_pol.append(" WALL ".center(10))
        else:
            row_pol.append(arrow[policy[s]].center(10))
    print(" | ".join(row_pol))