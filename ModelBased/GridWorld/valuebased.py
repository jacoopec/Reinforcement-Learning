# Simple Value Iteration for the 3x4 Grid World

ROWS, COLS = 3, 4
WALL = (1, 1)
TERMINALS = {(0, 3): 1.0, (1, 3): -1.0}

ACTIONS = ['U', 'D', 'L', 'R']
MOVE = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

LEFT =  {'U': 'L', 'D': 'R', 'L': 'D', 'R': 'U'}
RIGHT = {'U': 'R', 'D': 'L', 'L': 'U', 'R': 'D'}

gamma = 0.99
step_reward = -0.04
threshold = 1e-4

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) != WALL

def next_state(state, action):
    if state in TERMINALS:
        return state

    r, c = state
    dr, dc = MOVE[action]
    nr, nc = r + dr, c + dc

    if is_valid(nr, nc):
        return (nr, nc)
    return state  # stay in place if hit wall/boundary

def get_transitions(state, action):
    # intended: 0.8, slip left/right: 0.1 each
    return [
        (0.8, next_state(state, action)),
        (0.1, next_state(state, LEFT[action])),
        (0.1, next_state(state, RIGHT[action]))
    ]

# initialize values
V = {}
for r in range(ROWS):
    for c in range(COLS):
        s = (r, c)
        if s == WALL:
            continue
        V[s] = TERMINALS.get(s, 0.0)

# value iteration
while True:
    delta = 0
    new_V = V.copy()

    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c)

            if s == WALL or s in TERMINALS:
                continue

            action_values = []
            for a in ACTIONS:
                val = 0
                for prob, ns in get_transitions(s, a):
                    reward = TERMINALS[ns] if ns in TERMINALS else step_reward
                    val += prob * (reward + gamma * V[ns])
                action_values.append(val)

            new_V[s] = max(action_values)
            delta = max(delta, abs(new_V[s] - V[s]))

    V = new_V
    if delta < threshold:
        break

# extract policy
policy = {}
for r in range(ROWS):
    for c in range(COLS):
        s = (r, c)

        if s == WALL:
            policy[s] = 'WALL'
        elif s in TERMINALS:
            policy[s] = 'T'
        else:
            best_action = None
            best_value = float('-inf')

            for a in ACTIONS:
                val = 0
                for prob, ns in get_transitions(s, a):
                    reward = TERMINALS[ns] if ns in TERMINALS else step_reward
                    val += prob * (reward + gamma * V[ns])

                if val > best_value:
                    best_value = val
                    best_action = a

            policy[s] = best_action

# print values
print("State Values:")
for r in range(ROWS):
    row = []
    for c in range(COLS):
        s = (r, c)
        if s == WALL:
            row.append(" WALL ")
        else:
            row.append(f"{V[s]:5.2f}")
    print(" ".join(row))

# print policy
symbols = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'T': 'T'}
print("\nPolicy:")
for r in range(ROWS):
    row = []
    for c in range(COLS):
        s = (r, c)
        if s == WALL:
            row.append(" WALL ")
        else:
            row.append(f"  {symbols[policy[s]]}  ")
    print(" ".join(row))