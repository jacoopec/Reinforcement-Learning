import random
from collections import defaultdict


# This is a Monte Carlo Control example on a 4×4 GridWorld using an ϵ-greedy policy.
# The algorithm repeatedly generates complete episodes, computes the return G, and estimates:

# Q(s,a)=expected return after taking action a in state s

# The learned policy then simply chooses:

# π(s)=argamax​Q(s,a)

# while ϵ-greedy exploration during training prevents the agent from always following its current best guess.

# -------------------------------------------------
# Environment
# -------------------------------------------------

GRID_SIZE = 4
START = (0, 0)
GOAL = (3, 3)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ACTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}


def step(state, action):
    """Apply an action and return next_state, reward, done."""

    row, col = state
    dr, dc = ACTION_DELTA[action]

    new_row = row + dr
    new_col = col + dc

    # Prevent leaving the grid
    new_row = max(0, min(GRID_SIZE - 1, new_row))
    new_col = max(0, min(GRID_SIZE - 1, new_col))

    next_state = (new_row, new_col)

    if next_state == GOAL:
        return next_state, 10, True

    # Small penalty encourages short paths
    return next_state, -1, False


# -------------------------------------------------
# Q table
# -------------------------------------------------

Q = defaultdict(lambda: {action: 0.0 for action in ACTIONS})

# Store number of visits to each state-action pair
visit_count = defaultdict(lambda: {action: 0 for action in ACTIONS})


# -------------------------------------------------
# Epsilon-greedy policy
# -------------------------------------------------

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    values = Q[state]
    max_value = max(values.values())

    # Handle ties randomly
    best_actions = [
        action
        for action, value in values.items()
        if value == max_value
    ]

    return random.choice(best_actions)


# -------------------------------------------------
# Generate one episode
# -------------------------------------------------

def generate_episode(epsilon):
    state = START
    episode = []

    max_steps = 100

    for _ in range(max_steps):

        action = choose_action(state, epsilon)

        next_state, reward, done = step(state, action)

        episode.append((state, action, reward))

        state = next_state

        if done:
            break

    return episode


# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

gamma = 0.9
epsilon = 0.1
num_episodes = 10000


for episode_number in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0

    # Used for first-visit Monte Carlo
    visited = set()

    # Traverse episode backwards
    for state, action, reward in reversed(episode):

        G = reward + gamma * G

        pair = (state, action)

        if pair not in visited:

            visited.add(pair)

            visit_count[state][action] += 1

            n = visit_count[state][action]

            # Incremental average
            Q[state][action] += (
                G - Q[state][action]
            ) / n


# -------------------------------------------------
# Extract learned policy
# -------------------------------------------------

policy = {}

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):

        state = (row, col)

        if state == GOAL:
            continue

        policy[state] = max(
            Q[state],
            key=Q[state].get
        )


# -------------------------------------------------
# Display learned policy
# -------------------------------------------------

symbols = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}

print("\nLearned policy:\n")

for row in range(GRID_SIZE):

    for col in range(GRID_SIZE):

        state = (row, col)

        if state == GOAL:
            print(" G ", end="")
        else:
            print(f" {symbols[policy[state]]} ", end="")

    print()


# -------------------------------------------------
# Test learned policy
# -------------------------------------------------

print("\nExample path:")

state = START

for _ in range(20):

    print(state, end=" -> ")

    if state == GOAL:
        break

    action = policy[state]

    state, reward, done = step(state, action)

    if done:
        print(state)
        break