import random
from collections import defaultdict



# Here the policy is not being learned. It is fixed:
# 80% -> move right
# 20% -> move left

# Monte Carlo prediction simply observes many episodes and estimates:

# V(s)=E[Gt∣St=s]

# So the distinction is:
# MC Prediction: fixed policy → learn V(s)	
# MC Control learns the policy itself.

# -----------------------------------
# Simple environment
# -----------------------------------

START = 0
TERMINAL = 4

# Fixed policy:
# from each state, move RIGHT with 80%
# and LEFT with 20%

def policy(state):
    if random.random() < 0.8:
        return 1   # RIGHT
    return -1      # LEFT


def step(state, action):
    next_state = state + action

    # Keep state inside [0, 4]
    next_state = max(0, min(TERMINAL, next_state))

    if next_state == TERMINAL:
        return next_state, 1, True

    return next_state, 0, False


# -----------------------------------
# Generate one episode
# -----------------------------------

def generate_episode():
    state = START
    episode = []

    while True:

        action = policy(state)

        next_state, reward, done = step(state, action)

        episode.append((state, reward))

        state = next_state

        if done:
            break

    return episode


# -----------------------------------
# Monte Carlo prediction
# -----------------------------------

V = defaultdict(float)
visit_count = defaultdict(int)

gamma = 0.9
num_episodes = 10000

for _ in range(num_episodes):

    episode = generate_episode()

    G = 0
    visited = set()

    # Compute returns backwards
    for state, reward in reversed(episode):

        G = reward + gamma * G

        # First-visit Monte Carlo
        if state not in visited:

            visited.add(state)

            visit_count[state] += 1

            # Incremental average
            V[state] += (
                G - V[state]
            ) / visit_count[state]


# -----------------------------------
# Results
# -----------------------------------

for state in range(TERMINAL):
    print(f"V({state}) = {V[state]:.3f}")