import random

# -----------------------------
# MRP definition (from diagram)
# -----------------------------

P = {
    "Facebook": [("Facebook", 0.9), ("Class1", 0.1)],
    "Class1":   [("Facebook", 0.5), ("Class2", 0.5)],
    "Class2":   [("Class3", 0.8), ("Sleep", 0.2)],
    "Class3":   [("Pass", 0.6), ("Pub", 0.4)],
    "Pub":      [("Class1", 0.2), ("Class2", 0.4), ("Class3", 0.4)],
    "Pass":     [("Sleep", 1.0)],
    "Sleep":    []  # terminal
}

R = {
    "Facebook": -1,
    "Class1":   -2,
    "Class2":   -2,
    "Class3":   -2,
    "Pub":      +1,
    "Pass":     +10,
    "Sleep":    0
}

GAMMA = 0.99
START_STATE = "Class1"
TERMINAL_STATE = "Sleep"

# -----------------------------
# Sampling helpers
# -----------------------------

def sample_next_state(state: str) -> str:
    """Sample next state using the transition probabilities P."""
    choices = P[state]
    if not choices:
        return state  # terminal or no outgoing edges
    r = random.random()
    cumulative = 0.0
    for s_next, prob in choices:
        cumulative += prob
        if r <= cumulative:
            return s_next
    # Safety fallback for floating point sums like 0.999999
    return choices[-1][0]

def simulate_episode(seed: int = None):
    """Simulate one episode until TERMINAL_STATE. Returns (path, rewards, G)."""
    if seed is not None:
        random.seed(seed)

    state = START_STATE
    path = [state]
    rewards = [R[state]]  # reward for being in the start state (state-reward convention)

    # Step until terminal
    while state != TERMINAL_STATE:
        state = sample_next_state(state)
        path.append(state)
        rewards.append(R[state])

    # Compute discounted return G = r0 + γ r1 + γ^2 r2 + ...
    G = 0.0
    for t, rt in enumerate(rewards):
        G += (GAMMA ** t) * rt

    return path, rewards, G

def run_many(num_episodes: int = 20, seed: int = 0):
    random.seed(seed)

    episodes = []
    best = None  # (G, path, rewards)

    for i in range(num_episodes):
        path, rewards, G = simulate_episode()
        episodes.append((path, rewards, G))

        if best is None or G > best[0]:
            best = (G, path, rewards)

    # Print all tested paths
    print(f"Tested {num_episodes} episodes (gamma={GAMMA})\n")
    for i, (path, rewards, G) in enumerate(episodes, start=1):
        print(f"Episode {i:02d}: G={G:7.3f} | Path: {' -> '.join(path)}")

    # Print best
    best_G, best_path, best_rewards = best
    print("\nBEST EPISODE")
    print(f"G={best_G:.3f}")
    print("Path:", " -> ".join(best_path))
    print("Rewards:", best_rewards)

if __name__ == "__main__":
    run_many(num_episodes=30, seed=42)