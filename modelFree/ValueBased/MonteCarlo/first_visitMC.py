#!/usr/bin/env python3
"""
First-visit Monte Carlo prediction (model-free) on a tiny episodic MRP.

States: "A", "B", "T" (terminal)
Policy: fixed (there are no actions here—just a Markov Reward Process).

Dynamics (unknown to the algorithm; we only *sample*):
- From A:
    70% -> B with reward +1
    30% -> T with reward  0
- From B:
    60% -> A with reward  0
    40% -> T with reward +2

Goal: estimate V(s) = E[G | S0=s] under this process using first-visit MC.
"""

import random
from collections import defaultdict

GAMMA = 0.95
EPISODES = 5000
MAX_STEPS_PER_EP = 100  # safety against rare long loops


def step(state: str, rng: random.Random):
    """Environment step: returns (next_state, reward)."""
    if state == "A":
        if rng.random() < 0.70:
            return "B", 1.0
        else:
            return "T", 0.0
    elif state == "B":
        if rng.random() < 0.60:
            return "A", 0.0
        else:
            return "T", 2.0
    else:
        raise ValueError("Terminal state has no outgoing transitions.")


def generate_episode(start_state: str, rng: random.Random):
    """
    Generate one episode.
    Returns:
      states:  [S0, S1, ..., ST] where ST is terminal "T"
      rewards: [R1, R2, ..., RT] aligned so Rt is reward after transitioning from S_{t-1} to S_t
    """
    states = [start_state]
    rewards = []
    s = start_state

    for _ in range(MAX_STEPS_PER_EP):
        ns, r = step(s, rng)
        rewards.append(r)
        states.append(ns)
        s = ns
        if s == "T":
            break

    return states, rewards


def first_visit_mc_prediction(num_episodes: int, gamma: float, seed: int = 0):
    rng = random.Random(seed)

    # We'll do sample-average MC:
    # V(s) = average of observed returns from first visit to s in each episode
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for ep in range(1, num_episodes + 1):
        start_state = rng.choice(["A", "B"])  # random starts
        states, rewards = generate_episode(start_state, rng)

        # Compute returns G_t for each timestep t (excluding the final terminal state index)
        # G_t = R_{t+1} + gamma R_{t+2} + ...
        G = 0.0
        returns_from_t = [0.0] * (len(states) - 1)  # one per nonterminal time step
        for t in reversed(range(len(states) - 1)):
            G = rewards[t] + gamma * G
            returns_from_t[t] = G

        # First-visit updates: only the first time a state appears in this episode
        visited = set()
        for t, s in enumerate(states[:-1]):  # ignore terminal
            if s in visited:
                continue
            visited.add(s)

            returns_sum[s] += returns_from_t[t]
            returns_count[s] += 1
            V[s] = returns_sum[s] / returns_count[s]

        # Optional progress output
        if ep in {1, 10, 100, 1000, num_episodes}:
            a = V["A"]
            b = V["B"]
            print(f"Episode {ep:5d} | V(A)={a: .4f}  V(B)={b: .4f}  (counts: A={returns_count['A']}, B={returns_count['B']})")

    return V, returns_count


if __name__ == "__main__":
    V, counts = first_v_
