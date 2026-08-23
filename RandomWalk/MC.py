"""
Monte Carlo policy evaluation for the classic Random Walk MDP (Sutton & Barto style)

States:
  Terminal:  s_down (value 0), s_up (value 1)
  Non-terminal: s^-2, s^-1, s^0, s^1, s^2  (5 states)

Dynamics:
  From any non-terminal state, move Left or Right with prob 0.5.
  Episode ends when hitting a terminal.
Rewards:
  Reward = 1 if you end in s_up, else 0 (i.e., return is 1 for success, 0 otherwise)

Goal:
  Estimate V(s) under the fixed random policy using Monte Carlo updates:
    V(s) <- V(s) + alpha * (G - V(s))   for each visited state s in the episode
  and plot:
    1) final V(s) vs true V(s)
    2) average absolute error |V - V_true| vs episode index for different alphas
"""

import random
import matplotlib.pyplot as plt

# ----- MDP setup -----
# Indexing: 0 = s_down (terminal), 6 = s_up (terminal), 1..5 = non-terminal
STATE_NAMES = {
    0: "s_down",
    1: "s^-2",
    2: "s^-1",
    3: "s^0",
    4: "s^1",
    5: "s^2",
    6: "s_up",
}

NON_TERMINAL = [1, 2, 3, 4, 5]
TERMINALS = {0, 6}

# True values for unbiased random walk:
# Probability of reaching s_up before s_down starting from position i in {1..5} is i/6
V_TRUE = {i: i / 6.0 for i in NON_TERMINAL}

def step(state: int) -> int:
    """One step of the random walk (equiprobable left/right)."""
    if state in TERMINALS:
        return state
    move = -1 if random.random() < 0.5 else +1
    return state + move

def run_episode(exploring_starts=True) -> list[int]:
    """
    Generate an episode as a sequence of visited states (non-terminal states only).
    We use exploring starts to visit all states: start state is uniform over non-terminals.
    """
    s = random.choice(NON_TERMINAL) if exploring_starts else 3  # default start s^0
    visited = []

    while s not in TERMINALS:
        visited.append(s)
        s = step(s)

    # Episode ended; s is terminal
    return visited + [s]  # include terminal at end for reward determination

def return_from_terminal(terminal_state: int) -> float:
    """Return G for the episode given the terminal state reached."""
    return 1.0 if terminal_state == 6 else 0.0

def mean_abs_error(V: dict[int, float]) -> float:
    """Mean absolute error over non-terminal states."""
    return sum(abs(V[s] - V_TRUE[s]) for s in NON_TERMINAL) / len(NON_TERMINAL)

# ----- Monte Carlo evaluation (constant-alpha incremental) -----
def mc_evaluate(alphas=(0.01, 0.02, 0.03), episodes=120, seed=0, exploring_starts=True):
    random.seed(seed)

    # Separate value tables per alpha
    V_tables = {a: {s: 0.5 for s in NON_TERMINAL} for a in alphas}  # common init (0.5)
    errors = {a: [] for a in alphas}

    for m in range(1, episodes + 1):
        ep = run_episode(exploring_starts=exploring_starts)
        terminal = ep[-1]
        G = return_from_terminal(terminal)

        # states visited in episode (excluding terminal at end)
        visited_states = ep[:-1]

        # First-visit MC update (first occurrence only)
        first_visit_set = set()
        for s in visited_states:
            if s in first_visit_set:
                continue
            first_visit_set.add(s)

            for a in alphas:
                V = V_tables[a]
                V[s] = V[s] + a * (G - V[s])

        # Track error after each episode
        for a in alphas:
            errors[a].append(mean_abs_error(V_tables[a]))

    return V_tables, errors

# ----- Plotting -----
def plot_results(V_tables, errors, alphas):
    # 1) Final V(s) plot (use the last alpha's V by default for the value plot)
    # You can change which alpha you want to show.
    alpha_for_value_plot = alphas[-1]
    V = V_tables[alpha_for_value_plot]

    x = NON_TERMINAL
    v_est = [V[s] for s in x]
    v_true = [V_TRUE[s] for s in x]
    labels = [STATE_NAMES[s] for s in x]

    plt.figure()
    plt.plot(x, v_est, marker="o", label=f"MC estimate (alpha={alpha_for_value_plot})")
    plt.plot(x, v_true, marker="o", label="True V(s)")
    plt.xticks(x, labels)
    plt.ylim(-0.05, 1.05)
    plt.title("State-Value Function V(s)")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2) Average absolute error vs episodes for each alpha
    plt.figure()
    for a in alphas:
        plt.plot(range(1, len(errors[a]) + 1), errors[a], label=f"alpha={a}")
    plt.title("Average Absolute Difference from True V(s)")
    plt.xlabel("Episode m")
    plt.ylabel("Mean |V - V_true| (non-terminal states)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

def main():
    alphas = (0.009, 0.01, 0.02, 0.03, 0.04, 0.05)
    episodes = 120
    seed = 1

    V_tables, errors = mc_evaluate(
        alphas=alphas,
        episodes=episodes,
        seed=seed,
        exploring_starts=True,  # visits all states over time
    )

    # Print final estimates for each alpha
    for a in alphas:
        print(f"\nFinal V estimates (alpha={a}):")
        for s in NON_TERMINAL:
            print(f"  {STATE_NAMES[s]:5s}: {V_tables[a][s]:.4f}   (true {V_TRUE[s]:.4f})")

    plot_results(V_tables, errors, alphas)

if __name__ == "__main__":
    main()