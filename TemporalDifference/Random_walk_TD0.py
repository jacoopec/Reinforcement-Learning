import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Random Walk setup (5-state random walk)
# ============================================================
# States: 0 1 2 3 4 5 6
# Non-terminal states: 1..5
# Terminal states: 0 and 6
# Start state: 3
# Policy: random, move left/right with equal probability
# Reward: 1 only when entering state 6, else 0
# True values for states 1..5: [1/6, 2/6, 3/6, 4/6, 5/6]
# ============================================================

N_STATES = 5
LEFT_TERMINAL = 0
RIGHT_TERMINAL = N_STATES + 1
START_STATE = 3

TRUE_VALUES = np.arange(1, N_STATES + 1) / (N_STATES + 1)

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
SEED = 123

# Graph 1: value estimates vs true values
ALPHA_FOR_VALUE_PLOT = 0.1
VALUE_SNAPSHOT_EPISODES = [0, 1, 10, 100]

# Graph 2: RMS error vs episodes
RMS_ALPHAS = [0.05, 0.10, 0.15]
RMS_EPISODES = 100
RMS_RUNS = 100

# Graph 3: average RMS error vs alpha
ALPHA_GRID = np.arange(0.01, 0.31, 0.01)
AVG_RMS_EPISODES = 100
AVG_RMS_RUNS = 100


def step(state, rng: np.random.Generator):
    """
    Take one step under the random policy.
    Returns: next_state, reward
    """
    action = rng.choice([-1, 1])
    next_state = state + action
    reward = 1.0 if next_state == RIGHT_TERMINAL else 0.0
    return next_state, reward


def td0_episode(values, alpha, rng: np.random.Generator):
    """
    Run one episode of TD(0) prediction.
    Update:
        V(S_t) <- V(S_t) + alpha * [R_{t+1} + V(S_{t+1}) - V(S_t)]
    with terminal values fixed at 0.
    """
    state = START_STATE

    while state not in (LEFT_TERMINAL, RIGHT_TERMINAL):
        next_state, reward = step(state, rng)

        td_target = reward + values[next_state]
        values[state] += alpha * (td_target - values[state])

        state = next_state


def rms_error(values):
    estimate = values[1: N_STATES + 1]
    return np.sqrt(np.mean((estimate - TRUE_VALUES) ** 2))


def run_td(alpha, episodes, seed, snapshots=None, init_value=0.5):
    """
    Run TD(0) for a number of episodes.

    Returns:
        values: final value array for states 0..6
        rms_history: RMS error after each episode
        saved_snapshots: dict {episode_number: value_array_copy}
    """
    rng = np.random.default_rng(seed)

    values = np.zeros(N_STATES + 2)
    values[1: N_STATES + 1] = init_value
    values[LEFT_TERMINAL] = 0.0
    values[RIGHT_TERMINAL] = 0.0

    snapshots = set(snapshots or [])
    saved_snapshots = {}

    if 0 in snapshots:
        saved_snapshots[0] = values.copy()

    rms_history = np.zeros(episodes)

    for ep in range(1, episodes + 1):
        td0_episode(values, alpha, rng)
        rms_history[ep - 1] = rms_error(values)

        if ep in snapshots:
            saved_snapshots[ep] = values.copy()

    return values, rms_history, saved_snapshots


def plot_estimated_vs_true():
    """
    Graph 1:
    Estimated value function vs true value function.
    """
    _, _, snapshots = run_td(
        alpha=ALPHA_FOR_VALUE_PLOT,
        episodes=max(VALUE_SNAPSHOT_EPISODES),
        seed=SEED,
        snapshots=VALUE_SNAPSHOT_EPISODES,
    )

    x = np.arange(1, N_STATES + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, TRUE_VALUES, marker="o", linewidth=2, label="True value")

    for ep in VALUE_SNAPSHOT_EPISODES:
        estimate = snapshots[ep][1: N_STATES + 1]
        plt.plot(x, estimate, marker="o", linestyle="--", label=f"TD(0) after {ep} episodes")

    plt.xticks(x)
    plt.xlabel("State")
    plt.ylabel("Estimated value")
    plt.title("Estimated value function vs true value function")
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_rms_error_vs_episodes():
    """
    Graph 2:
    RMS error vs number of episodes, for several alpha values.
    """
    avg_rms_by_alpha = {}

    for alpha in RMS_ALPHAS:
        avg_rms = np.zeros(RMS_EPISODES)

        for run in range(RMS_RUNS):
            run_seed = SEED + 1000 * run + int(alpha * 100000)
            _, rms_hist, _ = run_td(
                alpha=alpha,
                episodes=RMS_EPISODES,
                seed=run_seed,
            )
            avg_rms += rms_hist

        avg_rms /= RMS_RUNS
        avg_rms_by_alpha[alpha] = avg_rms

    episodes_axis = np.arange(1, RMS_EPISODES + 1)

    plt.figure(figsize=(8, 5))
    for alpha in RMS_ALPHAS:
        plt.plot(episodes_axis, avg_rms_by_alpha[alpha], label=f"TD(0) alpha={alpha:.2f}")

    plt.xlabel("Episodes")
    plt.ylabel("Average RMS error")
    plt.title("RMS error vs number of episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)


def plot_avg_rms_vs_alpha():
    """
    Graph 3:
    Average RMS error vs step-size alpha.
    Metric = average over runs and over all episodes.
    """
    avg_rms_per_alpha = []

    for alpha in ALPHA_GRID:
        total = 0.0

        for run in range(AVG_RMS_RUNS):
            run_seed = SEED + 5000 * run + int(alpha * 100000)
            _, rms_hist, _ = run_td(
                alpha=float(alpha),
                episodes=AVG_RMS_EPISODES,
                seed=run_seed,
            )
            total += np.mean(rms_hist)

        avg_rms_per_alpha.append(total / AVG_RMS_RUNS)

    plt.figure(figsize=(8, 5))
    plt.plot(ALPHA_GRID, avg_rms_per_alpha, marker="o")
    plt.xlabel(r"Step-size $\alpha$")
    plt.ylabel("Average RMS error")
    plt.title(r"Average RMS error vs step-size $\alpha$")
    plt.grid(True, alpha=0.3)


def main():
    plot_estimated_vs_true()
    plot_rms_error_vs_episodes()
    plot_avg_rms_vs_alpha()
    plt.show()


if __name__ == "__main__":
    main()