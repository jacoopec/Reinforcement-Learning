import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Gridworld setup
# -----------------------------

GRID_SIZE = 5
GAMMA = 0.9
THETA = 1e-4
MAX_ITERATIONS = 200

# Actions: name -> row change, col change
ACTIONS = {
    "↑": (-1, 0),
    "↓": (1, 0),
    "←": (0, -1),
    "→": (0, 1)
}

# Special states
A = (0, 1)
A_PRIME = (4, 1)
A_REWARD = 10

B = (0, 3)
B_PRIME = (2, 3)
B_REWARD = 5


def step(state, action):
    """
    Executes one action in the Gridworld.

    Returns:
        next_state, reward
    """

    # Special transition A -> A'
    if state == A:
        return A_PRIME, A_REWARD

    # Special transition B -> B'
    if state == B:
        return B_PRIME, B_REWARD

    row, col = state
    d_row, d_col = action

    next_row = row + d_row
    next_col = col + d_col

    # Moving outside the grid keeps the agent in the same state
    # and gives reward -1.
    if (
        next_row < 0 or next_row >= GRID_SIZE or
        next_col < 0 or next_col >= GRID_SIZE
    ):
        return state, -1

    # Normal transition
    return (next_row, next_col), 0


def value_iteration():
    """
    Value iteration for finding the optimal value function.

    Bellman optimality update:

        V(s) = max_a [ r + gamma * V(s') ]
    """

    values = np.zeros((GRID_SIZE, GRID_SIZE))
    history = [values.copy()]

    for iteration in range(MAX_ITERATIONS):
        new_values = np.zeros_like(values)
        delta = 0

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                state = (row, col)

                action_values = []

                for action in ACTIONS.values():
                    next_state, reward = step(state, action)
                    next_row, next_col = next_state

                    q_value = reward + GAMMA * values[next_row, next_col]
                    action_values.append(q_value)

                best_value = max(action_values)
                new_values[row, col] = best_value

                delta = max(delta, abs(best_value - values[row, col]))

        values = new_values
        history.append(values.copy())

        if delta < THETA:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return history


def get_optimal_policy(values):
    """
    Extracts the greedy optimal policy from the optimal value function.

    If multiple actions are equally good, all are shown.
    """

    policy = {}

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)

            action_scores = {}

            for action_symbol, action in ACTIONS.items():
                next_state, reward = step(state, action)
                next_row, next_col = next_state

                q_value = reward + GAMMA * values[next_row, next_col]
                action_scores[action_symbol] = q_value

            best_score = max(action_scores.values())

            best_actions = [
                action_symbol
                for action_symbol, score in action_scores.items()
                if np.isclose(score, best_score)
            ]

            policy[state] = best_actions

    return policy


def draw_values_grid(ax, values, iteration):
    """
    Draws the Gridworld values.
    """

    ax.clear()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Optimal Value Function\nIteration {iteration}")

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.plot([0, GRID_SIZE], [i, i], color="black", linewidth=1)
        ax.plot([i, i], [0, GRID_SIZE], color="black", linewidth=1)

    # Draw values
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = col + 0.5
            y = GRID_SIZE - row - 0.5

            ax.text(
                x,
                y,
                f"{values[row, col]:.1f}",
                ha="center",
                va="center",
                fontsize=15
            )

    # Mark special states
    labels = {
        A: "A",
        A_PRIME: "A'",
        B: "B",
        B_PRIME: "B'"
    }

    for state, label in labels.items():
        row, col = state
        x = col + 0.12
        y = GRID_SIZE - row - 0.15

        ax.text(
            x,
            y,
            label,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold"
        )


def draw_policy_grid(ax, values):
    """
    Draws the optimal policy using arrows.
    """

    policy = get_optimal_policy(values)

    ax.clear()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optimal Policy")

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.plot([0, GRID_SIZE], [i, i], color="black", linewidth=1)
        ax.plot([i, i], [0, GRID_SIZE], color="black", linewidth=1)

    arrow_offsets = {
        "↑": (0.5, 0.70),
        "↓": (0.5, 0.30),
        "←": (0.30, 0.5),
        "→": (0.70, 0.5)
    }

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)
            actions = policy[state]

            for action_symbol in actions:
                offset_x, offset_y = arrow_offsets[action_symbol]

                x = col + offset_x
                y = GRID_SIZE - row - offset_y

                ax.text(
                    x,
                    y,
                    action_symbol,
                    ha="center",
                    va="center",
                    fontsize=20
                )

    # Mark special states
    labels = {
        A: "A",
        A_PRIME: "A'",
        B: "B",
        B_PRIME: "B'"
    }

    for state, label in labels.items():
        row, col = state
        x = col + 0.12
        y = GRID_SIZE - row - 0.15

        ax.text(
            x,
            y,
            label,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold"
        )


def animate_value_iteration(history):
    """
    Creates an animation of the values changing at each iteration.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        draw_values_grid(ax, history[frame], frame)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=500,
        repeat=False
    )

    plt.show()

    return animation


def save_final_results(history):
    """
    Saves a final figure with both v* and pi*.
    """

    final_values = history[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    draw_values_grid(axes[0], final_values, len(history) - 1)
    draw_policy_grid(axes[1], final_values)

    plt.tight_layout()
    plt.savefig("optimal_gridworld_control.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved final figure as optimal_gridworld_control.png")


if __name__ == "__main__":
    history = value_iteration()
    final_values = history[-1]

    print("\nOptimal value function:")
    print(np.round(final_values, 1))

    print("\nOptimal policy:")
    optimal_policy = get_optimal_policy(final_values)

    for row in range(GRID_SIZE):
        row_policy = []
        for col in range(GRID_SIZE):
            actions = "".join(optimal_policy[(row, col)])
            row_policy.append(actions)
        print(row_policy)

    animate_value_iteration(history)
    save_final_results(history)