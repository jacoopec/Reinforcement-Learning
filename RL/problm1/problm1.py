import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Gridworld setup
# -----------------------------

GRID_SIZE = 5
GAMMA = 0.9
THETA = 1e-4
MAX_ITERATIONS = 100

# Actions: up, down, left, right
ACTIONS = [
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1)    # right
]

ACTION_PROB = 1.0 / len(ACTIONS)

# Special states from Sutton & Barto Gridworld
A = (0, 1)
A_PRIME = (4, 1)
A_REWARD = 10

B = (0, 3)
B_PRIME = (2, 3)
B_REWARD = 5


def step(state, action):
    """
    Executes one step in the Gridworld.

    Returns:
        next_state, reward
    """

    # Special state A
    if state == A:
        return A_PRIME, A_REWARD

    # Special state B
    if state == B:
        return B_PRIME, B_REWARD

    row, col = state
    d_row, d_col = action

    next_row = row + d_row
    next_col = col + d_col

    # If the action takes the agent outside the grid,
    # the agent remains in the same state and receives reward -1.
    if (
        next_row < 0 or next_row >= GRID_SIZE or
        next_col < 0 or next_col >= GRID_SIZE
    ):
        return state, -1

    # Normal transition
    return (next_row, next_col), 0


def policy_evaluation():
    """
    Iterative policy evaluation for the uniform random policy.

    Bellman expectation update:

        V(s) = sum_a pi(a|s) [ r + gamma * V(s') ]

    Since the policy is uniform random:

        pi(a|s) = 1 / 4
    """

    values = np.zeros((GRID_SIZE, GRID_SIZE))
    history = [values.copy()]

    for iteration in range(MAX_ITERATIONS):
        new_values = np.zeros_like(values)

        delta = 0

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                state = (row, col)

                value = 0

                for action in ACTIONS:
                    next_state, reward = step(state, action)
                    next_row, next_col = next_state

                    value += ACTION_PROB * (
                        reward + GAMMA * values[next_row, next_col]
                    )

                new_values[row, col] = value
                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        values = new_values
        history.append(values.copy())

        if delta < THETA:
            print(f"Converged after {iteration + 1} iterations.")
            break

    return history


def draw_grid(ax, values, iteration):
    """
    Draws the Gridworld with values inside each cell.
    """

    ax.clear()
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Uniform Random Policy Evaluation\nIteration {iteration}")

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        ax.plot([0, GRID_SIZE], [i, i], color="black", linewidth=1)
        ax.plot([i, i], [0, GRID_SIZE], color="black", linewidth=1)

    # Write values inside cells
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
                fontsize=16
            )

    # Mark special states
    special_states = {
        A: "A",
        A_PRIME: "A'",
        B: "B",
        B_PRIME: "B'"
    }

    for state, label in special_states.items():
        row, col = state
        x = col + 0.15
        y = GRID_SIZE - row - 0.2

        ax.text(
            x,
            y,
            label,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold"
        )


def animate_values(history):
    """
    Creates an animation showing how the value function changes.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        draw_grid(ax, history[frame], frame)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=500,
        repeat=False
    )

    plt.show()

    return animation


def save_final_value_function(history):
    """
    Saves the final converged value function as a PNG image.
    """

    final_values = history[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_grid(ax, final_values, len(history) - 1)

    plt.savefig("final_value_function.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved final figure as final_value_function.png")


if __name__ == "__main__":
    history = policy_evaluation()

    print("\nFinal value function:")
    print(np.round(history[-1], 1))

    animate_values(history)
    save_final_value_function(history)