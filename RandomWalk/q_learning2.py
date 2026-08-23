import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# States:
# 0 = losing terminal state
# 1, 2, 3, 4, 5 = normal states
# 6 = winning terminal state

START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

NUM_STATES = 7
NUM_ACTIONS = 2

LEFT = 0
RIGHT = 1

EPISODES = 10000
ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1

fig, ax = plt.subplots()

fig.patch.set_facecolor("#222222")   # outside graph area
ax.set_facecolor("#333333")          # inside graph area

# Initial y values
y = np.zeros(NUM_STATES)

text = ax.text(
    0.05, 0.95,
    f"Episode: 0/{EPISODES}",
    transform=ax.transAxes,
    color="white",
    fontsize=12,
    verticalalignment="top"
)

x = np.arange(NUM_STATES)  # x values for states 0 to 6
# Draw points
pointsa1, = ax.plot(
    x,
    y,
    marker="o",
    linestyle="-",
    markersize=10,
    color="cyan",
    markerfacecolor="orange",
    markeredgecolor="white",
    markeredgewidth=2
)

pointsa2, = ax.plot(
    x,
    y,
    marker="o",
    linestyle="-",
    markersize=10,
    color="cyan",
    markerfacecolor="orange",
    markeredgecolor="white",
    markeredgewidth=2
)

ax.set_xlim(-0.5, 7)
ax.set_ylim(-1.5, 1.5)
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")

ax.grid(
    True,
    color="gray",
    linestyle="--",
    linewidth=0.5,
    alpha=0.5
)

ax.set_xlabel("States")
ax.set_ylabel("Values")

def choose_action(q_table, state):
    """
    Epsilon-greedy action selection.
    """
    if random.random() < EPSILON:
        return random.choice([LEFT, RIGHT])
    return np.argmax(q_table[state])


def take_step(state, action):
    """
    Apply action and return next_state, reward, done.
    """
    if action == LEFT:
        next_state = state - 1
    else:
        next_state = state + 1

    if next_state == LEFT_TERMINAL:
        return next_state, -1, True

    if next_state == RIGHT_TERMINAL:
        return next_state, 1, True

    return next_state, 0, False


def run_episode(q_table):
    state = START_STATE
    done = False

    while not done:
        action = choose_action(q_table, state)
        next_state, reward, done = take_step(state, action)

        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(q_table[next_state])

        q_table[state, action] += ALPHA * (
            target - q_table[state, action]
        )

        state = next_state


def train():
    episode_count = 0
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    def init():
        pointsa1.set_data(x, q_table[:, LEFT])
        pointsa2.set_data(x, q_table[:, RIGHT])
        text.set_text(f"Episode: {episode_count}/{EPISODES}")
        return pointsa1, pointsa2, text

    def update(frame):
        nonlocal episode_count
        run_episode(q_table)
        episode_count += 1

        pointsa1.set_data(x, q_table[:, LEFT])
        pointsa2.set_data(x, q_table[:, RIGHT])
        text.set_text(f"Episode: {episode_count}/{EPISODES}")

        return pointsa1, pointsa2, text

    animation = FuncAnimation(
        fig,
        update,
        frames=EPISODES,
        init_func=init,
        interval=1,
        blit=True,
        repeat=False
    )
        
    plt.show()
    
    return q_table


def run_trained_agent(q_table):
    state = START_STATE
    path = [state]
    done = False

    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done = take_step(state, action)

        path.append(next_state)
        state = next_state

    return path, reward


if __name__ == "__main__":
    q_table = train()

    print("Learned Q-table:")
    print(q_table)

    print("\nLearned policy:")
    for state in range(1, 6):
        best_action = np.argmax(q_table[state])
        action_name = "left" if best_action == LEFT else "right"
        print(f"State {state}: move {action_name}")

    path, final_reward = run_trained_agent(q_table)

    print("\nPath taken by trained agent:")
    print(path)

    print("\nFinal reward:")
    print(final_reward)
