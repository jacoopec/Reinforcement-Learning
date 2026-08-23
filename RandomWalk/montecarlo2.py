import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

EPISODES = 2
START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

EPISODES = 120
ALPHA = 0.1

visited_states = set()


fig, ax = plt.subplots()

fig.patch.set_facecolor("#222222")   # outside graph area
ax.set_facecolor("#333333")          # inside graph area

# Initial y values
y = np.random.rand(7)

x = np.array([1, 2, 3, 4, 5,6,7])  # x values for states s^-2 to s^2
# Draw points
points, = ax.plot(
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

ax.set_xlim(0, 6)
ax.set_ylim(-1, 1)
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

counter  = 0





def run_episode():
    """
    Run one full random-walk episode.

    The agent moves randomly left or right until it reaches
    either terminal state 0 or terminal state 6.
    """
    state = START_STATE
    episode = []

    while state not in [LEFT_TERMINAL, RIGHT_TERMINAL]:
        old_state = state

        action = random.choice([-1, 1])
        state += action

        if state == LEFT_TERMINAL:
            reward = -1
        elif state == RIGHT_TERMINAL:
            reward = 1
        else:
            reward = 0

        episode.append((old_state, reward))

    return episode

def train():
    
    values = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
    }
    
    def update(frame):
        episode = run_episode()
        final_return = episode[-1][1]
        
        visited_states = set()

        for state, _ in episode:
            # First-visit Monte Carlo:
            # update each state only once per episode.
            if state not in visited_states:
                values[state] = values[state] + ALPHA * (final_return - values[state])
                visited_states.add(state)
        
        # Simulate changing values
        new_y = values[1], values[2], values[3], values[4], values[5], values[6], values[7]

        # Move the points
        points.set_data(x, new_y)
        global counter
        counter += 1
        print(f"Update {counter}: y values = {new_y}")
        
        if counter >= EPISODES:
            animation.event_source.stop()
            print("Animation stopped after 5 updates.")
        return points,



    animation = FuncAnimation(
        fig,
        update,
        interval=100,  # update every 500 ms
        blit=True
    )
        


    plt.show()


if __name__ == "__main__":
    values = train()
    
    print("Visited states during training:")

    print("Estimated state values:")
    for state in sorted(values):
        print(f"State {state}: {values[state]:.3f}")