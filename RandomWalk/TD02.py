import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

EPISODES = 2
START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

GAMMA = 1.0
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

fixed_x = np.array([1,2,3,4,5,6,7])
fixed_y = np.array([-1,-0.6667, -0.3333, 0.0000, 0.3333, 0.6667,1])

fixed_points = ax.scatter(
    fixed_x,
    fixed_y,
    s=120,              # point size
    color="white",
    alpha=0.25,         # transparency: 0 invisible, 1 solid
    marker="o",
    label="Fixed points"
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

counter  = 0


def train():
    
    values = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
    }
    
    def update(frame):
        state = START_STATE

        while state not in [LEFT_TERMINAL, RIGHT_TERMINAL]:
            old_state = state

            action = random.choice([-1, 1])
            next_state = state + action

            if next_state == LEFT_TERMINAL:
                reward = -1
            elif next_state == RIGHT_TERMINAL:
                reward = 1
            else:
                reward = 0
                
            f = values[next_state] 
            l = values[old_state]

            values[old_state] = values[old_state] + ALPHA * (
                reward + GAMMA * values[next_state] - values[old_state]
            )

            state = next_state
        
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