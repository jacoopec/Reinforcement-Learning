import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

NUM_STATES = 7
NUM_ACTIONS = 2

LEFT = 0
RIGHT = 1

EPISODES = 5000
ALPHA = 0.1
GAMMA = 1.0
EPSILON = 0.1

visited_states = set()


fig, ax = plt.subplots()

fig.patch.set_facecolor("#222222")   # outside graph area
ax.set_facecolor("#333333")          # inside graph area

# Initial y values
y = np.random.rand(7)

x = np.array([1, 2, 3, 4, 5,6,7])  # x values for states s^-2 to s^2
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
counter  = 0
def train():
    """
        SARSA learning.

        Update rule:

        Q(s, a) = Q(s, a) + alpha * [
            reward + gamma * Q(s_next, a_next) - Q(s, a)
        ]
    """
    
    q_table = np.array([[0.1,0.4],
                        [0.5,0.3],
                        [0.2,0.4],
                        [0.3,0.3],
                        [0.1,0.2],
                        [0.3,0.1],
                        [0.2,0.3]])
    
            

    def update(frame):
        state = START_STATE
        global counter 
        action = choose_action(q_table, state)
        done = False
        
        while not done:
            next_state, reward, done = take_step(state, action)
            
            # print(f"Episode {counter}: state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}")

            if done:
                target = reward
                
            else:
                next_action = choose_action(q_table, next_state)
                target = reward + GAMMA * q_table[next_state, next_action]
                
            # print(target)
            # print(state, action)

            q_table[state, action] += ALPHA * (
                target - q_table[state, action]
            )
            # Simulate changing values
            new_a1 = q_table[0][0], q_table[1][0], q_table[2][0], q_table[3][0], q_table[4][0], q_table[5][0], q_table[6][0]
            new_a2 = q_table[0][1], q_table[1][1], q_table[2][1], q_table[3][1], q_table[4][1], q_table[5][1], q_table[6][1]

            # Move the points
            pointsa1.set_data(x, new_a1)
            pointsa2.set_data(x, new_a2)

            if not done:
                state = next_state
                action = next_action

        if counter >= EPISODES:
            animation.event_source.stop()
        
        # print(f"Update {counter}: y values = {new_a1}, {new_a2}")
        
        counter += 1

        return pointsa1, pointsa2

    animation = FuncAnimation(
        fig,
        update,
        interval=1,  # update every 500 ms
        blit=True
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