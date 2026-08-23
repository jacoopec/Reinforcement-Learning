import random
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

EPISODES = 5000
ALPHA = 0.1
GAMMA = 1.0
EPSILON = 0.1


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


def train():
    """
    SARSA learning.

    Update rule:

    Q(s, a) = Q(s, a) + alpha * [
        reward + gamma * Q(s_next, a_next) - Q(s, a)
    ]
    """

    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    for _ in range(EPISODES):
        state = START_STATE
        action = choose_action(q_table, state)

        done = False

        while not done:
            next_state, reward, done = take_step(state, action)

            if done:
                target = reward
            else:
                next_action = choose_action(q_table, next_state)
                target = reward + GAMMA * q_table[next_state, next_action]

            q_table[state, action] += ALPHA * (
                target - q_table[state, action]
            )

            if not done:
                state = next_state
                action = next_action

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