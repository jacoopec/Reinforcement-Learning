import random


# States:
# 0 = losing terminal state
# 1, 2, 3, 4, 5 = normal states
# 6 = winning terminal state

START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

EPISODES = 5000
ALPHA = 0.1
GAMMA = 1.0


def train():
    """
    TD(0) state-value learning.

    The value of each state is updated immediately after each step,
    using the reward plus the estimated value of the next state.
    """

    values = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        6: 0.0,
    }

    for _ in range(EPISODES):
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

            values[old_state] = values[old_state] + ALPHA * (
                reward + GAMMA * values[next_state] - values[old_state]
            )

            state = next_state

    return values


if __name__ == "__main__":
    values = train()

    print("Estimated state values:")
    for state in range(1, 6):
        print(f"State {state}: {values[state]:.3f}")