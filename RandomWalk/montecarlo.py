import random


# States:
# 0 = losing terminal state
# 1, 2, 3, 4, 5 = normal states
# 6 = winning terminal state

START_STATE = 3
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6

EPISODES = 5
ALPHA = 0.1

visited_states = set()


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
    """
    Monte Carlo state-value learning.

    After each episode ends, every visited state is updated
    using the final return.
    """
    values = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
    }
    global  visited_states

    for _ in range(EPISODES):
        episode = run_episode()

        # In this simple problem, only the final reward matters.
        final_return = episode[-1][1]

        visited_states = set()

        for state, _ in episode:
            # First-visit Monte Carlo:
            # update each state only once per episode.
            if state not in visited_states:
                values[state] = values[state] + ALPHA * (final_return - values[state])
                visited_states.add(state)

    return values


if __name__ == "__main__":
    values = train()
    
    print("Visited states during training:")

    print("Estimated state values:")
    for state in sorted(values):
        print(f"State {state}: {values[state]:.3f}")