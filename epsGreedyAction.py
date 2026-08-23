import random

def epsilon_greedy_action(Q, s, actions, eps=0.1):
    """
    Q: dict like Q[(s,a)] -> value  OR  nested dict Q[s][a] -> value
    s: current state
    actions: list of available actions in state s
    eps: exploration rate (epsilon)
    """
    # explore
    if random.random() < eps:
        return random.choice(actions)

    # exploit (argmax Q)
    def q_value(a):
        try:
            return Q[(s, a)]
        except KeyError:
            return Q.get(s, {}).get(a, 0.0)

    best_a = max(actions, key=q_value)
    return best_a


# ---- example usage ----
actions = ["left", "right", "stay"]
s = "S1"

Q = {
    ("S1", "left"): 3.2,
    ("S1", "right"): 4.9,
    ("S1", "stay"): 4.1
}

eps = 0.1
print("Chosen action:", epsilon_greedy_action(Q, s, actions, eps=eps))
