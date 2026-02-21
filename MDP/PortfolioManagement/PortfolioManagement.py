import random

# ----- MDP definition (toy portfolio) -----
# State = (wealth_level, market_regime)
# wealth_level: Low / Mid / High (coarse discretization)
# market_regime: Bear / Bull
states = [(w, m) for w in ["Low", "Mid", "High"] for m in ["Bear", "Bull"]]

# Action = target allocation to risky asset
actions = ["RiskOff", "Balanced", "RiskOn"]  # ~ 0%, 50%, 100% risky

def transition(s, a):
    """
    Stochastic transition: returns next_state.
    - market regime follows a Markov chain
    - wealth changes depending on regime + action
    """
    wealth, market = s

    # 1) Market regime transition (Markov)
    u = random.random()
    if market == "Bear":
        market_next = "Bear" if u < 0.7 else "Bull"
    else:  # Bull
        market_next = "Bull" if u < 0.7 else "Bear"

    # 2) Wealth transition depends on action + current market (simple rules)
    # Think of this as "expected" wealth drift (no numbers, just buckets).
    if a == "RiskOff":
        # steady/safe: tends to stay or drift up slowly
        if wealth == "Low":
            wealth_next = "Mid" if random.random() < 0.4 else "Low"
        elif wealth == "Mid":
            wealth_next = "High" if random.random() < 0.2 else "Mid"
        else:  # High
            wealth_next = "High"
    elif a == "Balanced":
        if market == "Bull":
            # in bull markets, balanced tends to improve
            if wealth == "Low":
                wealth_next = "Mid" if random.random() < 0.7 else "Low"
            elif wealth == "Mid":
                wealth_next = "High" if random.random() < 0.5 else "Mid"
            else:
                wealth_next = "High" if random.random() < 0.8 else "Mid"
        else:
            # in bear markets, balanced can slip
            if wealth == "High":
                wealth_next = "Mid" if random.random() < 0.5 else "High"
            elif wealth == "Mid":
                wealth_next = "Low" if random.random() < 0.3 else "Mid"
            else:
                wealth_next = "Low"
    else:  # RiskOn
        if market == "Bull":
            # risk-on helps more often in bull markets
            if wealth == "Low":
                wealth_next = "Mid" if random.random() < 0.8 else "Low"
            elif wealth == "Mid":
                wealth_next = "High" if random.random() < 0.7 else "Mid"
            else:
                wealth_next = "High"
        else:
            # risk-on hurts more often in bear markets
            if wealth == "High":
                wealth_next = "Mid" if random.random() < 0.8 else "High"
            elif wealth == "Mid":
                wealth_next = "Low" if random.random() < 0.6 else "Mid"
            else:
                wealth_next = "Low"

    return (wealth_next, market_next)

def reward(s, a, s_next):
    """
    Reward: +1 if wealth goes up a level, -1 if it goes down, 0 otherwise.
    (Simple "make money / lose money" signal.)
    """
    order = {"Low": 0, "Mid": 1, "High": 2}
    w, _ = s
    w2, _ = s_next
    if order[w2] > order[w]:
        return 1
    if order[w2] < order[w]:
        return -1
    return 0

# ----- A simple rollout (simulate an agent) -----
def run_episode(start=("Mid", "Bear"), steps=10, policy=None, seed=0):
    random.seed(seed)
    s = start
    total = 0
    trajectory = [(s, None, 0)]

    for t in range(steps):
        # policy: a function that chooses an action given state
        if policy is None:
            a = random.choice(actions)  # random behavior
        else:
            a = policy(s)

        s_next = transition(s, a)
        r = reward(s, a, s_next)

        total += r
        trajectory.append((s_next, a, r))
        s = s_next

    return total, trajectory

# Example policy: be cautious in Bear, take risk in Bull
def simple_policy(s):
    wealth, market = s
    return "RiskOff" if market == "Bear" else "RiskOn"

if __name__ == "__main__":
    total, traj = run_episode(start=("Mid", "Bear"), steps=10, policy=simple_policy, seed=1)
    print("Total reward:", total)
    print("Trajectory (state, action_taken_to_get_here, reward):")
    for item in traj:
        print(item)