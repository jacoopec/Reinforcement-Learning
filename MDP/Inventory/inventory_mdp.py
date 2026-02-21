import random

# ----- MDP definition (toy inventory control) -----
# State = (inventory_level, demand_regime)
# inventory_level: 0..MAX_INV  (coarse, discrete)
# demand_regime: "Low" / "High" (simple demand Markov regime)

MAX_INV = 6
states = [(i, r) for i in range(MAX_INV + 1) for r in ["Low", "High"]]

# Action = order quantity (arrives immediately, for simplicity)
actions = [0, 1, 2, 3]  # units to order

# --- Simple cost / revenue parameters ---
PRICE_PER_UNIT = 5      # revenue per unit sold
ORDER_COST = 2          # cost per unit ordered
HOLDING_COST = 1        # cost per unit of leftover inventory
STOCKOUT_PENALTY = 4    # penalty per unit of unmet demand


def sample_next_regime(regime):
    """Demand regime follows a simple 2-state Markov chain."""
    u = random.random()
    if regime == "Low":
        return "Low" if u < 0.75 else "High"
    else:  # High
        return "High" if u < 0.75 else "Low"


def sample_demand(regime):
    """Stochastic demand conditional on regime (small integers)."""
    if regime == "Low":
        return random.choice([0, 1, 1, 2])   # mostly 0-2
    else:
        return random.choice([1, 2, 3, 3, 4])  # mostly 2-4


def transition(s, a):
    """
    Stochastic transition: returns next_state plus extra info:
    - demand sampled from current regime
    - next demand regime sampled from Markov chain
    Inventory evolves as:
      inv_after_order = min(MAX_INV, inv + order)
      inv_next = max(0, inv_after_order - demand)
    """
    inv, regime = s
    order_qty = a

    inv_after_order = min(MAX_INV, inv + order_qty)
    demand = sample_demand(regime)
    sales = min(inv_after_order, demand)
    unmet = max(0, demand - inv_after_order)

    inv_next = inv_after_order - sales
    regime_next = sample_next_regime(regime)

    s_next = (inv_next, regime_next)
    info = {"demand": demand, "sales": sales, "unmet": unmet, "inv_after_order": inv_after_order}
    return s_next, info


def reward(s, a, s_next, info):
    """
    Reward = profit (revenue - costs), using a simple one-step accounting:
    - revenue from sales
    - ordering cost
    - holding cost on leftover inventory
    - stockout penalty for unmet demand
    """
    order_qty = a
    sales = info["sales"]
    unmet = info["unmet"]
    inv_next, _ = s_next

    revenue = PRICE_PER_UNIT * sales
    ordering_cost = ORDER_COST * order_qty
    holding_cost = HOLDING_COST * inv_next
    stockout_cost = STOCKOUT_PENALTY * unmet

    return revenue - ordering_cost - holding_cost - stockout_cost


# ----- A simple rollout (simulate an agent) -----
def run_episode(start=(3, "Low"), steps=10, policy=None, seed=0):
    random.seed(seed)
    s = start
    total = 0
    trajectory = [(s, None, 0, {"note": "start"})]

    for t in range(steps):
        # policy: a function that chooses an action given state
        if policy is None:
            a = random.choice(actions)  # random behavior
        else:
            a = policy(s)

        s_next, info = transition(s, a)
        r = reward(s, a, s_next, info)

        total += r
        trajectory.append((s_next, a, r, info))
        s = s_next

    return total, trajectory


# Example policy: simple (s,S)-like heuristic
# - If inventory is low, order more; if high, order less.
def simple_reorder_policy(s):
    inv, regime = s
    if inv <= 1:
        return 3
    if inv == 2:
        return 2
    if inv == 3:
        return 1
    return 0


if __name__ == "__main__":
    total, traj = run_episode(start=(3, "Low"), steps=12, policy=simple_reorder_policy, seed=1)
    print("Total reward:", total)
    print("Trajectory (state, action_taken_to_get_here, reward, info):")
    for item in traj:
        print(item)
