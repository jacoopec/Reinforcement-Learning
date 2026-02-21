import random

# ----- MDP definition (toy Production Planning & Control) -----
# We model a single-product, single-period-per-step planning problem.
#
# State = (inventory_level, capacity_regime)
# - inventory_level: 0..MAX_INV  (discrete)
# - capacity_regime: "LowCap" / "HighCap" (machine availability regime)
#
# Action = production quantity (units produced this step), bounded by current capacity.
#
# Dynamics per step:
# 1) Observe state (inventory, capacity regime)
# 2) Choose production quantity (<= capacity)
# 3) Demand is realized (stochastic)
# 4) Inventory updates: inv_next = clamp(inv + prod - sales, 0..MAX_INV)
# 5) Capacity regime transitions via a Markov chain
#
# Reward = profit = revenue - production cost - holding cost - backlog penalty
# (Backlog is not carried; unmet demand is penalized and lost, for simplicity.)

MAX_INV = 8
CAP_LOW = 2
CAP_HIGH = 5

states = [(i, r) for i in range(MAX_INV + 1) for r in ["LowCap", "HighCap"]]

# Demand regimes are implicit in demand sampling; you can extend state to include it if desired.

# --- Simple parameters ---
PRICE_PER_UNIT = 6         # revenue per unit sold
PROD_COST = 2              # cost per unit produced
SETUP_COST = 2             # fixed cost if you produce > 0 (represents setup/changeover)
HOLDING_COST = 1           # cost per unit of ending inventory
BACKLOG_PENALTY = 5        # penalty per unit of unmet demand (lost sales / service penalty)

def capacity_from_regime(regime):
    return CAP_LOW if regime == "LowCap" else CAP_HIGH

def sample_next_capacity_regime(regime):
    """Capacity availability follows a 2-state Markov chain."""
    u = random.random()
    if regime == "LowCap":
        return "LowCap" if u < 0.75 else "HighCap"
    else:
        return "HighCap" if u < 0.75 else "LowCap"

def sample_demand():
    """Stochastic customer demand (small integers)."""
    # Simple demand distribution (can be swapped for something more realistic)
    return random.choice([0, 1, 2, 2, 3, 3, 4])

def available_actions(state):
    """Action set depends on capacity regime."""
    inv, cap_reg = state
    cap = capacity_from_regime(cap_reg)
    return list(range(cap + 1))  # produce 0..cap

def transition(s, a):
    """
    Stochastic transition: returns next_state plus info dict.
    - production 'a' must be within current capacity (use available_actions)
    - demand realized after production
    """
    inv, cap_reg = s
    cap = capacity_from_regime(cap_reg)
    if a < 0 or a > cap:
        raise ValueError(f"Action {a} exceeds current capacity {cap} for regime {cap_reg}")

    demand = sample_demand()

    inv_after_prod = min(MAX_INV, inv + a)
    sales = min(inv_after_prod, demand)
    unmet = max(0, demand - inv_after_prod)

    inv_next = inv_after_prod - sales
    cap_next = sample_next_capacity_regime(cap_reg)

    s_next = (inv_next, cap_next)
    info = {
        "capacity": cap,
        "produced": a,
        "demand": demand,
        "sales": sales,
        "unmet": unmet,
        "inv_after_prod": inv_after_prod,
    }
    return s_next, info

def reward(s, a, s_next, info):
    """
    Reward = one-step profit:
    - revenue from sales
    - variable production cost + setup cost if produced > 0
    - holding cost on ending inventory
    - penalty for unmet demand (lost sales/backlog penalty)
    """
    sales = info["sales"]
    unmet = info["unmet"]
    inv_next, _ = s_next

    revenue = PRICE_PER_UNIT * sales
    prod_cost = PROD_COST * a
    setup = SETUP_COST if a > 0 else 0
    holding = HOLDING_COST * inv_next
    backlog = BACKLOG_PENALTY * unmet

    return revenue - prod_cost - setup - holding - backlog

# ----- A simple rollout (simulate a planner/controller) -----
def run_episode(start=(3, "HighCap"), steps=12, policy=None, seed=0):
    random.seed(seed)
    s = start
    total = 0
    trajectory = [(s, None, 0, {"note": "start"})]

    for t in range(steps):
        # policy: chooses production quantity given state
        if policy is None:
            a = random.choice(available_actions(s))
        else:
            a = policy(s)

        s_next, info = transition(s, a)
        r = reward(s, a, s_next, info)

        total += r
        trajectory.append((s_next, a, r, info))
        s = s_next

    return total, trajectory

# Example heuristic policy:
# - target inventory around TARGET level
# - produce more if inventory is below target, within capacity
TARGET_INV = 4

def simple_production_policy(s):
    inv, cap_reg = s
    cap = capacity_from_regime(cap_reg)
    gap = TARGET_INV - inv
    if gap <= 0:
        return 0
    return min(cap, gap)

if __name__ == "__main__":
    total, traj = run_episode(start=(3, "HighCap"), steps=12, policy=simple_production_policy, seed=1)
    print("Total reward:", total)
    print("Trajectory (state, action_taken_to_get_here, reward, info):")
    for item in traj:
        print(item)
