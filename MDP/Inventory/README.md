# Toy Inventory Control MDP (Simple Rollout)

This project contains a small Python script that models a **toy inventory control problem** as a **Markov Decision Process (MDP)** and runs a simple **episode simulation** (“rollout”) using either a random policy or a user-defined policy.

It **does not solve** the MDP (no optimization, dynamic programming, or reinforcement learning). It only defines the MDP dynamics and simulates trajectories.

---

## What the script represents

### State space
Each state is a pair:

- **Inventory level**: an integer from 0 to a fixed maximum (a small discrete “bucketed” inventory)
- **Demand regime**: `Low` or `High`

So the state captures both the current stock on hand and whether demand is currently in a low-demand or high-demand mode.

### Action space
At every step the agent chooses an **order quantity** from a small discrete set (for example: 0, 1, 2, 3 units).

For simplicity, the script assumes orders **arrive immediately** (no lead time).

---

## How the environment evolves (transitions)

At each step:

1) **An order is placed** (your action). Inventory increases, but is capped at a maximum level.
2) **Demand is realized** stochastically, based on the current demand regime:
   - In the `Low` regime, demand is usually small.
   - In the `High` regime, demand is usually larger.
3) **Sales occur** up to the available inventory.
4) **Leftover inventory** becomes the next inventory level.
5) The **demand regime changes** using a simple 2-state Markov chain (it tends to persist, but can switch).

The script also records step information such as realized demand, sales, unmet demand, and inventory after ordering.

---

## Reward signal (profit)

The reward is a simple one-step profit calculation:

- **Revenue** from units sold (price × sales)
- **Ordering cost** (cost × units ordered)
- **Holding cost** on leftover inventory (cost × ending inventory)
- **Stockout penalty** for unmet demand (penalty × unmet units)

So each step reward reflects the trade-off between:
- ordering enough to meet demand,
- avoiding excess inventory,
- and avoiding stockouts.

The total episode reward is the sum of step rewards.

---

## Episode simulation (rollout)

An episode is a sequence of steps starting from an initial state. At each step:
1) the policy picks an order quantity
2) demand is sampled and inventory updates
3) a reward (profit) is produced
4) the trajectory is recorded and printed

The script prints:
- **Total reward** for the episode
- The **trajectory**, listing the next state plus the action and reward that led there, along with step details

Runs can be made reproducible by setting a random seed.

---

## Example policy

The script includes a simple heuristic policy similar to an (s,S)-style rule:

- If inventory is very low, order more
- If inventory is moderate, order a little
- If inventory is high, order nothing

You can replace it with any policy you like (random, heuristic, or learned elsewhere).

---

## Limitations (by design)

- This is a **toy** MDP: small, discrete, and not calibrated to real operational data.
- Inventory is represented as **small integer buckets**, not a full-scale continuous system.
- Orders are assumed to arrive **immediately** (no lead times).
- There is **no solver**; the script is meant to demonstrate MDP structure and simulation.

---

## Requirements

- Python 3.x
- No external libraries (standard library only)
