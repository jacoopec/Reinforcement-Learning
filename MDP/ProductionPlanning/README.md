# Toy Production Planning & Control MDP (Simple Rollout)

This project contains a small Python script that models a **toy production planning & control problem** as a **Markov Decision Process (MDP)** and runs a simple **episode simulation** (“rollout”) using either a random policy or a user-defined policy.

It **does not solve** the MDP (no optimization, dynamic programming, or reinforcement learning). It only defines the MDP dynamics and simulates trajectories.

---

## What the script represents

### State space
Each state is a pair:

- **Inventory level**: an integer from 0 up to a fixed maximum (discrete inventory buckets)
- **Capacity regime**: `LowCap` or `HighCap` (machine availability mode)

The capacity regime represents whether the system currently has low or high production capacity available.

### Action space
At every step the controller chooses a **production quantity** (units produced this step).

The available actions depend on current capacity:
- In `LowCap`, production is limited to a small maximum.
- In `HighCap`, production can be higher.

For simplicity, production is assumed to be available immediately within the step.

---

## How the environment evolves (transitions)

At each step:

1) **Choose production quantity** within the current capacity.
2) **Demand is realized** stochastically.
3) **Inventory updates**:
   - inventory increases by production (capped at a maximum)
   - inventory decreases by sales (sales are limited by available inventory)
4) **Unmet demand** is recorded (lost sales / service shortfall).
5) **Capacity regime changes** using a simple 2-state Markov chain (it tends to persist, but can switch).

The script records step details such as capacity, produced units, demand, sales, unmet demand, and inventory after production.

---

## Reward signal (profit)

The reward is a simple one-step profit calculation:

- **Revenue** from units sold (price × sales)
- **Variable production cost** (cost × produced units)
- **Setup cost** (a fixed cost whenever production is greater than zero)
- **Holding cost** on ending inventory
- **Penalty** for unmet demand (lost sales / backlog penalty)

This captures typical production planning trade-offs:
- producing enough to meet demand (avoid penalties)
- avoiding excess inventory (holding cost)
- avoiding excessive setups (setup cost)
- respecting capacity constraints

The total episode reward is the sum of step rewards.

---

## Episode simulation (rollout)

An episode is a sequence of steps starting from an initial state. At each step:
1) the policy selects production
2) demand is sampled and inventory updates
3) a reward (profit) is produced
4) the trajectory is recorded and printed

The script prints:
- **Total reward** for the episode
- The **trajectory**, listing the next state plus the action and reward that led there, along with step details

Runs can be made reproducible by setting a random seed.

---

## Example policy

The script includes a simple heuristic policy that aims for a target inventory level:

- If inventory is below the target, produce enough to close the gap (up to capacity).
- Otherwise, produce nothing.

You can replace it with any policy you like (random, heuristic, or learned elsewhere).

---

## Limitations (by design)

- This is a **toy** MDP: small, discrete, and not calibrated to real production data.
- Inventory is represented as **small integer buckets**, not a full-scale continuous system.
- Unmet demand is **penalized but not carried** as backlog (no backorders).
- There is **no solver**; the script is meant to demonstrate MDP structure and simulation.

---

## Requirements

- Python 3.x
- No external libraries (standard library only)
