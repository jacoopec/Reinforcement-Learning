# Toy Robot Walking MDP (Simple Rollout)

This project contains a small Python script that models a **toy robot walking (locomotion) problem** as a **Markov Decision Process (MDP)** and runs a simple **episode simulation** (“rollout”) using either a random policy or a user-defined policy.

It **does not solve** the MDP (no optimization or reinforcement learning). It only defines the MDP dynamics and simulates trajectories.

---

## What the script represents

### State space
Each state is a pair:

- **Position**: an integer from 0 to a fixed goal value (discrete progress along a 1D track)
- **Stability**: `Stable`, `Wobbly`, or `Fallen`

`Fallen` is a terminal/absorbing condition; the episode ends when the robot falls.

### Action space
At each step the controller chooses one of three gait options:

- **SmallStep**: safer, slower forward progress
- **BigStep**: faster progress, higher risk of instability/falling
- **Recover**: attempt to regain stability; may lose progress

---

## How the environment evolves (transitions)

At each step:

1) The policy selects an action (gait choice).
2) The robot's **next position** and **stability** are sampled stochastically based on the action and current stability.

Key intuition:
- If the robot is `Wobbly`, stepping actions are riskier and falling is more likely.
- `SmallStep` tends to move forward more safely.
- `BigStep` tends to move forward faster but increases the chance of `Wobbly` or `Fallen`.
- `Recover` often improves stability but can cost time and sometimes lose a bit of position.

The episode terminates when:
- the robot **reaches the goal position**, or
- the robot becomes **Fallen**.

---

## Reward signal

The reward encourages forward progress while discouraging falling:

- Positive reward proportional to **forward progress** each step
- Small penalty for being `Wobbly` (encourage stability)
- Large negative penalty for `Fallen`
- Bonus reward for reaching the goal

The total episode reward is the sum of step rewards.

---

## Episode simulation (rollout)

An episode simulates a sequence of steps from an initial state. At each step:
1) choose action via policy (or random)
2) sample next state
3) compute reward
4) record the trajectory

The script prints:
- **Total reward**
- The **trajectory**, showing each next state and the action/reward that led there

Runs can be made reproducible by setting a random seed.

---

## Example policy

The script includes a simple heuristic policy:

- If `Wobbly`, choose **Recover**
- Otherwise, choose **BigStep** to move quickly

You can replace it with any policy you like (random, heuristic, or learned elsewhere).

---

## Limitations (by design)

- This is a **toy** MDP: discrete states, simple probabilities, and not a physics simulator.
- Position is a small integer; real locomotion is continuous with many degrees of freedom.
- There is **no solver**; the script is meant to demonstrate MDP structure and simulation.

---

## Requirements

- Python 3.x
- No external libraries (standard library only)
