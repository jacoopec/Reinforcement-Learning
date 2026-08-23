# Uniform-Policy Gridworld: State-Value and Action-Value Functions

This README computes the **state-value function** $V_\pi(s)$ and **action-value function** $Q_\pi(s,a)$ for the pictured 4×4 gridworld example under a **uniform policy**:

- Policy: $\pi(a\mid s)=0.25$ for the 4 actions (Up, Right, Down, Left)
- Reward per step: $R_t = -1$
- Terminal states: **top-left** and **bottom-right** with value 0
- Transition model (standard gridworld):
  - actions move deterministically to the neighboring cell
  - if an action would go off the grid, the agent stays in the same cell
- Discount factor: $\gamma = 1$ (common for episodic gridworlds)

---

## Bellman relationships used

For a non-terminal state $s$, taking action $a$ leads to next state $s'$. Then:

$$
Q_\pi(s,a) = -1 + \gamma V_\pi(s')
$$

And with a uniform policy:

$$
V_\pi(s) = \frac{1}{4}\sum_{a\in\{U,R,D,L\}} Q_\pi(s,a)
$$

---

## State-value function $V_\pi(s)$

Values are listed row by row (terminal states are 0):

Row 1:  $[0.000,\ -0.775,\ -0.850,\ -0.700]$  
Row 2:  $[-0.750,\ -1.025,\ -0.525,\ -0.775]$  
Row 3:  $[-0.850,\ -0.325,\ -1.275,\ -0.625]$  
Row 4:  $[-0.500,\ -1.025,\ -0.850,\ 0.000]$

---

## Action-value function $Q_\pi(s,a)$

Action order used below: **(Up, Right, Down, Left)**

### Row 1
- (1,2): $(-0.600,\ -0.800,\ -0.700,\ -1.000)$
- (1,3): $(-0.800,\ -0.900,\ -1.100,\ -0.600)$
- (1,4): $(-0.900,\ -0.900,\ -0.200,\ -0.800)$

### Row 2
- (2,1): $(-1.000,\ -0.700,\ -0.100,\ -1.200)$
- (2,2): $(-0.600,\ -1.100,\ -1.200,\ -1.200)$
- (2,3): $(-0.800,\ -0.200,\ -0.400,\ -0.700)$
- (2,4): $(-0.900,\ -0.200,\ -0.900,\ -1.100)$

### Row 3
- (3,1): $(-1.200,\ -1.200,\ -0.900,\ -0.100)$
- (3,2): $(-0.700,\ -0.400,\ -0.100,\ -0.100)$
- (3,3): $(-1.100,\ -0.900,\ -1.900,\ -1.200)$
- (3,4): $(-0.200,\ -0.900,\ -1.000,\ -0.400)$

### Row 4
- (4,1): $(-0.100,\ -0.100,\ -0.900,\ -0.900)$
- (4,2): $(-1.200,\ -1.900,\ -0.100,\ -0.900)$
- (4,3): $(-0.400,\ -1.000,\ -1.900,\ -0.100)$

---

## Notes

- These calculations use $Q_\pi(s,a) = -1 + V(s')$ because $\gamma=1$.
- Terminal states have $V=0$ and no outgoing action-values listed.
- If you want a different discount (e.g., $\gamma=0.9$), replace the backup with:
  $$
  Q_\pi(s,a) = -1 + \gamma V(s')
  $$
  and recompute.


## “If the agent keeps following this random policy, what is the expected return from each state?”

terminal states at (0,0) and (3,3)
reward -1 on every move
uniform random policy π(a|s)=0.25
γ = 1.0
if an action would leave the grid, the agent stays in the same state.

It is policy evaluation, not full policy iteration.

Why:

Policy evaluation: compute V(s) or Q(s,a) for a given policy π
Policy iteration: repeatedly do
policy evaluation
policy improvement
until the policy becomes optimal

Your script only does step 1: it evaluates the fixed random policy. It does not improve the policy, so it is not policy iteration.

In one line:
This is iterative policy evaluation for a fixed policy.