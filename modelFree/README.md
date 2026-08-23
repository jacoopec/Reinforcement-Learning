##  Model-Free Methods in Reinforcement Learning

>Model-free methods learn optimal behavior directly from experience without building a model of the environment.

Optimize the value functionof an unknownMDP.

These models helps where:
 • MDP model is unknown, but experience can be sampled
 • MDP model isknown, but  is too big to use, excep tby samples
 • Model-free  control  can solve the seproblems


Model-free RL methods do not know the environment dynamics.
They do not have access to:

Transition probabilities $P(s′∣s,a)$, Reward function or $R(s,a)$. Instead, they learn purely from experience: $(s,a,r,s′)$

The agent improves by interacting with the environment (trial and error).

Core idea

A model-free agent learns either:

 - **Value functions** → how good states/actions are
 - **Policies** → what action to take directly

without ever building a full model of the environment.
---
## Main categories of model-free methods
---
### Value-Based Methods
Idea: Learn a function: $Q(s,a)$ that estimates the expected return of taking action a in state 𝑠.


`Model-free methods learn optimal behavior directly from experience without building a model of the environment.`
---
### Policy-Based Methods
Idea: Directly learn: $π(a∣s)$
Examples
REINFORCE (Monte Carlo policy gradient)
Key concept

Optimize expected reward via gradient: ∇J(θ)

 - Good for continuous actions
 - Can learn stochastic policies





