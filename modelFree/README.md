##  Model-Free Methods in Reinforcement Learning

Model-free RL does not explicitly learn the environment dynamics. It learns directly how good actions/states are or what action to take.

Examples:

 - Monte Carlo
 - SARSA
 - Q-learning
 - DQN
 - PPO
 - SAC

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

---
### Model-free control
Not estimate but evaluate a policy 

• On-Policy Monte-Carlo Control
• On-Policy Temporal-Difference Control (SARSA)
• Off-Policy Learning (Q-learning)

2 ways  to estimate a policy are state value funtction and action-value function
it is possible to improve the policy from the value function by acting greedily:
$$π′ = greedy(vπ)$$
I choose the policy which maximizes the reward.
Nei contesti model-free abbiamo a disposizione solo Q(s, a) perch´e per essere greedy rispetto alla state-value
function ho bisogno del modello e in particolare sapere quale azione mi porta in quale stato (probabilit`a di
transizione).

Per i metodi di model-free control abbiamo in generale due possibilit`a:
• On-policy learning: impara la policy migliore basandosi sui suoi episodi
• Off-policy learning: impara la policy migliore basandosi su episodi di un’altra policy
