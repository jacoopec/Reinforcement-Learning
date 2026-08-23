Reinforcement Learning (RL) agents learn to make decisions by interacting with an environment and maximizing cumulative reward. Different types of RL agents are categorized based on how they represent knowledge and learn optimal behavior.

Reinforcement learning agents can be categorized based on how they learn:

 - Value-based → learn “how good” states/actions are
 - Policy-based → learn “what to do” directly
 - Actor–Critic → combine both approaches
 - Model-based vs Model-free → differ in use of environment knowledge


------------------

## Value-Based Agents

Value-based agents learn a value function, which estimates how good a state or action is.

State value: 
V(s)

Action value:
Q(s,a)

The policy is derived by choosing the action with the highest value.

Examples
Q-learning
SARSA
Deep Q-Network (DQN)
Characteristics
Works well for discrete action spaces
Simple and stable
Indirectly learns policy
Limitations
Struggles with continuous actions
Can be inefficient in large state spaces


--------------------

## Policy-based agents 

directly learn a policy function:

π(a∣s)

This maps states directly to actions.

Examples
REINFORCE (Monte Carlo policy gradient)
Characteristics
Suitable for continuous action spaces
Can learn stochastic policies
Direct optimization of behavior
Limitations
High variance in learning
Can be unstable without improvements

--------------------

## Actor–Critic Agents
Definition

Actor–Critic methods combine value-based and policy-based approaches:

Actor: decides actions (policy)
Critic: evaluates actions (value function)
Examples
A2C (Advantage Actor-Critic)
A3C
DDPG
PPO
Characteristics
More stable than pure policy methods
Efficient learning
Handles both discrete and continuous spaces
Limitations
More complex to implement
Requires careful tuning

--------------------

## Model-Based vs Model-Free Agents
Model-Free RL
No knowledge of environment dynamics
Learns from experience only

Examples:

Q-learning
DQN
PPO
Model-Based RL
Uses or learns a model of environment transitions
Can plan ahead

Examples:

Value Iteration
Policy Iteration
Dyna-Q










A value-based agent in reinforcement learning is one that learns a value function (like 
𝑉
(
𝑠
)
V(s) or 
𝑄
(
𝑠
,
𝑎
)
Q(s,a)) and derives its policy from that—rather than learning the policy directly.

These methods work best in problems with:

discrete or manageable state/action spaces
clear reward signals
need for optimal decision sequences