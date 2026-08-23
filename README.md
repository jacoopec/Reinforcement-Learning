# Reinforcement learning

Reinforcement Learning (RL) agents learn to make decisions by interacting with an environment and maximizing cumulative reward. Different types of RL agents are categorized based on how they represent knowledge and learn optimal behavior.

Reinforcement learning agents can be categorized based on how they learn:

 - **Value-based** → learn “how good” states/actions are
 - **Policy-based** → learn “what to do” directly
 - **Actor–Critic** → combine both approaches
 - **Model-based vs Model-free** → differ in use of environment knowledge


>In practice, when someone describes an RL algorithm, a good set of questions is: Does it learn a model? Does it learn values or a policy? Is it on- or off-policy? Does it use MC or TD updates? Is learning online or offline?


Reinforcement Learning
│
├── Model-based RL
│   ├── Known environment model
│   │   └── Dynamic Programming
│   │
│   └── Learned environment model
│       ├── Dyna
│       ├── MuZero
│       └── Dreamer
│
└── Model-free RL
    │
    ├── Value-based
    │   ├── Monte Carlo Control
    │   ├── SARSA
    │   ├── Q-learning
    │   └── DQN
    │
    ├── Policy-based
    │   └── REINFORCE
    │
    └── Actor-Critic
        ├── A2C
        ├── PPO
        ├── DDPG
        ├── TD3
        └── SAC

| Method              | Model           | Main type                  | Update               | Policy          |
| ------------------- | --------------- | -------------------------- | -------------------- | --------------- |
| Monte Carlo Control | Model-free      | Value-based                | MC                   | On/Off          |
| SARSA               | Model-free      | Value-based                | TD                   | On-policy       |
| Q-learning          | Model-free      | Value-based                | TD                   | Off-policy      |
| DQN                 | Model-free      | Value-based                | TD                   | Off-policy      |
| REINFORCE           | Model-free      | Policy-based               | MC                   | On-policy       |
| PPO                 | Model-free      | Actor-Critic               | TD/GAE               | On-policy       |
| SAC                 | Model-free      | Actor-Critic               | TD                   | Off-policy      |
| DreamerV3           | **Model-based** | Actor-Critic + World Model | TD/imagined rollouts | Off-policy-like |
---

## On-policy vs. Off-policy

This classification asks whether the algorithm learns from the same policy that generated the data.

On-policy:

behavior policy=policy being learned

Examples:

SARSA
REINFORCE
PPO
On-policy Monte Carlo

Off-policy:

behavior policy

=target policy

Examples:

Q-learning
DQN
DDPG
SAC
Off-policy Monte Carlo

This is why algorithms such as DQN and SAC can effectively use a replay buffer containing older experiences.
---
## Monte Carlo vs. Temporal Difference

Another important distinction concerns how value estimates are updated.

Monte Carlo waits until the return is observed:
Gt	​=Rt+1	​+γRt+2	​+γ2Rt+3	​+⋯Then:V(St	​)←V(St	​)+α(Gt	​−V(St	​))

Examples:

 -MC Prediction
 -MC Control
 -REINFORCE

Temporal Difference (TD) bootstraps from another estimate:

$$V(St	​)←V(St	​)+α[Rt+1	​+γV(St+1	​)−V(St	​)]$$

Examples:

 - TD(0)
 - SARSA
 - Q-learning
 - DQN

---

## Tabular vs. Function Approximation

Tabular RL stores values explicitly: $Q(s,a)$ for every state-action combination.

Examples:

 - Tabular Q-learning
 - Tabular SARSA
 - Tabular Monte Carlo

It works well when the state space is small.

Function approximation represents values or policies with a parameterized function:

Qθ​(s,a) or πθ(a∣s)

When the function is a neural network, we usually talk about Deep Reinforcement Learning.

Examples:

DQN
PPO
SAC
Dreamer

---

## Online vs. Offline RL

Online RL allows the agent to interact with the environment while learning:

s
t
	​

→a
t
	​

→r
t
	​

,s
t+1
	​


Examples:

Q-learning during environment interaction
PPO
SAC

Offline RL trains entirely from an existing dataset:

D={(s,a,r,s
′
)}

without further environment interaction.

Examples:

CQL
IQL
Decision Transformer

This distinction is particularly important in robotics, autonomous driving, and embodied AI because collecting new interactions can be expensive or dangerous.
---

### Reward
### episode
sequence of steps come to end.
At each step:
 - the agent: receives a reward and an observation, emits an action
 - the environment: receives an action, emits a reward and an observation

### History
An agent will have  a sequence of observation, reward,  action

### State
Information used to decide the next action and the reward

### Policy
Decides  the agent's bhaviour and maps states to actions.

### State value function
expected return  being in state s and following policy pi.

### Action valu function 
expected return  being in state s, taking action a and following policy pi.

###  Model
It explicitly describeshow the environment works.

### Model-free prediction methods
They estimate the valu function given a policy in a non-observable environment
- montecarlo learning
- temporal diff learning



### Markov decision processes 
formally deescribe an environment for reinforcement learning


### L’equazione di Bellman 
nel Reinforcement Learning descrive il valore di uno stato come:

ricompensa immediata + valore atteso degli stati futuri

State value  function 
$$v_pi(s)=e[Gt|s]$$

Action value  function 
$$q_pi(s)=e[Gt|St=s,At=a]$$

Bellman's equations allow to convert infos about the environment into improvement of the agent's behaviour 


---

### Value function
misura quanto è buono trovarsi in uno stato s, seguendo una certa policy π.
Si indica con Vπ(s) e rappresenta il ritorno atteso partendo dallo stato s.
$Vπ(s)=a∑​π(a∣s)s′∑​P(s′∣s,a)[R(s,a,s′)+γVπ(s′)]$
---

### Bellman equation per la Q-function

Invece di valutare solo uno stato, possiamo valutare una coppia:
Qπ(s,a) cioè: quanto è buono fare l’azione a nello stato s, seguendo poi la policy π
$Qπ(s,a)=s′∑​P(s′∣s,a)[R(s,a,s′)+γa′∑​π(a′∣s′)Qπ(s′,a′)]$

---
### Bellman optimality equation

Quando vogliamo trovare la policy migliore, non usiamo più una policy fissata π, ma scegliamo sempre l’azione migliore.

Per la value function ottimale:
$$V∗(s)=amax​s′∑​P(s′∣s,a)[R(s,a,s′)+γV∗(s′)]$$
Per la Q-function ottimale:
$$Q∗(s,a)=s′∑​P(s′∣s,a)[R(s,a,s′)+γa′max​Q∗(s′,a′)]$$
Questa è la base di algoritmi come Q-learning.



---

# Agent categories:

## Value-Based Agents

Value-based agents learn a value function, which estimates how good a state or action is.

State value $V(s)$
Action value $Q(s,a)$

The policy is derived by choosing the action with the highest value.

Examples:
 - Q-learning
 - SARSA
 - Deep Q-Network (DQN)
Characteristics:
 - Works well for discrete action spaces
 - Simple and stable
 - Indirectly learns policy
 - Limitations
 - Struggles with continuous actions
 - Can be inefficient in large state spaces


--------------------

## Policy-based agents 

directly learn a policy function:  π(a∣s)
This maps states directly to actions.

Examples:
REINFORCE (Monte Carlo policy gradient)
Characteristics:
 - Suitable for continuous action spaces
 - Can learn stochastic policies
 - Direct optimization of behavior
 - Limitations
 - High variance in learning
 - Can be unstable without improvements

--------------------

## Actor–Critic Agents
Definition

>Actor–Critic methods combine value-based and policy-based approaches:

**Actor**: decides actions (policy)
**Critic**: evaluates actions (value function)
Examples:
 - A2C (Advantage Actor-Critic)
 - A3C
 - DDPG
 - PPO
Characteristics
 - More stable than pure policy methods
 - Efficient learning
 - Handles both discrete and continuous spaces
 - Limitations
 - More complex to implement
 - Requires careful tuning

--------------------

## Model-Based vs Model-Free Agents
Model-Free RL
No knowledge of environment dynamics
Learns from experience only

Examples:

 - Q-learning
 - DQN
 - PPO
 - Model-Based RL
 - Uses or learns a model of environment transitions
Can plan ahead

Examples:

 - Value Iteration
 - Policy Iteration
 - Dyna-Q




A value-based agent in reinforcement learning is one that learns a value function (like V(s) or Q(s,a)) and derives its policy from that—rather than learning the policy directly.

These methods work best in problems with:

 - discrete or manageable state/action spaces
 - clear reward signals
 - need for optimal decision sequences