Monte carlo methods **learn directly form episodes of experience**
MC is:
 - *model-free*,  no knowledge about MDP, transitions/rewards
 - *learns  from  complete episodes*: no bootstrapping
 - uses the simplest possible idea: *value = mean return*
 - can only be applied to episodic MDPs
 - Monte Carlo RL learns values from complete sampled episodes, without needing to know the environment's transition model.

---
# Monte Carlo control
>Monte Carlo control is a reinforcement learning method used to learn a policy, not just estimate state values.
The idea is:

Run a complete episode using the current policy.
Compute the return G obtained after each state-action pair (s,a).
Update the action-value estimate:
Q(s,a)←average return observed after taking a in s
Improve the policy by choosing the action with the highest Q(s,a), usually with some exploration such as ϵ-greedy.
Repeat for many episodes.

So compared with Monte Carlo prediction:

Monte Carlo prediction: learn V(s) or Q(s,a) for a fixed policy.
Monte Carlo control: learn Q(s,a) and continually improve the policy.

In short:

generate episodes→estimate Q(s,a)→improve policy

The final goal is to learn approximately:

$$π(s)=argamaxQ(s,a)$$

meaning: for each state, learn which action gives the highest expected return.


Usually, ϵ-greedy is used during training to balance exploration and exploitation.

During training:

π(a∣s)={random actionargmaxaQ(s,a)
	​

with probability ϵ
otherwise
	​


This helps the agent explore actions it might otherwise ignore.

During evaluation or deployment, you typically set:

$$ϵ=0$$

so the agent behaves greedily:

$$a=argamax	​Q(s,a)$$

A common approach is also to decay ϵ during training, for example from 1.0 to 0.05: lots of exploration early, mostly exploitation later.
---


Moonte-carlo policy evaluation:
The goal is to learn vπ from episodes of experience under the policy π
S1,A1,R1,S2,A2,R2,...
The return is the total discounted reward:
$$Gt = Rt+1 + gamma*Rt+2 + ...$$
The value function is the expected return:
$$vπ= Eπ[Gt|St=s]$$
Monte-carlo policy evaluation uses empirical mean return instead of expected return 

First-visit monte carlo policy evaluation 
To evaluate state  s 
The first timestep t  that state s is visited in an episode

In RL, the agent usually does not directly choose the next state. It chooses an action, and the environment determines the resulting state


``
---

Monte Carlo methods in reinforcement learning are generally model-free.

They do not need to know the environment's transition probabilities

$$P(s′∣s,a)$$

or reward model

$$R(s,a).$$

Instead, they learn directly from sampled episodes.

So:

MC prediction: model-free policy evaluation
MC control: model-free policy optimization

They estimate values from experience, for example:

V(s)≈average return observed after visiting s

or

Q(s,a)≈average return observed after taking a in s.

The key point is that Monte Carlo uses actual sampled trajectories rather than an explicit model of the environment.
---



Monte Carlo (MC) methods in reinforcement learning are a family of algorithms that learn from complete episodes by using the actual returns observed (sampled experience), instead of using a model of the environment.
The idea 
Run the policy, observe an episode, compute the return (discounted sum of rewards), and use it to update value estimates.

G(t) = Rt+1 + gamma*Rt+2 + gamma^2*Rt+3 + ... 

as a noisy sample of the true expected value.

Monte Carlo prediction (policy evaluation)
Estimate Vπ​(s) for a fixd policy π​

`Procedure:`
 - generate many episodes with the policy π​
 - for each visit to state s compute the return G.
set V(s) to the average of all observed returns s

first-visit MC: use only the first time s appears in each episode.
Every-visit MC: use every time s appears.

Incremental uupdate form (instead of storing all returns)
V(s) <- alfa(G -V(s))

  - MonteCarlo control (learning an optimal policy)

        Goal: learn Q(s,a) and improve the policy to become optimal.

        Qπ​(s,a) = E[Gt | St = s, At = a]

        πThen improve the policy by being greedy (or mostly greedy) with respect to 
        𝑄
        Q:

  - On-policy vs Off-policy MOnte Carlo
  on-policy: evaluate and improve the same policy used to generate data.
  off-policy: data come from a behaviour policy

  