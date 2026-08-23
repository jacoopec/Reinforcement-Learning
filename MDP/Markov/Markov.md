difference between MarkovProcess, MarkovRewardProcess and MarkovDecisionProcess
MP: “What happens next?”
MRP: “What happens next, and what reward do I get?”
MDP: “What happens next and reward, and I can choose actions to do better.”

They’re three closely related models for sequential dynamics, with increasing “structure” (and what you can compute from them).

examples: predicting state evolution, stationary distributions, hitting times, customer churn, inventory, traffic, weather regime switching, disease progression

----------------------------------------

MARKOV PROCESS / CHAIN
Models state dynamics only.
State-space: S
Transition model P(s'|s)
Markov properety: the next state depends only on the current state not the full history.

EXAMPLES

States:
Good, Degraded, Failed

Transition probabilities:
P matrix
    G   D    F
G   0.8 0.18 0.02
D   0   0.65 0.35
F   0   0    1

πt+1 = πt*P

if you are in G, the machine is good at week 0:
π0=[1 0 0]
THen, at week 1 it will be:
π1=[0.8 0.18 0.02]
And at week 2:
π2=[0.64, 0.261, 0.099]
After 3  weeks:
[0.512, 0.28485, 0.20315]
THere is 51,2% probability that the machine will be in a good state.
20,3% probability that it will fail.


----------------------------------------


MARKOV DECISION PROCESS
It formally describes an environment for reinforcement learning. The environment is fully observable so we get  to know everything that is needed to understand it. The current state fully characterises the process.
Almost all RL problems can be formalized as MDP.
It models dynamics + reewards  + actions 
State space S
Action space A
Transition model P(s'|s,a)
Reward model R(s,a)
discount factor gamma 

Actions are chose through a policy π(a|s)

key quantities:
-state value under policy π: Vπ(s)
-action value Qπ(s,a)

The goal is to find an optimal policy π* maximizing the expected return
Bellman optimality form:
V*(s)  = max[ R(s,a) + gamma*∑P(s'|s,a)V*(s')]

EXAMPLES

-State evolution prediction
if I take action a in state s what’s the probability of being in each next state s'?
An MDP is <S,A,P,R,gamma>
States: Good, degraded, failed
Actions: N do nothing, M preventive maintanance, R repair
Transition model, state evolution prediction:
If you do nothing:
    From G: 
        P(G|G,N) = 0.8
        P(D|G,N) = 0.18
        P(F|G,N) = 0.02
    From  D:
        P(D|G,N) = 0.65
        P(D|F,N) = 0.35
    From F:
        P(F|F,N) = 1
    
If you don preventive maintanance:
    From G: 
        P(G|G,M) = 0.95
        P(D|G,M) = 0.05
    From  D:
        P(G|D,M) = 0.60
        P(D|D,M) = 0.35
        P(F|D,M) = 0.05
    From F:
        P(F|F,M) = 1

If you repair:
    From G: 
        P(G|G,R) = 0.97
        P(D|G,R) = 0.03
    From F:
        P(G|F,R) = 0.9
        P(D|F,R) = 0.1
    From D:
        P(D|D,R) = 0.2
        P(G|D,R) = 0.75
        P(F|D,R) = 0.05

Rewards  R(s,a)
    If you want an MDP that can be optimized, not just predicted, add cost:
    operation value:
        -10 peeer tep in G
        -4 per step in D
        -0 in F
    action value:
        cost(N) = 0
        cost(M) = 3
        cost(R) = 12

    R(s,a) = value-cost

    R(G,N)  = 10
    R(D,M) =  4-3
    R(F,R) = -12



----------------------------------------
Markov reward process
It models state dynamics and rewards but still no actions (no control).
-State space: S
-Transition model: P(s'|s)
-Reward model: 
    R(s) expected immediate reward in s
    R(s,s')/R(s,s',r) reward depend on transition/distribution

Key-quantity: state-value function
V(s) =  Expected[∑ gamma^t * Rt]

and it satisfies the bellman expectation equation:

V(s) = R(s) + gamma *  ∑P(s'|s) * V(s')

Use it for long-term value when the dynamics aree fixed.



----------------------------------------

PLANNING

You have (or assume) a model of the environment.
Model = transition + reward, i.e. 


Goal: compute a good/optimal policy (or a good action sequence) by “thinking” ahead using the model.

Typical methods: dynamic programming (value iteration, policy iteration), tree search, Monte Carlo tree search, shortest-path / optimal control methods when structured.

Example: You know the rules of chess → search/planning chooses moves.


----------------------------------------
CONTROL 

You are acting to achieve good performance (optimize return).
“Control” is the objective/task: pick actions to maximize expected long-term reward.
Control can be done:
with a known model → model-based control (often looks like planning + execution)
with an unknown model → you still want control, but you need to learn while acting (that’s where RL often comes in)

So “control” is broader: it’s about choosing actions; planning and RL are two common ways to do it.

Example: Keeping a drone stable and following a path is a control problem (could use a physics model + MPC, or learn a policy).
----------------------------------------
REINFORCEMENT LEARNING

You don’t fully know the model; you learn from interaction/data.

Goal: achieve control (good decisions) while learning values/policies from experience.

Typical methods:

value-based: Q-learning, DQN

policy-based: REINFORCE

actor–critic: PPO, A2C, SAC

model-based RL: learn a model, then plan with it (hybrid)

Example: A robot learns to grasp objects by trial and error (or logged experience), without a perfect dynamics model.