Dynamic programming (DP) is a general problem-solving method where you solve a big problem by breaking it into overlapping subproblems, solving each subproblem once, and reusing those results.

Dynamic programming  applies when:
-Optimal substructure: the best solution to the whole  problem can be built from best solutions to parts.
-Overlappiing subproblems: the same subproblems show up  repeatedly

DP stores solution to subproblems in a table.

In RL, DP means methods that compute value functions and optimal policies by repeatedly applying
the bellman equations,assuming you know the environment model:
-transition probabilities P(s'|s,a)
-expected rewards R(s,a,s')


Examples of dynamic programming:  
shortest paths (Bellman–Ford)
knapsack
edit distance



Dynamic programming assumes full knowledge of the MDP, in dynamic programming you already know the environment model:
An MDP is usually written as: <S,A,P,R,GAMMA>

S set of states
A is the set of actions
P(s'|s,a) is the transition model, the probability of the next state.
R(s,a) is the reward model

So  DP is not trial and error learning from data.
That is why DP is used for planning.

Planning in an MDP:
Planning means computing  what to do using the model P,R instead of learning from experience.
If you can simulate or compute transitions and rewards, you can run  bellman backups  on all the  states

DP  can solve these tasks:

-Prediction (policy evaluation)
The goal is to evaluate a policy π.
The input is <S,A,P,R,gamma> and a policy π
The output is the value function for that policy:
vπ(s) is the expected return starting from s following π
sometimes also qπ(s,a)
This corresponds  to algorithms like iterative policy evaluation.

-Control(findde the best policy)
The goal is to find the optimal behaviour
The input is only the MDP <S,A,P,R,gamma>
THe output is the optimal value function v*(s) and the optimal policy π*(s)

This corresponds to algorithms  like:
-policy iteration (evaluate  -> improve -> repeat)
-value iteration  (apply bellman optimality  backups directly)

Prediction  is: "If I act like this, how good is it?"
Control is: "What is the best way to act?"

DP methods update values by sweeping over states.
------
Policy evaluation(prediction)
The goal is to compute a value function for a fixed policy π
The output is vπ(s) or qπ(s,a)
The typical method is iterative policy evaluation

The iterative policy evaluation gives the prediction,because you are not changing the policy, just 
determining how good is it.
------
Policy Iteration
The goal is to find an optimal policy.
It alternates 2 steps:
-evaluate current policy π -> get vπ
-improve policy π using greediness 







Dynamic programming (DP) is a family of methods that compute value functions and optimal policies
by repeatedly applying bellmans' equations, with a known moddel of the environment.

A model is known when you can query or compute:
-transition probabilities P(s'|s,a)
-expected rewards R(s,a,s')

Dynamic  programming is planning, not learning from raw experience.

The core idea of dynamic programming is to break-down long term return into:
-immediate reward
-discounted value of next state

Policy evaluation predicts values for a fixed policy 𝜋, optimality/control  finds the best possibble values.

DP algorithms solve this by iterating updates until values stabilize.

DP works well when the environment model P and R is known  and the state space is small enough to sweep over.
DP doesn't scale when state-space is huge or continous  or the model is unknown.

Main DP algorithms in RL:
-Policy evaluation(prediction)  Compute vπ for a given policy.
Iterative policy evaluation: starts with any values repeatedly apply Bellman expectation backup.
-Policy improvement
Given vπ make the policy greedier
-Policy iteration. Alterante: improve π,  update π until policy stops cchanging.

It is called dynamic programming because it solves a big sequential decision problem by:
-using recursion (Bellman equation)
-doing systematic backups




 way to solve problems by breaking them into smaller overlapping subproblems, solving each subproblem once, and reusing the results.

The two core ideas

Optimal substructure
The best solution to the whole problem can be built from best solutions to smaller parts.

Overlapping subproblems
The same smaller problems appear many times if you solve things naively, so DP saves time by storing answers.

Two common styles

-Memoization (top-down): write the recursive solution, but cache results so you don’t recompute.

-Tabulation (bottom-up): compute answers for small cases first in a table, then build up to the final answer.

In RL/planning, DP means computing value functions using Bellman updates when you know the model
R(s) + y*∑P​(s|s)*v(s')
You repeatedly update values for all states until they converge (or use policy iteration/value iteration).



If you already know how the environment works, you can compute the best behavior by repeatedly using a simple “update rule” for every state.
If you already know how the environment works, you can compute the best behavior by repeatedly using a simple “update rule” for every state.
“If I’m in state S, and I take action A, what reward do I get now, and what states can I reach next?
Use that to update how good S is."
These updates are called Bellman backups.

Two common DP methods

1) Policy Iteration
Pick a policy (a rule for what to do).
Evaluate it: compute how good each state is if you follow that policy.
Improve it: switch actions to the ones that look better using those values.
Repeat until it stops changing.

2) Value Iteration
Skip the “fully evaluate” step.
Directly update each state toward the best possible action.
Repeat until values stabilize, then pick the best action in each state.

When DP is useful
When the world is small enough to list all states
And you know the transition probabilities and rewards (the “model”)

Why it matters
Even if you don’t use DP in real problems, DP is the blueprint for most RL:
“values”
“best action”
“update using next-state value”