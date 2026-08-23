A Markov Decision Process is the standard mathematical way to describe the kind of decision making problems that RL tries to solve.

An  MDP models an agent interacting with the environment over time.
It is ddeffined by:
-states S: what the world looks like 
-Actions A: what the agent can do.
-Transition dynamics  P(s'|s,a): probability of landing in the  next state s' after taking action a  in state s 
-Reward function: R(s,a) immediate feedback  signal.
-Discount factor: how  much future rewards matter vs immediate ones.

The Markov property:
The future depends only  on the current state (and action), not the full history.
Formally P(st+1|st,at) is enough.

RL solves an MDP when you don't  fully know it.
-In classic planning, you may know P  and R and compute the ebst policy directly.
-In RL usually you don't  know P and R so you learn from experience.(trial and error)

Goal in MDP:
you want a policy π(a|s) that maximizes  total return:
Gt = ∑ gamma^k  * r(t+k+1)

So the objective is to find π∗ such that:
π∗ = argmax E(Gt)



