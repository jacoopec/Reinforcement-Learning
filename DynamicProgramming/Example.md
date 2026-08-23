The states are the possible siituations an agnet can be in:
positions on a grid
low/med/high inventory

Policy π(a|s) is a rule  that tells you how likely you are to chose each action when you are in a state s.
If deterministic 1 for a specific action, 0 for others.
If stochastic, the probability for each action.

what the bellman equation mean:
vπ(s) = ∑π(a|s) * ∑ P(s'|s,a)*(R(s,a,s') + γ*vπ(s'))

You are in state s, and you chose action a with probability π(a|s) 
The environment moves you to the next state s' with probability P(s'|s,a) and you get the 
reward R(s,a,s').
You add discounted future value γ*vπ(s')
This equation Takes expectation over both sources of randomness (your action choice + environment transition).

Example:

States = {A,B}
Actions={L,R}
Environment: A->B
𝛾=0.9


From A:
-if R, you go to B with prob  and reward +5
-if L, you stay in A with prob 1 and reward 0

From B:
-if L, you go to B with prob 1 and reward 1.
-if R, you stay in B with reward 0.

The policies are stochastic:
-in A: 𝜋(R|A) = 0.8 and 𝜋(L|A) = 0.2
-in B: 𝜋(R|B) = 0.5 and 𝜋(L|B) = 0.5

Bellman equations:

the first terms are related to action R, the second to action L

for state A:
v(A)  = 0.8[5+0.9*v(b)] + 0.2[0+0.9*v(a)]

for state B:
v(B)  = 0.5[a+0.9*v(b)] + 0.5[0+0.9*v(a)]

DP methods compute v𝜋 by iterating this update.

in dynamic programming, policy evaluation does this:
starts with:
Va(0) = 0
Vb(0) = 0

then, repeatedly update for all the states using bellman equation.

ITERATION 0
Va(0)=Vb(0) =0
ITERATION 1
v(A)  = 0.8[5+0.9*0] + 0.2[0+0.9*0]=4
v(B)  = 0.5[a+0.9*0] + 0.5[0+0.9*0]=0.5
ITERATION 2
v(A)  = 0.8[5+0.9*0.5] + 0.2[0+0.9*4]=5.08
v(B)  = 0.5[a+0.9*0.5] + 0.5[0+0.9*4]=2.525
ITERATION 3
v(A)  = 0.8[5+0.9*2.525] + 0.2[0+0.9*5.08]=6.73
v(B)  = 0.5[a+0.9*2.525] + 0.5[0+0.9*5.08]=3.92

These values are moving toward V𝜋, iteration after iteration.

Closed solution:

v(A)  = 0.8[5+0.9*v(b)] + 0.2[0+0.9*v(a)]
v(B)  = 0.5[a+0.9*v(b)] + 0.5[0+0.9*v(a)]

v(A) = 4 + 0.72v(b) + 0.18v(A) 
v(B) = 0.5 + 0.45v(B) + 0.45v(A) 

solving the linear system...
vb = 17.4
va = 20.157


v = r𝜋 + γ*P*v

v=(I-γ*P)^(-1)*r𝜋

If you take an MDP (states, actions, transition probabilities, rewards) and then fix a policy 
𝜋  (a rule for choosing actions), the remaining process over states only becomes a Markov Reward Process (MRP).
In an MDP the next state depends both on current state and the action.
