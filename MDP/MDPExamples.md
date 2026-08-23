smallest, cleanest  model-based RL example of an MDP: a 2 state chain where you either 
try to reach a reeewarding terminal state or loop around.
MDP definition:
States:
s0 and s1
Actions:
a = GO
a = STAY

This is an MDP(S,A,P,R,gamma)

TRANSITION MODEL P(s' | s,a)
From s0: 
P(s1|s0,GO) = 1
P(s0|s0,STAY) = 1
From s1:
P(s1|s1,a)= 1

REWARD MODEL R(s,a,s')
R(s0,GO,s1)  = +1
R(s0,STAY,s0) = 0
R(s1,STAY,s1) = 0
R(s1,STAY,s1) = 0

In model-based RL you assume you have the model P and R and then plan with it,and here planning is trivial:
Compute action values at s0

Bellman optimality:

Q*(s,a) = ∑ P(s'|s,a)[R(s',a,s)+gamma*Vstar(s')]


Vstar(s) = max Qstar(s,a)

in the terminal state Vstar(s1) = 0

Now, at s0:
-For GO:
    Q*(s0,GO) = 1 + gamma*Vstar(s1) = 1 + gamma*0 = 1
-For STAY:
    Q*(s0,STAY) = 0 + gamma*Vstar(s0) = gamma*Vstar(s0)

    Vstar(s0) = max(1,gamma*Vstar(s0))

Since 0 <  gamma < 1

The solution is Vstar(s0) = 1
and the optimal action is πstar (s0) = GO

GO gives you reward immediately at end.
STAY never  gives reward it is always worse.

---------------
---------------
---------------
---------------

MDP definition
states
S  = {A,B}
Actions:  in each state you can chose  L or R.
Discount: 0.9

Transitions and rewards:
From A:
R -> you go to B
    reward:+5
    P(B|A,Right) =1, reward: 5

L ->  you stay in  A
    reward +1
    P(A|A,Left) = 1, reward: 1

From B:
R -> you go to B
    reward:0
    P(B|B,Right) = 1, reward: 0

L ->  you stay in  B
    reward 0
    P(A|A,Left) = 1, reward: 0

Compute values for the choices in state A:
-Chose Right in A:
    Q(A,Right) = r + gamma*V(B) =5 + 0.9*0
-Chose Left in A:
    You keep collecting +1
    1 + 0.9*1 + 0.9^2*1 + ...

States A and B 
Actions: Right or Left
Discount: 0.9
V(B) = 0

Transitions + rewards from A
Action R
-probability of 0.7 of going to B, reward 5
-probability oof 0.3 of stay in A,  reward  0

Action L
-probability 1 to stay in A, reward 1

VL(A) = 1  + gamma*VL(A)

Q(A,Left) = 10

Compute value if you always chose R:
Bellman equation:
Vr(A) = E[r + gamma*Vr(s')]
Expected immediate reward  under Right:
E[R] = 0.7*5 + 0.3*0

---
---
---

simple 3-state MDP with stochastic transitions, and 3 iterations of value iteration.

States: A, B, T

Discount  0.9

Transitions and rewards:
From A:
FAST
With probability 0.4 go to B, reward 0
With probability 0.6 go to T, reward 6
SAFE
With probability 0.8 go to B, reward 0
With probability 0.2 go to T, reward 1

From B:
WORK
With probability 0.7 stay in B, reward 1
With probability 0.3 go to T, reward 2
SAFE
With probability 1 go to T, reward 0

Valuee iteration update 

Vk+1(s) = max ∑ P(s'|s,a)(R(s,a,a')+gamma*Vk(s'))
Starts with V0(A) = V0(B) = V0(T)

----
----
----

Value iteration update:
Vk+1(s) = max ∑​ P(s'|s,a) *(R(s'|s,a)+gamma*Vk(s'))

Iteration 1  from V0

State A
SAFE:
0.8*(0 + 0.9*V0(B)) + 0.2(1+0.9*V0(T)) = 0.8*0 + 0.2*1 = 0.2
FAST:
0.6*(3+0.9V0(T)) + 0.4(-1+0.9*V0(B)) = 0.6*3 + 0.4(-1)

State  B
WORK:
0.7*(1+0.9*V0(B)) + 0.3*(2 + 0.9V0(T))= 0.7 * 1 + 0.3 * 2
QUIT: 0






