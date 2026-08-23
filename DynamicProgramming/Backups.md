In Dynamic Programming for RL/MDPs, synchronous vs asynchronous is about how you apply Bellman backups (updates) across states.

-----------------

Synchronous DP

Idea: update all states “at once” using the old value function.
For every state s, compute a backup  using Vold and store it into Vnew
Synchronous DP

Idea: update all states “at once” using the old value function.


What it implies:

Conceptually clean, easy to vectorize / implement as “full sweeps”.

Information “propagates” one sweep per iteration.
-----------------

Asynchronous DP

Idea: update states one (or a few) at a time, and immediately use the newest values.

Single array V.
Pick a state s, do a Bellman backup, overwrite  V(s) right away.

Next update may use that freshly updated value.


Key difference in one sentence

Synchronous: “compute the next  V for all states using the previous  V.”
Asynchronous: “update  V state-by-state, reusing updates immediately.”


---
---
---

Example
A->B->T

γ=0.9

Start with V(A) = V(B) = 0

Synchronous sweep 1:
V1(B) = 1 + 0.9*0 = 1
V1(A) =  0 + 0.9*V0(B) = 0

After sweep 1:
V(A) = 0, V(B) = 1

V(B) = 1
V(A) = 0.9*V(B)
