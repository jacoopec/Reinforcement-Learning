
# Montecarlo vs temporal difference
>An RL agent learns by interacting with its environment 

Given the experience and the received reward the agent will update its value function or policy .

Montecarlo and temporal difference are 2 strategies on how to train  the value fucntion or the policy.

Both of them use experience to solve the RL problem.
## Montecarlo:
 - uses an entire episodde of experience before learning.
 - learns at the end of the episode.
 - waits until the end of the episode, Calculates Gt andd uses it as a target for updating V(St)
 - requires an entire episode of interaction before updating the value function.
 - waits until the endd of the episode, then calculates Gt and uses it as a target for its value or policy.

```
V(St)                <-    V(St)                 +    alfa*         [Gt      -              V(St)]
New value of state t     Former estimation          learning rate   return at timestap t
                            of value at state t
```

If we train a state-value function using montecarlo:
At the end of the episode we have a lsit of State, Action, Rewards and next state
The agent will sum the total rewards Gt

It will then update V(St)

By running more and more episodes the agent will learn how to play better.

With montecarlo, we update the value function for a complete episode, and so we use the actual accurate discounted return of this episode 
---

 ## Temporal difference
 - uses only a step to learn.(St,At,Rt+1,St+1), learning: learning at each step
 - waits for only one interaction St+1 to form a TD target and update V(St) using Rt+1 and gamma*V(St+1)

But  because we didn't experience an entire episode we don't have Gt.
Instead, we estimate Gt by adding Rt+1 and the discounted value for the next state 

This is calledd bootstrapping because TD bases its uupdate in part on an existing estimate V(st+1)
andd not a complet sampe Gt.


With TD we update the value function from a step and we replace Gt, whit an estimate return called TD target.



---

# MONTE CARLO

Environment:  **A-B-Terminal**

 - It starts in A and the only action is "move right" -> The reward is -1 per step.
gamma = 1
episode ends when  it reaches the terminal

 - From B: 1 step to terminal, return G=-1 V(B) = -1
FromA:
1steps to terminal, return G=-1 V(A) = -2

MC  learning (every-visit) step-size alfa=0.5

after an episode compute the return for each state visited, then update

V(s) <- V(s) + alfa(G-V(s))

V(A) = V(B) = 0

---
Episode 1: A->B->Terminal
Rewards: -1,-1

Returns:
For A:G(A) = -1 -1 = -2
For   B: G(B) = -1
---
Episode 2: 
Updates:
V(A)=-1 + 0.5(-2-(-1))=-1.5
V(B)= -0.5+0.5(-1-(-0.5))=-0.75

---
Episode3:
V(A) = -1.5 + 0.5(-2+1.5) = -1.75
V(B) = -0.75 + 0.5(-1+0.75) =-0.875

Values are convergin toward
V(A) -> -2
V(B) -> -1

Monte carlo updates using the complete sampled return from the episode, not a model and not bootstrapping from V(text)

---

# TEMPORAL DIFFERENCE

V(A) = V(B) = V(Terminal) =0

TD update rule
$$V(S) <- V(S) + alfa*(r + gamma*V(s')-V(s))$$

---
1 episode: A->B->Terminal

Step 1: from A to B
s=A,s'=B, r=-1
TD target = r + V(B) = -1 +0= -1
TD error = target - V(A) = -1 -0=-1
Update: V(A) = 0 + 0.5(-1) = -0.5

Step2: from B to terminal
s = B, s' = T, r =  -1

TD target = r + V(T) = -1 + 0
TD Error = -1-0 = -1
Update: V(B) = 0 + 0.5(-1) = -0.5


---
Episode 2
A -> B
Target = -1 + V(B) = -1 -0.5 = -1.5
Update:
V(A) = -0.5 + 0.5(-1.5-(0.5)) = -1
B -> teerminal
Target = -1
Update:
V(B) = -0.5 + 0.5(-1-(-0.5))= -0.75

V(A) = -1 
V(B) = -0.75

It keeps moving toward the true values:
V(B) -> -1
V(A) -> -2

The key difference is that TD updates after each episode and uses a bootstrapped target 
r + gamma*V'(s) instead of waiting for the full episode to return.


---

# TD  LAMBA

TD(lambda)  is  a way to  do prediction estimate Vπ that smoothly blends:
-TD(0):learn from the enxt step  (low variance, more bias )
-monte carlo, leaarn from the full return (high variance, less bias)

It operates assigning credit backkwards thorugh time with elegibility  traces.

The core idea is that after each transition st->st+1 with reward Rt+1
-TD error is compmuted  dt = rt+1 + gammaV(st+1)  -V(St)
-Keep an eligibility trace e(s) for each state ( a short term memory of recently visited states)
-increase trace for the current state 
-decay all traces each step





