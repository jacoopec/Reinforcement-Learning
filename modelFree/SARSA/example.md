SARSA on policy TD control

States: A,B,T
Actions:
-From A: go to B, stay in A
-From B finish

Dynamics and rewards:
A->B R: 0
A->A R:0
B->t R:1

γ=0.9
α=0.5

initialize all Q(s,a) =0

Q(s,a) <-  Q(s,a) + α*(r + γQ'(s',a')- Q(s,a))

Uses actual  next action a' you took so it is on policy.

---
---
---
EPISODE 1
--
step1
s=A
a=go to B
r= 0
s'=B
a'=finish
Q(A,go to B)=0, Q(B,finish) = 0
TD target = 0
Q(A,  go to B) = 0
--
step 2
s = B
a= finish
r = 1
s'=T
TD target 1+0  = 1
Q(B,finish) <- 0 + 0.5(1-0)

---
---
---
EPISODE  2
TD target for Q(A, go to B)  = 0 + 0.9*Q(B,finish) =0.9*0.5 = 0.45
Update: Q(A, go to B) <- 0 + 0.5*0.45 = 0.225