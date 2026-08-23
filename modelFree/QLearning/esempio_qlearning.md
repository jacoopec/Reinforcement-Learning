fully worked, pencil-and-paper Q-learning example on a 3×3 grid
gridworld  +   q-learning 

(0,0)=S   (0,1)     (0,2)
(1,0)     (1,1)     (1,2)
(2,0)     (2,1)     (2,2)=G

Actions: Up (U), Right (R), Down (D), Left (L)

The  dynamics  is deterministic, if you try to move outside of a cell you'll stay in the same cell.
Rewards:
-1 for  every normale  move
+10 for moving to the goal 


Q-learning update rule 
Q(s,a) <- Q(s,a) + alfa(r + gamma(Q(s',a')-Q(s,a)))

we pick:
alfa = 0.5 learning rate 
gamma = 0.9  discount rate 
Q(s,a) = 0

A fixed-trial path (what the agent experiences)
Suppose the agent (by exploration ) happns to take this shortst path each episode:
(0,0)->(0,1)->(0,2)->(1,2)->(2,2)=G

Th episode is:
A (Right, -1 ) -> B( Right, -1) -> C( Down, -1) -> D( Down, +10) -> Goal

---
Step 1:
update Q(A,R)
state is A, action is Right, reward is -1 and thee new  state is B

Thee target iss r + gamma*max  Q(B,*)  =  -1 + 0.9*0 = -1

The update is Q(A,R) = 0 + 0.5(-1-0) = -0.5

---
Step  2:
update Q(B,Right) 

Q(B,R) = 0 + 0.5 (-1) = -0.5

update Q(C,Down)

