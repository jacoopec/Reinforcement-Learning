Estimate V(A) and V(B) from experience, no model
Discount: 
γ=0.9
Step size: 
α=0.5
Initialize: 
V(A)=0, V(B)=0
Terminal state 
V(T)=0

A ->(R=1)->B->(R=2)->T

Transition A->B
TD target:
1 + 0.9*V(B) = 1
TD Error:
δ = 1 - V(A) = 0
Update 
V(A)<- V(A) + 0.5*1 = 0.5

Transition B->T
TD target:
2  + 0.9*V(T) = 2
TD error:
δ = 2 - V(B)  = 2
Update
V(B) <- V(B) + 0.5*2 =  1

Same episode Again:
Transition A->B
TD target:
1 + 0.9*V(B) = 1.9
TD Error:
δ = 1.9 - V(A) = 1.4
Update 
V(A)<- V(A) + 0.5*1.4 = 1.2

Transition B->T
TD target:
2  + 0.9*V(T) = 2
TD error:
δ = 2 - V(B)  = 2
Update
V(B) <- V(B) + 0.5*1 =  1.5

First time thorugh the episode:
Initially we had V(B) =  0
Updating from A -> B with reward 1
So TD thinks "From TD I get about 1" because V(B) è 0
So update gave V(A)  = 0
After finishing this episode V(B) got 1.
So the agennt now believes that being in B is good.

The  second time thoruhg the same episode,
now
“From A I get reward 1, and then I end up in B, and B is worth about 1, so A must be worth more than I thought.
V(A) increased on the second run because TD’s update for A depends on its current estimate of 
V(B), and that estimate got bigger after learning from the previous run.

TD learns from experience (the transitions it sees).

It bootstraps: V(A) improved in episode 2 because it used the current estimate V(B).

Over many episodes, the values approach the true returns:

True 
𝑉∗(B)=1

True 
𝑉∗(A)=0.9