POLICY EVALUATION
A-> B -> T

From A: with probability 1, go to B, reward: +1
From B: with probability 1, go to T, reward: +2

Bellman expectation equations for policy evaluation:
Vπ(A) = 1 + γ Vπ(B)
Vπ(B) = 2 + γ * 0
Vπ(B) = 2
Vπ(A) = 1 + 0.9*2 = 2.8

That’s policy evaluation: compute the value of states assuming the policy is fixed.

------------------------------------------------------------------------------
POLICY ITERATION


------------------------------------------------------------------------------
VALUE ITERATION