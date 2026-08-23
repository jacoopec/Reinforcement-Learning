Here's a simple, complete example of model-based RL in a tabular setting — perfect for understanding the concepts without heavy deep learning.
We use the FrozenLake-v1 environment (deterministic version for simplicity):

Small discrete state/action space (16 states, 4 actions).
Agent learns a model of transitions $  P(s'|s,a)  $ and rewards $  R(s,a)  $ from random experience.
Then uses value iteration (dynamic programming) on the learned model to compute the optimal policy — no more interaction needed!

This is classic model-based planning: learn dynamics → plan optimally.