This is classic model-based planning: learn dynamics → plan optimally.

Model-based RL in a tabular setting.
FrozenLake-v1 environment (deterministic version for simplicity)

Small discrete state/action space (16 states, 4 actions).
Agent learns a model of transitions $  P(s'|s,a)  $ and rewards $  R(s,a)  $ from random experience.
Then uses value iteration (dynamic programming) on the learned model to compute the optimal policy — no more interaction needed!
