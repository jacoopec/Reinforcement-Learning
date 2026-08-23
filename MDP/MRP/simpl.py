import numpy as np

# -------------------------------------------------
# Simple Markov Reward Process
# -------------------------------------------------

#This is an MRP because there are states, transition probabilities, and rewards, but no actions.

#The value of a state s is the average total future reward you expect to get if you start in state s.


# States:
#   0 = Sleep
#   1 = Eat
#   2 = Study
#   3 = Pass Exam
#
# The process moves between states according to P.
# Each state has an immediate reward R.
#
# We want to compute:
#
#   V(s) = R(s) + gamma * sum_s' P(s, s') * V(s')
#
# In matrix form:
#
#   V = R + gamma * P @ V
#
# Rearranged:
#
#   (I - gamma * P) V = R
#
# Therefore:
#
#   V = inverse(I - gamma * P) @ R
#

gamma = 0.8


states = ["Sleep", "Eat", "Study", "Pass Exam"]

# Transition matrix P
# P[i, j] = probability of moving from state i to state j
P = np.array([
    # Sleep  Eat   Study Pass
    [0.2,    0.6,  0.2,  0.0],  # Sleep
    [0.1,    0.2,  0.7,  0.0],  # Eat
    [0.0,    0.1,  0.6,  0.3],  # Study
    [0.0,    0.0,  0.0,  1.0],  # Pass Exam
])

# Reward vector R
# R[i] = immediate reward for being in state i
R = np.array([
    1.0,    # Sleep
    2.0,    # Eat
    4.0,    # Study
    10.0,   # Pass Exam
])

# Identity matrix
I = np.eye(len(states))

# Solve Bellman equation exactly
V = np.linalg.solve(I - gamma * P, R)

print("State-value function:")
for state, value in zip(states, V):
    print(f"V({state}) = {value:.2f}")