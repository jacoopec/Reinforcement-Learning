model based
 They known to be data efficient, but they fail when the state space is too large. Try to build a model-based algorithm to play Go. Not gonna happen.


--------------------------------
Model-free 
Model-free algorithms do not require to learn the environment and store all the combination of states and actions. The can be divided into two categories, based on the ultimate goal of the training.

Policy-based
Policy-based methods try to find the optimal policy, whether it’s stochastic or deterministic. Algorithms like policy gradients and REINFORCE belong in this category. Their advantages are better convergence and effectiveness on high dimensional or continuous action spaces.
Policy-based methods are essentially an optimization problem, where we find the maximum of a policy function. That’s why we also use algorithms like evolution strategies and hill climbing.

Value-based
Value-based methods, on the other hand, try to find the optimal value. A big part of this category is a family of algorithms called Q-learning, which learn to optimize the Q-Value. I plan to analyze Q-learning thoroughly on a next article because it is an essential aspect of Reinforcement learning. Other algorithms involve SARSA and value iteration.

At the intersection of policy and value-based method, we find the Actor-Critic methods, where the goal is to optimize both the policy and the value function.

--------------------------------
Dynamic programming

Dynamic programming methods are an example of model-based methods, as they require the complete knowledge of the environment, such as transition probabilities and rewards.
Dynamic programming amounts to breaking down an optimization problem into simpler sub-problems, and storing the solution to each sub-problem so that each sub-problem is only solved once.
DP is a useful technique for optimization problems, those problems that seek the maximum or minimum solution given certain constraints, because it looks through all possible sub-problems and never recomputes the solution to any sub-problem. This guarantees correctness and efficiency
storing solutions — a technique known as memoization
sub-problems build on each other in order to obtain the solution to the original problem

--------------------------------


Deep neural networks have been used to model the dynamics of the environment(mode-based), to enhance policy searches (policy-based) and to approximate the Value function (value-based).