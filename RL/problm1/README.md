problem solved:

What is the value function for the uniform random
policy?


iterative policy evaluation of the Gridworld example under the uniform random policy.

The difference is that the first problem is prediction, while the second  problem is control.

Prediction problem (1)
In the first Gridworld, the policy is already fixed.
The question is:
Given this policy, what values do the states have?
So the Bellman update is an average over all actions:
V(s) = average over actions of [reward + gamma * V(next_state)]

So we are not trying to find the best action.
We are only evaluating how good each state is if the agent behaves randomly.

Control problem (2)
In the second Gridworld, the policy is not fixed.
The question is:
What is the best possible value of each state, and what action should the agent choose?
So instead of averaging over actions, we take the maximum over actions:
V*(s) = max over actions [reward + gamma * V*(next_state)]
Then, after finding the best values, we extract the policy:
Choose the action or actions that achieve the maximum value.
So control solves two things:
optimal value function: V*
optimal policy: π*

| Aspect           | Prediction           | Control                                    |
| ---------------- | -------------------- | ------------------------------------------ |
| Policy           | Given/fixed          | Unknown; must be found                     |
| Goal             | Evaluate a policy    | Find the best policy                       |
| Update           | Average over actions | Maximum over actions                       |
| Result           | Value function `Vπ`  | Optimal value `V*` and optimal policy `π*` |
| Example behavior | Random movement      | Best possible movement                     |


Prediction: The agent moves randomly, so even from a good state it may take bad actions.

Control: The agent chooses the best available action, so it learns to move toward useful states like A and B.