1. Monte Carlo learning

Monte Carlo is probably the simplest RL method conceptually.

The idea is:

Let the agent play one full episode.
Observe the final reward.
Update the value of each visited state based on what happened.
Repeat many times.

It does not update during the episode. It waits until the episode ends.

Start at state 3
Move randomly until reaching 0 or 6
If it reaches 6, reward = +1
If it reaches 0, reward = -1
Then update the states it visited

In the Monte Carlo script, the policy is random. 50%.


--------------------------------


2. Dynamic Programming

This works if you know the full environment model.

That means you know:

From state 3, action right always moves to state 4
From state 4, action left always moves to state 3
Reaching state 6 gives +1
Reaching state 0 gives -1

Then you can solve it mathematically using value iteration or policy iteration.

This is simple in small examples, but less practical when the environment is unknown.

------------------------------

3. Temporal Difference Learning

TD learning updates while the episode is still running.

The most basic TD method is:

V(s) ← V(s) + α [r + γV(s') - V(s)]

This is called TD(0).

It is more efficient than Monte Carlo because it learns step by step instead of waiting until the end.
So TD(0) also evaluates a random policy in that script.

It estimates:
How good is each state if I continue moving randomly?



------------------------------
4. SARSA

SARSA is similar to Q-learning, but it updates using the action the agent actually takes next.

Its update rule is:

Q(s, a) ← Q(s, a) + α [r + γQ(s', a') - Q(s, a)]

The name comes from:

State, Action, Reward, State, Action

SARSA is on-policy, meaning it learns from the same behavior policy the agent is currently following.

So during training:

10% random action
90% best known action

SARSA learns the value of the policy it actually follows. That is why it is called on-policy.

-------------------------------


| Method              | Basic idea                                 |
| ------------------- | -------------------------------------------|
| Monte Carlo         | Learn from complete episodes               |
| TD(0)               | Learn after every step                     |
| SARSA               | Learn action values using actual next actio|
| Q-learning          | Learn action values using best next action |
| Dynamic Programming | Solve using known transition rules         |


SARSA uses the actual next action chosen by the current policy, while Q-learning uses the best possible next action.