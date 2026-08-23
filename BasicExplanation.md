Reinforcement learning is a way to train an agent to make decisions by trying actions, 
getting feedbacks and improving its decisions over time to maximize a long term reward.
The agent is the learnere/decision maker.
The environment is everything the agent interact with.
The state is what the agent observes about the environment 
Action a is the choice the agent can make.
The Reward is a numeric score from the environment after each action.
The policy is the agent's strategy. What action it tends to take in each state.

Reinforcement learning loop:
-The agent observes state st.
-The agent picks action At  (using its policy)
-Environment responds with reward rt+1  and state st+1
-The agent updates itself to ddo better next time.

The goal is  not to maximize immediate reward at each step, but to maximize expected total reward over time.
Gt  =  rt+1 + y*rt+2 +  y*rt+3 + ...
y gamma is the discount factor that controls how much future rewards matter.
Exploration: try new actions to discover better strategies.
Exploitation:  Use what it already believes works best.

RL methods typically learn one of these:
-value-based learning (learn how good states/actions are)
The agent learns a value function:
-state-value V(s) expected return from state s
-action-value Q(s,a) expected return fromtaking action a in state s.

Then it acts by choosing actions with high estimated value.

Q learning
It updates its estimate by using the bellman idea(bootstrapping from the next state):
Q(s,a) <- Q(s,a) + alfa*[r  + gamma*maxQ(s',a')-Q(s,a)]


Policy-based learning
Instead of learning values and picking max you directly adjust the policy parameters  to increase expected reward.
This is useful when:
-actions are continous
-best policies are stochastic (randomized)


Q represents the action-value function, it is the agent's estimate of how good it is to take 
action a in state s,  measured as the expected return.

If you are one step away from the goal, Q(state,move-to-goal) becomes large because that action leads to a future reward soon.


---
GRIDWORLD
The agent is on a grid,  and the actions are up,down,left,right
The reward if +1 for reaching the goal and there is -0.01 ffor each steps to maximize efficiency.

This is a full markov decision process
state -> action -> reward  -> next state
If you don't know the environment  dynamics you can use model-free RL:

-Montecarlo
Learn by running the full episodes and averaging the total return.
It  works best when episodes end naturally.
It is simple but it can be noisy (high average) 

-Temporal difference learning
It learns step by step  without waiting for episode end.
    SARSA (on policy)
    It updates  using the actions you take next.
    Q-learning(off policy)
    it updates using the best possible next action 
    It learns an optimal greedy policy more directly 

-DQN
Same as Q-learning but Q(s,a) is approximated by a neural network isntead of a table.
It is used when the state-space is  huge.






