# QLearning

QLearning is off-policy TD control

$$Q(s,a)<- Q(s,a) + α(r + γ*maxQ(s',a'')-Q(s,a))$$

Uses the best possible  next action via max.
will learn the Q value from trial and error? Exactly. We initialize the Q, we choose an action and perform it, we evaluate it by measuring the reward and we update the Q accordingly. In first, randomness will be a key player but as the agent explores the environment, the algorithm will find the best Q value for each state and action. Can we describe this mathematically?


The basic concept to understand here is that the Bellman equation relates states with each other and thus, it relates Action value functions.

But what if some action has a very small probability to produce a very large reward? The agent will never get there. This is fixed by adding random exploration. Every once in a while, the agent will perform a random move, without considering the optimal policy. But because we want the algorithm to converge at some point, we lower the probability to take a random action as the game proceeds.

Q learning is good. No one can deny that. But the fact that it is ineffective in big state spaces remains. Imagine a game with 1000 states and 1000 actions per state. We would need a table of 1 million cells. And that is a very small state space comparing to chess or Go. Also, Q learning can’t be used in unknown states because it can’t infer the Q value of new states from the previous ones.

What if we approximate the Q values using some machine learning model.
What if we approximate them using neural networks? 


In deep Q learning, we utilize a neural network to approximate the Q value function. The network receives the state as an input (whether is the frame of the current state or a single value) and outputs the Q values for all possible actions. The biggest output is our next action. We can see that we are not constrained to Fully Connected Neural Networks, but we can use Convolutional, Recurrent and whatever else type of model suits our needs.


