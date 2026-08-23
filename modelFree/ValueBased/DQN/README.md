# DQN in Reinforcement Learning

**DQN (Deep Q-Network)** is a reinforcement learning algorithm that combines **Q-learning** with a **deep neural network**.

In ordinary Q-learning, you store a table:

$$
Q(s,a)
$$

that estimates how good it is to take action \(a\) in state \(s\). This works when the number of states is small. For problems with very large state spaces, such as images, storing a Q-table becomes impossible.

DQN replaces the Q-table with a neural network:

$$
Q(s,a;\theta)
$$

The network receives a state \(s\) and outputs a Q-value for each possible action:

$$
s
\rightarrow
\text{Neural Network}
\rightarrow
[Q(s,a_1),Q(s,a_2),...,Q(s,a_n)]
$$

For example, in a game with three actions:

```text
State: image of game

          Neural Network
                |
                v
       [ Left   Right   Jump ]
       [  2.3    5.8     1.2 ]
```

The agent would normally choose **Right**, because:

$$
\arg\max_a Q(s,a)=\text{Right}
$$

During training, DQN usually uses an **ε-greedy policy**, so it sometimes chooses a random action to explore.

The network is trained using the Q-learning target:

$$
y =
r + \gamma \max_{a'} Q(s',a';\theta^-)
$$

and minimizes approximately:

$$
L =
\left(
y-Q(s,a;\theta)
\right)^2
$$

where:

- \(r\) is the reward,
- \(s'\) is the next state,
- \(\gamma\) is the discount factor.

## Key Techniques

Two techniques are especially important in DQN:

### Experience Replay

Transitions

$$
(s,a,r,s')
$$

are stored in a **replay buffer**, and random batches are sampled to train the network.

This reduces correlation between consecutive experiences and makes training more stable.

### Target Network

DQN keeps a second neural network:

$$
Q(s,a;\theta^-)
$$

whose parameters are updated less frequently.

It is used to compute the training target and helps stabilize learning.

## Basic DQN Loop

$$
s_t
\rightarrow Q\text{-network}
\rightarrow a_t
\rightarrow \text{environment}
\rightarrow (r_t,s_{t+1})
\rightarrow \text{replay buffer}
\rightarrow \text{network update}
$$

DQN is therefore a:

- **Model-free**
- **Value-based**
- **Off-policy**

reinforcement learning algorithm.

It does not learn the environment dynamics. Instead, it learns an approximation of the optimal action-value function:

$$
Q^*(s,a)
$$

A useful way to summarize DQN is:

$$
\boxed{
\text{Q-learning}
+
\text{Deep Neural Network}
+
\text{Experience Replay}
+
\text{Target Network}
=
\text{DQN}
}
$$
