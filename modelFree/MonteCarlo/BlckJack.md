Model-Free Monte Carlo Example: Blackjack

A classic example of model-free Monte Carlo (MC) reinforcement learning is Blackjack.

State

A state can be represented as:

[
s = (\text{player sum},\ \text{dealer visible card},\ \text{usable ace})
]

For example:

(15, 10, no usable ace)

means:

the player's current hand sums to 15,

the dealer is showing a 10,

the player has no usable ace.

Actions

The agent can choose:

hit
stick

Why it is model-free

The agent does not need to know the environment's transition probabilities:

[
P(s' \mid s,a)
]

For example, it does not need an explicit model describing the probability of reaching every possible next hand after choosing hit.

Instead, it simply plays many games and observes what happens.

Example episode

One sampled game could be:

[
(15,10,\text{no ace})
\xrightarrow{\text{hit}}
(19,10,\text{no ace})
\xrightarrow{\text{stick}}
\text{win}
]

Suppose the rewards are:

Win: +1

Draw: 0

Loss: -1

The final return for this episode is therefore:

[
G = +1
]

Monte Carlo learning can use this return to update estimates such as:

[
Q((15,10,\text{no ace}),\text{hit})
]

and:

[
Q((19,10,\text{no ace}),\text{stick})
]

After many complete Blackjack games:

[
Q(s,a) \approx \text{average return observed after taking action } a
\text{ in state } s
]

Main idea

Monte Carlo is model-free because it learns directly from complete sampled episodes instead of using an explicit model of the environment.

[
\boxed{\text{Experience} \rightarrow \text{Returns} \rightarrow Q(s,a)}
]