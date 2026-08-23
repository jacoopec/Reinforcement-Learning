>Both value iteration and policy iteration are classic dynamic programming methods for solving a Markov Decision Process (MDP). They assume you know the environment model (transitions + rewards), like in your maze.

# Value Iteration (VI)
`Optimize values → derive policy`
Combines evaluation + improvement in one step
`Jump directly toward optimality`


Core idea

Instead of explicitly maintaining a policy, you directly compute the optimal value function:

$$𝑉∗(𝑠) = max(𝑎)   $$ 

$$∑𝑠′ 𝑃(𝑠′∣𝑠,𝑎) [𝑅(𝑠,𝑎,𝑠′)+𝛾𝑉∗(𝑠′)]$$

This is the Bellman optimality equation.

How it works
You repeatedly update values:
$$𝑉𝑘+1(𝑠)  =  max(⁡𝑎)   [𝑅(𝑠,𝑎) +  𝛾∑𝑠′    𝑃(𝑠′∣𝑠,𝑎) 𝑉𝑘(𝑠′)]$$

Steps:

 - Initialize V(s) arbitrarily (e.g., 0)
 - Repeatedly update all states using the equation above
 - Stop when values converge
 - Extract policy afterward:
$$𝜋∗(𝑠) =  arg⁡max(⁡𝑎) 𝑄(𝑠,𝑎)π∗(s) =  argamax	​Q(s,a)$$

Intuition
“What is the best possible future value from this state?”
Policy is implicit during learning


---

# Policy Iteration (PI)
`Evaluate policy → improve policy`
Policy Iteration separates:


evaluation (how good is current policy?)
 and 
improvement (make it better)

Core idea:

Instead of optimizing values directly, you:

 - Start with a policy
 - Improve it step by step
 - Two alternating steps
`(1) Policy Evaluation`

Compute value of current policy:

$$𝑉𝜋(𝑠)=∑𝑠′𝑃(𝑠′∣𝑠,𝜋(𝑠))[𝑅+𝛾𝑉𝜋(𝑠′)]Vπ(s)$$

`(2) Policy Improvement`

Update policy greedily:
$$𝜋′(𝑠)=arg⁡max⁡𝑎𝑄(𝑠,𝑎)π′(s)=argamax	​Q(s,a)$$

Repeat until policy stops changing.

Intuition
“Given this policy, how good is it?”
“Can I improve it locally?”


Main object	Value function	Policy + value
Update style	One-step greedy backup	Alternate eval + improve
Policy handling	Derived at the end	Explicit at every step
Computation per iteration	Cheap	More expensive (evaluation step)
Convergence	Gradual	Often fewer iterations
Typical behavior	Many small updates	Fewer but heavier updates











