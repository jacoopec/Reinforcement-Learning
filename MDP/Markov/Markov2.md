A Markov process “models state dynamics” in the sense that it gives you a mathematical rule for how the state evolves over time.

1) You represent the system by states

Pick a set of possible states 

S={1,2,…}.
Example: weather 

S={Sunny,Cloudy,Rainy}.

Let 
𝑋
𝑡


 be the state at time 

t=0,1,2,… or continuous time 


2) You model how it changes (the “dynamics”)

Instead of a deterministic rule like “tomorrow is always sunny,” you use probabilities:

Pr
⁡
(
𝑋
𝑡
+
1
=
𝑗
∣
𝑋
𝑡
=
𝑖
)
=
𝑃
𝑖
𝑗
Pr(X
t+1
	​

=j∣X
t
	​

=i)=P
ij
	​


So if you’re in state 
𝑖
i now, 
𝑃
𝑖
𝑗
P
ij
	​

 is the chance you’ll be in state 
𝑗
j next step.
The matrix 
𝑃
P is the dynamics of the Markov chain.

3) The Markov property: “memoryless”

The key feature is that the future depends on the current state only, not the whole past:

Pr
⁡
(
𝑋
𝑡
+
1
=
𝑗
∣
𝑋
𝑡
=
𝑖
,
𝑋
𝑡
−
1
,
…
,
𝑋
0
)
=
Pr
⁡
(
𝑋
𝑡
+
1
=
𝑗
∣
𝑋
𝑡
=
𝑖
)
Pr(X
t+1
	​

=j∣X
t
	​

=i,X
t−1
	​

,…,X
0
	​

)=Pr(X
t+1
	​

=j∣X
t
	​

=i)

So the current state summarizes all the relevant information for predicting the next state.

4) What you can do with a state-dynamics model

Once you have the transition probabilities, you can compute:

Prediction over time: if your current distribution is 
𝑣
(
0
)
v
(0)
, then

𝑣
(
𝑛
)
=
𝑣
(
0
)
𝑃
𝑛
v
(n)
=v
(0)
P
n

Long-run behavior: does it converge to a stationary distribution 
𝜋
π with 
𝜋
𝑃
=
𝜋
πP=π?

Expected times: time to hit a state (failure, recovery, etc.), absorption probabilities, etc.

5) Tiny concrete example

States: {Working (W), Failed (F)}

𝑃
=
(
0.98
	
0.02


0.10
	
0.90
)
P=(
0.98
0.10
	​

0.02
0.90
	​

)

If it’s working now, it fails next step with probability 
0.02
0.02.

If it’s failed now, it gets repaired next step with probability 
0.10
0.10.

That’s “modeling state dynamics”: you’re capturing the system’s evolution as probabilistic jumps between states, governed by 
𝑃
P.
