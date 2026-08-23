Monte Carlo (MC) evaluation is a simple way to estimate the value of a state by running episodes and averaging the returns you actually observed.

What it estimates

For a fixed policy 
π, the value is:

Vπ(s)=E[G∣St=s]

MC estimates this expectation by sample averages.

How it works (step by step)

Pick a policy 
𝜋
π (you’re not improving it yet—just evaluating it).

Generate episodes by following 
S0,A0,R1,S1,...St

 (until terminal).

Every time you visit a state 𝑠 in an episode, compute the return from that point:

Gt = Rt+1 +yRt+2

Average those returns for each state:

𝑉(𝑠)≈average of all observed 𝐺 starting from 𝑠
V(s)≈average of all observed G starting from s

That’s it: value = average outcome you got after being in that state.

First-visit vs Every-visit

First-visit MC: in each episode, only use the first time you visit 
𝑠
s.

Every-visit MC: use every time you visit 
𝑠
s in that episode.

Both converge with enough episodes.

Why it’s called “Monte Carlo”

Because it uses random sampling (random episodes) to estimate expectations.
