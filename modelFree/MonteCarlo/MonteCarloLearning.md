Monte Carlo (MC) learning for model-free prediction is about estimating value functions from complete episodes, using sampled returns instead of a transition model or bootstrapping.
Setting: model-free prediction

You’re not trying to control (improve the policy), just evaluate a fixed policy 
𝜋
π.

You observe episodes: 
S0,A0,R1,S1,A1...ST

Goal: estimate vpi(s) = E[Gt|St = s]

where the return is Gt = Rt+1 + gamma*Rt+2 + ....

Core idea

Each time you visit a state 
𝑠
s in an episode, you can compute the return from that time step and treat it as a noisy sample of 
𝑣𝜋(s). Average many such samples → converges to the true expected value (under standard conditions).

Two common variants

First-visit MC prediction
Update V(s) only for the first time state s appears in the episode.

Every-visit MC prediction
Update V(s) for every occurrence of 𝑠 within the episode.

Both converge to vπ with enough episodes; every-visit often uses more data per episode.