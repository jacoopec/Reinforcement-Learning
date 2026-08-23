Temporal-Difference (TD) learning is a model-free way to learn value functions by updating from partial experience, using a learned estimate to “fill in” what you don’t know yet. It sits between Monte Carlo and pure dynamic programming.

Core idea: bootstrapping

Instead of waiting until the end of an episode (Monte Carlo), TD updates after each step using:

V(St) <-  V(St) + alfa*δt

if δt​ =  (Rt+1 + gamma*V(St+1) -V(St))
>0: things were better than expected, increase V(St)
<0: things were worse than expected, reduce V(St)

Compared to Monte Carlo (MC)

TD learns online: updates immediately after each transition.

Works naturally for continuing tasks (no need for episode termination).

Often lower variance than MC because it doesn’t use full returns.

Tradeoff: TD targets are biased early on (because V(St+1) is an estimate). MC targets are unbiased but high-variance.

γ=0.9 
α=0.1
Current estimate V(st)= 5 
observed reward Rt+1 = 2
next-state estimate V(St+1) = 6

TD taarget = 2 + 0.9*6 = 7.4

TD error  = 7.4-5 = 2.4

update 
V(St) <- 5 + 0.1*2.4 =5.24

TD for action values
Instead of values V(S) you can learn from action values Q(s,a)

-SARSA on-policy
learns the values on policy
Q(St,At) <- Q(St,At) +  α(Rt+1 + γQ(St+1,At+1) - Q(St,At))

-Q-learning off-policy
Learns the greedy optimal policy while behaving exploratorily:
Q(St,At) <- Q(St,At) +  α(Rt+1 + γ max(Q(St+1,a)-Q(St,At))

SARSA uses the next action actually taken.
Q-learning uses the best possible next action.
