## Temporal-Difference Learning

Il TD target dipende da una sola transizione, una sola azione e un solo reward quindi ha varianza inferiore
al ritorno finale e impara in meno tempo


Aggiorno il valore passo dopo passo, senza aspettare la fine dell’episodio.

La forma tipica è: $V(s)←V(s)+α[r+γV(s′)−V(s)]$

Temporal difference learning is  a  family of RL methhods that learn value functions from experience but update after every step instead of waiting for the end of   an episode.
After you moved from state s to s' and got reaward r, you update you r value estimate for  s toward:
target = r + gammaV(s')

this is called bootstrapping, using the current estimate V(s') to update V(s)

V(s): current value estimate

α: learning rate

γ: discount factor

r+γV(s′)−V(s): TD error (how wrong you were)

Intuition

If the outcome was better than expected, TD error > 0 → increase 


V(s)

If it was worse than expected, TD error < 0 → decrease 


V(s)

Why TD is useful

Doesn’t need a model (like Monte Carlo).

Learns online (step-by-step).

Often lower variance than Monte Carlo (but can have some bias due to bootstrapping).

TD vs Monte Carlo (practical difference)

Monte Carlo updates with the full return 

Gt after the episode ends.

TD updates immediately using a 1-step prediction.

Beyond TD(0): TD(λ)

TD(λ) mixes 1-step and multi-step updates using eligibility traces:

λ=0 → TD(0)


λ→1 → closer to Monte Carlo behavior
This is the “bridge” between TD and MC.




TD control (learning how to act)

When you use TD to learn action-values 

Q(s,a), you get famous algorithms:

SARSA (on-policy)

Q(s,a)←Q(s,a)+α(r+γQ(s′,a′)−Q(s,a))

Learns the value of the policy it actually follows (e.g., 
𝜖
ϵ-greedy).

Q-learning (off-policy)

Q(s,a)←Q(s,a)+α(r+γa′max	​Q(s′,a′)−Q(s,a))

Learns the greedy optimal policy while behaving exploratorily.





Temporal-Difference (TD) learning is a way to learn state values from experience while the episode is still running, by updating from a 1-step lookahead.

Core idea (one sentence)

Update your value estimate using:
what you got now + your current guess of the next state’s value.

TD(0) update rule

After you see a transition 
𝑠→𝑠′ with reward 𝑟

V(s) + alfa(r + gamma*V(s')-V(s))

V(s) Current estimate of value of state s
alfa learning rate
gamma discount factor
r+ gamma*V(s') target, better guess of what V(s) should be
r+ gamma*V(s') -Vd  TD Error

Monte Carlo waits until the episode ends to know the full return.

TD updates immediately, after each step, using a “bootstrap” guess V(s')

That’s TD learning: learn by stepping through experience and correcting your predictions each step.
