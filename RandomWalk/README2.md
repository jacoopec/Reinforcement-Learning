# Dynamic Programming Solution (Policy Evaluation at s^0)

This README explains how to solve the pictured MDP problem using **dynamic programming** (specifically, **policy evaluation** at the decision state \(s^0\)).

From the diagram:

- The only decision state is **\(s^0\)**.
- Available actions at \(s^0\): **Left** (←) and **Right** (→).
- The table gives **joint probabilities** \(p(s', r \mid s^0, a)\) for:
  - next state \(s' \in \{s^{-2}, s^{-1}, s^{1}, s^{2}\}\)
  - immediate reward \(r \in \{0, 1, 2\}\)
- Given policy at \(s^0\):
  - \(\pi(\leftarrow \mid s^0) = 0.4\)
  - \(\pi(\rightarrow \mid s^0) = 0.6\)
- Discount factor: \(\gamma = 0.95\)
- Given state values (from the figure):
  - \(V(s^{-2}) = 19.2\)
  - \(V(s^{-1}) = 16.5\)
  - \(V(s^{1}) = 18.1\)
  - \(V(s^{2}) = 16.2\)

---

## 1) Bellman Expectation Equation (Policy Evaluation)

For a fixed policy \(\pi\), the value of state \(s^0\) is:

\[
V(s^0) = \sum_{a} \pi(a\mid s^0) \sum_{s',r} p(s',r\mid s^0,a)\,[r + \gamma V(s')]
\]

It is convenient to compute **action-values**:

\[
Q(s^0,a) = \sum_{s',r} p(s',r\mid s^0,a)\,[r + \gamma V(s')]
\]

Then:

\[
V(s^0)=0.4\,Q(s^0,\leftarrow)+0.6\,Q(s^0,\rightarrow)
\]

---

## 2) Compute \(Q(s^0,\leftarrow)\)

From the table for **Left**, the next states are \(s^{-2}\) and \(s^{-1}\) with rewards \(0,1,2\):

- Reward 0: \(p(s^{-2},0)=0.34,\; p(s^{-1},0)=0.17\)
- Reward 1: \(p(s^{-2},1)=0.05,\; p(s^{-1},1)=0.23\)
- Reward 2: \(p(s^{-2},2)=0.17,\; p(s^{-1},2)=0.04\)

So:

\[
Q(s^0,\leftarrow)=\sum p(s',r)\,[r+0.95\,V(s')]
\]

Plugging in the given values:

\[
Q(s^0,\leftarrow)=17.8114
\]

---

## 3) Compute \(Q(s^0,\rightarrow)\)

From the table for **Right**, the next states are \(s^{1}\) and \(s^{2}\):

- Reward 0: \(p(s^{1},0)=0.12,\; p(s^{2},0)=0.09\)
- Reward 1: \(p(s^{1},1)=0.22,\; p(s^{2},1)=0.32\)
- Reward 2: \(p(s^{1},2)=0.20,\; p(s^{2},2)=0.05\)

So:

\[
Q(s^0,\rightarrow)=\sum p(s',r)\,[r+0.95\,V(s')]
\]

Numerically:

\[
Q(s^0,\rightarrow)=17.4047
\]

---

## 4) Compute \(V(s^0)\) Under the Given Policy

\[
V(s^0)=0.4(17.8114)+0.6(17.4047)=17.56738
\]

### Final Answer

\[
\boxed{V(s^0) \approx 17.57}
\]

---

## Optional: Policy Improvement Insight

Since:

- \(Q(s^0,\leftarrow)=17.8114\)
- \(Q(s^0,\rightarrow)=17.4047\)

We have \(Q(s^0,\leftarrow) > Q(s^0,\rightarrow)\), so a **greedy improvement step** would favor choosing **Left** more often (potentially always Left).
