import numpy as np

gamma = 0.9

# States: A=0, B=1
# Actions: L=0, R=1
A, B = 0, 1
L, R = 0, 1

# Fixed policy pi(a|s): rows=states, cols=actions
# pi[s,a]
pi = np.array([
    [0.2, 0.8],  # in A: P(L)=0.2, P(R)=0.8
    [0.5, 0.5],  # in B: P(L)=0.5, P(R)=0.5
], dtype=float)

# Transition model P[s,a,s'] = P(s' | s,a)
P = np.zeros((2, 2, 2), dtype=float)
# Reward model R[s,a,s'] = R(s,a,s')
Rwd = np.zeros((2, 2, 2), dtype=float)

# From A:
# - action R: go to B with reward +5
P[A, R, B] = 1.0
Rwd[A, R, B] = 5.0
# - action L: stay in A with reward 0
P[A, L, A] = 1.0
Rwd[A, L, A] = 0.0

# From B:
# - action R: stay in B with reward +1
P[B, R, B] = 1.0
Rwd[B, R, B] = 1.0
# - action L: go to A with reward 0
P[B, L, A] = 1.0
Rwd[B, L, A] = 0.0

def policy_evaluation_dp(pi, P, Rwd, gamma, iters=30):
    """
    Iterative DP policy evaluation:
      v_{k+1}(s) <- sum_a pi(a|s) sum_{s'} P(s'|s,a)[ R(s,a,s') + gamma v_k(s') ]
    """
    nS = pi.shape[0]
    v = np.zeros(nS, dtype=float)

    for k in range(iters):
        v_new = np.zeros_like(v)
        for s in range(nS):
            total = 0.0
            for a in range(pi.shape[1]):
                # expected over next state s'
                exp_next = 0.0
                for sp in range(nS):
                    exp_next += P[s, a, sp] * (Rwd[s, a, sp] + gamma * v[sp])
                total += pi[s, a] * exp_next
            v_new[s] = total

        v = v_new
        print(f"iter {k+1:2d}: v(A)={v[A]:.6f}, v(B)={v[B]:.6f}")

    return v

print("Policy evaluation (iterative DP) convergence:")
v_est = policy_evaluation_dp(pi, P, Rwd, gamma, iters=25)

# Also compute the exact solution by solving linear system:
# v = r_pi + gamma P_pi v  -> (I - gamma P_pi) v = r_pi
nS, nA = pi.shape
P_pi = np.zeros((nS, nS))
r_pi = np.zeros(nS)
for s in range(nS):
    for a in range(nA):
        # expected next-state distribution under pi
        P_pi[s, :] += pi[s, a] * P[s, a, :]
        # expected immediate reward under pi
        r_pi[s] += pi[s, a] * np.sum(P[s, a, :] * Rwd[s, a, :])

v_exact = np.linalg.solve(np.eye(nS) - gamma * P_pi, r_pi)

print("\nFinal (iterative) v:", v_est)
print("Exact v:", v_exact)
print("Abs diff:", np.abs(v_est - v_exact))
