import math

# ============================================
# Policy evaluation in a NxN Gridworld Policy Evaluation (Uniform π)
# ============================================
# Assumptions (standard textbook gridworld):
# - Grid size: NxN
# - Terminal states: (0,0) and (N-1,N-1) with V=0 and no updates
# - Actions: Up, Right, Down, Left
# - Deterministic transitions; if action hits wall -> stay in same cell
# - Reward each step: -1
# - Uniform policy: π(a|s)=0.25 for all 4 actions
# - Discount factor: gamma = 1.0
#
# We perform iterative policy evaluation:
#   V_{k+1}(s) = (1/4) * sum_a [ -1 + gamma * V_k(s') ]
# printing the full V-matrix at every iteration.

N = 4
GAMMA = 1.0
REWARD = -1.0
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
TERMINALS = {(0, 0), (N - 1, N - 1)}

def step(r, c, dr, dc):
    """Deterministic transition with wall bounce (stay put if off-grid)."""
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= N or nc < 0 or nc >= N:
        return r, c
    return nr, nc

def format_matrix(V):
    """Pretty print matrix with fixed width."""
    lines = []
    for r in range(N):
        row = []
        for c in range(N):
            row.append(f"{V[r][c]:7.3f}")
        lines.append(" ".join(row))
    return "\n".join(lines)

def policy_evaluation(max_iter=200, tol=1e-10, print_every_iter=True):
    # Initialize V(s) = 0 everywhere (including terminals)
    V = [[0.0 for _ in range(N)] for _ in range(N)]

    for it in range(1, max_iter + 1):
        V_new = [[V[r][c] for c in range(N)] for r in range(N)]
        delta = 0.0

        for r in range(N):
            for c in range(N):
                if (r, c) in TERMINALS:
                    V_new[r][c] = 0.0
                    continue

                # Bellman expectation update under uniform policy
                v = 0.0
                for dr, dc in ACTIONS:
                    nr, nc = step(r, c, dr, dc)
                    v += 0.25 * (REWARD + GAMMA * V[nr][nc])
                V_new[r][c] = v
                delta = max(delta, abs(V_new[r][c] - V[r][c]))

        V = V_new

        if print_every_iter:
            print(f"\nIteration {it} (max |Δ| = {delta:.3e})")
            print(format_matrix(V))

        if delta < tol:
            print(f"\nConverged in {it} iterations (tol={tol}).")
            break

    return V

if __name__ == "__main__":
    policy_evaluation(max_iter=200, tol=1e-2, print_every_iter=True)