import math

# ============================================
# 4x4 Gridworld Policy Iteration
# ============================================
# - Terminal states: (0,0) and (3,3) with V=0
# - Actions: Up, Right, Down, Left
# - Deterministic transitions; off-grid => stay
# - Reward each step: -1
# - Discount: gamma = 1.0 (episodic with terminals)
# - Policy iteration: eval (print V each sweep) + improve (print policy each round)

N = 4
GAMMA = 1.0
REWARD = -1.0

ACTIONS = ["U", "R", "D", "L"]
DELTAS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
TERMINALS = {(0, 0), (N - 1, N - 1)}


def step(r, c, a):
    dr, dc = DELTAS[a]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= N or nc < 0 or nc >= N:
        return r, c
    return nr, nc


def format_matrix(V):
    return "\n".join(" ".join(f"{V[r][c]:7.3f}" for c in range(N)) for r in range(N))


def format_policy(pi):
    """
    Show deterministic policy arrows for each state.
    Terminals shown as 'T'. If stochastic (ties), show concatenated actions.
    """
    arrow = {"U": "↑", "R": "→", "D": "↓", "L": "←"}
    lines = []
    for r in range(N):
        row = []
        for c in range(N):
            if (r, c) in TERMINALS:
                row.append("  T  ")
            else:
                # If policy is distribution, show all max-prob actions (often 1 after improvement)
                best_p = max(pi[r][c].values())
                acts = [a for a, p in pi[r][c].items() if abs(p - best_p) < 1e-12 and p > 0]
                row.append(" " + "".join(arrow[a] for a in acts).ljust(3) + " ")
        lines.append("".join(row))
    return "\n".join(lines)


def init_uniform_policy():
    # pi[r][c] is a dict action->prob
    pi = [[{a: 0.25 for a in ACTIONS} for _ in range(N)] for _ in range(N)]
    for (r, c) in TERMINALS:
        pi[r][c] = {a: 0.0 for a in ACTIONS}
    return pi


def policy_evaluation(pi, V, tol=1e-10, max_eval_iters=10_000, print_each_iter=True, label=""):
    """
    Iterative policy evaluation:
      V_{k+1}(s)= sum_a pi(a|s) [ -1 + gamma V_k(s') ]
    Prints the full V matrix each evaluation iteration if requested.
    """
    for it in range(1, max_eval_iters + 1):
        delta = 0.0
        V_new = [[V[r][c] for c in range(N)] for r in range(N)]

        for r in range(N):
            for c in range(N):
                if (r, c) in TERMINALS:
                    V_new[r][c] = 0.0
                    continue

                v = 0.0
                for a, p in pi[r][c].items():
                    if p == 0:
                        continue
                    nr, nc = step(r, c, a)
                    v += p * (REWARD + GAMMA * V[nr][nc])
                V_new[r][c] = v
                delta = max(delta, abs(V_new[r][c] - V[r][c]))

        V = V_new
        if print_each_iter:
            print(f"\nPolicy Evaluation{label} - iter {it} (max |Δ| = {delta:.3e})")
            print(format_matrix(V))

        if delta < tol:
            break

    return V


def policy_improvement(pi, V):
    """
    Greedy improvement:
      pi'(s) = argmax_a [ -1 + gamma V(s') ]
    If ties, split probability equally among best actions.
    Returns (new_pi, stable_flag).
    """
    stable = True
    new_pi = [[dict(pi[r][c]) for c in range(N)] for r in range(N)]

    for r in range(N):
        for c in range(N):
            if (r, c) in TERMINALS:
                continue

            # compute one-step lookahead q(s,a)
            qs = {}
            best = -1e18
            for a in ACTIONS:
                nr, nc = step(r, c, a)
                q = REWARD + GAMMA * V[nr][nc]
                qs[a] = q
                best = max(best, q)

            best_actions = [a for a, q in qs.items() if abs(q - best) < 1e-12]
            # create improved distribution (uniform over best actions)
            improved = {a: 0.0 for a in ACTIONS}
            for a in best_actions:
                improved[a] = 1.0 / len(best_actions)

            if improved != pi[r][c]:
                stable = False
            new_pi[r][c] = improved

    return new_pi, stable


def main():
    pi = init_uniform_policy()
    V = [[0.0 for _ in range(N)] for _ in range(N)]

    print("Initial policy (uniform; shown as all arrows):")
    print(format_policy(pi))

    max_policy_iters = 2
    for k in range(1, max_policy_iters + 1):
        # 1) Evaluate current policy (prints V each sweep)
        V = policy_evaluation(
            pi, V,
            tol=1e-10,
            max_eval_iters=10,
            print_each_iter=True,
            label=f" (policy iter {k})"
        )

        # 2) Improve policy
        pi_new, stable = policy_improvement(pi, V)
        print(f"\nPolicy Improvement (policy iter {k}) - stable: {stable}")
        print("Improved policy:")
        print(format_policy(pi_new))

        pi = pi_new
        if stable:
            print("\nPolicy iteration converged (policy stable).")
            break

    print("\nFinal value function V:")
    print(format_matrix(V))


if __name__ == "__main__":
    main()