import math

# ============================================
# DP / Policy Evaluation for the pictured MDP
# ============================================
#
# States: s^-2, s^-1, s^0, s^1, s^2
# Only decision state: s^0
# Actions at s^0: Left, Right
#
# The table in the image gives joint probabilities:
#   p(s', r | s^0, a)
# where r in {0,1,2} and s' depends on the action.
#
# We evaluate a FIXED policy at s^0:
#   pi(Left|s0)=0.4, pi(Right|s0)=0.6
#
# The outer-state values are taken from the figure:
#   V(s^-2)=19.2, V(s^-1)=16.5, V(s^1)=18.1, V(s^2)=16.2
#
# Then Bellman expectation at s^0 is:
#   V(s0) <- sum_a pi(a|s0) sum_{s',r} p(s',r|s0,a) [ r + gamma V(s') ]
#
# Note: since transitions from s^0 go only to outer states (not back to s^0),
# this converges in ONE update, but we still print iterations.

# Discount factor from figure
gamma = 0.95

# Fixed policy from figure
pi = {"Left": 0.4, "Right": 0.6}

# Given outer-state values from the figure
V_fixed = {
    "s-2": 19.2,
    "s-1": 16.5,
    "s1": 18.1,
    "s2": 16.2,
}

# Joint probabilities p(s', r | s0, a) from the table in the figure
# Left leads to s-2 or s-1; Right leads to s1 or s2; rewards r in {0,1,2}
P = {
    "Left": {
        ("s-2", 0): 0.34,
        ("s-2", 1): 0.05,
        ("s-2", 2): 0.17,
        ("s-1", 0): 0.17,
        ("s-1", 1): 0.23,
        ("s-1", 2): 0.04,
    },
    "Right": {
        ("s1", 0): 0.12,
        ("s1", 1): 0.22,
        ("s1", 2): 0.20,
        ("s2", 0): 0.09,
        ("s2", 1): 0.32,
        ("s2", 2): 0.05,
    },
}

def expected_return_for_action(action: str) -> float:
    """Compute Q(s0, action) = sum_{s',r} p(s',r|s0,a) [ r + gamma V(s') ]."""
    total = 0.0
    for (s_next, r), prob in P[action].items():
        total += prob * (r + gamma * V_fixed[s_next])
    return total

def bellman_update(V_s0: float) -> float:
    """One Bellman expectation update for V(s0) under the fixed policy pi."""
    # V_s0 isn't actually used in this particular MDP structure (no transitions back to s0),
    # but we keep it in the signature to emphasize it's an iterative DP update.
    q_left = expected_return_for_action("Left")
    q_right = expected_return_for_action("Right")
    return pi["Left"] * q_left + pi["Right"] * q_right

def main():
    tol = 1e-12
    max_iter = 20

    # Start with an arbitrary initial value for s0
    V_s0 = 0.0

    # Precompute action-values (these are constants here)
    q_left = expected_return_for_action("Left")
    q_right = expected_return_for_action("Right")

    print("Given:")
    print(f"  gamma = {gamma}")
    print(f"  pi(Left|s0) = {pi['Left']}, pi(Right|s0) = {pi['Right']}")
    print("  Outer state values:")
    for k in ["s-2", "s-1", "s1", "s2"]:
        print(f"    V({k}) = {V_fixed[k]}")
    print("\nAction-values (computed from the transition table):")
    print(f"  Q(s0, Left)  = {q_left:.6f}")
    print(f"  Q(s0, Right) = {q_right:.6f}")

    greedy = "Left" if q_left > q_right else "Right"
    print(f"  Greedy (policy improvement) would choose: {greedy}\n")

    print("Policy Evaluation Iterations for V(s0):")
    print("iter | V_old        -> V_new        | delta")
    print("-" * 52)

    for it in range(1, max_iter + 1):
        V_new = bellman_update(V_s0)
        delta = abs(V_new - V_s0)
        print(f"{it:4d} | {V_s0:12.8f} -> {V_new:12.8f} | {delta:.3e}")
        V_s0 = V_new
        if delta < tol:
            break

    print("\nFinal:")
    print(f"  V(s0) ≈ {V_s0:.6f}")

if __name__ == "__main__":
    main()