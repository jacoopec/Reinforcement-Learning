#!/usr/bin/env python3
"""
3-state Markov chain: stationary distribution + convergence by iteration.

Example:
P = [[0.5, 0.3, 0.2],
     [0.2, 0.5, 0.3],
     [0.1, 0.3, 0.6]]

We compute the stationary distribution pi satisfying pi P = pi, sum(pi)=1,
then show convergence of v_{n+1} = v_n P from different initial distributions.
"""

import numpy as np

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Solve for stationary distribution pi of an NxN Markov matrix P.

    We solve (P^T - I) pi^T = 0 with constraint sum(pi)=1 by replacing
    one equation with the normalization constraint.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if P.shape != (n, n):
        raise ValueError("P must be square.")
    if np.any(P < -1e-12):
        raise ValueError("P should not have negative entries.")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of P must sum to 1.")

    A = P.T - np.eye(n)
    b = np.zeros(n)

    # Replace last row with normalization constraint sum(pi)=1
    A[-1, :] = 1.0
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)

    # Clean tiny numerical noise and renormalize
    pi[abs(pi) < 1e-15] = 0.0
    pi = pi / pi.sum()
    return pi

def iterate(P: np.ndarray, v0: np.ndarray, steps: int = 25) -> np.ndarray:
    """
    Iterate v_{k+1} = v_k P, returning all vectors as an array of shape (steps+1, n).
    """
    P = np.asarray(P, dtype=float)
    v = np.asarray(v0, dtype=float)
    if v.ndim != 1 or v.shape[0] != P.shape[0]:
        raise ValueError("v0 must be a 1D vector of length n.")
    if not np.isclose(v.sum(), 1.0):
        raise ValueError("v0 must sum to 1.")
    if np.any(v < -1e-12):
        raise ValueError("v0 must be nonnegative.")

    out = [v.copy()]
    for _ in range(steps):
        v = v @ P
        out.append(v.copy())
    return np.vstack(out)

def print_convergence(traj: np.ndarray, pi: np.ndarray, every: int = 5) -> None:
    """
    Print selected iterates and the L1 distance to pi.
    """
    for k, v in enumerate(traj):
        if k % every == 0 or k == len(traj) - 1:
            l1 = np.sum(np.abs(v - pi))
            print(f"k={k:2d}  v={v}   L1_dist_to_pi={l1:.8f}")

def main() -> None:
    P = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6],
    ], dtype=float)

    pi = stationary_distribution(P)

    print("Transition matrix P:\n", P)
    print("\nStationary distribution pi (solution of pi P = pi):")
    print(pi)
    print("Check pi P:", pi @ P)
    print("Sum(pi):", pi.sum())

    # Show exact (rational) solution we derived earlier, for comparison:
    pi_exact = np.array([11/48, 3/8, 19/48], dtype=float)
    print("\nKnown exact solution (11/48, 3/8, 19/48):")
    print(pi_exact)
    print("Max abs diff:", np.max(np.abs(pi - pi_exact)))

    # Convergence from multiple initial distributions
    initials = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.2, 0.3, 0.5]),
    ]

    steps = 40
    for i, v0 in enumerate(initials, start=1):
        print(f"\n--- Convergence run #{i}: v0 = {v0} ---")
        traj = iterate(P, v0, steps=steps)
        print_convergence(traj, pi, every=5)

if __name__ == "__main__":
    main()
