import itertools
import numpy as np


# For this MDP, there are:

# 2^3=8

# deterministic policies, because there are 3 states and 2 possible actions per state.
# The ranking is based on the average value across all states:
# So the best policy is the one that gives the highest average expected discounted return.

# ============================================================
# Small finite Markov Decision Process
# ============================================================

states = ["Low", "Medium", "High"]
actions = ["Safe", "Risky"]

n_states = len(states)
n_actions = len(actions)

gamma = 0.9
theta = 1e-10
max_iterations = 1000

# ------------------------------------------------------------
# Transition probabilities P[s, a, s']
#
# P[s, a, s'] = probability of going to state s'
#               when taking action a in state s
# ------------------------------------------------------------

P = np.zeros((n_states, n_actions, n_states))

# State: Low
P[0, 0] = [0.7, 0.3, 0.0]  # Safe
P[0, 1] = [0.3, 0.6, 0.1]  # Risky

# State: Medium
P[1, 0] = [0.1, 0.7, 0.2]  # Safe
P[1, 1] = [0.2, 0.3, 0.5]  # Risky

# State: High
P[2, 0] = [0.0, 0.2, 0.8]  # Safe
P[2, 1] = [0.4, 0.2, 0.4]  # Risky

# ------------------------------------------------------------
# Rewards R[s, a, s']
#
# Reward depends on:
#   current state s,
#   action a,
#   next state s'
# ------------------------------------------------------------

R = np.zeros((n_states, n_actions, n_states))

# Safe gives smaller but stable rewards
R[:, 0, :] = [
    [1, 2, 0],   # from Low using Safe
    [1, 3, 4],   # from Medium using Safe
    [0, 4, 6],   # from High using Safe
]

# Risky gives bigger rewards, but can also fall back
R[:, 1, :] = [
    [-1, 3, 6],  # from Low using Risky
    [-2, 2, 8],  # from Medium using Risky
    [-5, 1, 10], # from High using Risky
]


def expected_return_for_action(s, a, V):
    """
    Computes:

        q(s, a) = sum_s' P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]

    This is the expected return of taking action a in state s,
    then following the current value function V afterward.
    """

    return np.sum(P[s, a] * (R[s, a] + gamma * V))


def value_iteration():
    """
    Solves the MDP using value iteration.

    Bellman optimality equation:

        V*(s) = max_a sum_s' P(s'|s,a)
                [R(s,a,s') + gamma * V*(s')]

    """

    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        new_V = np.zeros(n_states)

        for s in range(n_states):
            action_values = [
                expected_return_for_action(s, a, V)
                for a in range(n_actions)
            ]

            new_V[s] = max(action_values)

        delta = np.max(np.abs(new_V - V))
        V = new_V

        if delta < theta:
            print(f"Value iteration converged after {iteration + 1} iterations.")
            break

    return V


def extract_greedy_policy(V):
    """
    Extracts the optimal deterministic policy from V*.

    For each state, choose the action with the highest q(s, a).
    """

    policy = []

    for s in range(n_states):
        action_values = [
            expected_return_for_action(s, a, V)
            for a in range(n_actions)
        ]

        best_action = int(np.argmax(action_values))
        policy.append(best_action)

    return tuple(policy)


def evaluate_policy(policy):
    """
    Evaluates one deterministic policy exactly.

    For a fixed policy pi, the Bellman equation is:

        V_pi(s) = sum_s' P(s'|s,pi(s))
                  [R(s,pi(s),s') + gamma * V_pi(s')]

    In matrix form:

        V_pi = R_pi + gamma * P_pi V_pi

    Therefore:

        V_pi = inverse(I - gamma * P_pi) R_pi
    """

    P_pi = np.zeros((n_states, n_states))
    R_pi = np.zeros(n_states)

    for s in range(n_states):
        a = policy[s]

        P_pi[s] = P[s, a]

        R_pi[s] = np.sum(P[s, a] * R[s, a])

    I = np.eye(n_states)

    V_pi = np.linalg.solve(I - gamma * P_pi, R_pi)

    return V_pi


def policy_to_string(policy):
    """
    Converts a policy tuple into a readable string.
    Example:
        (0, 1, 0)
    means:
        Low -> Safe
        Medium -> Risky
        High -> Safe
    """

    parts = []

    for s, a in enumerate(policy):
        parts.append(f"{states[s]} -> {actions[a]}")

    return ", ".join(parts)


def rank_all_policies(start_state=None):
    """
    Enumerates and ranks all deterministic policies.

    If start_state is None:
        Policies are ranked by the average value across all states.

    If start_state is an integer:
        Policies are ranked by the value from that state only.
    """

    all_policies = list(itertools.product(range(n_actions), repeat=n_states))

    results = []

    for policy in all_policies:
        V_pi = evaluate_policy(policy)

        if start_state is None:
            score = np.mean(V_pi)
        else:
            score = V_pi[start_state]

        results.append((score, policy, V_pi))

    results.sort(reverse=True, key=lambda x: x[0])

    return results


if __name__ == "__main__":
    # --------------------------------------------------------
    # Solve the MDP
    # --------------------------------------------------------

    V_star = value_iteration()
    optimal_policy = extract_greedy_policy(V_star)

    print("\nOptimal value function V*:")
    for state, value in zip(states, V_star):
        print(f"V*({state}) = {value:.4f}")

    print("\nOptimal policy pi*:")
    print(policy_to_string(optimal_policy))

    # --------------------------------------------------------
    # Rank all deterministic policies
    # --------------------------------------------------------

    ranking = rank_all_policies(start_state=None)

    print("\nRanking of all deterministic policies")
    print("Ranking criterion: average value across all states\n")

    for rank, (score, policy, V_pi) in enumerate(ranking, start=1):
        print(f"Rank {rank}")
        print(f"Policy: {policy_to_string(policy)}")
        print(f"Average value: {score:.4f}")

        for state, value in zip(states, V_pi):
            print(f"  V_pi({state}) = {value:.4f}")

        print("-" * 60)