import random

# ----- MDP definition -----
states = ["A", "B", "C"]
actions = ["Left", "Right"]

def transition(s, a):
    """Deterministic transition function: returns next_state."""
    if s == "A":
        return "A" if a == "Left" else "B"
    if s == "B":
        return "A" if a == "Left" else "C"
    if s == "C":
        return "B" if a == "Left" else "D"
    if s == "D":
        return "C" if a == "Left" else "E"
    if s == "E":
        return "D" if a == "Left" else "E"   
    raise ValueError("Unknown state")

def reward(s, a, s_next):
    """Reward for arriving in E."""
    return 1 if s_next == "E" else 0

# ----- A simple rollout (simulate an agent) -----
def run_episode(start="A", steps=10, policy=None, seed=0):
    random.seed(seed)
    s = start
    total = 0
    trajectory = [(s, None, 0)]

    for t in range(steps):
        if s == "E":
            break  # episode ends when we reach C
        
        # policy: a function that chooses an action given state
        if policy is None:
            a = random.choice(actions)  # random behavior
        else:
            a = policy(s)

        s_next = transition(s, a)
        r = reward(s, a, s_next)

        total += r
        trajectory.append((s_next, a, r))
        s = s_next

    return total, trajectory

# Example: a greedy policy that always moves right
def always_right_policy(s):
    return "Right"

if __name__ == "__main__":
    total, traj = run_episode(start="A", steps=10, policy=always_right_policy)
    print("Total reward:", total)
    print("Trajectory (state, action_taken_to_get_here, reward):")
    for item in traj:
        print(item)