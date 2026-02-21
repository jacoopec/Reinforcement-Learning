import random

# ----- MDP definition (toy robot walking / locomotion) -----
# We model a very simple 1D "walker" that tries to move forward without falling.
#
# State = (position, stability)
# - position: 0..GOAL (discrete progress along a line)
# - stability: "Stable", "Wobbly", "Fallen"
#
# Action = gait choice:
# - "SmallStep": safer, slower
# - "BigStep": faster, riskier
# - "Recover": try to improve stability (can lose progress)
#
# Transition is stochastic: actions change position and stability with probabilities.
# Reward encourages forward progress, penalizes falling, and ends episode at goal or fall.

GOAL = 10
states = [(pos, st) for pos in range(GOAL + 1) for st in ["Stable", "Wobbly", "Fallen"]]
actions = ["SmallStep", "BigStep", "Recover"]

def is_terminal(s):
    pos, st = s
    return st == "Fallen" or pos >= GOAL

def transition(s, a):
    """Stochastic transition: returns next_state."""
    pos, st = s
    if st == "Fallen":
        return s  # absorbing

    # Helper random
    u = random.random()

    # If you're wobbly, you're more likely to fall on steps
    wobble_factor = 0.0 if st == "Stable" else 0.15

    if a == "SmallStep":
        # +1 progress most of the time, small chance of no progress
        # stability may worsen a bit; fall is rare
        if u < 0.75:
            pos_next = min(GOAL, pos + 1)
        else:
            pos_next = pos

        # stability transition
        v = random.random()
        if v < 0.80:
            st_next = st  # keep
        elif v < 0.95:
            st_next = "Wobbly"
        else:
            st_next = "Fallen" if random.random() < wobble_factor else "Wobbly"

    elif a == "BigStep":
        # More progress (+2) but higher chance to get wobbly or fall
        if u < 0.70:
            pos_next = min(GOAL, pos + 2)
        elif u < 0.90:
            pos_next = min(GOAL, pos + 1)
        else:
            pos_next = pos

        v = random.random()
        # chance to become wobbly or fall
        fall_chance = 0.05 + wobble_factor
        if v < 0.60:
            st_next = "Wobbly"
        elif v < 1.0 - fall_chance:
            st_next = st
        else:
            st_next = "Fallen"

    else:  # Recover
        # Often improves stability, but may lose 1 position due to corrective motion
        if u < 0.40:
            pos_next = max(0, pos - 1)
        else:
            pos_next = pos

        v = random.random()
        if v < 0.70:
            st_next = "Stable"
        elif v < 0.95:
            st_next = "Wobbly"
        else:
            st_next = "Fallen"  # rare slip during recovery

    return (pos_next, st_next)

def reward(s, a, s_next):
    """Reward encourages forward progress, penalizes falling."""
    pos, st = s
    pos2, st2 = s_next

    if st2 == "Fallen":
        return -10

    # reward for forward progress
    progress = pos2 - pos
    r = progress  # +1 or +2 typical

    # small penalty for spending time wobbly (encourage stability)
    if st2 == "Wobbly":
        r -= 0.5

    # bonus for reaching goal
    if pos2 >= GOAL:
        r += 5

    return r

# ----- A simple rollout (simulate an agent) -----
def run_episode(start=(0, "Stable"), steps=30, policy=None, seed=0):
    random.seed(seed)
    s = start
    total = 0
    trajectory = [(s, None, 0)]

    for t in range(steps):
        if is_terminal(s):
            break

        if policy is None:
            a = random.choice(actions)
        else:
            a = policy(s)

        s_next = transition(s, a)
        r = reward(s, a, s_next)

        total += r
        trajectory.append((s_next, a, r))
        s = s_next

    return total, trajectory

# Example policy:
# - If wobbly, recover; otherwise take big steps to move fast
def simple_walking_policy(s):
    pos, st = s
    if st == "Wobbly":
        return "Recover"
    return "BigStep"

if __name__ == "__main__":
    total, traj = run_episode(start=(0, "Stable"), steps=30, policy=simple_walking_policy, seed=1)
    print("Total reward:", total)
    print("Trajectory (state, action_taken_to_get_here, reward):")
    for item in traj:
        print(item)
