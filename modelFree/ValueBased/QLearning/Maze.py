import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 0 = free, 1 = wall
grid = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=int)

start = (0, 0)
goal  = (5, 5)

actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N,E,S,W
H, W = grid.shape

def step(state, a_idx):
    r, c = state
    dr, dc = actions[a_idx]
    nr, nc = r + dr, c + dc
    # illegal move -> stay
    if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == 1:
        nr, nc = r, c
    next_state = (nr, nc)
    reward = -1
    done = (next_state == goal)
    return next_state, reward, done

# --- Q-learning ---
Q   = np.zeros((H, W, 4), dtype=float)
rng = np.random.default_rng(0)

alpha, gamma            = 0.1, 0.99
eps, eps_min, eps_decay = 1.0, 0.05, 0.995

for _ in range(800):      # episodes
    s = start
    for _ in range(200):  # max steps
        r, c = s
        # epsilon-greedy
        if rng.random() < eps:
            a = int(rng.integers(4))
        else:
            a = int(np.argmax(Q[r, c]))
        s2, rew, done = step(s, a)
        r2, c2 = s2
        Q[r, c, a] += alpha * (rew + gamma * np.max(Q[r2, c2]) - Q[r, c, a])
        s = s2
        if done:
            break
    eps = max(eps_min, eps * eps_decay)

# --- Greedy path after learning ---
path = [start]
s = start
for _ in range(200):
    r, c = s
    a = int(np.argmax(Q[r, c]))
    s2, _, done = step(s, a)
    path.append(s2)
    s = s2
    if done:
        break

# --- Plot maze + learned path ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Q-learning on a tiny maze (path after training)")
ax.set_xlim(-0.5, W - 0.5)
ax.set_ylim(H - 0.5, -0.5)
ax.set_xticks(np.arange(W))
ax.set_yticks(np.arange(H))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

# walls (hatched squares)
for rr in range(H):
    for cc in range(W):
        if grid[rr, cc] == 1:
            ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, hatch="///", fill=False))

# draw path
xs = [c for r, c in path]
ys = [r for r, c in path]
ax.plot(xs, ys, lw=2)
ax.plot([xs[0]], [ys[0]], marker="o", markersize=10)
ax.plot([xs[-1]], [ys[-1]], marker="o", markersize=10)

ax.text(start[1], start[0], "S", ha="center", va="center", fontsize=14)
ax.text(goal[1], goal[0], "G", ha="center", va="center", fontsize=14)

plt.tight_layout()
plt.show()
