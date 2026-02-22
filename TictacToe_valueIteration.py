import random
from functools import lru_cache

# =========================
# Tic-Tac-Toe: Value Iteration (MDP)
# Agent = 'X'
# Opponent = 'O' with a fixed stochastic policy (default: random legal move)
# =========================

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diags
]

# Rewards from X's perspective
R_WIN = 1.0
R_LOSE = -1.0
R_DRAW = 0.0
R_STEP = 0.0  # optional small step cost, e.g. -0.01 to encourage faster wins

GAMMA = 1.0   # episodic, undiscounted is fine

def check_winner(board):
    """Return 'X', 'O', 'D' (draw), or None (game ongoing)."""
    for a, b, c in WIN_LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    if "." not in board:
        return "D"
    return None

def legal_moves(board):
    return [i for i, ch in enumerate(board) if ch == "."]

def apply_move(board, idx, player):
    b = list(board)
    b[idx] = player
    return "".join(b)

def is_terminal(board):
    return check_winner(board) is not None

def terminal_reward(board):
    w = check_winner(board)
    if w == "X":
        return R_WIN
    if w == "O":
        return R_LOSE
    return R_DRAW

# -------------------------
# Opponent policy (O)
# -------------------------
def opponent_policy_random(board):
    """Uniform random over legal moves."""
    moves = legal_moves(board)
    p = 1.0 / len(moves)
    return [(m, p) for m in moves]

# You can swap this in if you want a slightly smarter opponent (still fixed policy).
def opponent_policy_center_then_random(board):
    moves = legal_moves(board)
    if 4 in moves:
        # bias to center
        rest = [m for m in moves if m != 4]
        probs = [(4, 0.6)]
        if rest:
            p = 0.4 / len(rest)
            probs += [(m, p) for m in rest]
        else:
            probs = [(4, 1.0)]
        return probs
    # otherwise random
    p = 1.0 / len(moves)
    return [(m, p) for m in moves]

OPP_POLICY = opponent_policy_random  # <--- change if you want


# -------------------------
# MDP Dynamics (X acts, then O acts stochastically)
# -------------------------
@lru_cache(maxsize=None)
def transitions_after_x(board, x_move):
    """
    Given board and X's move, return list of (next_board, prob, reward, done).
    Opponent acts according to fixed stochastic policy.
    """
    if is_terminal(board):
        # no moves from terminal; keep absorbing
        return [(board, 1.0, terminal_reward(board), True)]

    # Apply X move
    b1 = apply_move(board, x_move, "X")

    # If X just ended the game
    if is_terminal(b1):
        return [(b1, 1.0, terminal_reward(b1), True)]

    # Otherwise O moves stochastically
    outs = []
    for o_move, p in OPP_POLICY(b1):
        b2 = apply_move(b1, o_move, "O")
        done = is_terminal(b2)
        r = terminal_reward(b2) if done else R_STEP
        outs.append((b2, p, r, done))

    # Numerical sanity: probs sum to ~1
    return outs


# -------------------------
# Enumerate reachable states (boards where it's X to move)
# -------------------------
def reachable_x_states():
    """
    Generate all reachable states under: X moves then O moves (any legal for both).
    We only keep states where it's X to move (i.e., counts are equal).
    """
    start = "........."
    seen = set()
    stack = [start]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)

        if is_terminal(b):
            continue

        # Ensure it's X to move: X count == O count
        if b.count("X") != b.count("O"):
            continue

        # For each possible X move, for each possible O move, add resulting X-turn state
        for xm in legal_moves(b):
            b1 = apply_move(b, xm, "X")
            if is_terminal(b1):
                continue
            for om in legal_moves(b1):
                b2 = apply_move(b1, om, "O")
                if is_terminal(b2):
                    continue
                # After O move, it's X's turn if counts equal again
                if b2.count("X") == b2.count("O"):
                    stack.append(b2)
    return seen

# -------------------------
# Value Iteration
# -------------------------
def value_iteration(tol=1e-10, max_iter=20000):
    states = reachable_x_states()
    # Include terminal states too (useful for evaluation)
    # But for VI we only update X-to-move nonterminal states; terminal are fixed.
    V = {s: (terminal_reward(s) if is_terminal(s) else 0.0) for s in states}
    pi = {s: None for s in states}

    for it in range(max_iter):
        delta = 0.0
        for s in states:
            if is_terminal(s):
                continue
            if s.count("X") != s.count("O"):
                continue  # not X turn

            moves = legal_moves(s)
            if not moves:
                continue

            best_val = -1e9
            best_move = moves[0]

            for a in moves:
                # Q(s,a) = E[r + gamma V(s')]
                q = 0.0
                for s2, p, r, done in transitions_after_x(s, a):
                    q += p * (r + (0.0 if done else GAMMA * V.get(s2, 0.0)))
                if q > best_val:
                    best_val = q
                    best_move = a

            delta = max(delta, abs(best_val - V[s]))
            V[s] = best_val
            pi[s] = best_move

        if delta < tol:
            # converged
            break

    return V, pi

# -------------------------
# Simulation to verify policy
# -------------------------
def play_game(policy, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
    b = "........."
    while True:
        w = check_winner(b)
        if w is not None:
            if w == "X":
                return 1
            if w == "O":
                return -1
            return 0

        # X move
        if b.count("X") != b.count("O"):
            # shouldn't happen in this loop
            raise RuntimeError("Not X's turn unexpectedly.")
        xm = policy.get(b, None)
        if xm is None:
            # fallback random
            xm = random.choice(legal_moves(b))
        b = apply_move(b, xm, "X")
        if verbose:
            print("X plays", xm, "\n", render(b), "\n")

        w = check_winner(b)
        if w is not None:
            continue

        # O move (fixed policy)
        dist = OPP_POLICY(b)
        u = random.random()
        cum = 0.0
        om = dist[-1][0]
        for m, p in dist:
            cum += p
            if u <= cum:
                om = m
                break
        b = apply_move(b, om, "O")
        if verbose:
            print("O plays", om, "\n", render(b), "\n")

def evaluate(policy, n=20000, seed=0):
    random.seed(seed)
    wins = losses = draws = 0
    for i in range(n):
        res = play_game(policy)
        if res == 1:
            wins += 1
        elif res == -1:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws

def render(board):
    rows = [board[i:i+3] for i in range(0, 9, 3)]
    return "\n".join(" ".join(r) for r in rows)

def show_first_move(policy):
    start = "........."
    move = policy.get(start, None)
    return move

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    V, pi = value_iteration()

    print("Opponent policy:", OPP_POLICY.__name__)
    print("Learned best first move index (0..8):", show_first_move(pi))

    # Evaluate learned policy vs the fixed opponent
    w, l, d = evaluate(pi, n=20000, seed=1)
    total = w + l + d
    print(f"Results vs opponent over {total} games:")
    print(f"  wins:  {w} ({w/total:.3f})")
    print(f"  losses:{l} ({l/total:.3f})")
    print(f"  draws: {d} ({d/total:.3f})")

    # Optional: play one verbose game
    # play_game(pi, seed=2, verbose=True)