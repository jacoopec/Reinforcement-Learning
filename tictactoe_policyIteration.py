import random
from functools import lru_cache

# =========================
# Tic-Tac-Toe: Policy Iteration (MDP)
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
R_STEP = 0.0  # optional step shaping, e.g. -0.01 to prefer faster wins

GAMMA = 1.0   # episodic, undiscounted


def check_winner(board: str):
    """Return 'X', 'O', 'D' (draw), or None (game ongoing)."""
    for a, b, c in WIN_LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    if "." not in board:
        return "D"
    return None


def is_terminal(board: str) -> bool:
    return check_winner(board) is not None


def terminal_reward(board: str) -> float:
    w = check_winner(board)
    if w == "X":
        return R_WIN
    if w == "O":
        return R_LOSE
    return R_DRAW


def legal_moves(board: str):
    return [i for i, ch in enumerate(board) if ch == "."]


def apply_move(board: str, idx: int, player: str) -> str:
    b = list(board)
    b[idx] = player
    return "".join(b)


# -------------------------
# Opponent policy (O) - fixed, stochastic
# -------------------------
def opponent_policy_random(board: str):
    """Uniform random over legal moves."""
    moves = legal_moves(board)
    p = 1.0 / len(moves)
    return [(m, p) for m in moves]


# Slightly different but still fixed opponent (optional)
def opponent_policy_center_then_random(board: str):
    moves = legal_moves(board)
    if 4 in moves:
        rest = [m for m in moves if m != 4]
        if not rest:
            return [(4, 1.0)]
        probs = [(4, 0.6)]
        p = 0.4 / len(rest)
        probs += [(m, p) for m in rest]
        return probs
    p = 1.0 / len(moves)
    return [(m, p) for m in moves]


OPP_POLICY = opponent_policy_random  # <--- swap if you want


# -------------------------
# MDP transitions: X acts, then O acts stochastically
# -------------------------
@lru_cache(maxsize=None)
def transitions_after_x(board: str, x_move: int):
    """
    Given board and X's move, return list of (next_board, prob, reward, done).
    """
    if is_terminal(board):
        return [(board, 1.0, terminal_reward(board), True)]

    b1 = apply_move(board, x_move, "X")

    if is_terminal(b1):
        return [(b1, 1.0, terminal_reward(b1), True)]

    outs = []
    for o_move, p in OPP_POLICY(b1):
        b2 = apply_move(b1, o_move, "O")
        done = is_terminal(b2)
        r = terminal_reward(b2) if done else R_STEP
        outs.append((b2, p, r, done))
    return outs


# -------------------------
# Enumerate states where it's X to move (reachable)
# -------------------------
def reachable_x_states():
    """
    Generate all reachable boards where it is X to move (X count == O count),
    under the assumption both players can pick any legal move.
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

        # Keep only X-to-move boards
        if b.count("X") != b.count("O"):
            continue

        # Expand: X moves then O moves (any legal), to another X-to-move board
        for xm in legal_moves(b):
            b1 = apply_move(b, xm, "X")
            if is_terminal(b1):
                continue
            for om in legal_moves(b1):
                b2 = apply_move(b1, om, "O")
                if is_terminal(b2):
                    continue
                if b2.count("X") == b2.count("O"):
                    stack.append(b2)

    return seen


# -------------------------
# Policy Iteration
# -------------------------
def policy_iteration(eval_tol=1e-12, eval_max_iter=200000, improve_max_iter=1000):
    states = list(reachable_x_states())

    # Initialize V and a random policy pi(s)
    V = {s: (terminal_reward(s) if is_terminal(s) else 0.0) for s in states}
    pi = {}
    for s in states:
        if is_terminal(s) or s.count("X") != s.count("O"):
            pi[s] = None
        else:
            pi[s] = random.choice(legal_moves(s))

    # Policy evaluation (iterative)
    def policy_evaluation():
        for _ in range(eval_max_iter):
            delta = 0.0
            for s in states:
                if is_terminal(s) or s.count("X") != s.count("O"):
                    continue
                a = pi[s]
                if a is None:
                    continue

                v_new = 0.0
                for s2, p, r, done in transitions_after_x(s, a):
                    v_new += p * (r + (0.0 if done else GAMMA * V.get(s2, 0.0)))

                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new

            if delta < eval_tol:
                break

    # Policy improvement
    def policy_improvement():
        stable = True
        for s in states:
            if is_terminal(s) or s.count("X") != s.count("O"):
                continue
            moves = legal_moves(s)
            if not moves:
                continue

            old_a = pi[s]
            best_a = old_a
            best_q = -1e18

            for a in moves:
                q = 0.0
                for s2, p, r, done in transitions_after_x(s, a):
                    q += p * (r + (0.0 if done else GAMMA * V.get(s2, 0.0)))
                if q > best_q:
                    best_q = q
                    best_a = a

            pi[s] = best_a
            if best_a != old_a:
                stable = False
        return stable

    # Main loop
    for _ in range(improve_max_iter):
        policy_evaluation()
        if policy_improvement():
            break

    return V, pi


# -------------------------
# Simulation / evaluation
# -------------------------
def play_game(policy, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
    b = "........."

    while True:
        w = check_winner(b)
        if w is not None:
            return 1 if w == "X" else (-1 if w == "O" else 0)

        # X move
        if b.count("X") != b.count("O"):
            raise RuntimeError("Not X's turn unexpectedly.")
        xm = policy.get(b)
        if xm is None:
            xm = random.choice(legal_moves(b))  # fallback
        b = apply_move(b, xm, "X")
        if verbose:
            print("X plays", xm)
            print(render(b), "\n")

        w = check_winner(b)
        if w is not None:
            continue

        # O move
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
            print("O plays", om)
            print(render(b), "\n")


def evaluate(policy, n=20000, seed=0):
    random.seed(seed)
    wins = losses = draws = 0
    for _ in range(n):
        res = play_game(policy)
        if res == 1:
            wins += 1
        elif res == -1:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


def render(board: str) -> str:
    return "\n".join(" ".join(board[i:i+3]) for i in range(0, 9, 3))


if __name__ == "__main__":
    V, pi = policy_iteration()

    start = "........."
    print("Opponent policy:", OPP_POLICY.__name__)
    print("Best first move index (0..8):", pi.get(start))

    w, l, d = evaluate(pi, n=20000, seed=1)
    total = w + l + d
    print(f"Results vs opponent over {total} games:")
    print(f"  wins:   {w} ({w/total:.3f})")
    print(f"  losses: {l} ({l/total:.3f})")
    print(f"  draws:  {d} ({d/total:.3f})")

    # Uncomment to watch one game:
    # play_game(pi, seed=2, verbose=True)