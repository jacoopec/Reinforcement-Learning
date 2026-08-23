import numpy as np
import random

# ============================================================
# Ambiente: griglia 3x3
# ============================================================

GRID_SIZE = 3

START = (0, 0)
GOAL = (2, 2)

ACTIONS = ["up", "down", "left", "right"]

ACTION_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

ACTION_SYMBOL = {
    "up": "U",
    "down": "D",
    "left": "L",
    "right": "R"
}

# ============================================================
# Parametri Q-learning
# ============================================================

alpha = 0.5
gamma = 0.9
epsilon = 0.2

episodes = 5
max_steps = 30

# Stampa dettagliata solo per i primi episodi
DETAILED_EPISODES = 5

# Q-table: Q[row, col, action]
Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Conta le visite stato-azione
N = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)), dtype=int)


# ============================================================
# Funzioni base
# ============================================================

def action_index(action):
    return ACTIONS.index(action)


def is_terminal(state):
    return state == GOAL


def valid_actions(state):
    """
    Restituisce solo le azioni che non fanno uscire dalla griglia.
    """
    row, col = state
    valid = []

    for action in ACTIONS:
        dr, dc = ACTION_TO_DELTA[action]
        new_row = row + dr
        new_col = col + dc

        if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
            valid.append(action)

    return valid


def step(state, action):
    """
    Esegue un'azione.

    Restituisce:
    - next_state
    - reward
    - done
    """

    if is_terminal(state):
        return state, 0, True

    row, col = state
    dr, dc = ACTION_TO_DELTA[action]

    new_row = row + dr
    new_col = col + dc

    # Se l'azione uscisse dalla griglia, l'agente resta fermo
    if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
        next_state = state
    else:
        next_state = (new_row, new_col)

    if next_state == GOAL:
        reward = 10
        done = True
    else:
        reward = -1
        done = False

    return next_state, reward, done


# ============================================================
# Policy epsilon-greedy
# ============================================================

def choose_action(state):
    """
    Sceglie un'azione con epsilon-greedy.

    - con probabilità epsilon esplora
    - altrimenti sceglie la migliore azione secondo Q
    """

    possible_actions = valid_actions(state)

    if random.random() < epsilon:
        return random.choice(possible_actions)

    row, col = state

    q_values = []
    for action in possible_actions:
        idx = action_index(action)
        q_values.append(Q[row, col, idx])

    max_q = max(q_values)

    best_actions = []
    for action, q in zip(possible_actions, q_values):
        if q == max_q:
            best_actions.append(action)

    return random.choice(best_actions)


def max_q_next_state(state):
    """
    Restituisce max_a Q(s', a), considerando solo azioni valide.
    """

    row, col = state
    possible_actions = valid_actions(state)

    q_values = []
    for action in possible_actions:
        idx = action_index(action)
        q_values.append(Q[row, col, idx])

    return max(q_values)


# ============================================================
# Visualizzazione
# ============================================================

def print_grid(agent_state=None):
    """
    Stampa la griglia.

    A = agente
    G = goal
    . = cella vuota
    """

    for row in range(GRID_SIZE):
        line = ""

        for col in range(GRID_SIZE):
            state = (row, col)

            if state == agent_state:
                line += " A "
            elif state == GOAL:
                line += " G "
            else:
                line += " . "

        print(line)


def print_q_values_for_state(state):
    row, col = state

    print(f"Stato {state}")

    for action in ACTIONS:
        idx = action_index(action)
        value = Q[row, col, idx]
        visits = N[row, col, idx]
        print(f"  {action:5s} | Q = {value:7.3f} | visite = {visits}")
        
        
def print_q_values():
    line = ""
    for col in range(GRID_SIZE):
        for row in range(GRID_SIZE):
            line += "|"
            for a in  ACTIONS:
                line += f"{Q[row, col,  action_index(a)]:7.1f}|"
        line += "\n"
    line += "\n"
    print(line)


# ============================================================
# Training Q-learning
# ============================================================

for episode in range(episodes):

    state = START

    if episode < DETAILED_EPISODES:
        print()
        print("=" * 60)
        print(f"EPISODIO {episode + 1}")
        print("=" * 60)

    for step_number in range(max_steps):

        row, col = state

        # Nel Q-learning l'azione corrente viene scelta epsilon-greedy
        action = choose_action(state)
        a_idx = action_index(action)

        next_state, reward, done = step(state, action)

        old_q = Q[row, col, a_idx]

        # Incremento visite
        N[row, col, a_idx] += 1

        if done:
            target = reward
            best_next_q = 0
        else:
            best_next_q = max_q_next_state(next_state)
            target = reward + gamma * best_next_q

        td_error = target - old_q

        Q[row, col, a_idx] = old_q + alpha * td_error

        if episode < DETAILED_EPISODES:
            print()
            print(f"Step {step_number + 1}")
            print("-" * 40)

            print("Griglia:")
            print_grid(agent_state=state)

            print()
            print(f"s       = {state}")
            print(f"a       = {action}")
            print(f"r       = {reward}")
            print(f"s'      = {next_state}")

            print()
            print("Aggiornamento Q-learning:")
            print(f"Q(s,a) vecchio       = {old_q:.3f}")

            if done:
                print(f"target               = r = {target:.3f}")
            else:
                print(f"max_a Q(s',a)        = {best_next_q:.3f}")
                print(f"target               = r + gamma * max_a Q(s',a)")
                print(f"target               = {reward} + {gamma} * {best_next_q:.3f}")
                print(f"target               = {target:.3f}")

            print(f"TD error             = {td_error:.3f}")
            print(f"Q(s,a) nuovo         = {Q[row, col, a_idx]:.3f}")

        state = next_state

        if done:
            break


# ============================================================
# Q-table finale
# ============================================================

print()
print("=" * 60)
print("Q-TABLE FINALE")
print("=" * 60)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        print()
        print_q_values_for_state((row, col))


# ============================================================
# Policy finale
# ============================================================

print()
print("=" * 60)
print("POLICY FINALE")
print("=" * 60)
print("Legenda: U=up, D=down, L=left, R=right, G=goal, ?=non esplorato")
print()

for row in range(GRID_SIZE):
    line = ""

    for col in range(GRID_SIZE):
        state = (row, col)

        if state == GOAL:
            line += " G "
            continue

        possible_actions = valid_actions(state)

        tried_actions = []
        for action in possible_actions:
            idx = action_index(action)

            if N[row, col, idx] > 0:
                tried_actions.append(action)

        if len(tried_actions) == 0:
            line += " ? "
        else:
            q_values = []
            for action in tried_actions:
                idx = action_index(action)
                q_values.append(Q[row, col, idx])

            max_q = max(q_values)

            best_actions = []
            for action, q in zip(tried_actions, q_values):
                if q == max_q:
                    best_actions.append(action)

            best_action = random.choice(best_actions)
            line += f" {ACTION_SYMBOL[best_action]} "

    print(line)


# ============================================================
# Percorso greedy finale
# ============================================================

print()
print("=" * 60)
print("PERCORSO GREEDY DALLA PARTENZA")
print("=" * 60)

state = START
path = [state]

for _ in range(20):

    if state == GOAL:
        break

    row, col = state
    possible_actions = valid_actions(state)

    tried_actions = []
    for action in possible_actions:
        idx = action_index(action)

        if N[row, col, idx] > 0:
            tried_actions.append(action)

    if len(tried_actions) == 0:
        print("Percorso interrotto: stato non esplorato.")
        break

    q_values = []
    for action in tried_actions:
        idx = action_index(action)
        q_values.append(Q[row, col, idx])

    max_q = max(q_values)

    best_actions = []
    for action, q in zip(tried_actions, q_values):
        if q == max_q:
            best_actions.append(action)

    best_action = random.choice(best_actions)

    next_state, reward, done = step(state, best_action)

    path.append(next_state)
    state = next_state

    if done:
        break

print("Percorso:")
print(path)

print()
print("Griglia finale:")
print_grid()

print_q_values()