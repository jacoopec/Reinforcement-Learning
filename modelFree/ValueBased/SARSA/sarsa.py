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

ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→"
}

# ============================================================
# Parametri SARSA
# ============================================================

alpha     = 0.5       # learning rate
gamma     = 0.9       # discount factor
epsilon   = 0.2     # probabilità di esplorare
episodes  = 5
max_steps = 30

# Q-table: Q[row, col, action]
Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Conta quante volte una coppia stato-azione viene aggiornata
N = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)), dtype=int)


# ============================================================
# Funzioni dell'ambiente
# ============================================================

def is_terminal(state):
    return state == GOAL


def action_index(action):
    return ACTIONS.index(action)


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
    Esegue un'azione nello stato corrente.

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

    # Se l'azione uscisse dalla griglia, resta fermo
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


def choose_action(state):
    """
    Policy epsilon-greedy.

    Con probabilità epsilon sceglie un'azione casuale valida.
    Altrimenti sceglie l'azione valida con valore Q massimo.
    """

    possible_actions = valid_actions(state)

    # Esplorazione
    if random.random() < epsilon:
        return random.choice(possible_actions)

    # Sfruttamento
    row, col = state

    q_values = []
    for action in possible_actions:
        idx = action_index(action)
        q_values.append(Q[row, col, idx])

    max_q = max(q_values)

    # Gestione dei pareggi: sceglie casualmente tra le azioni migliori
    best_actions = [
        action for action, q in zip(possible_actions, q_values)
        if q == max_q
    ]

    return random.choice(best_actions)


# ============================================================
# Stampa della griglia
# ============================================================

def print_grid(agent_state=None):
    """
    Visualizza la griglia.
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
    """
    Stampa i valori Q di uno stato.
    """

    row, col = state

    print(f"Valori Q nello stato {state}:")
    for action in ACTIONS:
        idx = action_index(action)
        print(f"  Q({state}, {action}) = {Q[row, col, idx]:.3f}")


# ============================================================
# Training SARSA
# ============================================================

for episode in range(episodes):
    state = START
    action = choose_action(state)

    print("\n" + "=" * 60)
    print(f"EPISODIO {episode + 1}")
    print("=" * 60)

    for step_number in range(max_steps):
        row, col = state
        a_idx = action_index(action)

        next_state, reward, done = step(state, action)

        old_q = Q[row, col, a_idx]

        # Conta che questa coppia stato-azione è stata visitata
        N[row, col, a_idx] += 1

        print(f"\nStep {step_number + 1}")
        print("-" * 40)
        print("Griglia prima dell'azione:")
        print_grid(agent_state=state)

        print(f"\nstato s        = {state}")
        print(f"azione a       = {action}")
        print(f"reward r       = {reward}")
        print(f"nuovo stato s' = {next_state}")

        if done:
            # Se l'episodio termina, non c'è azione successiva
            target = reward
            td_error = target - old_q

            Q[row, col, a_idx] = old_q + alpha * td_error

            print("\nepisodio terminato")

            print("\nAggiornamento SARSA:")
            print(f"Q(s,a) vecchio = {old_q:.3f}")
            print(f"target         = r = {target:.3f}")
            print(f"TD error       = target - Q(s,a) = {td_error:.3f}")
            print(f"Q(s,a) nuovo   = {Q[row, col, a_idx]:.3f}")

            break

        else:
            # SARSA sceglie davvero la prossima azione a'
            next_action = choose_action(next_state)

            next_row, next_col = next_state
            next_a_idx = action_index(next_action)

            target = reward + gamma * Q[next_row, next_col, next_a_idx]
            td_error = target - old_q

            Q[row, col, a_idx] = old_q + alpha * td_error

            print(f"nuova azione a'= {next_action}")

            print("\nAggiornamento SARSA:")
            print(f"Q(s,a) vecchio = {old_q:.3f}")
            print(f"Q(s',a')       = {Q[next_row, next_col, next_a_idx]:.3f}")
            print(f"target         = r + gamma * Q(s',a') = {target:.3f}")
            print(f"TD error       = target - Q(s,a) = {td_error:.3f}")
            print(f"Q(s,a) nuovo   = {Q[row, col, a_idx]:.3f}")

            state = next_state
            action = next_action


# ============================================================
# Q-table finale
# ============================================================

print("\n\n" + "=" * 60)
print("Q-TABLE FINALE")
print("=" * 60)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        state = (row, col)
        print()
        print_q_values_for_state(state)


# ============================================================
# Numero di visite
# ============================================================

print("\n\n" + "=" * 60)
print("VISITE STATO-AZIONE")
print("=" * 60)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        state = (row, col)
        print(f"\nStato {state}")
        for action in ACTIONS:
            idx = action_index(action)
            print(f"  N({state}, {action}) = {N[row, col, idx]}")


# ============================================================
# Policy finale corretta
# ============================================================

print("\n\n" + "=" * 60)
print("POLICY FINALE")
print("=" * 60)

for row in range(GRID_SIZE):
    line = ""

    for col in range(GRID_SIZE):
        state = (row, col)

        if state == GOAL:
            line += " G "
            continue

        possible_actions = valid_actions(state)

        # Considera solo azioni valide e già provate almeno una volta
        tried_actions = [
            action for action in possible_actions
            if N[row, col, action_index(action)] > 0
        ]

        if len(tried_actions) == 0:
            line += " ? "
        else:
            q_values = []
            for action in tried_actions:
                idx = action_index(action)
                q_values.append(Q[row, col, idx])

            max_q = max(q_values)

            best_actions = [
                action for action, q in zip(tried_actions, q_values)
                if q == max_q
            ]

            best_action = random.choice(best_actions)
            line += f" {ARROWS[best_action]} "

    print(line)


# ============================================================
# Percorso greedy finale dalla partenza
# ============================================================

print("\n\n" + "=" * 60)
print("PERCORSO GREEDY DALLA PARTENZA")
print("=" * 60)

state = START
path = [state]

for _ in range(20):
    if state == GOAL:
        break

    row, col = state
    possible_actions = valid_actions(state)

    tried_actions = [
        action for action in possible_actions
        if N[row, col, action_index(action)] > 0
    ]

    if len(tried_actions) == 0:
        print("Percorso interrotto: stato non esplorato.")
        break

    best_action = max(
        tried_actions,
        key=lambda action: Q[row, col, action_index(action)]
    )

    next_state, reward, done = step(state, best_action)

    path.append(next_state)
    state = next_state

    if done:
        break

print("Percorso:")
print(path)

print("\nGriglia finale con goal:")
print_grid()