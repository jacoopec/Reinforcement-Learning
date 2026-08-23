import random
import math
from collections import deque, namedtuple

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Hyperparameters
# ============================================================

NUM_EPISODES = 500

GAMMA = 0.99

LEARNING_RATE = 1e-3

BATCH_SIZE = 64

REPLAY_BUFFER_SIZE = 10_000

MIN_REPLAY_SIZE = 1_000

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Replay Buffer
# ============================================================

Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "terminated"
    ]
)


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):
        self.buffer.append(
            Transition(
                state,
                action,
                reward,
                next_state,
                terminated
            )
        )

    def sample(self, batch_size):
        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Deep Q-Network
# ============================================================

class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Epsilon
# ============================================================

def get_epsilon(step):

    return (
        EPSILON_END
        + (EPSILON_START - EPSILON_END)
        * math.exp(-step / EPSILON_DECAY)
    )


# ============================================================
# Select action: epsilon-greedy
# ============================================================

def select_action(
    state,
    policy_net,
    env,
    step
):

    epsilon = get_epsilon(step)

    # Exploration
    if random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = policy_net(state_tensor)

        action = q_values.argmax(dim=1).item()

    return action


# ============================================================
# Train DQN
# ============================================================

def train_step(
    policy_net,
    target_net,
    replay_buffer,
    optimizer
):

    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    transitions = replay_buffer.sample(
        BATCH_SIZE
    )

    batch = Transition(*zip(*transitions))

    states = torch.tensor(
        np.array(batch.state),
        dtype=torch.float32,
        device=device
    )

    actions = torch.tensor(
        batch.action,
        dtype=torch.int64,
        device=device
    ).unsqueeze(1)

    rewards = torch.tensor(
        batch.reward,
        dtype=torch.float32,
        device=device
    )

    next_states = torch.tensor(
        np.array(batch.next_state),
        dtype=torch.float32,
        device=device
    )

    terminated = torch.tensor(
        batch.terminated,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Current Q-value
    #
    # Q(s, a)
    # --------------------------------------------------------

    q_values = policy_net(states)

    current_q = q_values.gather(
        1,
        actions
    ).squeeze(1)

    # --------------------------------------------------------
    # Target
    #
    # y = r + gamma * max_a Q_target(s', a)
    # --------------------------------------------------------

    with torch.no_grad():

        next_q_values = target_net(
            next_states
        )

        max_next_q = next_q_values.max(
            dim=1
        ).values

        target_q = (
            rewards
            + GAMMA
            * max_next_q
            * (1 - terminated)
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = nn.functional.smooth_l1_loss(
        current_q,
        target_q
    )

    optimizer.zero_grad()

    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        10
    )

    optimizer.step()

    return loss.item()


# ============================================================
# Main
# ============================================================

def main():

    env = gym.make(
        "CartPole-v1"
    )

    # State:
    #
    # [
    #   cart position,
    #   cart velocity,
    #   pole angle,
    #   pole angular velocity
    # ]

    state_size = env.observation_space.shape[0]

    # Actions:
    #
    # 0 -> push left
    # 1 -> push right

    action_size = env.action_space.n

    print("State size:", state_size)
    print("Action size:", action_size)

    # --------------------------------------------------------
    # Networks
    # --------------------------------------------------------

    policy_net = DQN(
        state_size,
        action_size
    ).to(device)

    target_net = DQN(
        state_size,
        action_size
    ).to(device)

    # Initially identical
    target_net.load_state_dict(
        policy_net.state_dict()
    )

    target_net.eval()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        policy_net.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Replay Buffer
    # --------------------------------------------------------

    replay_buffer = ReplayBuffer(
        REPLAY_BUFFER_SIZE
    )

    global_step = 0

    # ========================================================
    # Training
    # ========================================================

    for episode in range(NUM_EPISODES):

        state, _ = env.reset()

        episode_reward = 0

        while True:

            # ------------------------------------------------
            # Choose action
            # ------------------------------------------------

            action = select_action(
                state,
                policy_net,
                env,
                global_step
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                _
            ) = env.step(action)

            # ------------------------------------------------
            # Store experience
            # ------------------------------------------------

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            # ------------------------------------------------
            # Train network
            # ------------------------------------------------

            loss = train_step(
                policy_net,
                target_net,
                replay_buffer,
                optimizer
            )

            state = next_state

            episode_reward += reward

            global_step += 1

            # ------------------------------------------------
            # Update target network
            # ------------------------------------------------

            if (
                global_step
                % TARGET_UPDATE_FREQUENCY
                == 0
            ):

                target_net.load_state_dict(
                    policy_net.state_dict()
                )

            # ------------------------------------------------
            # End episode
            # ------------------------------------------------

            if terminated or truncated:
                break

        epsilon = get_epsilon(
            global_step
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    # ========================================================
    # Save model
    # ========================================================

    torch.save(
        policy_net.state_dict(),
        "cartpole_dqn.pth"
    )

    env.close()

    print("\nTraining finished.")
    print("Model saved as cartpole_dqn.pth")


if __name__ == "__main__":
    main()