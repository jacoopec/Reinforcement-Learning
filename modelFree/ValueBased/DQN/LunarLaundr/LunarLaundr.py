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

NUM_EPISODES = 1000

GAMMA = 0.99

LEARNING_RATE = 5e-4

BATCH_SIZE = 64

REPLAY_BUFFER_SIZE = 100_000

MIN_REPLAY_SIZE = 1_000

TARGET_UPDATE_FREQUENCY = 1000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 50_000


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
# Epsilon schedule
# ============================================================

def get_epsilon(step):

    epsilon = (
        EPSILON_END
        + (EPSILON_START - EPSILON_END)
        * math.exp(-step / EPSILON_DECAY)
    )

    return epsilon


# ============================================================
# Epsilon-greedy action selection
# ============================================================

def select_action(
    state,
    policy_net,
    env,
    step
):

    epsilon = get_epsilon(step)

    # --------------------------------------------------------
    # Exploration
    # --------------------------------------------------------

    if random.random() < epsilon:

        return env.action_space.sample()

    # --------------------------------------------------------
    # Exploitation
    # --------------------------------------------------------

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = policy_net(
            state_tensor
        )

        action = q_values.argmax(
            dim=1
        ).item()

    return action


# ============================================================
# Training step
# ============================================================

def train_step(
    policy_net,
    target_net,
    replay_buffer,
    optimizer
):

    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    # --------------------------------------------------------
    # Sample mini-batch
    # --------------------------------------------------------

    transitions = replay_buffer.sample(
        BATCH_SIZE
    )

    batch = Transition(
        *zip(*transitions)
    )

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


    # ========================================================
    # Current Q value
    #
    # Q(s,a)
    # ========================================================

    all_q_values = policy_net(
        states
    )

    current_q = all_q_values.gather(
        1,
        actions
    ).squeeze(1)


    # ========================================================
    # Target Q value
    #
    # y = r + gamma max Q_target(s',a')
    # ========================================================

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


    # ========================================================
    # Loss
    # ========================================================

    loss = nn.functional.smooth_l1_loss(
        current_q,
        target_q
    )


    # ========================================================
    # Gradient descent
    # ========================================================

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        10
    )

    optimizer.step()

    return loss.item()


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    policy_net,
    episodes=5,
    render=False
):

    if render:
        env = gym.make(
            "LunarLander-v3",
            render_mode="human"
        )
    else:
        env = gym.make(
            "LunarLander-v3"
        )

    policy_net.eval()

    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()

        total_reward = 0

        while True:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # Greedy policy
            with torch.no_grad():

                q_values = policy_net(
                    state_tensor
                )

                action = q_values.argmax(
                    dim=1
                ).item()

            (
                next_state,
                reward,
                terminated,
                truncated,
                _
            ) = env.step(action)

            state = next_state

            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

        print(
            f"Evaluation episode {episode + 1}: "
            f"{total_reward:.1f}"
        )

    env.close()

    print(
        "Average evaluation reward:",
        np.mean(rewards)
    )

    policy_net.train()


# ============================================================
# Main
# ============================================================

def main():

    env = gym.make(
        "LunarLander-v3"
    )

    # ========================================================
    # Environment
    # ========================================================

    state_size = env.observation_space.shape[0]

    action_size = env.action_space.n

    print("State size:", state_size)
    print("Action size:", action_size)


    # ========================================================
    # Networks
    # ========================================================

    policy_net = DQN(
        state_size,
        action_size
    ).to(device)

    target_net = DQN(
        state_size,
        action_size
    ).to(device)


    # Initially both networks are identical

    target_net.load_state_dict(
        policy_net.state_dict()
    )

    target_net.eval()


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.Adam(
        policy_net.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # Replay Buffer
    # ========================================================

    replay_buffer = ReplayBuffer(
        REPLAY_BUFFER_SIZE
    )


    # ========================================================
    # Training
    # ========================================================

    global_step = 0

    recent_rewards = deque(
        maxlen=100
    )

    for episode in range(NUM_EPISODES):

        state, _ = env.reset()

        episode_reward = 0

        episode_losses = []

        while True:

            # ------------------------------------------------
            # 1. Choose action
            # ------------------------------------------------

            action = select_action(
                state,
                policy_net,
                env,
                global_step
            )


            # ------------------------------------------------
            # 2. Execute action
            # ------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                _
            ) = env.step(action)


            # ------------------------------------------------
            # 3. Store transition
            # ------------------------------------------------

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                terminated
            )


            # ------------------------------------------------
            # 4. Train DQN
            # ------------------------------------------------

            loss = train_step(
                policy_net,
                target_net,
                replay_buffer,
                optimizer
            )

            if loss is not None:
                episode_losses.append(loss)


            # ------------------------------------------------
            # 5. Move to next state
            # ------------------------------------------------

            state = next_state

            episode_reward += reward

            global_step += 1


            # ------------------------------------------------
            # 6. Update target network
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


        # ====================================================
        # Statistics
        # ====================================================

        recent_rewards.append(
            episode_reward
        )

        average_reward = np.mean(
            recent_rewards
        )

        epsilon = get_epsilon(
            global_step
        )

        if episode_losses:
            average_loss = np.mean(
                episode_losses
            )
        else:
            average_loss = 0


        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg(100): {average_reward:8.2f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {epsilon:.3f}"
        )


    # ========================================================
    # Save model
    # ========================================================

    torch.save(
        policy_net.state_dict(),
        "lunarlander_dqn.pth"
    )

    print(
        "\nModel saved as lunarlander_dqn.pth"
    )

    env.close()


    # ========================================================
    # Evaluate trained model
    # ========================================================

    evaluate(
        policy_net,
        episodes=5,
        render=False
    )


if __name__ == "__main__":
    main()