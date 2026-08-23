import numpy as np
import random

# Simple Multi-Armed Bandit Problem with Epsilon-Greedy Exploration

class Bandit:
    def __init__(self, num_arms=3):
        # True reward probabilities for each arm (unknown to the agent)
        self.true_rewards = np.random.rand(num_arms)  # Random values between 0 and 1
        self.num_arms = num_arms

    def pull(self, arm):
        # Simulate pulling an arm: reward 1 with probability true_reward, else 0
        return 1 if random.random() < self.true_rewards[arm] else 0

class EpsilonGreedyAgent:
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        # Initialize estimates and counts
        self.q_values = np.zeros(num_arms)  # Estimated value for each arm
        self.arm_counts = np.zeros(num_arms)  # Number of times each arm pulled

    def select_action(self):
        # Epsilon-greedy: explore with prob epsilon, exploit otherwise
        if random.random() < self.epsilon:
            return random.randint(0, self.num_arms - 1)  # Random arm
        else:
            return np.argmax(self.q_values)  # Greedy: best estimated arm

    def update(self, arm, reward):
        # Update Q-value using incremental average
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

# Simulation
num_steps = 1000
bandit = Bandit(num_arms=3)
agent = EpsilonGreedyAgent(num_arms=3, epsilon=0.1)

total_reward = 0
for step in range(num_steps):
    arm = agent.select_action()
    reward = bandit.pull(arm)
    agent.update(arm, reward)
    total_reward += reward

print(f"True rewards: {bandit.true_rewards}")
print(f"Estimated Q-values: {agent.q_values}")
print(f"Total reward: {total_reward}")
print(f"Average reward: {total_reward / num_steps}")