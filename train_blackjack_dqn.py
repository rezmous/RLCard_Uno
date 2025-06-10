# train_blackjack_dqn.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.utils import set_seed, tournament
from agents.blackjack_dqn_agent import make_blackjack_dqn_agent


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# Set seeds for reproducibility
set_global_seed(42)

# Create Blackjack environments
env = rlcard.make('blackjack')
eval_env = rlcard.make('blackjack')

# Instantiate agents
agent = make_blackjack_dqn_agent(env)

env.set_agents([agent])
eval_env.set_agents([agent])

# Training parameters
num_episodes = 20000
save_every = 1000
history = []

# Prepare directory for model checkpoints
os.makedirs('models_blackjack', exist_ok=True)

for episode in range(1, num_episodes + 1):
    # Run one episode (env handles dealer automatically)
    trajectories, _ = env.run(is_training=True)
    # env.run calls agent.step() and agent.feed() internally

    # Periodic evaluation
    if episode % save_every == 0:
        # Evaluate win rate over 500 games
        win_rate = tournament(eval_env, 500)[0]
        print(f"Episode {episode}: Win rate = {win_rate:.2f}")
        history.append((episode, win_rate))
        # Save agent checkpoint
        agent.save_checkpoint(path='models_blackjack', filename=f'dqn_{episode}.pt')

# Plot learning curve
eps, rates = zip(*history)
plt.figure(figsize=(8, 5))
plt.plot(eps, rates, marker='o')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Blackjack DQNAgent vs Dealer')
plt.grid(True)
plt.tight_layout()
plt.show()
