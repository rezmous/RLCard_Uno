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


set_global_seed(42)

env = rlcard.make('blackjack')
eval_env = rlcard.make('blackjack')

agent = make_blackjack_dqn_agent(env)

env.set_agents([agent])
eval_env.set_agents([agent])

num_episodes = 20000
save_every = 1000
history = []

os.makedirs('models_blackjack', exist_ok=True)

for episode in range(1, num_episodes + 1):
    trajectories, _ = env.run(is_training=True)

    if episode % save_every == 0:
        win_rate = tournament(eval_env, 500)[0]
        print(f"Episode {episode}: Win rate = {win_rate:.2f}")
        history.append((episode, win_rate))
        agent.save_checkpoint(path='models_blackjack', filename=f'dqn_{episode}.pt')

eps, rates = zip(*history)
plt.figure(figsize=(8, 5))
plt.plot(eps, rates, marker='o')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Blackjack DQNAgent vs Dealer')
plt.grid(True)
plt.tight_layout()
plt.show()
