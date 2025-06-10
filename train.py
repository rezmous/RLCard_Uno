import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.utils import tournament
from rlcard.agents import RandomAgent
from agents.masked_dqn_agent import MaskedDQNAgent
from agents.rule_based_agent import RuleBasedUNOAgent


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


set_global_seed(42)

# Environments
env = rlcard.make('uno', config={'env_class': 'uno_custom_train.UnoEnv'})
eval_env = rlcard.make('uno', config={'env_class': 'uno_custom_train.UnoEnv'})

# Agents
agent = MaskedDQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0])
random_agent = RandomAgent(num_actions=env.num_actions)
rule_agent = RuleBasedUNOAgent()

# Opponent-mixing setup
mixed_prob = 0.5

history = []
switch_ep = None
self_play_ep = None

# Training loop
for ep in range(1, 60001):
    # Mix opponents
    if random.random() < mixed_prob:
        opp = random_agent
    else:
        opp = rule_agent
    env.set_agents([agent, opp])
    eval_env.set_agents([agent, opp])

    # Run one episode
    trajectories, _ = env.run(is_training=True)
    for ts in trajectories[0]: agent.feed(ts)

    # Periodic evaluation
    if ep % 1000 == 0:
        win_rate = tournament(eval_env, 500)[0]
        history.append((ep, win_rate))
        print(f"Episode {ep}: Win rate = {win_rate:.2f} (mixed opponents)")

        # Save model
        os.makedirs('./models', exist_ok=True)
        torch.save(agent.q_estimator.qnet.state_dict(), f'./models/model_{ep}.pt')

# Plot performance
eps, rates = zip(*history)
plt.figure(figsize=(10, 5))
plt.plot(eps, rates, marker='o')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('UNO DQN with PER, n-Step, and Mixed Opponents')
plt.grid()
plt.tight_layout()
plt.show()
