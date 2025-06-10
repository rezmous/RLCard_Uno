
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.utils import tournament
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents import RandomAgent


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(42)

env = rlcard.make('gin-rummy')
eval_env = rlcard.make('gin-rummy')

agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[128, 128],
)

agent.memory_init_size = 1000
agent.batch_size = 64
agent.train_every = 1
agent.learning_rate = 1e-4
agent.gamma = 0.99
agent.target_update_freq = 100
agent.epsilon_start = 1.0
agent.epsilon_end = 0.05
agent.epsilon_decay_steps = 50000

random_agent = RandomAgent(num_actions=env.num_actions)

env.set_agents([agent, random_agent])
eval_env.set_agents([agent, random_agent])

num_episodes = 30000
save_every = 1000
history = []

for ep in range(1, num_episodes + 1):
    trajectories, _ = env.run(is_training=True)

    if ep % save_every == 0:
        win_rate = tournament(eval_env, 500)[0]
        print(f"Episode {ep}: Win rate vs Random = {win_rate:.2f}")
        history.append((ep, win_rate))

        os.makedirs('models_gin', exist_ok=True)
        torch.save(agent.qnet.state_dict(), f'models_gin/dqn_{ep}.pt')

eps, rates = zip(*history)
plt.figure(figsize=(8, 5))
plt.plot(eps, rates, marker='o')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Gin Rummy with Built-in DQNAgent')
plt.grid(True)
plt.tight_layout()
plt.show()
