# train_gin_rummy_dqn_builtin.py

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import rlcard
from rlcard.utils import tournament
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents import RandomAgent


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(42)

# 1) Create environments
env = rlcard.make('gin-rummy')
eval_env = rlcard.make('gin-rummy')

# 2) Instantiate the built-in DQNAgent
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[128, 128],  # two hidden layers of size 128
)

# 3) Tweak hyper-parameters on the agent
agent.memory_init_size = 1000  # start learning after these many transitions
agent.batch_size = 64  # samples per gradient step
agent.train_every = 1  # call `.train()` every step
agent.learning_rate = 1e-4  # optimizer LR
agent.gamma = 0.99  # discount factor
agent.target_update_freq = 100  # sync target every 100 updates
agent.epsilon_start = 1.0
agent.epsilon_end = 0.05
agent.epsilon_decay_steps = 50000

# 4) Set up a random opponent
random_agent = RandomAgent(num_actions=env.num_actions)

# 5) Assign agents to both train & eval envs
env.set_agents([agent, random_agent])
eval_env.set_agents([agent, random_agent])

# Prepare logging & checkpoint directory
num_episodes = 30000
save_every = 1000
history = []
ckpt_dir = 'models_gin'
os.makedirs(ckpt_dir, exist_ok=True)

# 6) Training loop
for ep in range(1, num_episodes + 1):
    # `env.run` calls agent.step() and agent.feed() internally
    trajectories, _ = env.run(is_training=True)

    if ep % save_every == 0:
        # Evaluate performance
        win_rate = tournament(eval_env, 500)[0]
        print(f"Episode {ep}: Win rate vs Random = {win_rate:.2f}")
        history.append((ep, win_rate))

        # Save a full agent checkpoint
        # This saves networks, optimizer state, and all hyper-params
        agent.save_checkpoint(path=ckpt_dir, filename=f'dqn_{ep}.pt')

# 7) Plot the learning curve
episodes, rates = zip(*history)
plt.figure(figsize=(8, 5))
plt.plot(episodes, rates, marker='o')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.title('Gin-Rummy with Built-in DQNAgent')
plt.grid(True)
plt.tight_layout()
plt.show()
