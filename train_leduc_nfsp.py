import os
import random
import numpy as np
import matplotlib.pyplot as plt

import rlcard
from rlcard.utils import set_seed, tournament
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents import RandomAgent


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


set_global_seed(42)

train_env = rlcard.make('leduc-holdem')
eval_env_self = rlcard.make('leduc-holdem')
eval_env_rand = rlcard.make('leduc-holdem')

agent1 = NFSPAgent(
    num_actions=train_env.num_actions,
    state_shape=train_env.state_shape[0],
    hidden_layers_sizes=[64, 64],
    q_mlp_layers=[64, 64],
    anticipatory_param=0.1
)
agent2 = NFSPAgent(
    num_actions=train_env.num_actions,
    state_shape=train_env.state_shape[0],
    hidden_layers_sizes=[64, 64],
    q_mlp_layers=[64, 64],
    anticipatory_param=0.1
)

random_agent = RandomAgent(num_actions=train_env.num_actions)

train_env.set_agents([agent1, agent2])
eval_env_self.set_agents([agent1, agent2])
eval_env_rand.set_agents([agent1, random_agent])

num_episodes = 50000
eval_every = 1000
self_rewards = []
rand_winrates = []
ckpt_dir = 'models_leduc_nfsp'
os.makedirs(ckpt_dir, exist_ok=True)

for ep in range(1, num_episodes + 1):
    trajectories, _ = train_env.run(is_training=True)

    if ep % eval_every == 0:
        self_rewards_ep = tournament(eval_env_self, 1000)
        avg_self = np.mean(self_rewards_ep)
        self_rewards.append((ep, avg_self))

        rand_wr = tournament(eval_env_rand, 1000)[0]
        rand_winrates.append((ep, rand_wr))

        print(f"Episode {ep}: Self avg payoff = {avg_self:.3f}, Win rate vs random = {rand_wr:.3f}")

        agent1.save_checkpoint(path=ckpt_dir, filename=f'nfsp1_{ep}.pt')
        agent2.save_checkpoint(path=ckpt_dir, filename=f'nfsp2_{ep}.pt')

if self_rewards:
    eps, payoffs = zip(*self_rewards)
    _, wrs = zip(*rand_winrates)
    plt.figure(figsize=(10, 5))
    plt.plot(eps, payoffs, label='Self-play avg payoff', marker='o')
    plt.plot(eps, wrs, label='Win rate vs random', marker='x')
    plt.xlabel('Episode')
    plt.legend()
    plt.title("Leduc Hold'em NFSP: Self-play vs Random Evaluation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
