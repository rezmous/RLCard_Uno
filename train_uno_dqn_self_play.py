# train_uno_dqn_self_play.py

import os
import random
import numpy as np
import matplotlib
import rlcard
from rlcard.utils import set_seed, tournament
from rlcard.agents.dqn_agent import DQNAgent

import matplotlib.pyplot as plt

matplotlib.use('Agg')


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def main():
    set_global_seed(42)

    # configuration for 2 players
    num_players = 2
    num_episodes = 50000
    eval_every = 5000
    eval_games = 200
    mlp_layers = [128, 128]
    save_dir = 'models_uno_selfplay'
    os.makedirs(save_dir, exist_ok=True)

    # make the environment for uno, one train & one eval
    env = rlcard.make('uno')
    eval_env = rlcard.make('uno')

    # initialize your dqn agent
    agents = []
    for pid in range(num_players):
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=mlp_layers
        )
        # adjust the hyper parameters as needed
        agent.memory_init_size = 1000
        agent.batch_size = 64
        agent.train_every = 1
        agent.learning_rate = 1e-4
        agent.gamma = 0.99
        agent.target_update_freq = 100
        agent.epsilon_start = 1.0
        agent.epsilon_end = 0.00001
        agent.epsilon_decay_steps = 20000
        agents.append(agent)

    # put agents into environment
    env.set_agents(agents)
    eval_env.set_agents(agents)

    # track history for plots
    history = []  # list of tuples (episode, avg_payoffs_array)

    # training loop
    for episode in range(1, num_episodes + 1):
        # run one self play episode (handles feed/train internally)
        env.run(is_training=True)

        # evaluate periodically
        if episode % eval_every == 0:
            avg_payoffs = tournament(eval_env, eval_games)
            print(f"Episode {episode}: Avg payoffs = {np.round(avg_payoffs, 3)}")
            history.append((episode, avg_payoffs))
            # add a checkpoint to save
            for idx, agent in enumerate(agents):
                agent.save_checkpoint(save_dir, f'agent{idx}_ep{episode}.pt')

    # once training is done, filter to best player and model
    if history:
        last_ep, last_payoffs = history[-1]
        best_idx = int(np.argmax(last_payoffs))
        print(f"Best agent at episode {last_ep}: Player {best_idx} with payoff {last_payoffs[best_idx]:.3f}")

        # learning curves
        episodes, payoffs = zip(*history)
        payoffs = np.array(payoffs)
        plt.figure(figsize=(10, 6))
        for pid in range(num_players):
            plt.plot(episodes, payoffs[:, pid], label=f'Player {pid}')
        plt.xlabel('Episode')
        plt.ylabel('Average Payoff')
        plt.title('UNO Self-Play DQN Performance (2 Players)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(save_dir, 'learning_curve.png')
        plt.savefig(plot_path)
        print(f"Learning curve saved to {plot_path}")
