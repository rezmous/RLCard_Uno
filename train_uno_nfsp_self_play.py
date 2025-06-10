import os
import random
import numpy as np
import rlcard
from rlcard.utils import set_seed, tournament
from rlcard.agents.nfsp_agent import NFSPAgent
import matplotlib

import matplotlib.pyplot as plt

matplotlib.use('Agg')


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def main():
    set_global_seed(42)

    # adjust hyperparameters
    num_episodes = 50000
    eval_every = 5000
    eval_games = 200
    save_dir = 'models_uno_nfsp'
    os.makedirs(save_dir, exist_ok=True)

    # set up environments
    env = rlcard.make('uno')
    eval_env = rlcard.make('uno')

    # create two nfsp agents
    agent1 = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[128, 128],
        q_mlp_layers=[128, 128],
        anticipatory_param=0.1,
    )
    agent2 = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[128, 128],
        q_mlp_layers=[128, 128],
        anticipatory_param=0.1,
    )

    # put agents into environments
    env.set_agents([agent1, agent2])
    eval_env.set_agents([agent1, agent2])

    # train then evaluate
    history = []  # (episode, avg_payoffs_array)
    for episode in range(1, num_episodes + 1):
        # run one self-play episode (handles feed/training internally)
        env.run(is_training=True)

        if episode % eval_every == 0:
            avg_payoffs = tournament(eval_env, eval_games)
            print(f"Episode {episode}: Avg payoffs = {np.round(avg_payoffs, 3)}")
            history.append((episode, avg_payoffs))

            # save checkpoints
            agent1.save_checkpoint(save_dir, f'agent1_ep{episode}.pt')
            agent2.save_checkpoint(save_dir, f'agent2_ep{episode}.pt')

    # filter the best player and model
    last_ep, last_payoffs = history[-1]
    best = int(np.argmax(last_payoffs))
    print(f"Best agent at episode {last_ep}: Player {best} with payoff {last_payoffs[best]:.3f}")

    # learning curves
    episodes, payoffs = zip(*history)
    payoffs = np.array(payoffs)
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, payoffs[:, 0], marker='o', label='Agent 1')
    plt.plot(episodes, payoffs[:, 1], marker='x', label='Agent 2')
    plt.xlabel('Episode')
    plt.ylabel('Average Payoff')
    plt.title('UNO NFSP Self-Play Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, 'learning_curve.png')
    plt.savefig(out_path)
    print(f"Learning curve saved to {out_path}")


main()
