import random
import numpy as np
import matplotlib.pyplot as plt
import rlcard
from rlcard.utils import set_seed


def train_tabular_q(num_episodes=50000, eval_interval=1000, eval_games=1000):
    seed = 42
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = rlcard.make('blackjack')

    Q = np.zeros((32, 11, 2, 2), dtype=np.float32)

    alpha = 0.1
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = (eps_start - eps_end) / num_episodes
    epsilon = eps_start

    history = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.init_game()
        done = False

        while not done:
            ps, dc, ua = state['obs']
            ps, dc, ua = int(ps), int(dc), int(ua)

            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                action = int(np.argmax(Q[ps, dc, ua]))

            next_state, _ = env.step(action)
            done = env.is_over()

            if done:
                reward = env.get_payoffs()[0]
                next_max = 0.0
            else:
                nps, ndc, nua = next_state['obs']
                next_max = np.max(Q[int(nps), int(ndc), int(nua)])
                reward = 0.0

            Q[ps, dc, ua, action] += alpha * (
                reward + gamma * next_max - Q[ps, dc, ua, action]
            )

            state = next_state

        epsilon = max(eps_end, epsilon - eps_decay)

        if ep % eval_interval == 0:
            wins = 0
            for _ in range(eval_games):
                s, _ = env.init_game()
                d = False
                while not d:
                    ps, dc, ua = s['obs']
                    ps, dc, ua = int(ps), int(dc), int(ua)
                    a = int(np.argmax(Q[ps, dc, ua]))
                    s, _ = env.step(a)
                    d = env.is_over()
                if env.get_payoffs()[0] == 1:
                    wins += 1
            win_rate = wins / eval_games
            print(f"Episode {ep}: Win rate = {win_rate:.2f}")
            history.append((ep, win_rate))

    if history:
        eps, rates = zip(*history)
        plt.figure(figsize=(8, 5))
        plt.plot(eps, rates, marker='o')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.title('Tabular Q-Learning on Blackjack')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    train_tabular_q()