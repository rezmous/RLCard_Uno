import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('./experiments/uno_dqn/performance.csv')

# Sanity check: clip reward to [-1, 1]
df['eval_reward'] = df['eval_reward'].clip(lower=-1.0, upper=1.0)

# Compute moving average
window_size = 10
df['moving_avg'] = df['eval_reward'].rolling(window=window_size).mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['episode'], df['eval_reward'], label='Average Game Reward', alpha=0.5)
plt.plot(df['episode'], df['moving_avg'], label=f'{window_size}-Episode Moving Average', linewidth=2)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Fix axis
plt.ylim(-1.1, 1.1)
plt.xlabel('Episode')
plt.ylabel('Average Game Reward')
plt.title('UNO DQN Agent Performance vs Random Agent')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to file
plt.savefig('winrate_plot_fixed.png')
print("✅ Fixed plot saved as winrate_plot_fixed.png")