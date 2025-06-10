# evaluate.py

import torch
import rlcard
from agents.masked_dqn_agent import MaskedDQNAgent
from agents.rule_based_agent import RuleBasedUNOAgent
from rlcard.utils import tournament

# Load environment with raw_action enabled
env = rlcard.make('uno', config={
    'env_class': 'uno_custom_train.UnoEnv',
    'raw_action': True
})

# Initialize agents
trained_agent = MaskedDQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0])
rule_based_agent = RuleBasedUNOAgent()

# Load trained model
model_path = './experiments/uno_dqn/model_episode_90000.pt'
trained_agent.q_estimator.qnet.load_state_dict(torch.load(model_path))

# Set agents for evaluation
env.set_agents([trained_agent, rule_based_agent])

# Run evaluation
win_rate = tournament(env, 5000)[0]
print(f'Trained DQN agent win rate vs RuleBasedUNOAgent: {win_rate:.2f}')