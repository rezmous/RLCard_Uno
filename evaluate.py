import torch
import rlcard
from agents.masked_dqn_agent import MaskedDQNAgent
from agents.rule_based_agent import RuleBasedUNOAgent
from rlcard.utils import tournament

env = rlcard.make('uno', config={
    'env_class': 'uno_custom_train.UnoEnv',
    'raw_action': True
})

trained_agent = MaskedDQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0])
rule_based_agent = RuleBasedUNOAgent()

model_path = './experiments/uno_dqn/model_episode_90000.pt'
trained_agent.q_estimator.qnet.load_state_dict(torch.load(model_path))

env.set_agents([trained_agent, rule_based_agent])

win_rate = tournament(env, 5000)[0]
print(f'Trained DQN agent win rate vs RuleBasedUNOAgent: {win_rate:.2f}')