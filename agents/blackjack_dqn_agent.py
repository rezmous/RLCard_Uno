from rlcard.agents.dqn_agent import DQNAgent


def make_blackjack_dqn_agent(env):
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
    )
    agent.memory_init_size = 500
    agent.batch_size = 64
    agent.train_every = 1
    agent.learning_rate = 1e-4
    agent.gamma = 0.99
    agent.target_update_freq = 100
    agent.epsilon_start = 1.0
    agent.epsilon_end = 0.1
    agent.epsilon_decay_steps = 20000
    return agent
