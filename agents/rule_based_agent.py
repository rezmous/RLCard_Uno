# agents/rule_based_uno_agent.py
import random


class RuleBasedUNOAgent:
    def __init__(self):
        self.use_raw = True

    def step(self, state):
        legal_actions = state['raw_legal_actions']
        if 'draw' in legal_actions:
            return 'draw'
        for action in legal_actions:
            if 'wild' in action:
                continue
            return action
        return random.choice(legal_actions)

    def eval_step(self, state):
        return self.step(state), []
