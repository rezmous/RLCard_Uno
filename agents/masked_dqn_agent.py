import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from rlcard.agents.dqn_agent import DQNAgent


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = idx
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]


# PER Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6

    def push(self, transition, error=None):
        p = (abs(error) + self.epsilon) ** self.alpha if error is not None else self.tree.tree.max() or 1.0
        self.tree.add(p, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_prob = np.array(priorities) / self.tree.total
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        weights = (self.tree.n_entries * sampling_prob) ** (-beta)
        weights /= weights.max()
        return batch, idxs, weights

    def update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


class MaskedDQNAgent(DQNAgent):
    def __init__(self, num_actions, state_shape, hidden_layers=None, device=None,
                 memory_capacity=20000, n_step=3):
        if hidden_layers is None:
            hidden_layers = [128, 128]
        super().__init__(
            num_actions=num_actions,
            state_shape=state_shape,
            mlp_layers=hidden_layers,
            device=device
        )
        # Override uniform memory with PER
        self.memory = PrioritizedReplayBuffer(memory_capacity)

        # Epsilon schedule
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 30000
        self.total_steps = 0

        # DQN hyperparams
        self.gamma = 0.99
        self.sync_freq = 100
        self.train_step = 0

        # N-step returns
        self.n_step = n_step
        self.n_buffer = deque(maxlen=n_step)

        # Learning rate
        for pg in self.q_estimator.optimizer.param_groups:
            pg['lr'] = 5e-4

    def step(self, state):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        epsilon = self.epsilon_start * (1 - frac) + self.epsilon_end * frac

        legal = list(state['legal_actions'].keys())
        if random.random() < epsilon:
            return int(random.choice(legal))

        obs = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
        q_vals = self.q_estimator.qnet(obs)[0].detach().cpu().numpy()
        mask = np.full_like(q_vals, -np.inf)
        mask[legal] = q_vals[legal]
        return int(np.argmax(mask))

    def eval_step(self, state):
        return self.step(state), []

    def feed(self, ts):
        # ts = (state, action, reward, next_state, done)
        if not (isinstance(ts, tuple) and len(ts) == 5):
            return
        state, action, reward, next_state, done = ts
        if not (isinstance(state, dict) and isinstance(next_state, dict)):
            return

        # Add to n-step buffer
        self.n_buffer.append((state, action, reward, next_state, done))
        if len(self.n_buffer) < self.n_step:
            return

        # Compute n-step transition
        R = sum([self.n_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        s0, a0 = self.n_buffer[0][0], self.n_buffer[0][1]
        ns, d = self.n_buffer[-1][3], self.n_buffer[-1][4]
        legal_ns = list(ns['legal_actions'].keys()) if not d else []
        transition = (s0['obs'], a0, R, ns['obs'], legal_ns, d)

        # Push with max-priority
        self.memory.push(transition)

        # Per-step train
        self.train()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        transitions, idxs, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, next_legals, dones = zip(*transitions)

        S = torch.FloatTensor(np.array(states)).to(self.device)
        A = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        R = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        S2 = torch.FloatTensor(np.array(next_states)).to(self.device)
        D = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        W = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Current Q
        Q = self.q_estimator.qnet(S).gather(1, A)

        # Double DQN target
        with torch.no_grad():
            Q_online = self.q_estimator.qnet(S2)
            Q_target = self.q_estimator.target_qnet(S2)
            next_actions = []
            for i, legal in enumerate(next_legals):
                if legal:
                    m = torch.full((self.num_actions,), -float('inf')).to(self.device)
                    m[list(legal)] = Q_online[i][list(legal)]
                    next_actions.append(torch.argmax(m).item())
                else:
                    next_actions.append(0)
            A2 = torch.LongTensor(next_actions).unsqueeze(1).to(self.device)
            Q2 = Q_target.gather(1, A2)
            target = R + self.gamma * (1 - D) * Q2

        # Loss with importance sampling weights
        loss = (W * F.mse_loss(Q, target, reduction='none')).mean()

        # Backprop
        self.q_estimator.optimizer.zero_grad()
        loss.backward()
        self.q_estimator.optimizer.step()

        # Update priorities
        td_errors = (target - Q).detach().cpu().numpy().flatten()
        self.memory.update(idxs, td_errors)

        # Target sync
        self.train_step += 1
        if self.train_step % self.sync_freq == 0:
            self.q_estimator.sync_target()
