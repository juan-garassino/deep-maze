"""Uniform transition replay buffer for DQN."""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of (state, action, reward, next_state, done) transitions.

    Capacity counts transitions. States are stored flattened as float32.
    """
    def __init__(self, capacity, state_size, action_dim=1):
        self.capacity = capacity
        self.state_size = state_size
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one transition; the oldest is evicted once `capacity` is reached."""
        self.buffer.append((
            np.asarray(state, dtype=np.float32).flatten(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).flatten(),
            float(done),
        ))

    def sample(self, batch_size):
        """Sample `batch_size` transitions uniformly without replacement.

        Returns a dict of numpy arrays: ``state``/``next_state`` float32 (B, state_size),
        ``action`` int64 (B, 1), ``reward`` and ``done`` float32 (B, 1).
        """
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return {
            'state': np.stack(s),
            'action': np.array(a, dtype=np.int64).reshape(-1, 1),
            'reward': np.array(r, dtype=np.float32).reshape(-1, 1),
            'next_state': np.stack(ns),
            'done': np.array(d, dtype=np.float32).reshape(-1, 1),
        }

    def __len__(self):
        return len(self.buffer)
