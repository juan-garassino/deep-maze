"""Abstract base class shared by every agent."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Common interface for all agents.

    Subclasses implement `move` and `update`; `q_values`, `policy_snapshot` and
    `memory_snapshot` are optional hooks used by the visualisation code.

    Parameters
    ----------
    action_size : int
        Number of discrete actions.
    """
    def __init__(self, action_size):
        self.action_size = action_size
        self.last_loss = None
        self.deterministic = False
        self._saved_epsilon = None

    def set_deterministic(self, flag: bool) -> None:
        """Toggle greedy action selection for evaluation/replay."""
        flag = bool(flag)
        if flag and not self.deterministic and hasattr(self, "epsilon"):
            self._saved_epsilon = self.epsilon
            self.epsilon = 0.0
        elif (not flag) and self.deterministic and self._saved_epsilon is not None \
                and hasattr(self, "epsilon"):
            self.epsilon = self._saved_epsilon
            self._saved_epsilon = None
        self.deterministic = flag

    def on_episode_end(self) -> None:
        """Called once per training episode by train_agent, after the step
        loop. Default: decay epsilon. No-op for agents without an epsilon
        schedule (PPO) and while deterministic (mid-run eval)."""
        if self.deterministic:
            return
        decay = getattr(self, "epsilon_decay", None)
        if decay is not None and hasattr(self, "epsilon"):
            self.epsilon = max(getattr(self, "min_epsilon", 0.0),
                               self.epsilon * decay)

    def q_values_batch(self, states):
        """Vectorised q_values. Default: Python loop. Net agents override."""
        import numpy as np
        return np.stack([self.q_values(s) for s in states])

    @abstractmethod
    def move(self, state):
        """Return the discrete action (int in ``[0, action_size)``) for `state`."""
        ...

    @abstractmethod
    def update(self, state, action, reward, next_state, done, truncated=False):
        """One environment transition. `truncated=True` marks a time-limit
        cut (episode did NOT end in the MDP): value targets still bootstrap,
        but multi-step credit (e.g. PPO's GAE carry) must not cross it."""
        ...

    def q_values(self, state):
        """Per-action values for `state` as a float array of shape (action_size,), or None if the agent has none."""
        return None

    def policy_snapshot(self):
        """Copy of the agent's parameters (or Q-table) for the event bus, or None."""
        return None

    def memory_snapshot(self):
        """Snapshot of the agent's current internal memory for live viz.
        Returns None unless the agent has a useful memory (DRQN/DTQN)."""
        return None
