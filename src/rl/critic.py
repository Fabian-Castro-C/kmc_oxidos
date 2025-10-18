"""
Critic network (value function) for SwarmThinkers.
"""

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Critic network for value function estimation.

    Architecture: MLP with 5 hidden layers, 256 units each.
    """

    def __init__(self, state_dim: int, hidden_size: int = 256) -> None:
        """Initialize critic network."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def create_critic_network(state_dim: int, hidden_size: int = 256) -> CriticNetwork:
    """Create critic network."""
    return CriticNetwork(state_dim, hidden_size)
