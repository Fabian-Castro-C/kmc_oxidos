"""
Actor network (policy) for SwarmThinkers.

Implements the MLP policy network with 5 layers and 256 hidden units.
"""

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Actor network for policy.

    Architecture: MLP with 5 hidden layers, 256 units each.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        """Initialize actor network."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def create_policy_network(
    observation_dim: int, action_dim: int, hidden_size: int = 256
) -> ActorNetwork:
    """Create policy network."""
    return ActorNetwork(observation_dim, action_dim, hidden_size)
