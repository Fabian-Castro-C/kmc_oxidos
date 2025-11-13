"""
Swarm policy networks for SwarmThinkers framework.

This module implements neural network policies that propose atomic transitions
in a swarm-based manner, enabling intelligent event selection while preserving
thermodynamic consistency through reweighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSwarmPolicy(nn.Module):
    """
    Swarm policy for diffusion-only events (Phase 1).

    This policy takes local observations and outputs logits for K possible
    diffusion directions. It follows the SwarmThinkers architecture with
    a deep MLP encoder.

    Architecture:
        - Input: Local observation (51 dimensions)
        - Encoder: 5 hidden layers × 256 units with ReLU
        - Head: Linear layer to K directions (12 for BCC neighbors)

    The output logits are later combined via global softmax across all
    (agent, direction) pairs to enable swarm-level coordination.
    """

    def __init__(
        self, observation_dim: int = 51, n_directions: int = 12, hidden_size: int = 256
    ) -> None:
        """
        Initialize diffusion swarm policy.

        Args:
            observation_dim: Dimension of local observation vector.
            n_directions: Number of diffusion directions (neighbors).
            hidden_size: Size of hidden layers.
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.n_directions = n_directions
        self.hidden_size = hidden_size

        # Shared encoder (5 layers as per SwarmThinkers paper)
        self.encoder = nn.Sequential(
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
        )

        # Direction head
        self.head_directions = nn.Linear(hidden_size, n_directions)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode observations and output direction logits.

        Args:
            observations: Tensor of shape (batch_size, observation_dim).

        Returns:
            Tensor of shape (batch_size, n_directions) with unnormalized logits.
            Softmax is applied later globally across all (agent, direction) pairs.

        Example:
            >>> policy = DiffusionSwarmPolicy()
            >>> obs = torch.randn(32, 51)  # Batch of 32 agents
            >>> logits = policy(obs)
            >>> logits.shape
            torch.Size([32, 12])
        """
        features = self.encoder(observations)
        logits = self.head_directions(features)
        return logits

    def get_action_logits_for_agent(
        self, observation: torch.Tensor
    ) -> torch.Tensor:
        """
        Get direction logits for a single agent.

        Args:
            observation: Tensor of shape (observation_dim,).

        Returns:
            Tensor of shape (n_directions,) with unnormalized logits.

        Example:
            >>> policy = DiffusionSwarmPolicy()
            >>> obs_single = torch.randn(51)
            >>> logits = policy.get_action_logits_for_agent(obs_single)
            >>> logits.shape
            torch.Size([12])
        """
        # Add batch dimension
        obs_batch = observation.unsqueeze(0)
        logits_batch = self.forward(obs_batch)
        return logits_batch.squeeze(0)


def create_diffusion_swarm_policy(
    observation_dim: int = 51, n_directions: int = 12, hidden_size: int = 256
) -> DiffusionSwarmPolicy:
    """
    Factory function to create a diffusion swarm policy.

    Args:
        observation_dim: Dimension of local observation vector.
        n_directions: Number of diffusion directions.
        hidden_size: Size of hidden layers.

    Returns:
        Initialized DiffusionSwarmPolicy.

    Example:
        >>> policy = create_diffusion_swarm_policy()
        >>> isinstance(policy, DiffusionSwarmPolicy)
        True
    """
    return DiffusionSwarmPolicy(
        observation_dim=observation_dim,
        n_directions=n_directions,
        hidden_size=hidden_size,
    )
