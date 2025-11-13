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


class AdsorptionSwarmPolicy(nn.Module):
    """
    Swarm policy for adsorption events.

    Proposes optimal surface sites for Ti/O adsorption based on local
    surface topology (heights, neighbors, curvature).

    Architecture matches SwarmThinkers: 5-layer MLP encoder.
    Output: Single logit per surface site (unnormalized preference).
    """

    def __init__(self, observation_dim: int = 51, hidden_size: int = 256) -> None:
        """
        Initialize adsorption swarm policy.

        Args:
            observation_dim: Dimension of local observation (surface site context).
            hidden_size: Size of hidden layers.
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_size = hidden_size

        # Shared encoder (5 layers)
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

        # Adsorption preference head (single logit per site)
        self.head_adsorption = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output adsorption logits for surface sites.

        Args:
            observations: Tensor of shape (batch_size, observation_dim).

        Returns:
            Tensor of shape (batch_size, 1) with unnormalized logits.
            Higher logit = more favorable adsorption site.
        """
        features = self.encoder(observations)
        logits = self.head_adsorption(features)
        return logits.squeeze(-1)  # (batch_size,)


class DesorptionSwarmPolicy(nn.Module):
    """
    Swarm policy for desorption events.

    Proposes which adsorbed atoms (Ti/O) should desorb based on
    local coordination, binding energy estimates, surface position.

    Architecture: 5-layer MLP → single logit per adsorbed atom.
    """

    def __init__(self, observation_dim: int = 51, hidden_size: int = 256) -> None:
        """
        Initialize desorption swarm policy.

        Args:
            observation_dim: Dimension of local observation (atom context).
            hidden_size: Size of hidden layers.
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_size = hidden_size

        # Shared encoder
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

        # Desorption preference head
        self.head_desorption = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output desorption logits for adsorbed atoms.

        Args:
            observations: Tensor of shape (batch_size, observation_dim).

        Returns:
            Tensor of shape (batch_size,) with unnormalized logits.
            Higher logit = more likely to desorb.
        """
        features = self.encoder(observations)
        logits = self.head_desorption(features)
        return logits.squeeze(-1)


class ReactionSwarmPolicy(nn.Module):
    """
    Swarm policy for reaction events (Ti + 2O → TiO₂).

    Proposes Ti atoms that should react with neighboring O atoms to form TiO₂.
    Requires analyzing multi-site coordination (Ti must have 2+ O neighbors).

    Architecture: 5-layer MLP → single logit per Ti atom candidate.
    """

    def __init__(self, observation_dim: int = 51, hidden_size: int = 256) -> None:
        """
        Initialize reaction swarm policy.

        Args:
            observation_dim: Dimension of local observation (Ti-O configuration).
            hidden_size: Size of hidden layers.
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.hidden_size = hidden_size

        # Shared encoder
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

        # Reaction preference head
        self.head_reaction = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output reaction logits for Ti atoms.

        Args:
            observations: Tensor of shape (batch_size, observation_dim).

        Returns:
            Tensor of shape (batch_size,) with unnormalized logits.
            Higher logit = more favorable to react.
        """
        features = self.encoder(observations)
        logits = self.head_reaction(features)
        return logits.squeeze(-1)


# Factory functions
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


def create_adsorption_swarm_policy(
    observation_dim: int = 51, hidden_size: int = 256
) -> AdsorptionSwarmPolicy:
    """Factory to create adsorption swarm policy."""
    return AdsorptionSwarmPolicy(observation_dim=observation_dim, hidden_size=hidden_size)


def create_desorption_swarm_policy(
    observation_dim: int = 51, hidden_size: int = 256
) -> DesorptionSwarmPolicy:
    """Factory to create desorption swarm policy."""
    return DesorptionSwarmPolicy(observation_dim=observation_dim, hidden_size=hidden_size)


def create_reaction_swarm_policy(
    observation_dim: int = 51, hidden_size: int = 256
) -> ReactionSwarmPolicy:
    """Factory to create reaction swarm policy."""
    return ReactionSwarmPolicy(observation_dim=observation_dim, hidden_size=hidden_size)
