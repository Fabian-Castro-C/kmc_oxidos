"""
Actor-Critic Networks for SwarmThinkers.

This module defines the Actor and Critic networks based on the SwarmThinkers
paper (Li et al., 2025).

- Actor: A decentralized policy network shared by all agents. It takes local
  observations and outputs action logits.
- Critic: A centralized value network that takes an aggregated representation
  of all agent observations and outputs a single value for the state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Common network architecture for both Actor and Critic
HIDDEN_DIM = 256
N_HIDDEN_LAYERS = 5


class Actor(nn.Module):
    """
    Decentralized Actor (Policy) Network.

    Shared by all particle agents. Maps local observations to action logits.

    Architecture (from SwarmThinkers paper):
    - Input: Local observation vector (58 dims)
    - 5 hidden layers with 256 units each
    - Activation: ReLU
    - Output: Action logits (10 dims)
    """

    def __init__(self, obs_dim: int = 58, action_dim: int = 10) -> None:
        """
        Initialize the Actor network.

        Args:
            obs_dim: Dimension of the local observation vector.
            action_dim: Number of possible actions for an agent.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = self._build_layers(obs_dim, action_dim)
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _build_layers(self, input_dim: int, output_dim: int) -> list[nn.Module]:
        """Construct the network layers."""
        layers = [nn.Linear(input_dim, HIDDEN_DIM), nn.ReLU()]
        for _ in range(N_HIDDEN_LAYERS - 1):
            layers.extend([nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()])
        layers.append(nn.Linear(HIDDEN_DIM, output_dim))
        return layers

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: maps observations to action logits.

        Args:
            observations: Batch of local observations [batch_size, obs_dim].

        Returns:
            Action logits [batch_size, action_dim].
        """
        return self.network(observations)


class Critic(nn.Module):
    """
    Centralized Critic (Value) Network.

    Estimates the value of a global state representation. The global state is
    formed by aggregating the local observations of all agents (e.g., by mean).

    Architecture (from SwarmThinkers paper):
    - Input: Aggregated observation vector (58 dims)
    - 5 hidden layers with 256 units each
    - Activation: ReLU
    - Output: Single state value (1 dim)
    """

    def __init__(self, obs_dim: int = 58) -> None:
        """
        Initialize the Critic network.

        Args:
            obs_dim: Dimension of the aggregated observation vector.
        """
        super().__init__()
        self.obs_dim = obs_dim

        layers = self._build_layers(obs_dim, 1)  # Output is a single value
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _build_layers(self, input_dim: int, output_dim: int) -> list[nn.Module]:
        """Construct the network layers."""
        layers = [nn.Linear(input_dim, HIDDEN_DIM), nn.ReLU()]
        for _ in range(N_HIDDEN_LAYERS - 1):
            layers.extend([nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU()])
        layers.append(nn.Linear(HIDDEN_DIM, output_dim))
        return layers

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, aggregated_observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: maps an aggregated observation to a state value.

        Args:
            aggregated_observation: A tensor representing the global state,
                                    e.g., the mean of all agent observations.
                                    Shape: [batch_size, obs_dim].

        Returns:
            The estimated state value. Shape: [batch_size, 1].
        """
        return self.network(aggregated_observation)
