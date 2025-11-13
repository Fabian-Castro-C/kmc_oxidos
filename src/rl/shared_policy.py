"""
Shared Policy Network for SwarmThinkers.

A single MLP network shared by all particle agents. Takes local observations
and outputs action logits. The network architecture follows the SwarmThinkers
paper (Li et al., 2025): 5 hidden layers with 256 units each.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt


class SharedPolicyNetwork(nn.Module):
    """
    Shared policy network for all particle agents.

    Architecture (from SwarmThinkers paper):
    - Input: local observation vector (58 dims)
    - 5 hidden layers: 256 units each
    - Activation: ReLU
    - Output: action logits (10 dims)

    All agents share the same weights, enabling transfer learning across
    different local configurations.
    """

    def __init__(
        self,
        obs_dim: int = 58,
        action_dim: int = 10,
        hidden_dim: int = 256,
        n_hidden_layers: int = 5,
    ) -> None:
        """
        Initialize the shared policy network.

        Args:
            obs_dim: Dimension of observation vector (default: 58).
            action_dim: Number of possible actions (default: 10).
            hidden_dim: Number of units per hidden layer (default: 256).
            n_hidden_layers: Number of hidden layers (default: 5).
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (no activation - raw logits)
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier/Glorot initialization)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            observations: Batch of observations [batch_size, obs_dim].

        Returns:
            Action logits [batch_size, action_dim].
        """
        return self.network(observations)

    def get_action_logits(
        self,
        observations: npt.NDArray[np.float32],
        action_masks: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Get action logits for a batch of observations.

        Args:
            observations: Numpy array of observations [batch_size, obs_dim].
            action_masks: Optional boolean mask [batch_size, action_dim].
                          True = action is valid, False = action is invalid.
                          Invalid actions get logit = -inf.

        Returns:
            Action logits [batch_size, action_dim] as numpy array.
        """
        # Convert to torch tensor
        obs_tensor = torch.from_numpy(observations).float()

        # Forward pass
        with torch.no_grad():
            logits = self.forward(obs_tensor)

        # Apply action masking if provided
        if action_masks is not None:
            # Convert mask to tensor
            mask_tensor = torch.from_numpy(action_masks).bool()
            # Set invalid actions to -inf
            logits = torch.where(mask_tensor, logits, torch.tensor(-float("inf")))

        return logits.numpy()

    def get_action_probabilities(
        self,
        observations: npt.NDArray[np.float32],
        action_masks: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Get action probabilities for a batch of observations.

        Args:
            observations: Numpy array of observations [batch_size, obs_dim].
            action_masks: Optional boolean mask for valid actions.

        Returns:
            Action probabilities [batch_size, action_dim] (sums to 1 per row).
        """
        logits = self.get_action_logits(observations, action_masks)

        # Apply softmax
        logits_tensor = torch.from_numpy(logits).float()
        probs = torch.softmax(logits_tensor, dim=-1)

        return probs.numpy()

    def sample_actions(
        self,
        observations: npt.NDArray[np.float32],
        action_masks: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """
        Sample actions from the policy distribution.

        Args:
            observations: Numpy array of observations [batch_size, obs_dim].
            action_masks: Optional boolean mask for valid actions.

        Returns:
            Tuple of:
            - Sampled actions [batch_size] as integers.
            - Log probabilities [batch_size] of sampled actions.
        """
        probs = self.get_action_probabilities(observations, action_masks)

        # Sample actions
        probs_tensor = torch.from_numpy(probs).float()
        dist = torch.distributions.Categorical(probs_tensor)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions.numpy(), log_probs.numpy()

    def __repr__(self) -> str:
        """String representation."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"SharedPolicyNetwork(\n"
            f"  obs_dim={self.obs_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  n_hidden_layers={self.n_hidden_layers},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )
