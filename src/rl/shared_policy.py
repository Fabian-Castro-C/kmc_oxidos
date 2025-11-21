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

# Common network architecture for both Actor and Critic (defaults from SwarmThinkers paper)
DEFAULT_HIDDEN_DIMS = [256, 256, 256, 256, 256]


class Actor(nn.Module):
    """
    Decentralized Actor (Policy) Network.

    Shared by all particle agents. Maps local observations to action logits.

    Architecture (from SwarmThinkers paper):
    - Input: Local observation vector (58 dims)
    - 5 hidden layers with 256 units each (configurable)
    - Activation: ReLU or tanh (configurable)
    - Output: Action logits (10 dims)
    """

    def __init__(
        self,
        obs_dim: int = 58,
        action_dim: int = 10,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        """
        Initialize the Actor network.

        Args:
            obs_dim: Dimension of the local observation vector.
            action_dim: Number of possible actions for an agent.
            hidden_dims: List of hidden layer dimensions. If None, uses paper defaults.
            activation: Activation function ('relu' or 'tanh').
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS
        self.activation = activation

        layers = self._build_layers(obs_dim, action_dim)
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.activation.lower() == "relu":
            return nn.ReLU()
        elif self.activation.lower() == "tanh":
            return nn.Tanh()
        elif self.activation.lower() == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _build_layers(self, input_dim: int, output_dim: int) -> list[nn.Module]:
        """Construct the network layers."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self._get_activation()])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
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
    Centralized Critic (Value) Network with Deep Sets Architecture.

    This critic is designed to be invariant to the number of agents (permutation invariant).
    It processes each agent's observation independently and then aggregates them
    using a symmetric pooling operation (Max Pooling) to form a global context vector.
    This allows the critic to scale from small training lattices to large inference lattices.

    Architecture:
    1. Agent Encoder (Shared): Maps local obs -> latent vector
    2. Symmetric Pooling: Max(latent vectors) -> Swarm Context
    3. Global Fusion: Concat(Swarm Context, Global Features) -> Value Head
    4. Value Head: MLP -> Value
    """

    def __init__(
        self,
        obs_dim: int = 58,
        global_obs_dim: int = 12,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS
        self.activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        # 1. Agent Encoder: Processes each agent's local observation
        # We use a smaller network for the encoder to keep parameter count reasonable
        encoder_hidden = 128
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_hidden),
            self.activation_fn,
            nn.Linear(encoder_hidden, encoder_hidden),
            self.activation_fn,
        )

        # 2. Value Head: Processes the combined global state
        # Input = Swarm Context (encoder_hidden) + Global Features (global_obs_dim)
        input_dim = encoder_hidden + global_obs_dim
        layers = []
        prev_dim = input_dim

        for dim in self.hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), self.activation_fn])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        global_features: torch.Tensor,
        agent_observations: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the centralized critic.

        Args:
            global_features: Batch of global features [batch_size, global_obs_dim]
            agent_observations:
                - If Tensor: [batch_size, max_agents, obs_dim] (padded)
                - If List: List of [n_agents, obs_dim] tensors (variable length)
                - If None: Assumes 0 agents (empty surface)

        Returns:
            State value estimates [batch_size, 1]
        """
        batch_size = global_features.shape[0]
        device = global_features.device

        # Handle case with no agents (e.g. initial empty lattice)
        if agent_observations is None or (
            isinstance(agent_observations, list) and len(agent_observations) == 0
        ):
            swarm_context = torch.zeros(batch_size, 128, device=device)

        elif isinstance(agent_observations, list):
            # Variable number of agents per batch element
            swarm_contexts = []
            for _i, obs in enumerate(agent_observations):
                if obs.shape[0] == 0:
                    # No agents in this environment
                    ctx = torch.zeros(128, device=device)
                else:
                    # Encode all agents: [n_agents, 128]
                    encoded = self.agent_encoder(obs)
                    # Max Pooling: [128]
                    ctx, _ = torch.max(encoded, dim=0)
                swarm_contexts.append(ctx)
            swarm_context = torch.stack(swarm_contexts)

        else:
            # Tensor input [batch, max_agents, obs_dim]
            # Assuming 0-padding for non-existent agents is handled by masking or ignored
            # For simplicity, if using tensor, we assume all agents are valid or padding doesn't affect max significantly
            # (Ideally, we should use a mask)
            encoded = self.agent_encoder(agent_observations)  # [batch, max_agents, 128]
            swarm_context, _ = torch.max(encoded, dim=1)  # [batch, 128]

        # Concatenate Swarm Context with Global Features
        combined_state = torch.cat([swarm_context, global_features], dim=1)

        return self.value_head(combined_state)
