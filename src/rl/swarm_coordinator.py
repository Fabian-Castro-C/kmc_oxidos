"""
Global Softmax Coordinator for SwarmThinkers.

This coordinator implements the key innovation of SwarmThinkers:
a global softmax over ALL agent×action pairs, enabling coordinated
decision-making across the entire swarm.

Key differences from traditional KMC:
- Traditional: Each site independently samples from local rates
- SwarmThinkers: Global softmax selects ONE event from ALL possibilities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.kmc.lattice import Lattice
    from src.rl.particle_agent import ActionType, ParticleAgent
    from src.rl.shared_policy import SharedPolicyNetwork


@dataclass
class SelectedEvent:
    """
    Represents a selected event from global softmax.

    Attributes:
        agent_idx: Index of agent in agent list.
        action: Action type to execute.
        probability: Softmax probability of this event.
        logit: Raw logit before softmax.
        global_rank: Rank among all agent×action pairs (0 = highest prob).
    """

    agent_idx: int
    action: ActionType
    probability: float
    logit: float
    global_rank: int


class SwarmCoordinator:
    """
    Global coordinator for SwarmThinkers architecture.

    Collects observations from all agents, runs them through the shared policy,
    and selects a single event via global softmax across ALL agent×action pairs.
    """

    def __init__(self, policy: SharedPolicyNetwork) -> None:
        """
        Initialize coordinator.

        Args:
            policy: Shared policy network used by all agents.
        """
        self.policy = policy
        self.step_count = 0

    def select_event(
        self,
        agents: list[ParticleAgent],
        temperature: float = 1.0,
    ) -> SelectedEvent:
        """
        Select a single event via global softmax.

        Process:
        1. Collect observations from all agents
        2. Get action masks (valid actions only)
        3. Forward pass through policy network → logits
        4. Flatten to [N_agents × N_actions] array
        5. Apply global softmax
        6. Sample one event

        Args:
            agents: List of all active agents.
            temperature: Softmax temperature (default: 1.0).
                        Higher = more exploration, Lower = more exploitation.

        Returns:
            Selected event with agent index and action.
        """
        from src.rl.action_space import get_batch_action_masks

        if len(agents) == 0:
            raise ValueError("Cannot select event: no agents provided")

        # Step 1: Collect observations
        observations = self._collect_observations(agents)

        # Step 2: Get action masks
        action_masks = get_batch_action_masks(agents)

        # Step 3: Policy forward pass
        logits = self.policy.get_action_logits(observations, action_masks)

        # Step 4: Flatten to 1D array [N_agents * N_actions]
        flat_logits = logits.flatten()
        flat_mask = action_masks.flatten()

        # Set invalid actions to -inf
        flat_logits[~flat_mask] = -np.inf

        # Step 5: Apply global softmax with temperature
        flat_logits_temp = flat_logits / temperature
        max_logit = np.max(flat_logits_temp[flat_mask])
        exp_logits = np.exp(flat_logits_temp - max_logit)
        exp_logits[~flat_mask] = 0.0
        probabilities = exp_logits / np.sum(exp_logits)

        # Step 6: Sample event
        flat_idx = np.random.choice(len(flat_logits), p=probabilities)

        # Convert flat index back to (agent_idx, action_idx)
        n_actions = logits.shape[1]
        agent_idx = flat_idx // n_actions
        action_idx = flat_idx % n_actions

        # Get action enum
        from src.rl.particle_agent import ActionType

        action = ActionType(action_idx)

        # Calculate global rank
        sorted_indices = np.argsort(probabilities)[::-1]
        global_rank = int(np.where(sorted_indices == flat_idx)[0][0])

        self.step_count += 1

        return SelectedEvent(
            agent_idx=agent_idx,
            action=action,
            probability=float(probabilities[flat_idx]),
            logit=float(flat_logits[flat_idx]),
            global_rank=global_rank,
        )

    def get_global_action_distribution(
        self,
        agents: list[ParticleAgent],
        temperature: float = 1.0,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Get full global action distribution without sampling.

        Useful for analysis and visualization.

        Args:
            agents: List of all active agents.
            temperature: Softmax temperature.

        Returns:
            Tuple of:
            - Logits [N_agents, N_actions]
            - Probabilities [N_agents, N_actions] (global softmax)
        """
        from src.rl.action_space import get_batch_action_masks

        # Collect observations and masks
        observations = self._collect_observations(agents)
        action_masks = get_batch_action_masks(agents)

        # Policy forward pass
        logits = self.policy.get_action_logits(observations, action_masks)

        # Global softmax
        flat_logits = logits.flatten()
        flat_mask = action_masks.flatten()
        flat_logits[~flat_mask] = -np.inf

        flat_logits_temp = flat_logits / temperature
        max_logit = np.max(flat_logits_temp[flat_mask])
        exp_logits = np.exp(flat_logits_temp - max_logit)
        exp_logits[~flat_mask] = 0.0
        flat_probs = exp_logits / np.sum(exp_logits)

        # Reshape back
        probabilities = flat_probs.reshape(logits.shape)

        return logits, probabilities

    def _collect_observations(
        self, agents: list[ParticleAgent]
    ) -> npt.NDArray[np.float32]:
        """
        Collect observations from all agents.

        Args:
            agents: List of agents.

        Returns:
            Observation batch [N_agents, obs_dim].
        """
        observations = []
        for agent in agents:
            obs = agent.observe()
            obs_vector = obs.to_vector()
            observations.append(obs_vector)

        return np.array(observations, dtype=np.float32)

    def get_stats(self) -> dict[str, int]:
        """
        Get coordinator statistics.

        Returns:
            Dictionary with step count and other metrics.
        """
        return {
            "step_count": self.step_count,
        }

    def reset_stats(self) -> None:
        """Reset coordinator statistics."""
        self.step_count = 0

    def __repr__(self) -> str:
        """String representation."""
        return f"SwarmCoordinator(steps={self.step_count})"
