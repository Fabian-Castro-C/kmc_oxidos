"""
Action space utilities for SwarmThinkers.

Defines action sets per species type and provides masking utilities
to ensure only valid actions are considered.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.rl.particle_agent import ParticleAgent


# Total number of possible actions
N_ACTIONS = 10


class ActionType(Enum):
    """Types of actions an agent can take."""

    # Diffusion actions (for Ti/O particles)
    DIFFUSE_X_POS = 0
    DIFFUSE_X_NEG = 1
    DIFFUSE_Y_POS = 2
    DIFFUSE_Y_NEG = 3
    DIFFUSE_Z_POS = 4
    DIFFUSE_Z_NEG = 5

    # Adsorption actions (for vacant sites)
    ADSORB_TI = 6
    ADSORB_O = 7

    # Desorption action (for Ti/O particles)
    DESORB = 8

    # Reaction action (for Ti particles with O neighbors)
    REACT_TIO2 = 9


def create_action_mask(agents: list[ParticleAgent]) -> npt.NDArray[np.bool_]:
    """
    Creates a boolean action mask for a list of agents.

    Args:
        agents: A list of ParticleAgent instances.

    Returns:
        A boolean numpy array of shape (n_agents, N_ACTIONS), where True indicates
        a valid action.
    """
    from src.kmc.lattice import SpeciesType

    if not agents:
        return np.zeros((0, N_ACTIONS), dtype=bool)

    n_agents = len(agents)
    mask = np.zeros((n_agents, N_ACTIONS), dtype=bool)

    # Define valid actions for occupied and vacant sites
    # Note: REACT_TIO2 is excluded as it's not implemented in the environment
    occupied_actions = [
        ActionType.DIFFUSE_X_POS.value,
        ActionType.DIFFUSE_X_NEG.value,
        ActionType.DIFFUSE_Y_POS.value,
        ActionType.DIFFUSE_Y_NEG.value,
        ActionType.DIFFUSE_Z_POS.value,
        ActionType.DIFFUSE_Z_NEG.value,
        ActionType.DESORB.value,
    ]
    vacant_actions = [
        ActionType.ADSORB_TI.value,
        ActionType.ADSORB_O.value,
    ]

    for i, agent in enumerate(agents):
        if agent.site.species == SpeciesType.VACANT:
            mask[i, vacant_actions] = True
        else:
            mask[i, occupied_actions] = True

    return mask


def get_action_mask(agent: ParticleAgent) -> npt.NDArray[np.bool_]:
    """
    Get boolean action mask for an agent.

    Args:
        agent: Particle agent to get mask for.

    Returns:
        Boolean array [N_ACTIONS] where True = valid action.
    """

    # Initialize all actions as invalid
    mask = np.zeros(N_ACTIONS, dtype=bool)

    # Get valid actions for this agent
    valid_actions = agent.get_valid_actions()

    # Set valid actions to True
    for action in valid_actions:
        mask[action.value] = True

    return mask


def get_batch_action_masks(agents: list[ParticleAgent]) -> npt.NDArray[np.bool_]:
    """
    Get action masks for a batch of agents.

    Args:
        agents: List of particle agents.

    Returns:
        Boolean array [n_agents, N_ACTIONS] where True = valid action.
    """
    masks = np.zeros((len(agents), N_ACTIONS), dtype=bool)

    for i, agent in enumerate(agents):
        masks[i] = get_action_mask(agent)

    return masks


def action_to_string(action_idx: int) -> str:
    """
    Convert action index to human-readable string.

    Args:
        action_idx: Action index (0-9).

    Returns:
        String description of the action.
    """
    from src.rl.particle_agent import ActionType

    try:
        action = ActionType(action_idx)
        return action.name
    except ValueError:
        return f"INVALID_ACTION_{action_idx}"


def get_species_action_counts() -> dict[str, list[str]]:
    """
    Get available actions for each species type.

    Returns:
        Dictionary mapping species name to list of action names.
    """
    from src.rl.particle_agent import ActionType

    result = {
        "VACANT": [ActionType.ADSORB_TI.name, ActionType.ADSORB_O.name],
        "TI": [
            ActionType.DIFFUSE_X_POS.name,
            ActionType.DIFFUSE_X_NEG.name,
            ActionType.DIFFUSE_Y_POS.name,
            ActionType.DIFFUSE_Y_NEG.name,
            ActionType.DIFFUSE_Z_POS.name,
            ActionType.DIFFUSE_Z_NEG.name,
            ActionType.DESORB.name,
            ActionType.REACT_TIO2.name,  # Conditional on O neighbors
        ],
        "O": [
            ActionType.DIFFUSE_X_POS.name,
            ActionType.DIFFUSE_X_NEG.name,
            ActionType.DIFFUSE_Y_POS.name,
            ActionType.DIFFUSE_Y_NEG.name,
            ActionType.DIFFUSE_Z_POS.name,
            ActionType.DIFFUSE_Z_NEG.name,
            ActionType.DESORB.name,
        ],
    }

    return result


def validate_action_mask(mask: npt.NDArray[np.bool_]) -> bool:
    """
    Validate that an action mask has at least one valid action.

    Args:
        mask: Boolean action mask [N_ACTIONS].

    Returns:
        True if mask has at least one True value.
    """
    return np.any(mask)
