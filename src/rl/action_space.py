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

from src.kmc.lattice import SpeciesType

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


# --- Action Sets for Masking ---

DIFFUSION_ACTIONS: set[ActionType] = {
    ActionType.DIFFUSE_X_POS,
    ActionType.DIFFUSE_X_NEG,
    ActionType.DIFFUSE_Y_POS,
    ActionType.DIFFUSE_Y_NEG,
    ActionType.DIFFUSE_Z_POS,
    ActionType.DIFFUSE_Z_NEG,
}

ADSORPTION_ACTIONS: set[ActionType] = {
    ActionType.ADSORB_TI,
    ActionType.ADSORB_O,
}

# Actions valid for an agent on an OCCUPIED site
VALID_ACTIONS_OCCUPIED: set[ActionType] = DIFFUSION_ACTIONS | {ActionType.DESORB}

# Actions valid for an agent on a VACANT site
VALID_ACTIONS_VACANT: set[ActionType] = ADSORPTION_ACTIONS


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


def create_action_mask(agents: list[ParticleAgent]) -> npt.NDArray[np.bool_]:
    """
    Creates a boolean mask for valid actions for each agent.

    The mask has shape (num_agents, N_ACTIONS).
    `mask[i, j] = True` if action `j` is valid for agent `i`.

    Args:
        agents: A list of active ParticleAgent instances.

    Returns:
        A numpy boolean array representing the action mask.
    """
    if not agents:
        return np.empty((0, N_ACTIONS), dtype=bool)

    num_agents = len(agents)
    mask = np.zeros((num_agents, N_ACTIONS), dtype=bool)

    for i, agent in enumerate(agents):
        # Determine the set of valid actions based on the site's occupancy
        valid_actions = (
            VALID_ACTIONS_OCCUPIED
            if agent.site.species != SpeciesType.VACANT
            else VALID_ACTIONS_VACANT
        )

        # Set the corresponding indices in the mask to True
        for action in valid_actions:
            mask[i, action.value] = True

    return mask
