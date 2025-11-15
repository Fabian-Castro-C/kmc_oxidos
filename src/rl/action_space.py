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


# Total number of possible actions for an agent
N_ACTIONS = 7


class ActionType(Enum):
    """Types of actions an agent can take."""

    # Diffusion actions (for Ti/O particles)
    DIFFUSE_X_POS = 0
    DIFFUSE_X_NEG = 1
    DIFFUSE_Y_POS = 2
    DIFFUSE_Y_NEG = 3
    DIFFUSE_Z_POS = 4
    DIFFUSE_Z_NEG = 5

    # Desorption action (for Ti/O particles)
    DESORB = 6

    # The REACT_TIO2 action is currently handled implicitly by the environment
    # when conditions are met, not as a direct agent choice.
    # Adsorption is a global event (deposition), not an agent choice.


def create_action_mask(
    agents: list[ParticleAgent], lattice_size: tuple[int, int, int]
) -> npt.NDArray[np.bool_]:
    """
    Creates a boolean mask for valid actions for each agent.

    An action is invalid if:
    - It's a diffusion action to an already occupied site.
    - It's a diffusion action out of the lattice boundaries.

    Args:
        agents: A list of `ParticleAgent` instances.
        lattice_size: The (x, y, z) dimensions of the lattice.

    Returns:
        A boolean numpy array of shape (num_agents, N_ACTIONS).
        `True` indicates a valid action, `False` indicates an invalid one.
    """
    num_agents = len(agents)
    mask = np.zeros((num_agents, N_ACTIONS), dtype=bool)

    if num_agents == 0:
        return mask

    # Get all agent positions and occupied sites for efficient checking
    agent_positions = np.array([agent.site_index for agent in agents])
    occupied_sites = {agent.site_index for agent in agents}
    lx, ly, lz = lattice_size

    # Vectorized boundary and neighbor calculations
    x = agent_positions % lx
    y = (agent_positions // lx) % ly
    z = agent_positions // (lx * ly)

    # --- Diffusion Actions ---
    # Check boundaries
    mask[:, ActionType.DIFFUSE_X_POS.value] = x + 1 < lx
    mask[:, ActionType.DIFFUSE_X_NEG.value] = x - 1 >= 0
    mask[:, ActionType.DIFFUSE_Y_POS.value] = y + 1 < ly
    mask[:, ActionType.DIFFUSE_Y_NEG.value] = y - 1 >= 0
    mask[:, ActionType.DIFFUSE_Z_POS.value] = z + 1 < lz
    mask[:, ActionType.DIFFUSE_Z_NEG.value] = z - 1 >= 0

    # Check for collisions with other agents
    # This part remains iterative as neighbor lookups are complex to vectorize simply
    for i, agent in enumerate(agents):
        neighbors = agent.get_neighbors(lattice_size)
        for action_type, neighbor_idx in neighbors.items():
            if neighbor_idx in occupied_sites:
                mask[i, action_type.value] = False

    # --- Desorption Action ---
    # Desorption is always considered a valid action for an existing particle.
    mask[:, ActionType.DESORB.value] = True

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
