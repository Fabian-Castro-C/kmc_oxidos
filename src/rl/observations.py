"""
Local observation extraction for SwarmThinkers agents.

This module provides functions to extract local observations from the lattice
for use in the swarm policy network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..kmc.lattice import Lattice

from ..kmc.lattice import SpeciesType


def get_local_observation(lattice: Lattice, site_idx: int) -> npt.NDArray[np.float32]:
    """
    Extract local observation for a site/atom.

    This observation is designed for the diffusion-only swarm policy (Phase 1).
    It captures species composition, local topology, and geometric information
    necessary for intelligent diffusion proposals.

    Args:
        lattice: The lattice structure.
        site_idx: Index of the site to observe.

    Returns:
        Observation vector with shape (51,) containing:
        - [0:36]: Species of 1st shell neighbors (12 neighbors × 3 species one-hot)
                  Order: [VACANT, TI, O] for each neighbor
        - [36:48]: Heights relative to site (z_neighbor - z_site) for ES barrier detection
        - [48:50]: Local composition (n_Ti, n_O in 1st shell)
        - [50]: Absolute z-height of site

    Example:
        >>> obs = get_local_observation(lattice, 42)
        >>> obs.shape
        (51,)
        >>> # First neighbor is Ti: obs[0:3] = [0, 1, 0]
        >>> # Second neighbor is vacant: obs[3:6] = [1, 0, 0]
    """
    site = lattice.sites[site_idx]

    # Initialize observation array
    obs = np.zeros(51, dtype=np.float32)

    # 1. Species of 1st shell neighbors (one-hot encoding)
    # Assuming 12 nearest neighbors in BCC-like structure
    neighbors_1st = site.neighbors[:12] if len(site.neighbors) >= 12 else site.neighbors

    for i, neighbor_idx in enumerate(neighbors_1st):
        neighbor_site = lattice.sites[neighbor_idx]
        species = neighbor_site.species

        # One-hot encoding: [VACANT, TI, O]
        if species == SpeciesType.VACANT:
            obs[i * 3] = 1.0
        elif species == SpeciesType.TI:
            obs[i * 3 + 1] = 1.0
        elif species == SpeciesType.O:
            obs[i * 3 + 2] = 1.0

    # Pad if fewer than 12 neighbors (edge case)
    # Already zeros, no action needed

    # 2. Heights relative to site (for ES barrier detection)
    z_site = site.position[2]
    for i, neighbor_idx in enumerate(neighbors_1st):
        neighbor_site = lattice.sites[neighbor_idx]
        z_neighbor = neighbor_site.position[2]
        obs[36 + i] = float(z_neighbor - z_site)

    # 3. Local composition (counts)
    n_ti = sum(
        1
        for neighbor_idx in neighbors_1st
        if lattice.sites[neighbor_idx].species == SpeciesType.TI
    )
    n_o = sum(
        1
        for neighbor_idx in neighbors_1st
        if lattice.sites[neighbor_idx].species == SpeciesType.O
    )

    obs[48] = float(n_ti)
    obs[49] = float(n_o)

    # 4. Absolute z-height
    obs[50] = float(z_site)

    return obs


def get_batch_observations(
    lattice: Lattice, site_indices: list[int]
) -> npt.NDArray[np.float32]:
    """
    Extract observations for multiple sites in batch.

    Args:
        lattice: The lattice structure.
        site_indices: List of site indices.

    Returns:
        Observation array with shape (len(site_indices), 51).

    Example:
        >>> agent_indices = [10, 20, 30]
        >>> obs_batch = get_batch_observations(lattice, agent_indices)
        >>> obs_batch.shape
        (3, 51)
    """
    observations = np.array(
        [get_local_observation(lattice, idx) for idx in site_indices], dtype=np.float32
    )
    return observations


def get_diffusable_atoms(lattice: Lattice) -> list[int]:
    """
    Get indices of all atoms that can diffuse.

    For Phase 1 (diffusion-only), these are all non-vacant sites.
    In later phases, this will be extended to distinguish surface sites
    for adsorption events.

    Args:
        lattice: The lattice structure.

    Returns:
        List of site indices containing Ti or O atoms.

    Example:
        >>> agents = get_diffusable_atoms(lattice)
        >>> len(agents)
        245  # Number of adsorbed atoms
    """
    diffusable = []
    for idx, site in enumerate(lattice.sites):
        if site.species in (SpeciesType.TI, SpeciesType.O):
            diffusable.append(idx)
    return diffusable
