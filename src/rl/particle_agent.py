"""
Particle Agent for SwarmThinkers.

Each particle (Ti, O, or vacant site) is modeled as an autonomous agent that:
- Observes its local neighborhood (1st and 2nd nearest neighbors)
- Proposes actions via a shared policy network
- Contributes to global softmax decision making

This follows the architecture from the SwarmThinkers paper (Li et al., 2025),
extended to handle deposition systems (adsorption/desorption/reaction).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.kmc.lattice import Lattice, SpeciesType


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


@dataclass
class LocalObservation:
    """
    Local observation for a particle agent.

    Includes:
    - Species of 1st nearest neighbors (6 in 3D cubic lattice)
    - Species of 2nd nearest neighbors (12 in 3D cubic lattice)
    - Agent's own species
    - Agent's height (z-coordinate)
    """

    neighbors_1st: npt.NDArray[np.int8]  # Shape: (6,), species IDs
    neighbors_2nd: npt.NDArray[np.int8]  # Shape: (12,), species IDs
    own_species: int  # Species ID of this agent
    height: int  # Z-coordinate

    def to_vector(self) -> npt.NDArray[np.float32]:
        """
        Convert observation to a flat vector for policy network.

        Returns:
            Flattened observation vector (one-hot encoded species + normalized height).
        """
        # One-hot encode species (3 species: VACANT=0, Ti=1, O=2)
        n_species = 3

        # Encode 1st neighbors (6 neighbors × 3 species = 18 dims)
        neighbors_1st_onehot = np.zeros(6 * n_species, dtype=np.float32)
        for i, species in enumerate(self.neighbors_1st):
            if 0 <= species < n_species:
                neighbors_1st_onehot[i * n_species + species] = 1.0

        # Encode 2nd neighbors (12 neighbors × 3 species = 36 dims)
        neighbors_2nd_onehot = np.zeros(12 * n_species, dtype=np.float32)
        for i, species in enumerate(self.neighbors_2nd):
            if 0 <= species < n_species:
                neighbors_2nd_onehot[i * n_species + species] = 1.0

        # Encode own species (3 dims)
        own_species_onehot = np.zeros(n_species, dtype=np.float32)
        if 0 <= self.own_species < n_species:
            own_species_onehot[self.own_species] = 1.0

        # Normalized height (1 dim, assuming max height ~20 layers)
        normalized_height = np.array([self.height / 20.0], dtype=np.float32)

        # Concatenate: 18 + 36 + 3 + 1 = 58 dims
        return np.concatenate(
            [neighbors_1st_onehot, neighbors_2nd_onehot, own_species_onehot, normalized_height]
        )


class ParticleAgent:
    """
    Agent representing a single particle or vacant site in the lattice.

    Each agent:
    - Has a unique site index in the lattice
    - Has a species (VACANT, TI, or O)
    - Can observe its local neighborhood
    - Can propose actions based on its species
    """

    def __init__(self, site_idx: int, lattice: Lattice) -> None:
        """
        Initialize a particle agent.

        Args:
            site_idx: Index of the site in lattice.sites.
            lattice: Reference to the lattice.
        """
        self.site_idx = site_idx
        self.lattice = lattice
        self.site = lattice.sites[site_idx]

    @property
    def position(self) -> tuple[int, int, int]:
        """Get (x, y, z) position."""
        return self.site.position

    @property
    def species(self) -> SpeciesType:
        """Get species."""
        return self.site.species

    @property
    def x(self) -> int:
        """X coordinate."""
        return self.site.position[0]

    @property
    def y(self) -> int:
        """Y coordinate."""
        return self.site.position[1]

    @property
    def z(self) -> int:
        """Z coordinate (height)."""
        return self.site.position[2]

    def observe(self) -> LocalObservation:
        """
        Get local observation of neighborhood.

        Returns:
            LocalObservation containing neighbor species and own state.
        """
        # Get 1st nearest neighbors (6 in 3D cubic)
        neighbors_1st = self._get_1st_neighbors()

        # Get 2nd nearest neighbors (12 in 3D cubic)
        neighbors_2nd = self._get_2nd_neighbors()

        return LocalObservation(
            neighbors_1st=neighbors_1st,
            neighbors_2nd=neighbors_2nd,
            own_species=self.species.value,
            height=self.z,
        )

    def _get_1st_neighbors(self) -> npt.NDArray[np.int8]:
        """
        Get species of 1st nearest neighbors.

        Uses the neighbor list from the Site object.

        Returns:
            Array of species IDs (padded to 6 for consistency).
        """
        neighbor_species = []

        # Get neighbors from site's neighbor list
        for neighbor_idx in self.site.neighbors:
            neighbor_site = self.lattice.sites[neighbor_idx]
            neighbor_species.append(neighbor_site.species.value)

        # Pad to 6 (some sites may have fewer neighbors at boundaries)
        while len(neighbor_species) < 6:
            neighbor_species.append(0)  # Pad with VACANT

        return np.array(neighbor_species[:6], dtype=np.int8)

    def _get_2nd_neighbors(self) -> npt.NDArray[np.int8]:
        """
        Get species of 2nd nearest neighbors.

        For simplicity, use a geometric approach with position offsets.

        Returns:
            Array of 12 species IDs.
        """
        x, y, z = self.position
        nx, ny, nz = self.lattice.size

        # 2nd neighbors: edge-sharing (12 directions)
        offsets = [
            # Along x-y plane
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            # Along x-z plane
            (1, 0, 1),
            (1, 0, -1),
            (-1, 0, 1),
            (-1, 0, -1),
            # Along y-z plane
            (0, 1, 1),
            (0, 1, -1),
            (0, -1, 1),
            (0, -1, -1),
        ]

        species_list = []
        for dx, dy, dz in offsets:
            nx_pos = (x + dx) % nx
            ny_pos = (y + dy) % ny
            nz_pos = z + dz

            # Handle z boundaries (no periodic in z)
            if 0 <= nz_pos < nz:
                # Find site index from position
                neighbor_idx = nx_pos + ny_pos * nx + nz_pos * nx * ny
                if 0 <= neighbor_idx < len(self.lattice.sites):
                    neighbor_site = self.lattice.sites[neighbor_idx]
                    species_list.append(neighbor_site.species.value)
                else:
                    species_list.append(0)
            else:
                # Out of bounds in z → treat as VACANT
                species_list.append(0)

        return np.array(species_list, dtype=np.int8)

    def get_valid_actions(self) -> list[ActionType]:
        """
        Get list of valid actions for this agent based on its species.

        For Ti atoms, includes REACT_TIO2 only if there are at least 2 O neighbors.

        Returns:
            List of valid ActionType values.
        """
        from src.kmc.lattice import SpeciesType

        if self.species == SpeciesType.VACANT:
            # Vacant sites can adsorb Ti or O
            return [ActionType.ADSORB_TI, ActionType.ADSORB_O]
        elif self.species == SpeciesType.TI:
            # Ti can diffuse, desorb, or react with O neighbors
            actions = [
                ActionType.DIFFUSE_X_POS,
                ActionType.DIFFUSE_X_NEG,
                ActionType.DIFFUSE_Y_POS,
                ActionType.DIFFUSE_Y_NEG,
                ActionType.DIFFUSE_Z_POS,
                ActionType.DIFFUSE_Z_NEG,
                ActionType.DESORB,
            ]
            # Check if reaction is possible (need at least 2 O neighbors)
            obs = self.observe()
            o_neighbors = np.sum(obs.neighbors_1st == SpeciesType.O.value)
            if o_neighbors >= 2:
                actions.append(ActionType.REACT_TIO2)
            return actions
        elif self.species == SpeciesType.O:
            # O can diffuse or desorb (no reaction for O)
            return [
                ActionType.DIFFUSE_X_POS,
                ActionType.DIFFUSE_X_NEG,
                ActionType.DIFFUSE_Y_POS,
                ActionType.DIFFUSE_Y_NEG,
                ActionType.DIFFUSE_Z_POS,
                ActionType.DIFFUSE_Z_NEG,
                ActionType.DESORB,
            ]
        else:
            return []

    def __repr__(self) -> str:
        """String representation."""
        return f"ParticleAgent(pos={self.position}, species={self.species.name})"


def create_agents_from_lattice(lattice: Lattice) -> list[ParticleAgent]:
    """
    Create particle agents for all occupied sites and surface vacant sites.

    Args:
        lattice: The KMC lattice.

    Returns:
        List of ParticleAgent instances.
    """
    from src.kmc.lattice import SpeciesType

    agents = []
    nx, ny, nz = lattice.size

    for idx, site in enumerate(lattice.sites):
        # Create agent for occupied sites (Ti, O)
        if site.species in [SpeciesType.TI, SpeciesType.O]:
            agents.append(ParticleAgent(idx, lattice))

        # Create agent for surface vacant sites
        elif site.species == SpeciesType.VACANT:
            x, y, z = site.position
            # Surface: z == 0 or has occupied neighbor below
            is_surface = z == 0
            if not is_surface and z > 0:
                idx_below = x + y * nx + (z - 1) * nx * ny
                if 0 <= idx_below < len(lattice.sites):
                    site_below = lattice.sites[idx_below]
                    is_surface = site_below.species != SpeciesType.VACANT

            if is_surface:
                agents.append(ParticleAgent(idx, lattice))

    return agents
