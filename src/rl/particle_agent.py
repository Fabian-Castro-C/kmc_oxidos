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
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .action_space import ActionType

if TYPE_CHECKING:
    from src.kmc.lattice import Lattice, SpeciesType


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
        Optimized with numpy vectorization.
        """
        n_species = 3
        # Pre-allocate the final vector
        vector = np.zeros(58, dtype=np.float32)

        # Vectorized one-hot encoding for 1st neighbors
        # 6 neighbors, each can be one of 3 species. Total 18 dimensions.
        valid_neighbors_1st_mask = self.neighbors_1st < n_species
        indices_1st = np.arange(6)[valid_neighbors_1st_mask] * n_species + self.neighbors_1st[valid_neighbors_1st_mask]
        vector[indices_1st] = 1.0

        # Vectorized one-hot encoding for 2nd neighbors
        # 12 neighbors, each can be one of 3 species. Total 36 dimensions.
        # Offset by 18 (size of 1st neighbors part)
        valid_neighbors_2nd_mask = self.neighbors_2nd < n_species
        indices_2nd = 18 + np.arange(12)[valid_neighbors_2nd_mask] * n_species + self.neighbors_2nd[valid_neighbors_2nd_mask]
        vector[indices_2nd] = 1.0

        # One-hot encode own species
        # Offset by 18 + 36 = 54
        if 0 <= self.own_species < n_species:
            vector[54 + self.own_species] = 1.0

        # Normalized height
        vector[57] = self.height / 20.0

        return vector


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
        Get species of 1st nearest neighbors. Vectorized with numpy.

        Uses the neighbor list from the Site object.

        Returns:
            Array of species IDs (padded to 6 for consistency).
        """
        # Pre-allocate result array with padding (VACANT = 0)
        result = np.zeros(6, dtype=np.int8)

        # Get neighbor indices as array
        neighbor_indices = np.array(self.site.neighbors, dtype=np.int32)
        n_neighbors = len(neighbor_indices)

        if n_neighbors > 0:
            # Vectorized species lookup
            neighbor_species = np.array(
                [self.lattice.sites[idx].species.value for idx in neighbor_indices[:6]],
                dtype=np.int8
            )
            result[:len(neighbor_species)] = neighbor_species

        return result

    def _get_2nd_neighbors(self) -> npt.NDArray[np.int8]:
        """
        Get species of 2nd nearest neighbors. Vectorized with numpy.

        For simplicity, use a geometric approach with position offsets.

        Returns:
            Array of 12 species IDs.
        """
        x, y, z = self.position
        nx, ny, nz = self.lattice.size

        # 2nd neighbors: edge-sharing (12 directions) - pre-computed as numpy array
        offsets = np.array([
            # Along x-y plane
            [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
            # Along x-z plane
            [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
            # Along y-z plane
            [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        ], dtype=np.int16)

        # Vectorized position calculation
        pos = np.array([x, y, z], dtype=np.int16)
        neighbor_positions = pos + offsets

        # Apply periodic boundary conditions for x and y
        neighbor_positions[:, 0] = neighbor_positions[:, 0] % nx
        neighbor_positions[:, 1] = neighbor_positions[:, 1] % ny

        # Check z boundaries (no PBC in z)
        valid_z = (neighbor_positions[:, 2] >= 0) & (neighbor_positions[:, 2] < nz)

        # Calculate indices
        species_list = np.zeros(12, dtype=np.int8)
        valid_positions = neighbor_positions[valid_z]

        if len(valid_positions) > 0:
            indices = (
                valid_positions[:, 0] +
                valid_positions[:, 1] * nx +
                valid_positions[:, 2] * nx * ny
            )
            # Vectorized species lookup for valid positions
            for i, idx in enumerate(indices):
                if 0 <= idx < len(self.lattice.sites):
                    species_list[np.where(valid_z)[0][i]] = self.lattice.sites[idx].species.value

        return species_list

    def get_valid_actions(self) -> list[ActionType]:
        """
        Get list of valid actions for this agent based on its species.

        Returns:
            List of valid ActionType values.
        """
        from src.kmc.lattice import SpeciesType

        if self.species == SpeciesType.VACANT:
            # Vacant sites cannot be agents
            return []
        elif self.species == SpeciesType.TI or self.species == SpeciesType.O:
            # Atoms can diffuse or desorb
            actions = [
                ActionType.DIFFUSE_X_POS,
                ActionType.DIFFUSE_X_NEG,
                ActionType.DIFFUSE_Y_POS,
                ActionType.DIFFUSE_Y_NEG,
                ActionType.DIFFUSE_Z_POS,
                ActionType.DIFFUSE_Z_NEG,
                ActionType.DESORB,
            ]
            return actions
        
        return []

    def get_neighbors(self, lattice_size: tuple[int, int, int]) -> dict[ActionType, int]:
        """
        Gets a dictionary mapping diffusion actions to neighbor site indices.

        Args:
            lattice_size: The (nx, ny, nz) dimensions of the lattice.

        Returns:
            A dictionary where keys are diffusion ActionTypes and values are
            the corresponding neighbor site indices.
        """
        neighbors = {}
        nx, ny, _ = lattice_size
        x, y, z = self.position

        # Map action to position change
        action_offsets = {
            ActionType.DIFFUSE_X_POS: (1, 0, 0),
            ActionType.DIFFUSE_X_NEG: (-1, 0, 0),
            ActionType.DIFFUSE_Y_POS: (0, 1, 0),
            ActionType.DIFFUSE_Y_NEG: (0, -1, 0),
            ActionType.DIFFUSE_Z_POS: (0, 0, 1),
            ActionType.DIFFUSE_Z_NEG: (0, 0, -1),
        }

        for action, (dx, dy, dz) in action_offsets.items():
            nx_pos, ny_pos, nz_pos = x + dx, y + dy, z + dz

            # Apply periodic boundary conditions in x and y (but not z)
            nx_pos = nx_pos % nx
            ny_pos = ny_pos % ny

            # Check z boundary only (no periodic boundary in z)
            if 0 <= nz_pos < lattice_size[2]:
                neighbor_idx = nx_pos + ny_pos * nx + nz_pos * nx * ny
                neighbors[action] = neighbor_idx

        return neighbors

    def get_neighbor_site(
        self, action: ActionType, lattice_size: tuple[int, int, int]
    ) -> int | None:
        """
        Gets the site index for a specific diffusion action.

        Args:
            action: The diffusion action to perform.
            lattice_size: The (nx, ny, nz) dimensions of the lattice.

        Returns:
            The neighbor site index, or None if the action is invalid or out of bounds.
        """
        neighbors = self.get_neighbors(lattice_size)
        return neighbors.get(action, None)

    def __repr__(self) -> str:
        """String representation."""
        return f"ParticleAgent(pos={self.position}, species={self.species.name})"


def create_agents_from_lattice(lattice: Lattice) -> list[ParticleAgent]:
    """
    Create a list of agents from the current lattice state.

    Agents are defined as all occupied sites on the surface (top-most atoms).
    With the new deposition model, vacant sites do not have agents since
    deposition is a global event, not an agent decision.

    This uses the optimized `get_surface_sites` method from the Lattice class.
    """
    agents = []
    nx, ny, _ = lattice.size

    # A helper function to calculate index from position
    def get_site_index(pos: tuple[int, int, int]) -> int:
        x, y, z = pos
        return x + y * nx + z * nx * ny

    # Add agents only for occupied surface sites
    surface_sites = lattice.get_surface_sites()
    for site in surface_sites:
        if site.is_occupied():
            site_idx = get_site_index(site.position)
            agents.append(ParticleAgent(site_idx=site_idx, lattice=lattice))

    return agents
