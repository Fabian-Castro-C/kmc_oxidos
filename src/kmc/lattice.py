"""
Lattice structure for KMC simulation.

This module defines the lattice structure for TiO2 thin film growth simulation,
including site representation and neighbor connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator


class SpeciesType(Enum):
    """Enumeration of atomic species."""

    VACANT = 0
    TI = 1  # Titanium
    O = 2  # Oxygen  # noqa: E741
    SUBSTRATE = 3  # Substrate atoms (fixed)


@dataclass
class Site:
    """
    Represents a single lattice site.

    Attributes:
        position: 3D coordinates (x, y, z) of the site.
        species: Type of species occupying the site.
        coordination: Number of occupied nearest neighbors.
        energy: Local energy of the site (eV).
        neighbors: List of neighboring site indices.
        bonded_to: List of site indices that this site is chemically bonded to.
        is_in_oxide: Flag indicating if this atom is part of a formed TiO2 molecule.
        site_id: Unique identifier for this site (for tracking purposes only).
    """

    position: tuple[int, int, int]
    species: SpeciesType = SpeciesType.VACANT
    coordination: int = 0
    energy: float = 0.0
    neighbors: list[int] = field(default_factory=list)
    bonded_to: list[int] = field(default_factory=list)
    is_in_oxide: bool = False
    site_id: int | None = None

    def is_occupied(self) -> bool:
        """Check if site is occupied by an atom."""
        return self.species not in (SpeciesType.VACANT, SpeciesType.SUBSTRATE)

    def is_surface(self) -> bool:
        """Check if site is on the surface (has vacant neighbors above)."""
        return self.coordination < 6  # For simple cubic lattice

    def is_bonded(self) -> bool:
        """Check if site is chemically bonded to other atoms."""
        return len(self.bonded_to) > 0


class Lattice:
    """
    3D lattice structure for TiO2 thin film growth.

    This class manages the lattice geometry, site connectivity, and provides
    methods for accessing and modifying the lattice state.

    Attributes:
        size: Tuple of (nx, ny, nz) lattice dimensions.
        sites: 1D array of Site objects.
        surface_sites: Indices of sites on the growth surface.
    """

    def __init__(self, size: tuple[int, int, int], lattice_constant: float = 4.59) -> None:
        """
        Initialize the lattice.

        Args:
            size: Lattice dimensions (nx, ny, nz).
            lattice_constant: Lattice constant in Angstroms (default: TiO2 rutilo).
        """
        self.size = size
        self.nx, self.ny, self.nz = size
        self.lattice_constant = lattice_constant
        self.n_sites = self.nx * self.ny * self.nz

        # Initialize sites
        self.sites: list[Site] = []
        self._initialize_sites()
        self._build_neighbor_lists()

        # Track surface sites
        self.surface_sites: set[int] = set()
        self._update_surface_sites()

    def _initialize_sites(self) -> None:
        """Initialize all lattice sites with unique IDs."""
        site_id = 0
        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    position = (ix, iy, iz)
                    # Bottom layer is substrate
                    species = SpeciesType.SUBSTRATE if iz == 0 else SpeciesType.VACANT
                    site = Site(position=position, species=species, site_id=site_id)
                    self.sites.append(site)
                    site_id += 1

    def _build_neighbor_lists(self) -> None:
        """Build nearest neighbor lists for all sites (simple cubic)."""
        # Define neighbor offsets (6 nearest neighbors in cubic lattice)
        neighbor_offsets = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        for _idx, site in enumerate(self.sites):
            x, y, z = site.position
            for dx, dy, dz in neighbor_offsets:
                nx, ny, nz = x + dx, y + dy, z + dz

                # Periodic boundary conditions in x and y
                nx = nx % self.nx
                ny = ny % self.ny

                # No PBC in z (growth direction)
                if 0 <= nz < self.nz:
                    neighbor_idx = self._get_index(nx, ny, nz)
                    site.neighbors.append(neighbor_idx)

    def _get_index(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates to 1D index."""
        return x + y * self.nx + z * self.nx * self.ny

    def get_site(self, x: int, y: int, z: int) -> Site:
        """Get site at given coordinates."""
        idx = self._get_index(x, y, z)
        return self.sites[idx]

    def get_site_by_index(self, idx: int) -> Site:
        """Get site by 1D index."""
        return self.sites[idx]

    def _update_surface_sites(self) -> None:
        """Update the set of surface sites (vacant sites available for deposition)."""
        self.surface_sites.clear()
        for idx, site in enumerate(self.sites):
            _x, _y, z = site.position

            if site.species != SpeciesType.VACANT:
                continue

            if z == 0:
                continue

            has_support = any(
                self.sites[n_idx].species not in (SpeciesType.VACANT, None)
                and self.sites[n_idx].position[2] <= z
                for n_idx in site.neighbors
            )

            if has_support:
                self.surface_sites.add(idx)

    def deposit_atom(self, site_idx: int, species: SpeciesType) -> None:
        """
        Deposit an atom at a given site.

        Args:
            site_idx: Index of the site.
            species: Species to deposit.
        """
        site = self.sites[site_idx]
        if site.species != SpeciesType.VACANT:
            raise ValueError(f"Site {site_idx} is already occupied")

        site.species = species
        self._update_coordination(site_idx)
        self._update_surface_sites()

    def remove_atom(self, site_idx: int) -> SpeciesType:
        """
        Remove an atom from a site (desorption).

        Args:
            site_idx: Index of the site.

        Returns:
            Species that was removed.
        """
        site = self.sites[site_idx]
        if not site.is_occupied():
            raise ValueError(f"Site {site_idx} is vacant")

        old_species = site.species
        site.species = SpeciesType.VACANT
        self._update_coordination(site_idx)
        self._update_surface_sites()

        return old_species

    def move_atom(self, from_idx: int, to_idx: int) -> None:
        """
        Move an atom from one site to another (diffusion).

        Args:
            from_idx: Source site index.
            to_idx: Destination site index.
        """
        from_site = self.sites[from_idx]
        to_site = self.sites[to_idx]

        if not from_site.is_occupied():
            raise ValueError(f"Source site {from_idx} is vacant")
        if to_site.is_occupied():
            raise ValueError(f"Destination site {to_idx} is occupied")

        to_site.species = from_site.species
        from_site.species = SpeciesType.VACANT

        self._update_coordination(from_idx)
        self._update_coordination(to_idx)
        self._update_surface_sites()

    def _update_coordination(self, site_idx: int) -> None:
        """Update coordination number for a site and its neighbors."""
        for idx in [site_idx] + self.sites[site_idx].neighbors:
            site = self.sites[idx]
            site.coordination = sum(
                1 for n_idx in site.neighbors if self.sites[n_idx].is_occupied()
            )

    def get_height_profile(self) -> npt.NDArray[np.float64]:
        """
        Calculate surface height profile.

        Returns:
            2D array of surface heights.
        """
        heights = np.zeros((self.nx, self.ny))
        for site in self.sites:
            if site.is_occupied():
                x, y, z = site.position
                heights[x, y] = max(heights[x, y], z)
        return heights

    def get_composition(self) -> dict[SpeciesType, int]:
        """
        Get composition of the lattice.

        Returns:
            Dictionary mapping species to counts.
        """
        composition = dict.fromkeys(SpeciesType, 0)
        for site in self.sites:
            composition[site.species] += 1
        return composition

    def get_composition_detailed(self) -> dict[str, int]:
        """
        Get detailed composition separating free atoms from bonded (oxide) atoms.

        Returns:
            Dictionary with counts:
            - 'ti_free': Ti atoms not in oxide
            - 'ti_oxide': Ti atoms in oxide (bonded)
            - 'o_free': O atoms not in oxide
            - 'o_oxide': O atoms in oxide (bonded)
            - 'vacant': Vacant sites
        """
        counts = {
            'ti_free': 0,
            'ti_oxide': 0,
            'o_free': 0,
            'o_oxide': 0,
            'vacant': 0,
        }

        for site in self.sites:
            if site.species == SpeciesType.TI:
                if site.is_in_oxide:
                    counts['ti_oxide'] += 1
                else:
                    counts['ti_free'] += 1
            elif site.species == SpeciesType.O:
                if site.is_in_oxide:
                    counts['o_oxide'] += 1
                else:
                    counts['o_free'] += 1
            elif site.species == SpeciesType.VACANT:
                counts['vacant'] += 1

        return counts

    def get_neighbor_species(self, site_idx: int) -> list[SpeciesType]:
        """
        Get list of species occupying neighboring sites.

        Args:
            site_idx: Index of the site.

        Returns:
            List of neighboring species.
        """
        site = self.sites[site_idx]
        return [self.sites[n_idx].species for n_idx in site.neighbors]

    def iter_occupied_sites(self) -> Iterator[tuple[int, Site]]:
        """Iterate over occupied sites."""
        for idx, site in enumerate(self.sites):
            if site.is_occupied():
                yield idx, site

    def create_bond(self, site_idx1: int, site_idx2: int) -> None:
        """
        Create a chemical bond between two sites.

        Args:
            site_idx1: Index of first site.
            site_idx2: Index of second site.
        """
        site1 = self.sites[site_idx1]
        site2 = self.sites[site_idx2]

        # Add bidirectional bond if not already present
        if site_idx2 not in site1.bonded_to:
            site1.bonded_to.append(site_idx2)
        if site_idx1 not in site2.bonded_to:
            site2.bonded_to.append(site_idx1)

    def break_bond(self, site_idx1: int, site_idx2: int) -> None:
        """
        Break a chemical bond between two sites.

        Args:
            site_idx1: Index of first site.
            site_idx2: Index of second site.
        """
        site1 = self.sites[site_idx1]
        site2 = self.sites[site_idx2]

        # Remove bidirectional bond
        if site_idx2 in site1.bonded_to:
            site1.bonded_to.remove(site_idx2)
        if site_idx1 in site2.bonded_to:
            site2.bonded_to.remove(site_idx1)

    def get_bonded_neighbors(self, site_idx: int, species_filter: SpeciesType | None = None) -> list[int]:
        """
        Get list of neighboring sites that could form bonds.

        Args:
            site_idx: Index of the site.
            species_filter: If provided, only return neighbors of this species.

        Returns:
            List of neighboring site indices.
        """
        site = self.sites[site_idx]
        candidates = []

        for neighbor_idx in site.neighbors:
            neighbor = self.sites[neighbor_idx]
            if neighbor.is_occupied() and (species_filter is None or neighbor.species == species_filter):
                candidates.append(neighbor_idx)

        return candidates

    def __repr__(self) -> str:
        """String representation."""
        comp = self.get_composition()
        return (
            f"Lattice(size={self.size}, "
            f"Ti={comp[SpeciesType.TI]}, "
            f"O={comp[SpeciesType.O]}, "
            f"Vacant={comp[SpeciesType.VACANT]})"
        )
