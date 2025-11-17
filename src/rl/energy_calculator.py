"""
System energy calculator for SwarmThinkers reward - OPEN SYSTEM.

Implements physics-grounded reward for ballistic deposition:
    r_t = -ΔΩ_t where Ω = E - µN (grand canonical ensemble)

For OPEN SYSTEM with particle reservoir:
- Adsorption: ΔΩ = ΔE_bonds - µ (cost to bring atom from reservoir)
- Diffusion: ΔΩ = ΔE_barrier (no change in N)
- Desorption: ΔΩ = -ΔE_bonds + µ (return atom to reservoir)

Energy includes:
- Bond energies (Ti-O, Ti-Ti, O-O)
- Chemical potential µ (reference: isolated gas-phase atoms)
- Local coordination environment (not global surface count)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.tio2_parameters import TiO2Parameters, get_tio2_parameters
from src.kmc.lattice import SpeciesType

if TYPE_CHECKING:
    from src.kmc.lattice import Lattice


class SystemEnergyCalculator:
    """
    Calculate grand potential change for open system (ballistic deposition).

    Following grand canonical ensemble: Ω = E - µN
    Reward: r_t = -ΔΩ (favor thermodynamically favorable events)

    For deposition from reservoir:
    - Chemical potential µ ≈ 0 (reference: isolated gas atoms)
    - Adsorption: ΔΩ = ΔE_local - µ ≈ ΔE_local (energy gain from bonding)
    - Diffusion: ΔΩ = E_barrier (activation energy, N unchanged)
    """

    def __init__(self, params: TiO2Parameters | None = None):
        """
        Initialize energy calculator.

        Args:
            params: TiO2 physical parameters
        """
        if params is None:
            params = get_tio2_parameters()
        self.params = params

        # Chemical potentials: Realistic adsorption energies on a clean surface.
        # This represents the energy cost/gain of bringing an atom from the
        # gas phase (reservoir) to the surface, before it forms bonds.
        # Values are now read from the parameters object.
        self.mu_ti = params.mu_ti
        self.mu_o = params.mu_o

        # Bond energy lookup (from DFT)
        self.bond_energies = {
            (SpeciesType.TI, SpeciesType.O): params.bond_energy_ti_o,  # -4.5 eV
            (SpeciesType.O, SpeciesType.TI): params.bond_energy_ti_o,  # -4.5 eV
            (SpeciesType.TI, SpeciesType.TI): params.bond_energy_ti_ti,  # -2.0 eV
            (SpeciesType.O, SpeciesType.O): params.bond_energy_o_o,  # -1.5 eV
        }

    def calculate_local_energy(self, lattice: Lattice, site_idx: int) -> float:
        """
        Calculate LOCAL energy of a site (bonds with neighbors + substrate interaction).

        For open system, we care about LOCAL energy changes, not global.
        Atoms at z=0 (in contact with substrate) receive additional stabilization.

        Args:
            lattice: Current lattice
            site_idx: Index of site to calculate energy for

        Returns:
            Local energy (sum of bonds to neighbors + substrate adsorption) in eV
        """
        if site_idx >= len(lattice.sites):
            return 0.0

        site = lattice.sites[site_idx]
        if not site.is_occupied():
            return 0.0

        local_energy = 0.0
        species_i = site.species

        # Sum bond energies with occupied neighbors
        for neighbor_idx in site.neighbors:
            if neighbor_idx >= len(lattice.sites):
                continue

            neighbor = lattice.sites[neighbor_idx]
            if not neighbor.is_occupied():
                continue

            species_j = neighbor.species
            bond_type = (species_i, species_j)

            if bond_type in self.bond_energies:
                # Divide by 2 to avoid double-counting (each bond shared by 2 sites)
                local_energy += self.bond_energies[bond_type] / 2.0

        # Add substrate adsorption energy for atoms at z=0 (substrate layer)
        if site.position[2] == 0:
            if species_i == SpeciesType.TI:
                local_energy += self.params.substrate_ads_ti
            elif species_i == SpeciesType.O:
                local_energy += self.params.substrate_ads_o

        return local_energy

    def calculate_system_energy(self, lattice: Lattice) -> float:
        """
        Calculate total system energy for compatibility.

        NOTE: For open system, LOCAL energy changes are more relevant.
        This sums all local energies (each bond counted once via /2).

        Args:
            lattice: Current lattice state

        Returns:
            Total bond energy (eV)
        """
        total_energy = 0.0

        # Sum local energies (bonds already divided by 2 in calculate_local_energy)
        for site_idx in range(len(lattice.sites)):
            total_energy += self.calculate_local_energy(lattice, site_idx)

        return total_energy

    def calculate_grand_potential(self, lattice: Lattice) -> float:
        """
        Calculate the Grand Potential Ω = E - µN.

        This is the key thermodynamic potential for an open system.
        - E: Total bond energy of the system.
        - µN: Sum of chemical potentials for all atoms present.

        Args:
            lattice: Current lattice state.

        Returns:
            The grand potential Ω (eV).
        """
        # E: Total bond energy
        total_bond_energy = self.calculate_system_energy(lattice)

        # N: Number of particles of each species
        # Use cached species counts from lattice if available (O(1) instead of O(N))
        if hasattr(lattice, '_species_counts_cache'):
            n_ti = lattice._species_counts_cache.get(SpeciesType.TI, 0)
            n_o = lattice._species_counts_cache.get(SpeciesType.O, 0)
        else:
            # Fallback: O(N) iteration (only if cache not available)
            n_ti = sum(1 for site in lattice.sites if site.species == SpeciesType.TI)
            n_o = sum(1 for site in lattice.sites if site.species == SpeciesType.O)

        # µN term
        mu_n_term = self.mu_ti * n_ti + self.mu_o * n_o

        # Grand Potential Ω = E - µN
        return total_bond_energy - mu_n_term

    def calculate_grand_potential_change(self, lattice_before: Lattice, lattice_after: Lattice) -> float:
        """
        Calculate the change in grand potential: ΔΩ = Ω_after - Ω_before.

        Args:
            lattice_before: Lattice state before event.
            lattice_after: Lattice state after event.

        Returns:
            The change in grand potential ΔΩ (eV).
        """
        omega_before = self.calculate_grand_potential(lattice_before)
        omega_after = self.calculate_grand_potential(lattice_after)
        return omega_after - omega_before


def calculate_swarmthinkers_reward(
    lattice_before: Lattice,
    lattice_after: Lattice,
    params: TiO2Parameters | None = None,
) -> float:
    """
    Calculate SwarmThinkers reward for an open system: r_t = -ΔΩ.

    A negative change in grand potential (system becomes more stable)
    results in a positive reward.

    Args:
        lattice_before: State before event.
        lattice_after: State after event.
        params: TiO2 parameters.

    Returns:
        Reward r_t = -ΔΩ (eV).
    """
    calculator = SystemEnergyCalculator(params)
    delta_omega = calculator.calculate_grand_potential_change(lattice_before, lattice_after)
    return -delta_omega  # Reward is the negative change in grand potential
