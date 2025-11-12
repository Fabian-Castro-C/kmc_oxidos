"""
Physical parameters for TiO2 rutile (110) surface.

This module contains experimentally and theoretically derived parameters
for KMC simulation of TiO2 thin film growth.

References:
    - DFT calculations for TiO2 surfaces
    - Experimental data from literature
    - Materials Project database
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class TiO2Parameters:
    """
    Physical parameters for TiO2 rutile (110).

    All energies in eV, frequencies in Hz, distances in Angstroms.

    Attributes:
        lattice_constant_a: Lattice parameter a (Angstrom).
        lattice_constant_c: Lattice parameter c (Angstrom).
        attempt_frequency: Typical attempt frequency (Hz).

        # Activation energies for diffusion
        ea_diff_ti: Activation energy for Ti diffusion (eV).
        ea_diff_o: Activation energy for O diffusion (eV).

        # Activation energies for desorption
        ea_des_ti: Activation energy for Ti desorption (eV).
        ea_des_o: Activation energy for O desorption (eV).

        # Reaction energies
        ea_reaction_tio2: Activation energy for TiO2 formation (eV).

        # Bond energies
        bond_energy_ti_o: Ti-O bond energy (eV).
        bond_energy_ti_ti: Ti-Ti bond energy (eV).
        bond_energy_o_o: O-O bond energy (eV).

        # Formation energies
        formation_energy_tio2: Formation energy of bulk TiO2 (eV/formula unit).
    """

    # Lattice parameters (rutile structure)
    lattice_constant_a: float = 4.59  # Angstrom
    lattice_constant_c: float = 2.96  # Angstrom

    # Attempt frequencies (typical phonon frequencies)
    attempt_frequency: float = 1e13  # Hz
    nu_reaction: float = 1e13  # Hz (reaction attempt frequency)

    # Diffusion barriers
    # Note: These are approximate values. Actual values depend on:
    # - Surface structure
    # - Local coordination
    # - Charge state
    # - Coverage
    ea_diff_ti: float = 0.6  # eV (typical range: 0.4-0.8 eV)
    ea_diff_o: float = 0.8  # eV (typically higher than Ti)

    # Desorption barriers
    # Higher than diffusion, strongly coordination-dependent
    ea_des_ti: float = 2.0  # eV (typical range: 1.5-2.5 eV)
    ea_des_o: float = 2.5  # eV (O binds more strongly)

    # Reaction barriers
    ea_reaction_tio2: float = 0.3  # eV (formation is typically facile)

    # Bond energies (from DFT or empirical potentials)
    bond_energy_ti_o: float = -4.5  # eV (strong ionic bond)
    bond_energy_ti_ti: float = -2.0  # eV (metallic bond)
    bond_energy_o_o: float = -1.5  # eV (weak interaction)

    # Formation energies
    formation_energy_tio2: float = -9.7  # eV per formula unit (from Materials Project)

    # Physical constants
    k_boltzmann: ClassVar[float] = 8.617333e-5  # eV/K


def get_tio2_parameters() -> TiO2Parameters:
    """
    Get default TiO2 parameters.

    Returns:
        TiO2Parameters instance with default values.
    """
    return TiO2Parameters()


# Parameter sets for different conditions or surfaces
RUTILE_110_PARAMETERS = TiO2Parameters()

RUTILE_100_PARAMETERS = TiO2Parameters(
    ea_diff_ti=0.7,  # Different surface -> different barriers
    ea_diff_o=0.9,
    ea_des_ti=2.2,
    ea_des_o=2.7,
)

ANATASE_101_PARAMETERS = TiO2Parameters(
    lattice_constant_a=3.78,
    lattice_constant_c=9.51,
    ea_diff_ti=0.5,  # Anatase typically has lower barriers
    ea_diff_o=0.7,
    ea_des_ti=1.8,
    ea_des_o=2.3,
    formation_energy_tio2=-9.5,  # Slightly less stable than rutile
)


def get_parameters_for_surface(surface: str = "rutile_110") -> TiO2Parameters:
    """
    Get parameters for a specific TiO2 surface.

    Args:
        surface: Surface type. Options: 'rutile_110', 'rutile_100', 'anatase_101'.

    Returns:
        TiO2Parameters for the specified surface.

    Raises:
        ValueError: If surface type is not recognized.
    """
    surface_params = {
        "rutile_110": RUTILE_110_PARAMETERS,
        "rutile_100": RUTILE_100_PARAMETERS,
        "anatase_101": ANATASE_101_PARAMETERS,
    }

    if surface not in surface_params:
        raise ValueError(f"Unknown surface: {surface}. Available: {list(surface_params.keys())}")

    return surface_params[surface]


# Temperature-dependent parameters
def get_effective_diffusion_barrier(
    base_barrier: float, coordination: int, max_coordination: int = 6
) -> float:
    """
    Calculate effective diffusion barrier based on local coordination.

    Lower coordination -> easier diffusion.

    Args:
        base_barrier: Base activation energy (eV).
        coordination: Number of nearest neighbors.
        max_coordination: Maximum possible coordination.

    Returns:
        Effective activation energy (eV).
    """
    coordination_factor = coordination / max_coordination
    # Barrier decreases with lower coordination
    return base_barrier * (0.5 + 0.5 * coordination_factor)


def get_effective_desorption_barrier(
    base_barrier: float, coordination: int, max_coordination: int = 6
) -> float:
    """
    Calculate effective desorption barrier based on local coordination.

    Higher coordination -> harder to desorb.

    Args:
        base_barrier: Base activation energy (eV).
        coordination: Number of nearest neighbors.
        max_coordination: Maximum possible coordination.

    Returns:
        Effective activation energy (eV).
    """
    coordination_factor = coordination / max_coordination
    # Barrier increases with higher coordination
    return base_barrier * (1.0 + 0.5 * coordination_factor)
