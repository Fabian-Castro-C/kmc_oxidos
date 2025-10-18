"""
Rate calculations for KMC events using Arrhenius equation.

This module handles the calculation of event rates based on activation energies
and temperature using the Arrhenius equation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .lattice import SpeciesType

if TYPE_CHECKING:
    from .lattice import Lattice, Site


class ArrheniusRate:
    """
    Calculate event rates using Arrhenius equation.

    The rate is given by: Γ = Γ₀ * exp(-Ea / (kB * T))

    Attributes:
        attempt_frequency: Pre-exponential factor Γ₀ (Hz).
        activation_energy: Activation energy Ea (eV).
        temperature: Temperature T (K).
        k_boltzmann: Boltzmann constant (eV/K).
    """

    def __init__(
        self,
        attempt_frequency: float,
        activation_energy: float,
        temperature: float,
        k_boltzmann: float = 8.617333e-5,
    ) -> None:
        """
        Initialize Arrhenius rate calculator.

        Args:
            attempt_frequency: Pre-exponential factor (Hz), typically 10^12 - 10^14.
            activation_energy: Activation energy (eV).
            temperature: Temperature (K).
            k_boltzmann: Boltzmann constant (eV/K).
        """
        self.attempt_frequency = attempt_frequency
        self.activation_energy = activation_energy
        self.temperature = temperature
        self.k_boltzmann = k_boltzmann

    def calculate_rate(self) -> float:
        """
        Calculate the event rate.

        Returns:
            Event rate in Hz.
        """
        exponent = -self.activation_energy / (self.k_boltzmann * self.temperature)
        return self.attempt_frequency * math.exp(exponent)

    def __repr__(self) -> str:
        """String representation."""
        rate = self.calculate_rate()
        return (
            f"ArrheniusRate(Ea={self.activation_energy:.3f} eV, "
            f"T={self.temperature:.1f} K, rate={rate:.2e} Hz)"
        )


class RateCalculator:
    """
    Calculate rates for all types of events in the KMC simulation.

    This class encapsulates the logic for calculating event rates based on
    local environment and system parameters.
    """

    def __init__(
        self,
        temperature: float,
        deposition_rate: float,
        k_boltzmann: float = 8.617333e-5,
    ) -> None:
        """
        Initialize rate calculator.

        Args:
            temperature: System temperature (K).
            deposition_rate: Deposition rate (ML/s).
            k_boltzmann: Boltzmann constant (eV/K).
        """
        self.temperature = temperature
        self.deposition_rate = deposition_rate
        self.k_boltzmann = k_boltzmann

    def calculate_adsorption_rate(self, site: Site) -> float:
        """
        Calculate adsorption rate.

        The adsorption rate is proportional to the deposition flux and the
        availability of the site.

        Args:
            site: Target site for adsorption.

        Returns:
            Adsorption rate (Hz).
        """
        # Base rate proportional to deposition flux
        # Adjust based on site coordination (sticking probability)
        sticking_coefficient = 1.0 - (site.coordination / 6.0) * 0.5
        return self.deposition_rate * sticking_coefficient

    def calculate_diffusion_rate(
        self,
        site: Site,
        target_site: Site,
        activation_energy: float,
        attempt_frequency: float = 1e13,
    ) -> float:
        """
        Calculate diffusion rate using Arrhenius equation.

        The activation energy may depend on local coordination.

        Args:
            site: Source site.
            target_site: Destination site.
            activation_energy: Base activation energy (eV).
            attempt_frequency: Attempt frequency (Hz).

        Returns:
            Diffusion rate (Hz).
        """
        # Lower coordination → easier diffusion
        coordination_factor = site.coordination / 6.0
        # Target site coordination affects barrier (higher target coordination = harder to hop into)
        target_factor = target_site.coordination / 6.0
        effective_ea = activation_energy * (0.7 + 0.3 * coordination_factor + 0.1 * target_factor)

        arrhenius = ArrheniusRate(
            attempt_frequency=attempt_frequency,
            activation_energy=effective_ea,
            temperature=self.temperature,
            k_boltzmann=self.k_boltzmann,
        )
        return arrhenius.calculate_rate()

    def calculate_reaction_rate(
        self,
        activation_energy: float,
        attempt_frequency: float = 1e13,
    ) -> float:
        """
        Calculate reaction rate (e.g., Ti + 2O → TiO2).

        Args:
            activation_energy: Activation energy (eV).
            attempt_frequency: Attempt frequency (Hz).

        Returns:
            Reaction rate (Hz).
        """
        arrhenius = ArrheniusRate(
            attempt_frequency=attempt_frequency,
            activation_energy=activation_energy,
            temperature=self.temperature,
            k_boltzmann=self.k_boltzmann,
        )
        return arrhenius.calculate_rate()

    def calculate_desorption_rate(
        self,
        activation_energy: float,
        attempt_frequency: float = 1e13,
    ) -> float:
        """
        Calculate desorption rate.

        Desorption energy typically depends strongly on coordination.

        Args:
            activation_energy: Base activation energy (eV).
            attempt_frequency: Attempt frequency (Hz).

        Returns:
            Desorption rate (Hz).
        """
        # Base desorption rate calculation
        effective_ea = activation_energy

        arrhenius = ArrheniusRate(
            attempt_frequency=attempt_frequency,
            activation_energy=effective_ea,
            temperature=self.temperature,
            k_boltzmann=self.k_boltzmann,
        )
        return arrhenius.calculate_rate()

    def calculate_energy_change(
        self, lattice: Lattice, site_index: int, target_index: int | None = None
    ) -> float:
        """
        Calculate energy change for an event.

        This is a simplified energy model based on nearest-neighbor bonds.

        Args:
            lattice: Lattice object.
            site_index: Primary site index.
            target_index: Secondary site index (for diffusion).

        Returns:
            Energy change in eV (negative means energy decrease).
        """
        # Simplified bond-counting model
        # Bond energies (eV): Ti-O: -4.5, Ti-Ti: -2.0, O-O: -1.5

        # Calculate initial energy
        initial_energy = self._calculate_site_energy(lattice, site_index)

        if target_index is not None:
            # For diffusion events
            target_energy = self._calculate_site_energy(lattice, target_index)
            # Approximate energy change
            return -(initial_energy - target_energy) * 0.5
        else:
            # For adsorption/desorption/reaction
            return -initial_energy * 0.1  # Simplified

    def _calculate_site_energy(self, lattice: Lattice, site_index: int) -> float:
        """Calculate local energy at a site based on neighbors."""
        site = lattice.get_site_by_index(site_index)
        neighbors = lattice.get_neighbor_species(site_index)

        energy = 0.0
        for neighbor_species in neighbors:
            if site.species == SpeciesType.TI:
                if neighbor_species == SpeciesType.O:
                    energy += -4.5  # Ti-O bond
                elif neighbor_species == SpeciesType.TI:
                    energy += -2.0  # Ti-Ti bond
            elif site.species == SpeciesType.O:
                if neighbor_species == SpeciesType.TI:
                    energy += -4.5  # O-Ti bond
                elif neighbor_species == SpeciesType.O:
                    energy += -1.5  # O-O bond

        return energy
