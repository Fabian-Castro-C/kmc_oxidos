"""
Physics-based reward calculator for agent-based training.

Calculates reward weights from first principles using:
- Formation energies from DFT
- Surface energies
- Thermal energy scale kB*T
- Bond energies

All rewards normalized by kB*T at reference temperature.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.tio2_parameters import TiO2Parameters, get_tio2_parameters


@dataclass
class PhysicsBasedRewards:
    """
    Physics-based reward weights.

    All weights are in units of kB*T (thermal energy scale).
    This ensures rewards are physically meaningful and temperature-aware.
    """

    coverage_weight: float
    roughness_weight: float
    stoichiometry_weight: float
    ess_weight: float
    temperature: float

    @classmethod
    def from_physics(
        cls,
        temperature: float = 600.0,
        params: TiO2Parameters | None = None,
    ) -> PhysicsBasedRewards:
        """
        Calculate reward weights from physical parameters.

        Args:
            temperature: Temperature in Kelvin
            params: TiO2 physical parameters

        Returns:
            PhysicsBasedRewards instance with calculated weights
        """
        if params is None:
            params = get_tio2_parameters()

        # Thermal energy at this temperature (eV)
        kT = params.k_boltzmann * temperature

        # 1. COVERAGE REWARD
        # Reward for depositing atoms = formation energy / kT
        # TiO2 formation: Ti + 2O -> TiO2, ΔH = -9.7 eV
        # Per atom deposited: ΔH/3 ≈ -3.23 eV (exothermic, favorable)
        # Reward should be positive and large to encourage growth
        formation_energy_per_atom = abs(params.formation_energy_tio2) / 3.0  # eV/atom
        coverage_weight = formation_energy_per_atom / kT
        # At 600K: 3.23 / 0.052 ≈ 62 (strong reward for growth)

        # 2. ROUGHNESS PENALTY
        # Penalty for rough surfaces = surface energy excess / kT
        # Rough surface has ~2x more dangling bonds than flat
        # Energy cost ≈ bond_energy_ti_o / 2 per rough site
        surface_energy_penalty = abs(params.bond_energy_ti_o) / 2.0  # eV
        roughness_weight = -(surface_energy_penalty / kT)
        # At 600K: -2.25 / 0.052 ≈ -43 (moderate penalty)

        # 3. STOICHIOMETRY PENALTY
        # Penalty for wrong Ti:O ratio = bond formation deficit
        # Missing O means Ti with dangling bond = ~bond_energy_ti_o
        stoich_energy_cost = abs(params.bond_energy_ti_o)  # eV
        stoichiometry_weight = -(stoich_energy_cost / kT)
        # At 600K: -4.5 / 0.052 ≈ -87 (strong penalty for wrong ratio)

        # 4. ESS PENALTY
        # Small penalty for low ESS (concentrated policy)
        # This is not directly physical, but helps exploration
        # Scale: ~10% of coverage reward to not dominate
        ess_weight = -coverage_weight * 0.1

        return cls(
            coverage_weight=coverage_weight,
            roughness_weight=roughness_weight,
            stoichiometry_weight=stoichiometry_weight,
            ess_weight=ess_weight,
            temperature=temperature,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for YAML config."""
        return {
            "coverage_weight": self.coverage_weight,
            "roughness_weight": self.roughness_weight,
            "stoichiometry_weight": self.stoichiometry_weight,
            "ess_weight": self.ess_weight,
        }

    def summary(self) -> str:
        """Print summary of physics-based rewards."""
        return f"""
Physics-Based Reward Weights (T={self.temperature:.0f}K)
=========================================
Coverage reward:      {self.coverage_weight:+8.2f}  (formation energy / kT)
Roughness penalty:    {self.roughness_weight:+8.2f}  (surface energy / kT)
Stoichiometry penalty:{self.stoichiometry_weight:+8.2f}  (bond energy / kT)
ESS penalty:          {self.ess_weight:+8.2f}  (exploration term)

Physical interpretation:
- Coverage: Reward = {abs(self.coverage_weight):.1f} kT per atom deposited
- Roughness: Penalty = {abs(self.roughness_weight):.1f} kT per rough site
- Stoichiometry: Penalty = {abs(self.stoichiometry_weight):.1f} kT per wrong ratio
- ESS: Penalty = {abs(self.ess_weight):.1f} kT for concentrated policy
"""


def calculate_rewards_for_temperature_range(
    T_min: float = 300.0,
    T_max: float = 900.0,
    n_points: int = 5,
) -> list[PhysicsBasedRewards]:
    """
    Calculate physics-based rewards across temperature range.

    Args:
        T_min: Minimum temperature (K)
        T_max: Maximum temperature (K)
        n_points: Number of temperature points

    Returns:
        List of PhysicsBasedRewards at different temperatures
    """
    import numpy as np

    temperatures = np.linspace(T_min, T_max, n_points)
    return [PhysicsBasedRewards.from_physics(T) for T in temperatures]


if __name__ == "__main__":
    # Example: Calculate rewards for different temperatures
    print("Physics-Based Reward Calculation")
    print("=" * 50)

    for T in [300, 600, 900]:
        rewards = PhysicsBasedRewards.from_physics(temperature=T)
        print(rewards.summary())
        print()
