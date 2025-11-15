"""
Rate calculator for RL actions.

Maps ActionType (RL policy actions) to physical event rates using the
existing KMC RateCalculator infrastructure. This enables importance sampling
reweighting in the SwarmThinkers framework: P(a) = π_θ(a)·Γ_a / Σ[π_θ(a')·Γ_a'].

Physical rates come from Arrhenius equation: Γ = Γ₀ · exp(-Ea / kB·T)
Policy logits come from neural network: π_θ(a) ∝ exp(logit_a)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.tio2_parameters import get_tio2_parameters
from src.kmc.lattice import SpeciesType
from src.kmc.rates import RateCalculator
from src.rl.particle_agent import ActionType

if TYPE_CHECKING:
    from src.kmc.lattice import Lattice
    from src.rl.particle_agent import ParticleAgent


class ActionRateCalculator:
    """
    Calculate physical rates for RL actions.

    Bridges the gap between policy actions (ActionType) and physical event rates (Hz).
    Uses the existing KMC RateCalculator to ensure thermodynamic consistency.

    Attributes:
        rate_calculator: KMC rate calculator with temperature and physical parameters
        params: TiO2 physical parameters (activation energies, attempt frequencies)
    """

    def __init__(
        self,
        temperature: float,
        deposition_rate: float = 0.1,  # ML/s (monolayers per second)
    ):
        """
        Initialize action rate calculator.

        Args:
            temperature: System temperature in Kelvin
            deposition_rate: Deposition flux in ML/s (for adsorption rates)
        """
        self.params = get_tio2_parameters()
        self.rate_calculator = RateCalculator(
            temperature=temperature,
            deposition_rate=deposition_rate,
            params=self.params,
        )

    def calculate_action_rate(
        self,
        agent: ParticleAgent,
        action: ActionType,
        lattice: Lattice,
    ) -> float:
        """
        Calculate physical rate for a given agent action.

        Maps RL actions to KMC event rates:
        - DIFFUSE_* → diffusion rate (with ES barrier for step-down)
        - ADSORB_TI/O → adsorption rate (deposition-limited)
        - DESORB → desorption rate (high activation barrier)
        - REACT_TIO2 → reaction rate (Ti + 2O → TiO₂)

        Args:
            agent: Particle agent attempting the action
            action: Action type from agent's action space
            lattice: Current lattice state

        Returns:
            Rate in Hz (events per second). Returns 0.0 for invalid actions.

        Note:
            This method assumes the action is valid (use agent.get_valid_actions()
            to check first). Invalid actions return 0.0 rate.
        """
        _site = lattice.get_site_by_index(agent.site_idx)

        # DIFFUSION ACTIONS (6 directions)
        if action in [
            ActionType.DIFFUSE_X_POS,
            ActionType.DIFFUSE_X_NEG,
            ActionType.DIFFUSE_Y_POS,
            ActionType.DIFFUSE_Y_NEG,
            ActionType.DIFFUSE_Z_POS,
            ActionType.DIFFUSE_Z_NEG,
        ]:
            return self._calculate_diffusion_rate(agent, action, lattice)

        # ADSORPTION ACTIONS (Ti or O onto vacant site)
        elif action in [ActionType.ADSORB_TI, ActionType.ADSORB_O]:
            return self._calculate_adsorption_rate(agent, action, lattice)

        # DESORPTION ACTION (Ti or O removal)
        elif action == ActionType.DESORB:
            return self._calculate_desorption_rate(agent, lattice)

        # REACTION ACTION (Ti + 2O → TiO₂)
        elif action == ActionType.REACT_TIO2:
            return self._calculate_reaction_rate(agent, lattice)

        else:
            # Unknown action
            return 0.0

    def _calculate_diffusion_rate(
        self,
        agent: ParticleAgent,
        action: ActionType,
        lattice: Lattice,
    ) -> float:
        """Calculate diffusion rate for directional move."""
        site = lattice.get_site_by_index(agent.site_idx)

        # Get target site index based on action direction
        direction_map = {
            ActionType.DIFFUSE_X_POS: 0,  # +X neighbor
            ActionType.DIFFUSE_X_NEG: 1,  # -X neighbor
            ActionType.DIFFUSE_Y_POS: 2,  # +Y neighbor
            ActionType.DIFFUSE_Y_NEG: 3,  # -Y neighbor
            ActionType.DIFFUSE_Z_POS: 4,  # +Z neighbor
            ActionType.DIFFUSE_Z_NEG: 5,  # -Z neighbor
        }

        neighbor_idx_in_list = direction_map.get(action)
        if neighbor_idx_in_list is None or neighbor_idx_in_list >= len(site.neighbors):
            return 0.0

        target_idx = site.neighbors[neighbor_idx_in_list]
        target_site = lattice.get_site_by_index(target_idx)

        # Check if target is vacant
        if target_site.species != SpeciesType.VACANT:
            return 0.0

        # Select activation energy based on species
        if site.species == SpeciesType.TI:
            ea = self.params.ea_diff_ti
            nu = self.params.attempt_frequency
        elif site.species == SpeciesType.O:
            ea = self.params.ea_diff_o
            nu = self.params.attempt_frequency
        else:
            return 0.0  # VACANT or SUBSTRATE can't diffuse

        # Use KMC rate calculator (includes ES barrier, coordination effects)
        rate = self.rate_calculator.calculate_diffusion_rate(
            site=site,
            target_site=target_site,
            activation_energy=ea,
            attempt_frequency=nu,
            lattice_sites=lattice.sites,
        )

        return rate

    def _calculate_adsorption_rate(
        self,
        agent: ParticleAgent,
        action: ActionType,
        lattice: Lattice,
    ) -> float:
        """Calculate adsorption rate for deposition."""
        site = lattice.get_site_by_index(agent.site_idx)

        # Only vacant sites can adsorb
        if site.species != SpeciesType.VACANT:
            return 0.0

        # Determine species being adsorbed
        if action == ActionType.ADSORB_TI:
            species = SpeciesType.TI
        elif action == ActionType.ADSORB_O:
            species = SpeciesType.O
        else:
            return 0.0

        # Use KMC rate calculator (includes sticking coefficient, coordination effects)
        rate = self.rate_calculator.calculate_adsorption_rate(
            site=site,
            species=species,
        )

        return rate

    def _calculate_desorption_rate(
        self,
        agent: ParticleAgent,
        lattice: Lattice,
    ) -> float:
        """Calculate desorption rate for atom removal."""
        site = lattice.get_site_by_index(agent.site_idx)

        # Select activation energy based on species
        if site.species == SpeciesType.TI:
            ea = self.params.ea_des_ti
            nu = self.params.attempt_frequency
        elif site.species == SpeciesType.O:
            ea = self.params.ea_des_o
            nu = self.params.attempt_frequency
        else:
            return 0.0  # VACANT or SUBSTRATE can't desorb

        # Use KMC rate calculator
        rate = self.rate_calculator.calculate_desorption_rate(
            activation_energy=ea,
            attempt_frequency=nu,
        )

        return rate

    def _calculate_reaction_rate(
        self,
        agent: ParticleAgent,
        lattice: Lattice,
    ) -> float:
        """Calculate Ti + 2O → TiO₂ reaction rate."""
        site = lattice.get_site_by_index(agent.site_idx)

        # Only Ti can initiate reaction
        if site.species != SpeciesType.TI:
            return 0.0

        # Find oxygen neighbors
        o_neighbors = []
        for neighbor_idx in site.neighbors:
            neighbor_site = lattice.get_site_by_index(neighbor_idx)
            if neighbor_site.species == SpeciesType.O:
                o_neighbors.append(neighbor_idx)

        # Need at least 2 oxygen neighbors
        if len(o_neighbors) < 2:
            return 0.0

        # Use KMC rate calculator (includes coordination effects, bonding checks)
        rate = self.rate_calculator.calculate_tio2_formation_rate(
            lattice=lattice,
            ti_site_index=agent.site_idx,
            o_site_indices=o_neighbors,
        )

        # calculate_tio2_formation_rate returns None if reaction is invalid
        return rate if rate is not None else 0.0

    def calculate_batch_rates(
        self,
        agents: list[ParticleAgent],
        actions: list[ActionType],
        lattice: Lattice,
    ) -> list[float]:
        """
        Calculate rates for multiple agent-action pairs.

        Efficient batch version for reweighting in SwarmCoordinator.

        Args:
            agents: List of N particle agents
            actions: List of N corresponding actions
            lattice: Current lattice state

        Returns:
            List of N rates in Hz
        """
        if len(agents) != len(actions):
            raise ValueError(
                f"agents and actions must have same length: "
                f"got {len(agents)} agents, {len(actions)} actions"
            )

        rates = []
        for agent, action in zip(agents, actions):
            rate = self.calculate_action_rate(agent, action, lattice)
            rates.append(rate)

        return rates
