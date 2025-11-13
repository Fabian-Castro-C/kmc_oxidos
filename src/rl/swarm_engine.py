"""
Swarm engine for intelligent event selection in KMC.

This module implements the core SwarmThinkers algorithm: generating proposals
from a learned policy, computing physical rates, reweighting, and selecting
events with importance sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from ..kmc.lattice import Lattice
    from ..kmc.rates import RateCalculator
    from .swarm_policy import DiffusionSwarmPolicy

from ..kmc.events import Event, EventType
from ..kmc.lattice import SpeciesType
from .observations import get_batch_observations, get_diffusable_atoms
from .reweighting import ReweightingMechanism


@dataclass
class SwarmProposal:
    """
    A proposed diffusion event from the swarm.

    Attributes:
        agent_idx: Index of the diffusing atom.
        direction_idx: Index of the target neighbor (0-11 for 12 neighbors).
        target_idx: Absolute index of target site in lattice.
        logit: Raw logit from policy network for this (agent, direction) pair.
        species: Species of the diffusing atom.
    """

    agent_idx: int
    direction_idx: int
    target_idx: int
    logit: float
    species: SpeciesType


class SwarmEngine:
    """
    Core engine for SwarmThinkers event selection.

    This class orchestrates the swarm-based proposal, reweighting, and selection
    process. It does NOT modify the KMCSimulator directly - instead it generates
    events that can be executed by the simulator.

    Phase 1 (diffusion-only) workflow:
        1. Get diffusable atoms from lattice
        2. Extract local observations
        3. Policy proposes directions for each agent
        4. Sample N proposals from the swarm
        5. Compute physical rates for proposals only
        6. Reweight: P(a) = π(a)·Γ_a / Z
        7. Select event via reweighted sampling
        8. Return event + importance weight

    Attributes:
        policy: The diffusion swarm policy network.
        rate_calculator: Calculator for physical transition rates.
        reweighting: Mechanism for combining policy and rates.
        device: Torch device (cpu or cuda).
    """

    def __init__(
        self,
        policy: DiffusionSwarmPolicy,
        rate_calculator: RateCalculator,
        device: str = "cpu",
    ) -> None:
        """
        Initialize swarm engine.

        Args:
            policy: Trained or initialized diffusion policy.
            rate_calculator: Rate calculator from KMC simulator.
            device: Device for torch computations.
        """
        self.policy = policy.to(device)
        self.policy.eval()  # Set to evaluation mode
        self.rate_calculator = rate_calculator
        self.reweighting = ReweightingMechanism()
        self.device = device

    def generate_proposals(
        self, lattice: Lattice, n_swarm: int
    ) -> tuple[list[SwarmProposal], npt.NDArray[np.float64]]:
        """
        Generate N swarm proposals from policy.

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals to generate.

        Returns:
            Tuple of:
                - List of SwarmProposal objects
                - Array of logits (before softmax) for each proposal

        Note:
            In SwarmThinkers, we apply global softmax across ALL proposals,
            not per-agent. This enables swarm-level coordination.
        """
        # 1. Get all diffusable atoms
        agent_indices = get_diffusable_atoms(lattice)

        if len(agent_indices) == 0:
            # No diffusable atoms - return empty proposals
            return [], np.array([])

        # 2. Extract observations for all agents
        observations = get_batch_observations(lattice, agent_indices)
        obs_tensor = torch.from_numpy(observations).float().to(self.device)

        # 3. Get logits from policy (batch_size, n_directions)
        with torch.no_grad():
            logits_batch = self.policy(obs_tensor)  # Shape: (n_agents, 12)

        logits_np = logits_batch.cpu().numpy()

        # 4. Flatten to (agent, direction) pairs
        all_proposals = []
        all_logits = []

        for agent_local_idx, agent_global_idx in enumerate(agent_indices):
            site = lattice.sites[agent_global_idx]
            species = site.species

            # Get neighbors for this agent
            neighbors = site.neighbors[:12]  # First 12 neighbors

            for dir_idx, neighbor_idx in enumerate(neighbors):
                # Check if diffusion is valid (target must be vacant)
                target_site = lattice.sites[neighbor_idx]
                if target_site.species != SpeciesType.VACANT:
                    continue  # Skip occupied targets

                proposal = SwarmProposal(
                    agent_idx=agent_global_idx,
                    direction_idx=dir_idx,
                    target_idx=neighbor_idx,
                    logit=logits_np[agent_local_idx, dir_idx],
                    species=species,
                )

                all_proposals.append(proposal)
                all_logits.append(proposal.logit)

        if len(all_proposals) == 0:
            return [], np.array([])

        all_logits_array = np.array(all_logits)

        # 5. Sample n_swarm proposals
        # Apply softmax to get probabilities
        logits_shifted = all_logits_array - np.max(all_logits_array)  # Numerical stability
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

        # Sample proposals
        n_to_sample = min(n_swarm, len(all_proposals))
        sampled_indices = np.random.choice(
            len(all_proposals), size=n_to_sample, replace=False, p=probs
        )

        sampled_proposals = [all_proposals[i] for i in sampled_indices]
        sampled_logits = all_logits_array[sampled_indices]

        return sampled_proposals, sampled_logits

    def compute_rates_for_proposals(
        self, proposals: list[SwarmProposal], lattice: Lattice
    ) -> npt.NDArray[np.float64]:
        """
        Compute physical transition rates for swarm proposals.

        This is where we call the existing RateCalculator, which includes
        ES barrier detection and all physical correctness.

        Args:
            proposals: List of swarm proposals.
            lattice: Current lattice state.

        Returns:
            Array of rates (Hz) for each proposal.
        """
        rates = np.zeros(len(proposals))

        for i, proposal in enumerate(proposals):
            # Get Site objects from lattice
            source_site = lattice.sites[proposal.agent_idx]
            target_site = lattice.sites[proposal.target_idx]

            # Get activation energy for the species
            if proposal.species == SpeciesType.TI:
                ea = self.rate_calculator.params.ea_diff_ti
            elif proposal.species == SpeciesType.O:
                ea = self.rate_calculator.params.ea_diff_o
            else:
                raise ValueError(f"Invalid species for diffusion: {proposal.species}")

            # Calculate rate with ES barrier check
            rate = self.rate_calculator.calculate_diffusion_rate(
                site=source_site,
                target_site=target_site,
                activation_energy=ea,
                lattice_sites=lattice.sites,
            )
            rates[i] = rate

        return rates

    def reweight_and_select(
        self,
        proposals: list[SwarmProposal],
        logits: npt.NDArray[np.float64],
        rates: npt.NDArray[np.float64],
    ) -> tuple[SwarmProposal, float]:
        """
        Reweight proposals and select event via importance sampling.

        Implements: P(a) = π(a) · Γ_a / Σ π(a')·Γ_a'

        Args:
            proposals: List of swarm proposals.
            logits: Logits from policy for each proposal.
            rates: Physical rates for each proposal.

        Returns:
            Tuple of:
                - Selected proposal
                - Importance weight (for unbiased estimation)
        """
        # 1. Convert logits to probabilities (global softmax)
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        policy_probs = exp_logits / np.sum(exp_logits)

        # 2. Reweight with physical rates
        reweighted_probs = self.reweighting.compute_reweighted_distribution(
            policy_probs, rates
        )

        # 3. Select event
        selected_idx = np.random.choice(len(proposals), p=reweighted_probs)
        selected_proposal = proposals[selected_idx]

        # 4. Compute importance weight: w = 1 / π(a)
        # This is used to correct bias in observable estimation
        importance_weight = 1.0 / policy_probs[selected_idx]

        return selected_proposal, importance_weight

    def run_step(
        self, lattice: Lattice, n_swarm: int = 32
    ) -> tuple[Event | None, float]:
        """
        Execute one SwarmThinkers step: propose, reweight, select.

        This is the main entry point for swarm-based event selection.

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals in the swarm.

        Returns:
            Tuple of:
                - Selected Event (compatible with KMCSimulator.execute_event)
                - Importance weight for this step

        Example:
            >>> engine = SwarmEngine(policy, rate_calc)
            >>> event, weight = engine.run_step(lattice, n_swarm=32)
            >>> simulator.execute_event(event)  # Execute in classic simulator
        """
        # 1. Generate proposals
        proposals, logits = self.generate_proposals(lattice, n_swarm)

        if len(proposals) == 0:
            # No valid proposals (e.g., no diffusable atoms)
            return None, 1.0

        # 2. Compute rates
        rates = self.compute_rates_for_proposals(proposals, lattice)

        # 3. Reweight and select
        selected_proposal, importance_weight = self.reweight_and_select(
            proposals, logits, rates
        )

        # 4. Convert to Event object (compatible with KMCSimulator)
        event_type = (
            EventType.DIFFUSION_TI
            if selected_proposal.species == SpeciesType.TI
            else EventType.DIFFUSION_O
        )

        event = Event(
            event_type=event_type,
            site_index=selected_proposal.agent_idx,
            target_index=selected_proposal.target_idx,
            rate=rates[proposals.index(selected_proposal)],
            species=selected_proposal.species,
        )

        return event, importance_weight
