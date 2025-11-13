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
    from .swarm_policy import (
        AdsorptionSwarmPolicy,
        DesorptionSwarmPolicy,
        DiffusionSwarmPolicy,
        ReactionSwarmPolicy,
    )

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


@dataclass
class EventProposal:
    """
    Generic event proposal from any swarm policy.

    Attributes:
        event_type: Type of event (ADSORPTION, DIFFUSION, DESORPTION, REACTION).
        site_index: Primary site index.
        target_index: Target site index (for diffusion) or None.
        logit: Raw logit from policy network.
        species: Species involved (Ti or O).
    """

    event_type: EventType
    site_index: int
    target_index: int | None
    logit: float
    species: SpeciesType


class SwarmEngine:
    """
    Core engine for SwarmThinkers event selection with ALL event types.

    This class orchestrates swarm-based proposal, reweighting, and selection
    for ALL events: diffusion, adsorption, desorption, and reaction.
    Each event type has its own learned policy network.

    Workflow:
        1. Generate proposals from each policy:
           - Diffusion: propose directions for mobile atoms
           - Adsorption: propose surface sites for incoming atoms
           - Desorption: propose atoms to desorb
           - Reaction: propose Ti atoms to react with O
        2. Compute physical rates for ALL proposals
        3. Global reweighting: P(a) = π(a)·Γ_a / Z
        4. Select event via reweighted sampling
        5. Return event + importance weight

    Attributes:
        diffusion_policy: Policy for diffusion events.
        adsorption_policy: Policy for adsorption events.
        desorption_policy: Policy for desorption events.
        reaction_policy: Policy for reaction events.
        rate_calculator: Calculator for physical transition rates.
        reweighting: Mechanism for combining policy and rates.
        device: Torch device (cpu or cuda).
    """

    def __init__(
        self,
        diffusion_policy: DiffusionSwarmPolicy,
        adsorption_policy: AdsorptionSwarmPolicy,
        desorption_policy: DesorptionSwarmPolicy,
        reaction_policy: ReactionSwarmPolicy,
        rate_calculator: RateCalculator,
        device: str = "cpu",
    ) -> None:
        """
        Initialize swarm engine with all policies.

        Args:
            diffusion_policy: Policy for diffusion events.
            adsorption_policy: Policy for adsorption events.
            desorption_policy: Policy for desorption events.
            reaction_policy: Policy for reaction events.
            rate_calculator: Rate calculator from KMC simulator.
            device: Device for torch computations.
        """
        self.diffusion_policy = diffusion_policy.to(device)
        self.diffusion_policy.eval()

        self.adsorption_policy = adsorption_policy.to(device)
        self.adsorption_policy.eval()

        self.desorption_policy = desorption_policy.to(device)
        self.desorption_policy.eval()

        self.reaction_policy = reaction_policy.to(device)
        self.reaction_policy.eval()

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
            logits_batch = self.diffusion_policy(obs_tensor)  # Shape: (n_agents, 12)

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
        reweighted_probs = self.reweighting.compute_reweighted_distribution(policy_probs, rates)

        # 3. Select event
        selected_idx = np.random.choice(len(proposals), p=reweighted_probs)
        selected_proposal = proposals[selected_idx]

        # 4. Compute importance weight: w = 1 / π(a)
        # This is used to correct bias in observable estimation
        importance_weight = 1.0 / policy_probs[selected_idx]

        return selected_proposal, importance_weight

    def run_step(
        self, lattice: Lattice, event_catalog, n_swarm: int = 32
    ) -> tuple[Event | None, float]:
        """
        Execute one SwarmThinkers step: propose diffusions + include all other events, reweight, select.

        This implements the full SwarmThinkers framework:
        - Diffusion events: proposed by learned policy with logits π_θ(a)
        - Other events (adsorption, desorption, reaction): included with uniform logits (0.0)
        - All events compete via reweighting: P(a) = exp(logit_a) · Γ_a / Z
        - Importance sampling: w = 1/π_θ(a_selected)

        Args:
            lattice: Current lattice state.
            event_catalog: EventCatalog with all possible events (from KMC simulator).
            n_swarm: Number of diffusion proposals in the swarm.

        Returns:
            Tuple of:
                - Selected Event (compatible with KMCSimulator.execute_event)
                - Importance weight for this step

        Example:
            >>> engine = SwarmEngine(policy, rate_calc)
            >>> event, weight = engine.run_step(lattice, sim.event_catalog, n_swarm=32)
            >>> simulator.execute_event(event)
        """
        # 1. Generate diffusion proposals (policy-driven)
        diff_proposals, diff_logits = self.generate_proposals(lattice, n_swarm)

        # 2. Get all non-diffusion events from catalog (uniform policy)
        non_diff_events = []
        non_diff_logits = []
        for event in event_catalog.events:
            if event.event_type not in [EventType.DIFFUSION_TI, EventType.DIFFUSION_O]:
                non_diff_events.append(event)
                non_diff_logits.append(0.0)  # Uniform logit (exp(0) = 1)

        # 3. Combine: proposals → Events for unified handling
        all_events = []
        all_logits = []
        all_rates = []

        # Add diffusion proposals as Events
        if len(diff_proposals) > 0:
            diff_rates = self.compute_rates_for_proposals(diff_proposals, lattice)
            for i, proposal in enumerate(diff_proposals):
                event_type = (
                    EventType.DIFFUSION_TI
                    if proposal.species == SpeciesType.TI
                    else EventType.DIFFUSION_O
                )
                event = Event(
                    event_type=event_type,
                    site_index=proposal.agent_idx,
                    target_index=proposal.target_idx,
                    rate=diff_rates[i],
                    species=proposal.species,
                )
                all_events.append(event)
                all_logits.append(diff_logits[i])
                all_rates.append(diff_rates[i])

        # Add non-diffusion events
        all_events.extend(non_diff_events)
        all_logits.extend(non_diff_logits)
        all_rates.extend([e.rate for e in non_diff_events])

        if len(all_events) == 0:
            # No events available
            return None, 1.0

        # 4. Reweight and select from ALL events
        all_logits_np = np.array(all_logits)
        all_rates_np = np.array(all_rates)

        # Reweighting: P(a) = exp(logit_a) · rate_a / Z
        logits_shifted = all_logits_np - np.max(all_logits_np)
        exp_logits = np.exp(logits_shifted)
        weights = exp_logits * all_rates_np
        total_weight = np.sum(weights)

        if total_weight == 0:
            return None, 1.0

        probs = weights / total_weight

        # Sample event
        selected_idx = np.random.choice(len(all_events), p=probs)
        selected_event = all_events[selected_idx]

        # Importance weight: w = 1 / π_θ(a_selected)
        # π_θ(a) = exp(logit_a) / Σ exp(logit_a')
        policy_probs = exp_logits / np.sum(exp_logits)
        importance_weight = 1.0 / policy_probs[selected_idx]

        return selected_event, importance_weight
