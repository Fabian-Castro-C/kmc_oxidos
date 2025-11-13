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

    def generate_diffusion_proposals(
        self, lattice: Lattice, n_swarm: int
    ) -> tuple[list[EventProposal], npt.NDArray[np.float64]]:
        """
        Generate N diffusion proposals from policy.

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals to generate.

        Returns:
            Tuple of:
                - List of EventProposal objects (diffusion events)
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

                proposal = EventProposal(
                    event_type=EventType.DIFFUSION_TI if species == SpeciesType.TI else EventType.DIFFUSION_O,
                    site_index=agent_global_idx,
                    target_index=neighbor_idx,
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

    def generate_adsorption_proposals(
        self, lattice: Lattice, n_swarm: int
    ) -> tuple[list[EventProposal], npt.NDArray[np.float64]]:
        """
        Generate N adsorption proposals from policy.

        Proposes optimal surface sites for Ti/O adsorption based on
        local topology (heights, neighbors, curvature).

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals to generate.

        Returns:
            Tuple of (EventProposal list, logits array).
        """
        # 1. Get all surface sites (vacant sites at maximum height)
        surface_indices = []
        for idx, site in enumerate(lattice.sites):
            if site.species == SpeciesType.VACANT and site.position[2] == lattice.size[2] - 1:
                surface_indices.append(idx)

        if len(surface_indices) == 0:
            return [], np.array([])

        # 2. Extract observations for surface sites
        observations = get_batch_observations(lattice, surface_indices)
        obs_tensor = torch.from_numpy(observations).float().to(self.device)

        # 3. Get logits from adsorption policy (1 logit per site)
        with torch.no_grad():
            logits_batch = self.adsorption_policy(obs_tensor)  # Shape: (n_sites,)

        logits_np = logits_batch.cpu().numpy()

        # 4. Create proposals (one per site, alternating Ti/O)
        all_proposals = []
        all_logits = []

        for site_local_idx, site_global_idx in enumerate(surface_indices):
            # Alternate between Ti and O adsorption
            for species, event_type in [
                (SpeciesType.TI, EventType.ADSORPTION_TI),
                (SpeciesType.O, EventType.ADSORPTION_O),
            ]:
                proposal = EventProposal(
                    event_type=event_type,
                    site_index=site_global_idx,
                    target_index=None,
                    logit=logits_np[site_local_idx],
                    species=species,
                )
                all_proposals.append(proposal)
                all_logits.append(proposal.logit)

        if len(all_proposals) == 0:
            return [], np.array([])

        all_logits_array = np.array(all_logits)

        # 5. Sample n_swarm proposals
        logits_shifted = all_logits_array - np.max(all_logits_array)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

        n_to_sample = min(n_swarm, len(all_proposals))
        sampled_indices = np.random.choice(
            len(all_proposals), size=n_to_sample, replace=False, p=probs
        )

        sampled_proposals = [all_proposals[i] for i in sampled_indices]
        sampled_logits = all_logits_array[sampled_indices]

        return sampled_proposals, sampled_logits

    def generate_desorption_proposals(
        self, lattice: Lattice, n_swarm: int
    ) -> tuple[list[EventProposal], npt.NDArray[np.float64]]:
        """
        Generate N desorption proposals from policy.

        Proposes which adsorbed atoms (Ti/O) should desorb based on
        local coordination, binding energy estimates, surface position.

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals to generate.

        Returns:
            Tuple of (EventProposal list, logits array).
        """
        # 1. Get all adsorbed atoms (non-vacant surface sites)
        adsorbed_indices = []
        for idx, site in enumerate(lattice.sites):
            if site.species != SpeciesType.VACANT and site.position[2] == lattice.size[2] - 1:
                adsorbed_indices.append(idx)

        if len(adsorbed_indices) == 0:
            return [], np.array([])

        # 2. Extract observations
        observations = get_batch_observations(lattice, adsorbed_indices)
        obs_tensor = torch.from_numpy(observations).float().to(self.device)

        # 3. Get logits from desorption policy
        with torch.no_grad():
            logits_batch = self.desorption_policy(obs_tensor)  # Shape: (n_atoms,)

        logits_np = logits_batch.cpu().numpy()

        # 4. Create proposals (one per adsorbed atom)
        all_proposals = []
        all_logits = []

        for atom_local_idx, atom_global_idx in enumerate(adsorbed_indices):
            site = lattice.sites[atom_global_idx]
            species = site.species

            event_type = (
                EventType.DESORPTION_TI if species == SpeciesType.TI else EventType.DESORPTION_O
            )

            proposal = EventProposal(
                event_type=event_type,
                site_index=atom_global_idx,
                target_index=None,
                logit=logits_np[atom_local_idx],
                species=species,
            )
            all_proposals.append(proposal)
            all_logits.append(proposal.logit)

        if len(all_proposals) == 0:
            return [], np.array([])

        all_logits_array = np.array(all_logits)

        # 5. Sample n_swarm proposals
        logits_shifted = all_logits_array - np.max(all_logits_array)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

        n_to_sample = min(n_swarm, len(all_proposals))
        sampled_indices = np.random.choice(
            len(all_proposals), size=n_to_sample, replace=False, p=probs
        )

        sampled_proposals = [all_proposals[i] for i in sampled_indices]
        sampled_logits = all_logits_array[sampled_indices]

        return sampled_proposals, sampled_logits

    def generate_reaction_proposals(
        self, lattice: Lattice, n_swarm: int
    ) -> tuple[list[EventProposal], npt.NDArray[np.float64]]:
        """
        Generate N reaction proposals from policy.

        Proposes Ti atoms that should react with neighboring O atoms
        to form TiO₂. Requires Ti to have ≥2 O neighbors.

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals to generate.

        Returns:
            Tuple of (EventProposal list, logits array).
        """
        # 1. Get all Ti atoms with sufficient O neighbors
        reaction_candidates = []
        for idx, site in enumerate(lattice.sites):
            if site.species != SpeciesType.TI:
                continue

            # Count O neighbors
            n_o_neighbors = 0
            for neighbor_idx in site.neighbors:
                neighbor_site = lattice.sites[neighbor_idx]
                if neighbor_site.species == SpeciesType.O:
                    n_o_neighbors += 1

            # Need at least 2 O neighbors for TiO2 reaction
            if n_o_neighbors >= 2:
                reaction_candidates.append(idx)

        if len(reaction_candidates) == 0:
            return [], np.array([])

        # 2. Extract observations
        observations = get_batch_observations(lattice, reaction_candidates)
        obs_tensor = torch.from_numpy(observations).float().to(self.device)

        # 3. Get logits from reaction policy
        with torch.no_grad():
            logits_batch = self.reaction_policy(obs_tensor)  # Shape: (n_candidates,)

        logits_np = logits_batch.cpu().numpy()

        # 4. Create proposals (one per Ti candidate)
        all_proposals = []
        all_logits = []

        for candidate_local_idx, candidate_global_idx in enumerate(reaction_candidates):
            proposal = EventProposal(
                event_type=EventType.REACTION_TIO2,
                site_index=candidate_global_idx,
                target_index=None,
                logit=logits_np[candidate_local_idx],
                species=SpeciesType.TI,
            )
            all_proposals.append(proposal)
            all_logits.append(proposal.logit)

        if len(all_proposals) == 0:
            return [], np.array([])

        all_logits_array = np.array(all_logits)

        # 5. Sample n_swarm proposals
        logits_shifted = all_logits_array - np.max(all_logits_array)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

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

    def compute_rates_for_event_proposals(
        self, proposals: list[EventProposal], lattice: Lattice
    ) -> npt.NDArray[np.float64]:
        """
        Compute physical transition rates for generic event proposals.

        Handles ALL event types: diffusion, adsorption, desorption, reaction.

        Args:
            proposals: List of EventProposal objects.
            lattice: Current lattice state.

        Returns:
            Array of rates (Hz) for each proposal.
        """
        rates = np.zeros(len(proposals))

        for i, proposal in enumerate(proposals):
            site = lattice.sites[proposal.site_index]

            if proposal.event_type in [EventType.DIFFUSION_TI, EventType.DIFFUSION_O]:
                # Diffusion: need source + target sites
                target_site = lattice.sites[proposal.target_index]
                ea = (
                    self.rate_calculator.params.ea_diff_ti
                    if proposal.species == SpeciesType.TI
                    else self.rate_calculator.params.ea_diff_o
                )
                rate = self.rate_calculator.calculate_diffusion_rate(
                    site=site,
                    target_site=target_site,
                    activation_energy=ea,
                    lattice_sites=lattice.sites,
                )

            elif proposal.event_type in [EventType.ADSORPTION_TI, EventType.ADSORPTION_O]:
                # Adsorption: constant rate from deposition_rate
                rate = self.rate_calculator.deposition_rate

            elif proposal.event_type in [EventType.DESORPTION_TI, EventType.DESORPTION_O]:
                # Desorption: use calculate_desorption_rate
                ea = (
                    self.rate_calculator.params.ea_des_ti
                    if proposal.species == SpeciesType.TI
                    else self.rate_calculator.params.ea_des_o
                )
                rate = self.rate_calculator.calculate_desorption_rate(
                    activation_energy=ea
                )

            elif proposal.event_type == EventType.REACTION_TIO2:
                # Reaction: use calculate_reaction_rate
                rate = self.rate_calculator.calculate_reaction_rate(
                    site=site,
                    lattice_sites=lattice.sites,
                )

            else:
                raise ValueError(f"Unknown event type: {proposal.event_type}")

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
        self, lattice: Lattice, n_swarm: int = 32
    ) -> tuple[Event | None, float]:
        """
        Execute one SwarmThinkers step using ALL policies (full framework).

        This implements the complete multi-policy SwarmThinkers framework:
        - ALL events (diffusion, adsorption, desorption, reaction) proposed by learned policies
        - Global reweighting: P(a) = π_θ(a) · Γ_a / Z
        - Importance sampling: w = 1/π_θ(a_selected)
        - FULLY DECOUPLED from KMC classic event catalog

        Args:
            lattice: Current lattice state.
            n_swarm: Number of proposals per event type.

        Returns:
            Tuple of:
                - Selected Event (compatible with KMCSimulator.execute_event)
                - Importance weight for this step

        Example:
            >>> engine = SwarmEngine(diff_pol, ads_pol, des_pol, react_pol, rate_calc)
            >>> event, weight = engine.run_step(lattice, n_swarm=32)
            >>> simulator.execute_event(event)
        """
        # 1. Generate proposals from ALL policies
        diff_proposals, diff_logits = self.generate_diffusion_proposals(lattice, n_swarm)
        ads_proposals, ads_logits = self.generate_adsorption_proposals(lattice, n_swarm)
        des_proposals, des_logits = self.generate_desorption_proposals(lattice, n_swarm)
        react_proposals, react_logits = self.generate_reaction_proposals(lattice, n_swarm)

        # 2. Combine all proposals
        all_proposals = []
        all_logits = []

        all_proposals.extend(diff_proposals)
        all_logits.extend(diff_logits.tolist() if len(diff_logits) > 0 else [])

        all_proposals.extend(ads_proposals)
        all_logits.extend(ads_logits.tolist() if len(ads_logits) > 0 else [])

        all_proposals.extend(des_proposals)
        all_logits.extend(des_logits.tolist() if len(des_logits) > 0 else [])

        all_proposals.extend(react_proposals)
        all_logits.extend(react_logits.tolist() if len(react_logits) > 0 else [])

        if len(all_proposals) == 0:
            # No proposals available
            return None, 1.0

        # 3. Compute physical rates for ALL proposals
        all_rates = self.compute_rates_for_event_proposals(all_proposals, lattice)

        # 4. Global reweighting and selection
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
        selected_idx = np.random.choice(len(all_proposals), p=probs)
        selected_proposal = all_proposals[selected_idx]

        # 5. Convert EventProposal to Event (for KMCSimulator compatibility)
        event = Event(
            event_type=selected_proposal.event_type,
            site_index=selected_proposal.site_index,
            target_index=selected_proposal.target_index,
            rate=all_rates_np[selected_idx],
            species=selected_proposal.species,
        )

        # 6. Compute importance weight: w = 1 / π_θ(a_selected)
        policy_probs = exp_logits / np.sum(exp_logits)
        importance_weight = 1.0 / policy_probs[selected_idx]

        return event, importance_weight
