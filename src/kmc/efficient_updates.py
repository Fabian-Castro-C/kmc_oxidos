"""
Efficient local event update methods for KMC simulator.

This module provides methods for updating only the events affected by a change,
rather than rebuilding the entire event list. This is critical for performance
in large systems.

Based on best practices from:
- Bortz-Kalos-Lebowitz (BKL) algorithm
- UTK paper: "Kinetic Monte Carlo simulations with minimal searching"
- Standard KMC implementations for thin film growth
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .events import Event, EventType
from .lattice import SpeciesType

if TYPE_CHECKING:
    from .simulator import KMCSimulator


def update_events_for_site(simulator: KMCSimulator, site_idx: int) -> None:
    """
    Update only events affected by changes at site_idx.

    This is the core of the efficient BKL algorithm - only updating
    events for sites affected by the last event, rather than rebuilding
    the entire event list.

    Args:
        simulator: KMC simulator instance.
        site_idx: Index of site whose events need updating.
    """
    site = simulator.lattice.get_site_by_index(site_idx)

    # Remove old events for this site
    if site_idx in simulator.event_map:
        # Remove in reverse order to maintain valid indices
        for event_idx in sorted(simulator.event_map[site_idx], reverse=True):
            if event_idx < len(simulator.event_catalog.events):
                simulator.event_catalog.remove_event(event_idx)
        del simulator.event_map[site_idx]

    # Add new events for this site
    new_event_indices: list[int] = []

    # Case 1: Site is vacant and on surface -> adsorption events
    if site.species == SpeciesType.VACANT and site_idx in simulator.lattice.surface_sites:
        # Ti adsorption
        rate_ti = simulator.rate_calculator.calculate_adsorption_rate(site, SpeciesType.TI)
        event_ti = Event(
            event_type=EventType.ADSORPTION_TI,
            site_index=site_idx,
            rate=rate_ti,
            species=SpeciesType.TI,
        )
        simulator.event_catalog.add_event(event_ti)
        new_event_indices.append(len(simulator.event_catalog) - 1)

        # O adsorption
        rate_o = simulator.rate_calculator.calculate_adsorption_rate(site, SpeciesType.O)
        event_o = Event(
            event_type=EventType.ADSORPTION_O,
            site_index=site_idx,
            rate=rate_o,
            species=SpeciesType.O,
        )
        simulator.event_catalog.add_event(event_o)
        new_event_indices.append(len(simulator.event_catalog) - 1)

    # Case 2: Site is occupied -> diffusion and desorption events
    elif site.is_occupied():
        # Diffusion events to neighboring vacant sites
        for neighbor_idx in site.neighbors:
            neighbor = simulator.lattice.get_site_by_index(neighbor_idx)
            if neighbor.species == SpeciesType.VACANT:
                event_type = (
                    EventType.DIFFUSION_TI
                    if site.species == SpeciesType.TI
                    else EventType.DIFFUSION_O
                )

                ea_diff = (
                    simulator.params.ea_diff_ti
                    if site.species == SpeciesType.TI
                    else simulator.params.ea_diff_o
                )

                rate = simulator.rate_calculator.calculate_diffusion_rate(
                    site, neighbor, activation_energy=ea_diff
                )

                event = Event(
                    event_type=event_type,
                    site_index=site_idx,
                    target_index=neighbor_idx,
                    rate=rate,
                    species=site.species,
                )
                simulator.event_catalog.add_event(event)
                new_event_indices.append(len(simulator.event_catalog) - 1)

        # Desorption event
        event_type = (
            EventType.DESORPTION_TI if site.species == SpeciesType.TI else EventType.DESORPTION_O
        )

        ea_des = (
            simulator.params.ea_des_ti
            if site.species == SpeciesType.TI
            else simulator.params.ea_des_o
        )

        rate = simulator.rate_calculator.calculate_desorption_rate(activation_energy=ea_des)

        event = Event(
            event_type=event_type,
            site_index=site_idx,
            rate=rate,
            species=site.species,
        )
        simulator.event_catalog.add_event(event)
        new_event_indices.append(len(simulator.event_catalog) - 1)

    # Update event map
    if new_event_indices:
        simulator.event_map[site_idx] = new_event_indices


def get_affected_sites(simulator: KMCSimulator, event: Event) -> set[int]:
    """
    Get set of sites whose events are affected by executing this event.

    According to BKL algorithm, we only need to update events for:
    - The primary site (where event occurred)
    - The target site (for diffusion events)
    - All neighbors of both sites

    Args:
        simulator: KMC simulator instance.
        event: Event that was executed.

    Returns:
        Set of site indices that need event updates.
    """
    affected = {event.site_index}

    # Add neighbors of primary site
    primary_site = simulator.lattice.get_site_by_index(event.site_index)
    affected.update(primary_site.neighbors)

    # For diffusion, also add target site and its neighbors
    if event.target_index is not None:
        affected.add(event.target_index)
        target_site = simulator.lattice.get_site_by_index(event.target_index)
        affected.update(target_site.neighbors)

    return affected


def update_events_after_execution(simulator: KMCSimulator, event: Event) -> None:  # noqa: ARG001
    """
    Update all events affected by executing the given event.

    TEMPORARY WORKAROUND: Due to event_map index invalidation bugs,
    this currently rebuilds the entire event list (O(N)) instead of
    local updates (O(1)). This ensures correctness but reduces performance.

    TODO: Fix event_map index tracking to enable O(1) local updates.

    Args:
        simulator: KMC simulator instance.
        event: Event that was just executed (currently unused in workaround).
    """
    # DEBUG logging
    import logging

    logger = logging.getLogger(__name__)

    before_n_events = len(simulator.event_catalog)
    before_total_rate = simulator.event_catalog.total_rate
    before_surface = len(simulator.lattice.surface_sites)

    # Temporary workaround: full rebuild to avoid index bugs
    # The 'event' parameter is kept for interface compatibility but not used
    simulator.build_event_list()

    after_n_events = len(simulator.event_catalog)
    after_total_rate = simulator.event_catalog.total_rate
    after_surface = len(simulator.lattice.surface_sites)

    logger.debug(
        f"Event update: events {before_n_events}->{after_n_events}, "
        f"total_rate {before_total_rate:.6f}->{after_total_rate:.6f}, "
        f"surface_sites {before_surface}->{after_surface}"
    )


def initialize_all_events(simulator: KMCSimulator) -> None:
    """
    Build the initial complete event list.

    This should only be called once at the start of simulation or after
    a complete lattice reset.

    Args:
        simulator: KMC simulator instance.
    """
    simulator.event_catalog.clear()
    simulator.event_map.clear()

    # Add events for all surface sites
    for site_idx in simulator.lattice.surface_sites:
        update_events_for_site(simulator, site_idx)

    # Add events for all occupied sites
    for site_idx, _site in simulator.lattice.iter_occupied_sites():
        update_events_for_site(simulator, site_idx)
