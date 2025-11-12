"""
Main KMC simulator engine.

This module implements the core kinetic Monte Carlo algorithm for simulating
thin film growth.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np

from ..data.tio2_parameters import TiO2Parameters, get_tio2_parameters
from .efficient_updates import initialize_all_events, update_events_after_execution
from .events import Event, EventCatalog, EventType
from .lattice import Lattice, SpeciesType
from .rates import RateCalculator

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class KMCSimulator:
    """
    Kinetic Monte Carlo simulator for thin film growth.

    This class implements the rejection-free (BKL) algorithm for KMC simulation.

    Attributes:
        lattice: Lattice structure.
        rate_calculator: Rate calculator for events.
        event_catalog: Catalog of all possible events.
        time: Current simulation time (s).
        step: Current simulation step.
    """

    def __init__(
        self,
        lattice_size: tuple[int, int, int],
        temperature: float,
        deposition_rate: float,
        params: TiO2Parameters | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize KMC simulator.

        Args:
            lattice_size: Lattice dimensions (nx, ny, nz).
            temperature: Temperature (K).
            deposition_rate: Deposition rate (ML/s).
            params: Physical parameters for TiO2. If None, uses default parameters.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.params = params if params is not None else get_tio2_parameters()

        self.lattice = Lattice(size=lattice_size)
        self.rate_calculator = RateCalculator(
            temperature=temperature,
            deposition_rate=deposition_rate,
            params=self.params,
        )

        self.event_catalog = EventCatalog()
        self.time = 0.0
        self.step = 0
        self.event_map: dict[int, list[int]] = {}

        self.events_executed: dict[EventType, int] = dict.fromkeys(EventType, 0)

        # Initialize event catalog with initial surface sites
        initialize_all_events(self)

        logger.info(
            f"Initialized KMC simulator: {lattice_size}, T={temperature}K, "
            f"deposition_rate={deposition_rate}ML/s, initial_events={len(self.event_catalog)}"
        )

    def build_event_list(self) -> None:
        """
        Build the complete list of possible events.

        This method examines the current lattice state and generates all possible
        events with their rates.
        """
        self.event_catalog.clear()

        # Add adsorption events for all surface sites
        for site_idx in self.lattice.surface_sites:
            site = self.lattice.get_site_by_index(site_idx)
            if site.species == SpeciesType.VACANT:
                # Ti adsorption
                rate_ti = self.rate_calculator.calculate_adsorption_rate(site, SpeciesType.TI)
                self.event_catalog.add_event(
                    Event(
                        event_type=EventType.ADSORPTION_TI,
                        site_index=site_idx,
                        rate=rate_ti,
                        species=SpeciesType.TI,
                    )
                )

                # O adsorption
                rate_o = self.rate_calculator.calculate_adsorption_rate(site, SpeciesType.O)
                self.event_catalog.add_event(
                    Event(
                        event_type=EventType.ADSORPTION_O,
                        site_index=site_idx,
                        rate=rate_o,
                        species=SpeciesType.O,
                    )
                )

        # Add diffusion and desorption events for occupied sites
        for site_idx, site in self.lattice.iter_occupied_sites():
            # Diffusion to neighboring vacant sites
            for neighbor_idx in site.neighbors:
                neighbor = self.lattice.get_site_by_index(neighbor_idx)
                if neighbor.species == SpeciesType.VACANT:
                    event_type = (
                        EventType.DIFFUSION_TI
                        if site.species == SpeciesType.TI
                        else EventType.DIFFUSION_O
                    )

                    ea_diff = (
                        self.params.ea_diff_ti
                        if site.species == SpeciesType.TI
                        else self.params.ea_diff_o
                    )

                    rate = self.rate_calculator.calculate_diffusion_rate(
                        site, neighbor, activation_energy=ea_diff
                    )

                    self.event_catalog.add_event(
                        Event(
                            event_type=event_type,
                            site_index=site_idx,
                            target_index=neighbor_idx,
                            rate=rate,
                            species=site.species,
                        )
                    )

            event_type = (
                EventType.DESORPTION_TI
                if site.species == SpeciesType.TI
                else EventType.DESORPTION_O
            )
            ea_des = (
                self.params.ea_des_ti if site.species == SpeciesType.TI else self.params.ea_des_o
            )

            rate = self.rate_calculator.calculate_desorption_rate(activation_energy=ea_des)

            self.event_catalog.add_event(
                Event(
                    event_type=event_type,
                    site_index=site_idx,
                    rate=rate,
                    species=site.species,
                )
            )

    def select_event(self) -> Event | None:
        """
        Select an event using the BKL algorithm.

        Returns:
            Selected event, or None if no events available.
        """
        if len(self.event_catalog) == 0:
            return None

        # Generate random number
        r = random.random() * self.event_catalog.total_rate

        # Binary search for event
        cumulative_rates = self.event_catalog.get_cumulative_rates()
        idx = np.searchsorted(cumulative_rates, r)

        return self.event_catalog.events[idx]

    def execute_event(self, event: Event) -> None:
        """
        Execute a selected event.

        Args:
            event: Event to execute.
        """
        if event.event_type in (EventType.ADSORPTION_TI, EventType.ADSORPTION_O):
            # Deposit atom
            species = SpeciesType.TI if event.species == SpeciesType.TI else SpeciesType.O
            self.lattice.deposit_atom(event.site_index, species)

        elif event.event_type in (EventType.DIFFUSION_TI, EventType.DIFFUSION_O):
            # Move atom
            if event.target_index is not None:
                self.lattice.move_atom(event.site_index, event.target_index)

        elif event.event_type in (EventType.DESORPTION_TI, EventType.DESORPTION_O):
            # Remove atom
            self.lattice.remove_atom(event.site_index)

        # Update statistics
        self.events_executed[event.event_type] += 1

    def advance_time(self) -> None:
        """Advance simulation time using BKL algorithm."""
        if self.event_catalog.total_rate > 0:
            dt = -np.log(random.random()) / self.event_catalog.total_rate
            self.time += dt

    def run_step(self) -> bool:
        """Execute one KMC step using efficient event updates."""
        event = self.select_event()
        if event is None:
            logger.warning("No events available, stopping simulation")
            return False

        self.execute_event(event)

        update_events_after_execution(self, event)

        self.advance_time()
        self.step += 1

        return True

    def run(
        self,
        max_steps: int = 10000,
        max_time: float | None = None,
        callback: Callable[[KMCSimulator], None] | None = None,
        snapshot_interval: int = 100,
    ) -> None:
        """
        Run KMC simulation.

        Args:
            max_steps: Maximum number of steps.
            max_time: Maximum simulation time (s).
            callback: Optional callback function called at snapshot intervals.
            snapshot_interval: Interval for calling callback.
        """
        logger.info(f"Starting KMC simulation: max_steps={max_steps}, max_time={max_time}")

        while self.step < max_steps:
            # Check time limit
            if max_time is not None and self.time >= max_time:
                logger.info(f"Reached time limit: {self.time:.2e}s")
                break

            # Execute step
            success = self.run_step()
            if not success:
                break

            # Snapshot callback
            if callback is not None and self.step % snapshot_interval == 0:
                callback(self)

        logger.info(
            f"Simulation completed: {self.step} steps, {self.time:.2e}s, "
            f"{self.lattice.get_composition()}"
        )

    def get_statistics(self) -> dict[str, int | float]:
        """
        Get simulation statistics.

        Returns:
            Dictionary of statistics.
        """
        composition = self.lattice.get_composition()
        return {
            "step": self.step,
            "time": self.time,
            "n_ti": composition[SpeciesType.TI],
            "n_o": composition[SpeciesType.O],
            "n_vacant": composition[SpeciesType.VACANT],
            **{f"events_{et.value}": count for et, count in self.events_executed.items()},
        }

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.lattice = Lattice(size=self.lattice.size)
        self.event_catalog.clear()
        self.event_map.clear()
        self.time = 0.0
        self.step = 0
        self.events_executed = dict.fromkeys(EventType, 0)

        initialize_all_events(self)

        logger.info("Simulator reset")
