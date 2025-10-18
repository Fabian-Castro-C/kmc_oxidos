"""
Event definitions for KMC simulation.

This module defines the types of atomic events that can occur during thin film
growth, including adsorption, diffusion, reaction, and desorption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lattice import SpeciesType


class EventType(Enum):
    """Types of atomic events in KMC simulation."""

    ADSORPTION_TI = "adsorption_ti"  # Ti atom adsorption
    ADSORPTION_O = "adsorption_o"  # O atom adsorption
    DIFFUSION_TI = "diffusion_ti"  # Ti atom diffusion
    DIFFUSION_O = "diffusion_o"  # O atom diffusion
    REACTION_TIO2 = "reaction_tio2"  # Ti + 2O → TiO2 formation
    DESORPTION_TI = "desorption_ti"  # Ti atom desorption
    DESORPTION_O = "desorption_o"  # O atom desorption


@dataclass
class Event:
    """
    Represents a single KMC event.

    Attributes:
        event_type: Type of the event.
        site_index: Primary site index where event occurs.
        target_index: Secondary site index (for diffusion).
        rate: Event rate (Hz).
        species: Species involved in the event.
        delta_energy: Energy change from this event (eV).
    """

    event_type: EventType
    site_index: int
    target_index: int | None = None
    rate: float = 0.0
    species: SpeciesType | None = None
    delta_energy: float = 0.0

    def __repr__(self) -> str:
        """String representation."""
        if self.target_index is not None:
            return (
                f"Event({self.event_type.value}, "
                f"site={self.site_index}->{self.target_index}, "
                f"rate={self.rate:.2e})"
            )
        return f"Event({self.event_type.value}, site={self.site_index}, rate={self.rate:.2e})"


class EventCatalog:
    """
    Catalog of all possible events in the system.

    This class maintains a list of all possible events and their rates,
    providing efficient access and updating capabilities.
    """

    def __init__(self) -> None:
        """Initialize empty event catalog."""
        self.events: list[Event] = []
        self.total_rate: float = 0.0
        self._cumulative_rates: list[float] = []

    def add_event(self, event: Event) -> None:
        """
        Add an event to the catalog.

        Args:
            event: Event to add.
        """
        self.events.append(event)
        self.total_rate += event.rate
        self._update_cumulative_rates()

    def remove_event(self, index: int) -> None:
        """
        Remove an event from the catalog.

        Args:
            index: Index of event to remove.
        """
        event = self.events.pop(index)
        self.total_rate -= event.rate
        self._update_cumulative_rates()

    def update_event_rate(self, index: int, new_rate: float) -> None:
        """
        Update the rate of an existing event.

        Args:
            index: Index of event to update.
            new_rate: New rate value.
        """
        old_rate = self.events[index].rate
        self.events[index].rate = new_rate
        self.total_rate += new_rate - old_rate
        self._update_cumulative_rates()

    def clear(self) -> None:
        """Clear all events from catalog."""
        self.events.clear()
        self.total_rate = 0.0
        self._cumulative_rates.clear()

    def _update_cumulative_rates(self) -> None:
        """Update cumulative rate list for event selection."""
        self._cumulative_rates = []
        cumsum = 0.0
        for event in self.events:
            cumsum += event.rate
            self._cumulative_rates.append(cumsum)

    def get_cumulative_rates(self) -> list[float]:
        """Get cumulative rates for event selection."""
        return self._cumulative_rates

    def __len__(self) -> int:
        """Number of events in catalog."""
        return len(self.events)

    def __repr__(self) -> str:
        """String representation."""
        return f"EventCatalog(n_events={len(self)}, total_rate={self.total_rate:.2e})"
