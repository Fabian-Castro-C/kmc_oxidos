"""
Training module for SwarmThinkers policy optimization.

This module contains all components needed for training RL policies:
- Reward function engineering
- Custom callbacks for monitoring
- Curriculum learning utilities

All dependencies are included in the base project installation.
"""

from __future__ import annotations

from .callbacks import (
    EpisodeSummaryCallback,
    ESSMonitorCallback,
    MorphologyLoggerCallback,
)

__all__ = [
    "ESSMonitorCallback",
    "MorphologyLoggerCallback",
    "EpisodeSummaryCallback",
]
