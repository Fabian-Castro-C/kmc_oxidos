"""
Reweighting mechanism and importance sampling for SwarmThinkers.

Implements the fusion of learned policy with physical rates.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class ReweightingMechanism:
    """
    Reweighting mechanism to combine policy preferences with physical rates.

    P(a) = π_θ(a|o) · Γ_a / Σ π_θ(a'|o) · Γ_a'
    """

    def compute_reweighted_distribution(
        self,
        policy_probs: npt.NDArray[np.float64],
        physical_rates: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Compute reweighted probability distribution.

        Args:
            policy_probs: Policy probabilities π_θ(a|o).
            physical_rates: Physical rates Γ_a.

        Returns:
            Reweighted probabilities P(a).
        """
        numerator = policy_probs * physical_rates
        denominator = np.sum(numerator)

        if denominator == 0:
            return np.ones_like(policy_probs) / len(policy_probs)

        return numerator / denominator


class ImportanceSampler:
    """
    Importance sampling to correct bias from reweighting.

    Uses inverse policy weights to compute unbiased estimates.
    """

    def __init__(self) -> None:
        """Initialize importance sampler."""
        self.weights: list[float] = []

    def compute_importance_weight(
        self,
        reweighted_prob: float,
        physical_prob: float,
    ) -> float:
        """
        Compute importance weight for a sampled action.

        w = P_physical(a) / P_reweighted(a)

        Args:
            reweighted_prob: Reweighted probability.
            physical_prob: True physical probability.

        Returns:
            Importance weight.
        """
        if reweighted_prob == 0:
            return 0.0

        return physical_prob / reweighted_prob

    def add_weight(self, weight: float) -> None:
        """Add a weight to the trajectory."""
        self.weights.append(weight)

    def get_cumulative_weight(self) -> float:
        """Get cumulative importance weight for trajectory."""
        if not self.weights:
            return 1.0
        return float(np.prod(self.weights))

    def reset(self) -> None:
        """Reset weights."""
        self.weights.clear()
