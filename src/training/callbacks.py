"""
Custom callbacks for Stable-Baselines3 training.

This module provides callbacks for monitoring SwarmThinkers-specific metrics
during training, including ESS tracking and morphology logging.
"""

from __future__ import annotations

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class ESSMonitorCallback(BaseCallback):
    """
    Callback to monitor Effective Sample Size (ESS) during training.

    ESS measures the diversity of SwarmEngine proposals. Low ESS (<0.5)
    indicates concentrated importance weights, which can hurt learning.

    This callback:
    - Tracks ESS at each step from environment info
    - Logs ESS statistics to TensorBoard
    - Optionally stops training if ESS remains too low

    Attributes:
        ess_threshold: Minimum acceptable mean ESS (default 0.3).
        patience: Number of evaluations below threshold before stopping (default 10).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        ess_threshold: float = 0.3,
        patience: int = 10,
        verbose: int = 0,
    ) -> None:
        """
        Initialize ESS monitor callback.

        Args:
            ess_threshold: Minimum acceptable mean ESS before early stopping.
            patience: Number of consecutive low-ESS evaluations before stopping.
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
        """
        super().__init__(verbose)
        self.ess_threshold = ess_threshold
        self.patience = patience
        self.ess_values: list[float] = []
        self.low_ess_count = 0

    def _on_step(self) -> bool:
        """
        Called at each environment step.

        Returns:
            True to continue training, False to stop.
        """
        # Extract ESS from info dict (if available)
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "ess" in info:
                    ess = float(info["ess"])
                    self.ess_values.append(ess)

        # Log ESS statistics every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.ess_values) > 0:
            mean_ess = np.mean(self.ess_values[-1000:])
            min_ess = np.min(self.ess_values[-1000:])
            max_ess = np.max(self.ess_values[-1000:])

            # Log to TensorBoard
            self.logger.record("train/ess_mean", mean_ess)
            self.logger.record("train/ess_min", min_ess)
            self.logger.record("train/ess_max", max_ess)

            if self.verbose >= 1:
                logger.info(
                    f"Step {self.n_calls}: ESS mean={mean_ess:.4f}, "
                    f"min={min_ess:.4f}, max={max_ess:.4f}"
                )

            # Check for early stopping
            if mean_ess < self.ess_threshold:
                self.low_ess_count += 1
                if self.verbose >= 1:
                    logger.warning(
                        f"Low ESS detected ({mean_ess:.4f} < {self.ess_threshold}), "
                        f"count={self.low_ess_count}/{self.patience}"
                    )

                if self.low_ess_count >= self.patience:
                    logger.error(
                        f"Training stopped: ESS below {self.ess_threshold} "
                        f"for {self.patience} consecutive evaluations"
                    )
                    return False
            else:
                # Reset counter if ESS recovers
                self.low_ess_count = 0

        return True


class MorphologyLoggerCallback(BaseCallback):
    """
    Callback to log morphology metrics during training.

    Tracks and logs:
    - Surface roughness
    - Surface coverage
    - Roughness/coverage ratio
    - Ti/O composition

    Logs to TensorBoard for monitoring film quality during training.

    Attributes:
        log_freq: Frequency (in steps) for logging metrics.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 0,
    ) -> None:
        """
        Initialize morphology logger callback.

        Args:
            log_freq: Frequency (in steps) for logging metrics to TensorBoard.
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.roughness_values: list[float] = []
        self.coverage_values: list[float] = []
        self.n_ti_values: list[float] = []
        self.n_o_values: list[float] = []

    def _on_step(self) -> bool:
        """
        Called at each environment step.

        Returns:
            True to continue training.
        """
        # Extract morphology metrics from info dict
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "roughness" in info:
                    self.roughness_values.append(float(info["roughness"]))
                if "coverage" in info:
                    self.coverage_values.append(float(info["coverage"]))
                if "n_Ti" in info:
                    self.n_ti_values.append(float(info["n_Ti"]))
                if "n_O" in info:
                    self.n_o_values.append(float(info["n_O"]))

        # Log statistics at specified frequency
        if self.n_calls % self.log_freq == 0 and len(self.roughness_values) > 0:
            # Compute statistics over last log_freq steps
            recent_roughness = self.roughness_values[-self.log_freq :]
            recent_coverage = self.coverage_values[-self.log_freq :]
            recent_n_ti = self.n_ti_values[-self.log_freq :]
            recent_n_o = self.n_o_values[-self.log_freq :]

            mean_roughness = np.mean(recent_roughness)
            mean_coverage = np.mean(recent_coverage)
            mean_n_ti = np.mean(recent_n_ti)
            mean_n_o = np.mean(recent_n_o)

            # Compute roughness/coverage ratio
            ratio = mean_roughness / mean_coverage if mean_coverage > 0 else 0.0

            # Log to TensorBoard
            self.logger.record("morphology/roughness_mean", mean_roughness)
            self.logger.record("morphology/roughness_std", np.std(recent_roughness))
            self.logger.record("morphology/coverage_mean", mean_coverage)
            self.logger.record("morphology/coverage_std", np.std(recent_coverage))
            self.logger.record("morphology/roughness_coverage_ratio", ratio)
            self.logger.record("morphology/n_Ti_mean", mean_n_ti)
            self.logger.record("morphology/n_O_mean", mean_n_o)

            # Compute Ti/O stoichiometry (should approach 0.5 for TiO2)
            stoichiometry = mean_n_ti / (mean_n_ti + mean_n_o) if (mean_n_ti + mean_n_o) > 0 else 0.0
            self.logger.record("morphology/Ti_fraction", stoichiometry)

            if self.verbose >= 1:
                logger.info(
                    f"Step {self.n_calls}: roughness={mean_roughness:.4f}, "
                    f"coverage={mean_coverage:.4f}, ratio={ratio:.4f}, "
                    f"Ti_frac={stoichiometry:.4f}"
                )

        return True


class EpisodeSummaryCallback(BaseCallback):
    """
    Callback to log episode-level statistics.

    Logs summary metrics at the end of each episode, including:
    - Episode reward
    - Episode length
    - Final roughness and coverage
    - Mean ESS over the episode

    Attributes:
        verbose: Verbosity level.
    """

    def __init__(self, verbose: int = 1) -> None:
        """
        Initialize episode summary callback.

        Args:
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
        """
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called at each environment step.

        Returns:
            True to continue training.
        """
        # Check if any environment finished an episode
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_data = info["episode"]

                    # Only log if this is our custom episode dict (with final_roughness)
                    # Skip SB3's default episode wrapper dict
                    if "final_roughness" not in episode_data:
                        continue

                    self.episode_count += 1

                    # Log episode summary
                    self.logger.record("episode/reward", episode_data["r"])
                    self.logger.record("episode/length", episode_data["l"])
                    self.logger.record("episode/final_roughness", episode_data["final_roughness"])
                    self.logger.record("episode/final_coverage", episode_data["final_coverage"])
                    self.logger.record("episode/mean_ess", episode_data["mean_ess"])
                    self.logger.record("episode/count", self.episode_count)

                    if self.verbose >= 1:
                        logger.info(
                            f"Episode {self.episode_count} finished: "
                            f"reward={episode_data['r']:.2f}, "
                            f"length={episode_data['l']}, "
                            f"roughness={episode_data['final_roughness']:.4f}, "
                            f"coverage={episode_data['final_coverage']:.4f}, "
                            f"ESS={episode_data['mean_ess']:.4f}"
                        )

        return True
