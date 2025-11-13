"""
Validation Experiment: SwarmThinkers Integration

This experiment validates the SwarmThinkers implementation by comparing
classical KMC (unbiased sampling from full catalog) against SwarmThinkers
(policy-driven proposals with reweighting).

Phase 1: Diffusion-only events
- Generate proposals from DiffusionSwarmPolicy
- Compute physical rates Γ_a
- Reweight with P(a) = π_θ(a) · Γ_a / Z
- Validate statistical correctness (KS test)
- Measure effective sample size (ESS)

Output: Logs, metrics JSON, and plots in experiments/results/validate_swarmthinkers/{timestamp}/
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import calculate_roughness
from src.data.tio2_parameters import TiO2Parameters
from src.kmc.efficient_updates import update_events_after_execution
from src.kmc.lattice import SpeciesType
from src.kmc.rates import RateCalculator
from src.kmc.simulator import KMCSimulator
from src.rl import (
    AdsorptionSwarmPolicy,
    DesorptionSwarmPolicy,
    DiffusionSwarmPolicy,
    ReactionSwarmPolicy,
    SwarmEngine,
    create_adsorption_swarm_policy,
    create_desorption_swarm_policy,
    create_diffusion_swarm_policy,
    create_reaction_swarm_policy,
)
from src.settings import settings

# Setup logging from settings (.env file)
logger = settings.setup_logging()


class ExperimentConfig:
    """Configuration for SwarmThinkers validation experiment."""

    def __init__(
        self,
        lattice_size=(20, 20, 10),
        temperature=180.0,
        deposition_rate=0.1,
        max_steps=1000,
        seed=42,
        n_snapshots=10,
        n_trials=50,
        swarm_size=32,
        policy_checkpoint=None,
    ):
        self.name = "validate_swarmthinkers"
        self.lattice_size = lattice_size
        self.temperature = temperature  # K
        self.deposition_rate = deposition_rate  # ML/s
        self.max_steps = max_steps
        self.seed = seed
        self.n_snapshots = n_snapshots
        self.n_trials = n_trials  # Number of independent runs for statistical validation
        self.swarm_size = swarm_size  # Number of proposals per step
        self.policy_checkpoint = policy_checkpoint  # Path to pretrained policy (None = random init)

    def to_dict(self):
        return {
            "name": self.name,
            "lattice_size": list(self.lattice_size),
            "temperature": self.temperature,
            "deposition_rate": self.deposition_rate,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "n_snapshots": self.n_snapshots,
            "n_trials": self.n_trials,
            "swarm_size": self.swarm_size,
            "policy_checkpoint": str(self.policy_checkpoint) if self.policy_checkpoint else None,
        }


class TrialResults:
    """Results from a single trial (KMC or SwarmThinkers)."""

    def __init__(self, trial_id: int, mode: str):
        self.trial_id = trial_id
        self.mode = mode  # "kmc_classic" or "swarmthinkers"

        # Time series
        self.times = []
        self.steps = []
        self.roughnesses = []
        self.coverages = []
        self.n_ti_list = []
        self.n_o_list = []

        # SwarmThinkers-specific
        self.importance_weights = []  # Track importance weights w = 1/π(a)

        # Final state
        self.final_roughness = 0.0
        self.final_coverage = 0.0
        self.final_n_ti = 0
        self.final_n_o = 0
        self.duration_s = 0.0

    def record_snapshot(self, sim, importance_weight=None):
        """Record current state."""
        height_profile = sim.lattice.get_height_profile()
        roughness = calculate_roughness(height_profile)
        coverage = float(height_profile.mean())

        n_ti = sum(1 for site in sim.lattice.sites if site.species == SpeciesType.TI)
        n_o = sum(1 for site in sim.lattice.sites if site.species == SpeciesType.O)

        self.times.append(sim.time)
        self.steps.append(sim.step)
        self.roughnesses.append(roughness)
        self.coverages.append(coverage)
        self.n_ti_list.append(n_ti)
        self.n_o_list.append(n_o)

        if importance_weight is not None:
            self.importance_weights.append(importance_weight)

    def finalize(self, duration_s: float):
        """Compute final metrics."""
        self.final_roughness = self.roughnesses[-1] if self.roughnesses else 0.0
        self.final_coverage = self.coverages[-1] if self.coverages else 0.0
        self.final_n_ti = self.n_ti_list[-1] if self.n_ti_list else 0
        self.final_n_o = self.n_o_list[-1] if self.n_o_list else 0
        self.duration_s = duration_s


class ExperimentResults:
    """Container for multi-trial experiment results."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(__file__).parent / "results" / config.name / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store all trial results
        self.kmc_trials: list[TrialResults] = []
        self.swarm_trials: list[TrialResults] = []

        # Statistical comparison
        self.ks_test_results = {}
        self.ess_metrics = {}
        self.performance_comparison = {}
        self.validation_status = {}

    def add_trial(self, trial: TrialResults):
        """Add completed trial to results."""
        if trial.mode == "kmc_classic":
            self.kmc_trials.append(trial)
        elif trial.mode == "swarmthinkers":
            self.swarm_trials.append(trial)

    def compute_statistical_validation(self):
        """Run statistical tests comparing KMC vs SwarmThinkers distributions."""
        logger.info("Computing statistical validation (Kolmogorov-Smirnov tests)...")

        # Extract final distributions
        kmc_roughness = np.array([t.final_roughness for t in self.kmc_trials])
        swarm_roughness = np.array([t.final_roughness for t in self.swarm_trials])

        kmc_coverage = np.array([t.final_coverage for t in self.kmc_trials])
        swarm_coverage = np.array([t.final_coverage for t in self.swarm_trials])

        kmc_n_ti = np.array([t.final_n_ti for t in self.kmc_trials])
        swarm_n_ti = np.array([t.final_n_ti for t in self.swarm_trials])

        kmc_n_o = np.array([t.final_n_o for t in self.kmc_trials])
        swarm_n_o = np.array([t.final_n_o for t in self.swarm_trials])

        # Kolmogorov-Smirnov tests (H0: distributions are identical)
        # p-value > 0.05 => cannot reject H0 => distributions are statistically identical
        ks_roughness = stats.ks_2samp(kmc_roughness, swarm_roughness)
        ks_coverage = stats.ks_2samp(kmc_coverage, swarm_coverage)
        ks_n_ti = stats.ks_2samp(kmc_n_ti, swarm_n_ti)
        ks_n_o = stats.ks_2samp(kmc_n_o, swarm_n_o)

        self.ks_test_results = {
            "roughness": {
                "statistic": float(ks_roughness.statistic),
                "pvalue": float(ks_roughness.pvalue),
                "pass": ks_roughness.pvalue > 0.05,
            },
            "coverage": {
                "statistic": float(ks_coverage.statistic),
                "pvalue": float(ks_coverage.pvalue),
                "pass": ks_coverage.pvalue > 0.05,
            },
            "n_ti": {
                "statistic": float(ks_n_ti.statistic),
                "pvalue": float(ks_n_ti.pvalue),
                "pass": ks_n_ti.pvalue > 0.05,
            },
            "n_o": {
                "statistic": float(ks_n_o.statistic),
                "pvalue": float(ks_n_o.pvalue),
                "pass": ks_n_o.pvalue > 0.05,
            },
        }

        logger.info(
            f"  KS test (roughness): D={ks_roughness.statistic:.4f}, p={ks_roughness.pvalue:.4f}"
        )
        logger.info(
            f"  KS test (coverage): D={ks_coverage.statistic:.4f}, p={ks_coverage.pvalue:.4f}"
        )
        logger.info(f"  KS test (n_Ti): D={ks_n_ti.statistic:.4f}, p={ks_n_ti.pvalue:.4f}")
        logger.info(f"  KS test (n_O): D={ks_n_o.statistic:.4f}, p={ks_n_o.pvalue:.4f}")

    def compute_ess_metrics(self):
        """Compute effective sample size (ESS) from importance weights."""
        logger.info("Computing effective sample size (ESS) metrics...")

        if not self.swarm_trials:
            logger.warning("No SwarmThinkers trials found, skipping ESS calculation")
            return

        # Pool all importance weights across trials
        all_weights = []
        for trial in self.swarm_trials:
            all_weights.extend(trial.importance_weights)

        if not all_weights:
            logger.warning("No importance weights found, skipping ESS calculation")
            return

        weights = np.array(all_weights)

        # ESS = (Σw)² / Σ(w²)
        # ESS close to 1.0 => high variance (bad)
        # ESS close to N => low variance (good)
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights**2)
        ess = (sum_w**2) / sum_w2 if sum_w2 > 0 else 0.0
        ess_normalized = ess / len(weights)  # Normalize by total number of samples

        self.ess_metrics = {
            "total_samples": len(weights),
            "ess": float(ess),
            "ess_normalized": float(ess_normalized),
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "target_ess_normalized": 0.5,  # Target threshold
            "pass": ess_normalized > 0.5,
        }

        logger.info(f"  ESS: {ess:.1f} / {len(weights)} = {ess_normalized:.4f}")
        logger.info(f"  Mean weight: {np.mean(weights):.4f} ± {np.std(weights):.4f}")
        logger.info(f"  Weight range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")

    def compute_performance_comparison(self):
        """Compare walltime performance between KMC and SwarmThinkers."""
        logger.info("Computing performance comparison...")

        kmc_times = [t.duration_s for t in self.kmc_trials]
        swarm_times = [t.duration_s for t in self.swarm_trials]

        self.performance_comparison = {
            "kmc_mean_time_s": float(np.mean(kmc_times)) if kmc_times else 0.0,
            "kmc_std_time_s": float(np.std(kmc_times)) if kmc_times else 0.0,
            "swarm_mean_time_s": float(np.mean(swarm_times)) if swarm_times else 0.0,
            "swarm_std_time_s": float(np.std(swarm_times)) if swarm_times else 0.0,
            "speedup_ratio": (
                np.mean(kmc_times) / np.mean(swarm_times)
                if swarm_times and np.mean(swarm_times) > 0
                else 0.0
            ),
        }

        logger.info(f"  KMC mean time: {np.mean(kmc_times):.2f} ± {np.std(kmc_times):.2f} s")
        logger.info(f"  Swarm mean time: {np.mean(swarm_times):.2f} ± {np.std(swarm_times):.2f} s")
        logger.info(f"  Speedup ratio: {self.performance_comparison['speedup_ratio']:.2f}x")

    def run_validation_checks(self):
        """Run final validation assertions."""
        logger.info("Running validation checks...")

        checks = {}

        # 1. All KS tests should pass (p > 0.05)
        ks_pass = all(result["pass"] for result in self.ks_test_results.values())
        checks["ks_tests_pass"] = ks_pass

        # 2. ESS should be > 0.5 (target threshold)
        ess_pass = self.ess_metrics.get("pass", False) if self.ess_metrics else False
        checks["ess_acceptable"] = ess_pass

        # 3. Both methods should complete successfully
        checks["kmc_trials_complete"] = len(self.kmc_trials) == self.config.n_trials
        checks["swarm_trials_complete"] = len(self.swarm_trials) == self.config.n_trials

        # 4. Distributions should have reasonable means
        kmc_roughness_mean = np.mean([t.final_roughness for t in self.kmc_trials])
        swarm_roughness_mean = np.mean([t.final_roughness for t in self.swarm_trials])
        checks["roughness_positive"] = kmc_roughness_mean > 0 and swarm_roughness_mean > 0

        # Overall pass
        critical_checks = [
            "ks_tests_pass",
            "ess_acceptable",
            "kmc_trials_complete",
            "swarm_trials_complete",
            "roughness_positive",
        ]
        checks["_overall_pass"] = all(checks.get(k, False) for k in critical_checks)

        self.validation_status = checks

        for check_name, result in checks.items():
            status = "[PASS]" if result else "[FAIL]"
            logger.info(f"  {status} {check_name}")

    def generate_plots(self):
        """Generate comparison plots."""
        logger.info(f"Generating plots in {self.output_dir}")

        # Plot 1: Final roughness distributions (histogram + KS test)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        kmc_roughness = [t.final_roughness for t in self.kmc_trials]
        swarm_roughness = [t.final_roughness for t in self.swarm_trials]

        axes[0].hist(
            kmc_roughness, bins=15, alpha=0.6, label="KMC Classic", color="blue", density=True
        )
        axes[0].hist(
            swarm_roughness, bins=15, alpha=0.6, label="SwarmThinkers", color="red", density=True
        )
        axes[0].set_xlabel("Final Roughness (Å)", fontsize=12)
        axes[0].set_ylabel("Density", fontsize=12)
        axes[0].set_title("Roughness Distribution Comparison", fontsize=13, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Q-Q plot for roughness
        stats.probplot(kmc_roughness, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot: KMC Roughness vs Normal", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_01_roughness_comparison.png", dpi=150)
        plt.close()

        # Plot 2: Coverage distributions
        fig, ax = plt.subplots(figsize=(10, 6))

        kmc_coverage = [t.final_coverage for t in self.kmc_trials]
        swarm_coverage = [t.final_coverage for t in self.swarm_trials]

        ax.hist(kmc_coverage, bins=15, alpha=0.6, label="KMC Classic", color="blue", density=True)
        ax.hist(
            swarm_coverage, bins=15, alpha=0.6, label="SwarmThinkers", color="red", density=True
        )
        ax.set_xlabel("Final Coverage (ML)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Coverage Distribution Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_02_coverage_comparison.png", dpi=150)
        plt.close()

        # Plot 3: Importance weights distribution (SwarmThinkers only)
        if self.swarm_trials and self.swarm_trials[0].importance_weights:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            all_weights = []
            for trial in self.swarm_trials:
                all_weights.extend(trial.importance_weights)

            axes[0].hist(all_weights, bins=50, alpha=0.7, color="green", density=True)
            axes[0].set_xlabel("Importance Weight w = 1/π(a)", fontsize=12)
            axes[0].set_ylabel("Density", fontsize=12)
            axes[0].set_title("Importance Weights Distribution", fontsize=13, fontweight="bold")
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(
                np.mean(all_weights),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean = {np.mean(all_weights):.3f}",
            )
            axes[0].legend()

            # Cumulative ESS over time
            cumulative_ess = []
            for i in range(1, len(all_weights) + 1):
                w = np.array(all_weights[:i])
                ess_i = (np.sum(w) ** 2) / np.sum(w**2) / i if i > 0 else 0.0
                cumulative_ess.append(ess_i)

            axes[1].plot(cumulative_ess, color="purple", linewidth=2)
            axes[1].axhline(0.5, color="red", linestyle="--", linewidth=2, label="Target ESS = 0.5")
            axes[1].set_xlabel("Number of Samples", fontsize=12)
            axes[1].set_ylabel("Normalized ESS", fontsize=12)
            axes[1].set_title("Cumulative ESS Evolution", fontsize=13, fontweight="bold")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / "plot_03_importance_weights.png", dpi=150)
            plt.close()

        # Plot 4: Example trajectory comparison (first trial from each)
        if self.kmc_trials and self.swarm_trials:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            kmc_example = self.kmc_trials[0]
            swarm_example = self.swarm_trials[0]

            # Roughness evolution
            axes[0].plot(
                kmc_example.steps,
                kmc_example.roughnesses,
                "b-",
                linewidth=2,
                label="KMC Classic",
                alpha=0.8,
            )
            axes[0].plot(
                swarm_example.steps,
                swarm_example.roughnesses,
                "r--",
                linewidth=2,
                label="SwarmThinkers",
                alpha=0.8,
            )
            axes[0].set_xlabel("KMC Steps", fontsize=12)
            axes[0].set_ylabel("Roughness (Å)", fontsize=12)
            axes[0].set_title(
                "Example Trajectory: Roughness Evolution", fontsize=13, fontweight="bold"
            )
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Composition evolution
            axes[1].plot(
                kmc_example.steps,
                kmc_example.n_ti_list,
                "b-",
                linewidth=2,
                label="KMC Ti",
                alpha=0.8,
            )
            axes[1].plot(
                kmc_example.steps,
                kmc_example.n_o_list,
                "b--",
                linewidth=2,
                label="KMC O",
                alpha=0.8,
            )
            axes[1].plot(
                swarm_example.steps,
                swarm_example.n_ti_list,
                "r-",
                linewidth=2,
                label="Swarm Ti",
                alpha=0.8,
            )
            axes[1].plot(
                swarm_example.steps,
                swarm_example.n_o_list,
                "r--",
                linewidth=2,
                label="Swarm O",
                alpha=0.8,
            )
            axes[1].set_xlabel("KMC Steps", fontsize=12)
            axes[1].set_ylabel("Number of Atoms", fontsize=12)
            axes[1].set_title(
                "Example Trajectory: Composition Evolution", fontsize=13, fontweight="bold"
            )
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "plot_04_example_trajectories.png", dpi=150)
            plt.close()

        logger.info("[OK] All plots generated successfully")

    def save_results(self):
        """Save all results to JSON files."""
        # Helper to convert numpy types to Python types
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Save configuration
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save metrics
        metrics = {
            "experiment_name": self.config.name,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "ks_test_results": convert_to_serializable(self.ks_test_results),
            "ess_metrics": convert_to_serializable(self.ess_metrics),
            "performance_comparison": convert_to_serializable(self.performance_comparison),
            "validation_status": convert_to_serializable(self.validation_status),
            "n_kmc_trials": len(self.kmc_trials),
            "n_swarm_trials": len(self.swarm_trials),
        }

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"[OK] Results saved to {self.output_dir}")


def run_kmc_trial(config: ExperimentConfig, trial_id: int) -> TrialResults:
    """Run single KMC classic trial."""
    trial = TrialResults(trial_id, "kmc_classic")

    # Create simulator with unique seed
    params = TiO2Parameters()
    sim = KMCSimulator(
        lattice_size=config.lattice_size,
        temperature=config.temperature,
        deposition_rate=config.deposition_rate,
        params=params,
        seed=config.seed + trial_id,
    )

    steps_between = config.max_steps // config.n_snapshots
    start_time = time.time()

    for i in range(config.n_snapshots):
        target_step = (i + 1) * steps_between
        # Run KMC steps until target
        while sim.step < target_step:
            sim.run_step()
        trial.record_snapshot(sim)

    duration = time.time() - start_time
    trial.finalize(duration)

    return trial


def run_swarmthinkers_trial(
    config: ExperimentConfig,
    trial_id: int,
    diffusion_policy: DiffusionSwarmPolicy,
    adsorption_policy: AdsorptionSwarmPolicy,
    desorption_policy: DesorptionSwarmPolicy,
    reaction_policy: ReactionSwarmPolicy,
    device: torch.device,
) -> TrialResults:
    """Run single SwarmThinkers trial."""
    trial = TrialResults(trial_id, "swarmthinkers")

    # Create simulator with unique seed
    params = TiO2Parameters()
    sim = KMCSimulator(
        lattice_size=config.lattice_size,
        temperature=config.temperature,
        deposition_rate=config.deposition_rate,
        params=params,
        seed=config.seed + trial_id + 1000,  # Offset to avoid overlap with KMC seeds
    )

    # Create rate calculator and swarm engine with ALL 4 policies
    rate_calculator = RateCalculator(
        temperature=config.temperature,
        deposition_rate=config.deposition_rate,
        params=params,
    )
    swarm_engine = SwarmEngine(
        diffusion_policy=diffusion_policy,
        adsorption_policy=adsorption_policy,
        desorption_policy=desorption_policy,
        reaction_policy=reaction_policy,
        rate_calculator=rate_calculator,
        device=device,
    )

    steps_between = config.max_steps // config.n_snapshots
    start_time = time.time()

    # Diagnostic counters
    event_counts = {
        "adsorption_ti": 0,
        "adsorption_o": 0,
        "diffusion_ti": 0,
        "diffusion_o": 0,
        "desorption_ti": 0,
        "desorption_o": 0,
        "reaction_tio2": 0,
    }

    for i in range(config.n_snapshots):
        target_step = (i + 1) * steps_between

        # Run swarm-based steps
        while sim.step < target_step:
            # Generate event via swarm engine (ALL events policy-driven, NO event_catalog)
            event, importance_weight = swarm_engine.run_step(
                sim.lattice, n_swarm=config.swarm_size
            )

            # If no valid events available
            if event is None:
                logger.warning("No events available in SwarmThinkers, stopping trial")
                break

            # Execute swarm-selected event using full KMC step mechanics
            # (execute event, update catalog, advance time, increment step)
            sim.execute_event(event)
            update_events_after_execution(sim, event)
            sim.advance_time()
            sim.step += 1

            # Count event type
            event_counts[event.event_type.value] += 1

            # Store importance weight for ESS calculation
            trial.importance_weights.append(importance_weight)

        trial.record_snapshot(sim)

    duration = time.time() - start_time
    trial.finalize(duration)

    # Log event distribution
    total_events = sum(event_counts.values())
    logger.info(f"    [DEBUG] Event distribution ({total_events} total):")
    for event_type, count in event_counts.items():
        pct = 100 * count / total_events if total_events > 0 else 0
        logger.info(f"      {event_type}: {count} ({pct:.1f}%)")

    return trial


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run the full validation experiment."""
    logger.info("=" * 80)
    logger.info("VALIDATION EXPERIMENT: SwarmThinkers vs KMC Classic")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config.to_dict()}")

    results = ExperimentResults(config)
    device = torch.device(settings.get_device())

    # Initialize ALL 4 policies (complete SwarmThinkers framework)
    logger.info(f"Initializing 4 SwarmThinkers policies on {device}")
    diffusion_policy = create_diffusion_swarm_policy()
    adsorption_policy = create_adsorption_swarm_policy()
    desorption_policy = create_desorption_swarm_policy()
    reaction_policy = create_reaction_swarm_policy()

    # Move to device and set eval mode
    diffusion_policy.to(device).eval()
    adsorption_policy.to(device).eval()
    desorption_policy.to(device).eval()
    reaction_policy.to(device).eval()

    if config.policy_checkpoint:
        logger.info(f"Loading policies from {config.policy_checkpoint}")
        # TODO: Load checkpoint with all 4 policies
        # For now, only diffusion policy checkpoint is supported
        diffusion_policy.load_state_dict(torch.load(config.policy_checkpoint, map_location=device))
    else:
        logger.info("Using randomly initialized policies (Phase 1 validation)")

    # Run KMC classic trials
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running {config.n_trials} KMC Classic trials...")
    logger.info(f"{'=' * 80}")

    for trial_id in range(config.n_trials):
        logger.info(f"  Trial {trial_id + 1}/{config.n_trials}...")
        trial = run_kmc_trial(config, trial_id)
        results.add_trial(trial)
        logger.info(
            f"    Completed: roughness={trial.final_roughness:.3f}, "
            f"coverage={trial.final_coverage:.2f}, "
            f"time={trial.duration_s:.2f}s"
        )

    # Run SwarmThinkers trials
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running {config.n_trials} SwarmThinkers trials...")
    logger.info(f"{'=' * 80}")

    for trial_id in range(config.n_trials):
        logger.info(f"  Trial {trial_id + 1}/{config.n_trials}...")
        trial = run_swarmthinkers_trial(
            config,
            trial_id,
            diffusion_policy,
            adsorption_policy,
            desorption_policy,
            reaction_policy,
            device,
        )
        results.add_trial(trial)
        logger.info(
            f"    Completed: roughness={trial.final_roughness:.3f}, "
            f"coverage={trial.final_coverage:.2f}, "
            f"time={trial.duration_s:.2f}s, "
            f"ESS_samples={len(trial.importance_weights)}"
        )

    # Statistical validation
    logger.info(f"\n{'=' * 80}")
    logger.info("STATISTICAL VALIDATION")
    logger.info(f"{'=' * 80}")

    results.compute_statistical_validation()
    results.compute_ess_metrics()
    results.compute_performance_comparison()
    results.run_validation_checks()

    # Generate plots
    results.generate_plots()

    # Save results
    results.save_results()

    # Final summary
    logger.info(f"\n{'=' * 80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'=' * 80}")

    overall_pass = results.validation_status.get("_overall_pass", False)
    if overall_pass:
        logger.info(
            "[PASS] VALIDATION PASSED: SwarmThinkers produces statistically identical distributions"
        )
    else:
        logger.warning("[FAIL] VALIDATION FAILED: Check metrics and plots for details")

    logger.info(f"\nResults saved to: {results.output_dir}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate SwarmThinkers implementation against KMC classic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        nargs=3,
        default=[20, 20, 10],
        metavar=("X", "Y", "Z"),
        help="Lattice dimensions (X Y Z)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=180.0,
        help="Temperature in Kelvin",
    )
    parser.add_argument(
        "--deposition-rate",
        type=float,
        default=0.1,
        help="Deposition rate in ML/s",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of KMC steps per trial",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=10,
        help="Number of snapshots per trial",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of independent trials for each method",
    )
    parser.add_argument(
        "--swarm-size",
        type=int,
        default=32,
        help="Number of swarm proposals per step",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained policy checkpoint (None = random init)",
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        lattice_size=tuple(args.lattice_size),
        temperature=args.temperature,
        deposition_rate=args.deposition_rate,
        max_steps=args.max_steps,
        seed=args.seed,
        n_snapshots=args.n_snapshots,
        n_trials=args.n_trials,
        swarm_size=args.swarm_size,
        policy_checkpoint=args.policy_checkpoint,
    )

    results = run_experiment(config)

    if results.validation_status.get("_overall_pass", False):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
