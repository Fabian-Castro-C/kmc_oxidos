"""
Validation Experiment 1: Basic KMC Simulation (No RL)

This experiment validates the core KMC simulator functionality:
- Surface site identification
- Adsorption events (Ti vs O rates)
- Diffusion events
- Temporal evolution (roughness, coverage)
- Performance metrics

Output: Logs, metrics JSON, and plots in experiments/results/validate_kmc_basic/{timestamp}/
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import calculate_roughness, fit_family_vicsek
from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import SpeciesType
from src.kmc.simulator import KMCSimulator
from src.settings import settings

# Setup logging from settings (.env file)
logger = settings.setup_logging()


class ExperimentConfig:
    """Configuration for the validation experiment."""

    def __init__(
        self,
        lattice_size=(30, 30, 20),
        temperature=600.0,
        deposition_rate=0.1,
        max_steps=100,
        seed=42,
        n_snapshots=10,
    ):
        self.name = "validate_kmc_basic"
        self.lattice_size = lattice_size
        self.temperature = temperature  # K
        self.deposition_rate = deposition_rate  # ML/s
        self.max_steps = max_steps
        self.seed = seed
        self.n_snapshots = n_snapshots

    def to_dict(self):
        return {
            "name": self.name,
            "lattice_size": list(self.lattice_size),
            "temperature": self.temperature,
            "deposition_rate": self.deposition_rate,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "n_snapshots": self.n_snapshots,
        }


class ExperimentResults:
    """Container for experiment results and validation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = (
            Path(__file__).parent / "results" / config.name / self.timestamp
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Time series data
        self.times = []
        self.steps = []
        self.roughnesses = []
        self.coverages = []
        self.n_ti_list = []
        self.n_o_list = []
        self.n_ti_free_list = []
        self.n_ti_oxide_list = []
        self.n_o_free_list = []
        self.n_o_oxide_list = []

        # Final metrics
        self.final_metrics = {}
        self.validation_status = {}
        self.performance_metrics = {}

    def record_snapshot(self, sim: KMCSimulator):
        """Record current state of simulation."""
        height_profile = sim.lattice.get_height_profile()
        roughness = calculate_roughness(height_profile)
        coverage = float(height_profile.mean())

        # Count species (total and detailed)
        n_ti = sum(1 for site in sim.lattice.sites if site.species == SpeciesType.TI)
        n_o = sum(1 for site in sim.lattice.sites if site.species == SpeciesType.O)

        # Get detailed composition (free vs oxide)
        detailed = sim.lattice.get_composition_detailed()

        self.times.append(sim.time)
        self.steps.append(sim.step)
        self.roughnesses.append(roughness)
        self.coverages.append(coverage)
        self.n_ti_list.append(n_ti)
        self.n_o_list.append(n_o)
        self.n_ti_free_list.append(detailed['ti_free'])
        self.n_ti_oxide_list.append(detailed['ti_oxide'])
        self.n_o_free_list.append(detailed['o_free'])
        self.n_o_oxide_list.append(detailed['o_oxide'])

    def compute_final_metrics(self, sim: KMCSimulator, duration_s: float):
        """Compute final metrics and validation checks."""
        # Final state
        self.final_metrics = {
            "final_step": sim.step,
            "final_time": sim.time,
            "final_roughness": self.roughnesses[-1] if self.roughnesses else 0.0,
            "final_coverage": self.coverages[-1] if self.coverages else 0.0,
            "composition": {
                "n_ti": self.n_ti_list[-1] if self.n_ti_list else 0,
                "n_o": self.n_o_list[-1] if self.n_o_list else 0,
                "ratio_o_ti": (
                    self.n_o_list[-1] / self.n_ti_list[-1]
                    if self.n_ti_list and self.n_ti_list[-1] > 0
                    else 0.0
                ),
            },
        }

        # Scaling exponents (if enough data)
        if len(self.times) > 5 and len(self.roughnesses) > 5:
            try:
                times_array = np.array(self.times)
                roughnesses_array = np.array(self.roughnesses)
                system_size = float(
                    np.sqrt(self.config.lattice_size[0] * self.config.lattice_size[1])
                )

                scaling = fit_family_vicsek(times_array, roughnesses_array, system_size)
                self.final_metrics["scaling_exponents"] = {
                    "alpha": float(scaling["alpha"]),
                    "beta": float(scaling["beta"]),
                }
            except Exception as e:
                logger.warning(f"Could not fit scaling exponents: {e}")
                self.final_metrics["scaling_exponents"] = {
                    "alpha": None,
                    "beta": None,
                    "error": str(e),
                }
        else:
            self.final_metrics["scaling_exponents"] = {
                "alpha": None,
                "beta": None,
                "note": "Not enough data points",
            }

        # Performance metrics
        self.performance_metrics = {
            "steps_per_second": sim.step / duration_s if duration_s > 0 else 0.0,
            "total_duration_s": duration_s,
            "avg_time_per_step_ms": (duration_s / sim.step * 1000) if sim.step > 0 else 0.0,
        }

        # Validation checks
        self.validation_status = self._run_validation_checks(sim)

    def _run_validation_checks(self, sim: KMCSimulator) -> dict:
        """Run validation assertions."""
        checks = {}

        # 1. Roughness should increase over time
        checks["roughness_increased"] = (
            self.roughnesses[-1] > self.roughnesses[0] if len(self.roughnesses) > 1 else False
        )

        # 2. Coverage should be positive and increasing
        checks["coverage_positive"] = self.coverages[-1] > 0 if self.coverages else False
        checks["coverage_increased"] = (
            self.coverages[-1] > self.coverages[0] if len(self.coverages) > 1 else False
        )

        # 3. Ti and O should have different adsorption rates
        # Check by comparing rates for a surface site
        if sim.lattice.surface_sites:
            site_idx = list(sim.lattice.surface_sites)[0]
            site = sim.lattice.get_site_by_index(site_idx)
            rate_ti = sim.rate_calculator.calculate_adsorption_rate(site, SpeciesType.TI)
            rate_o = sim.rate_calculator.calculate_adsorption_rate(site, SpeciesType.O)
            checks["rates_ti_neq_o"] = abs(rate_ti - rate_o) > 1e-6
            checks["rate_ti"] = float(rate_ti)
            checks["rate_o"] = float(rate_o)
        else:
            checks["rates_ti_neq_o"] = False
            checks["rate_ti"] = None
            checks["rate_o"] = None

        # 4. Performance should be acceptable (>10 steps/s for this size)
        checks["performance_acceptable"] = self.performance_metrics["steps_per_second"] > 10.0

        # 5. Both species should be present
        checks["both_species_present"] = (
            self.n_ti_list[-1] > 0 and self.n_o_list[-1] > 0 if self.n_ti_list and self.n_o_list else False
        )

        # Overall status
        critical_checks = [
            "roughness_increased",
            "coverage_positive",
            "rates_ti_neq_o",
            "performance_acceptable",
            "both_species_present",
        ]
        checks["_overall_pass"] = all(checks.get(k, False) for k in critical_checks)

        return checks

    def generate_plots(self, sim: KMCSimulator):
        """Generate all validation plots."""
        logger.info(f"Generating plots in {self.output_dir}")

        # Plot 1: Roughness evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.times, self.roughnesses, "b-", linewidth=2, label="Roughness")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Roughness (Å)", fontsize=12)
        ax.set_title("Roughness Evolution - KMC Basic Validation", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_01_roughness_evolution.png", dpi=150)
        plt.close()

        # Plot 2: Coverage evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.times, self.coverages, "g-", linewidth=2, label="Coverage")
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Coverage (ML)", fontsize=12)
        ax.set_title("Surface Coverage Evolution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_02_coverage_evolution.png", dpi=150)
        plt.close()

        # Plot 3: Composition evolution (Ti vs O, free vs oxide)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot stacked areas or lines
        ax.plot(self.steps, self.n_ti_free_list, "-", color="#ff6b6b", linewidth=2,
                label="Ti free", marker="o", markersize=3, markevery=max(1, len(self.steps)//20))
        ax.plot(self.steps, self.n_ti_oxide_list, "-", color="#8b0000", linewidth=2,
                label="Ti in TiO2", marker="s", markersize=3, markevery=max(1, len(self.steps)//20))
        ax.plot(self.steps, self.n_o_free_list, "-", color="#4dabf7", linewidth=2,
                label="O free", marker="^", markersize=3, markevery=max(1, len(self.steps)//20))
        ax.plot(self.steps, self.n_o_oxide_list, "-", color="#1864ab", linewidth=2,
                label="O in TiO2", marker="D", markersize=3, markevery=max(1, len(self.steps)//20))

        ax.set_xlabel("KMC Steps", fontsize=12)
        ax.set_ylabel("Number of Atoms", fontsize=12)
        ax.set_title("Composition Evolution (Free vs Bonded)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_03_composition_evolution.png", dpi=150)
        plt.close()

        # Plot 4: Final height profile
        height_profile = sim.lattice.get_height_profile()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(height_profile, cmap="viridis", interpolation="nearest")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title("Final Height Profile", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Height (ML)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_04_height_profile_final.png", dpi=150)
        plt.close()

        logger.info("[OK] All plots generated successfully")

    def save_results(self):
        """Save all results to JSON files."""
        # Save configuration
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save metrics
        metrics = {
            "experiment_name": self.config.name,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "results": self.final_metrics,
            "performance": self.performance_metrics,
            "validation_status": self.validation_status,
        }

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save time series data
        timeseries = {
            "times": self.times,
            "steps": self.steps,
            "roughnesses": self.roughnesses,
            "coverages": self.coverages,
            "n_ti": self.n_ti_list,
            "n_o": self.n_o_list,
            "n_ti_free": self.n_ti_free_list,
            "n_ti_oxide": self.n_ti_oxide_list,
            "n_o_free": self.n_o_free_list,
            "n_o_oxide": self.n_o_oxide_list,
        }

        timeseries_path = self.output_dir / "timeseries.json"
        with open(timeseries_path, "w") as f:
            json.dump(timeseries, f, indent=2)

        logger.info(f"[OK] Results saved to {self.output_dir}")

        return metrics


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run the validation experiment."""
    logger.info("=" * 80)
    logger.info("VALIDATION EXPERIMENT: Basic KMC Simulation (No RL)")
    logger.info("=" * 80)

    # Initialize results container
    results = ExperimentResults(config)

    # Create simulator
    logger.info(f"Creating simulator with config: {config.to_dict()}")
    params = TiO2Parameters()
    sim = KMCSimulator(
        lattice_size=config.lattice_size,
        temperature=config.temperature,
        deposition_rate=config.deposition_rate,
        params=params,
        seed=config.seed,
    )

    # Log initial state
    logger.info(f"Initial surface sites: {len(sim.lattice.surface_sites)}")
    logger.info(f"Total lattice sites: {len(sim.lattice.sites)}")

    # Run simulation with periodic snapshots
    steps_between = config.max_steps // config.n_snapshots
    logger.info(
        f"Running {config.max_steps} steps with {config.n_snapshots} snapshots "
        f"(every {steps_between} steps)"
    )

    start_time = time.time()

    for i in range(config.n_snapshots):
        # Run batch of steps
        for _ in range(steps_between):
            sim.run_step()

        # Record snapshot
        results.record_snapshot(sim)

        # Log progress
        progress_pct = (i + 1) / config.n_snapshots * 100
        logger.info(
            f"Snapshot {i+1}/{config.n_snapshots} ({progress_pct:.1f}%): "
            f"Step={sim.step}, Time={sim.time:.2f}s, "
            f"Roughness={results.roughnesses[-1]:.3f}Å, "
            f"Coverage={results.coverages[-1]:.2f}ML"
        )

    duration = time.time() - start_time

    # Compute final metrics
    results.compute_final_metrics(sim, duration)

    # Log final results
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Steps completed: {sim.step}")
    logger.info(f"Simulation time: {sim.time:.2f} s")
    logger.info(f"Wall time: {duration:.2f} s")
    logger.info(f"Performance: {results.performance_metrics['steps_per_second']:.1f} steps/s")
    logger.info(f"Final roughness: {results.final_metrics['final_roughness']:.3f} Å")
    logger.info(f"Final coverage: {results.final_metrics['final_coverage']:.2f} ML")
    logger.info(
        f"Composition: Ti={results.final_metrics['composition']['n_ti']}, "
        f"O={results.final_metrics['composition']['n_o']}, "
        f"O/Ti ratio={results.final_metrics['composition']['ratio_o_ti']:.3f}"
    )

    if results.final_metrics["scaling_exponents"]["alpha"] is not None:
        logger.info(
            f"Scaling exponents: alpha={results.final_metrics['scaling_exponents']['alpha']:.3f}, "
            f"beta={results.final_metrics['scaling_exponents']['beta']:.3f}"
        )

    # Print event statistics
    logger.info("=" * 80)
    sim.print_event_statistics()
    logger.info("=" * 80)

    # Log validation status
    logger.info("=" * 80)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 80)
    for check_name, check_result in results.validation_status.items():
        if check_name.startswith("_"):
            continue  # Skip internal keys
        status_symbol = "[PASS]" if check_result else "[FAIL]"
        logger.info(f"{status_symbol} {check_name}: {check_result}")

    overall_pass = results.validation_status.get("_overall_pass", False)
    if overall_pass:
        logger.info("=" * 80)
        logger.info("VALIDATION: PASS - All critical checks passed!")
        logger.info("=" * 80)
    else:
        logger.warning("=" * 80)
        logger.warning("VALIDATION: FAIL - Some checks did not pass")
        logger.warning("=" * 80)

    # Generate plots
    results.generate_plots(sim)

    # Save results
    results.save_results()

    return results


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Validate basic KMC simulation without RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        nargs=3,
        default=[30, 30, 20],
        metavar=("X", "Y", "Z"),
        help="Lattice dimensions (X Y Z)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=600.0,
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
        default=100,
        help="Maximum number of KMC steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-snapshots",
        type=int,
        default=10,
        help="Number of snapshots to record during simulation",
    )

    args = parser.parse_args()

    # Create configuration from arguments
    config = ExperimentConfig(
        lattice_size=tuple(args.lattice_size),
        temperature=args.temperature,
        deposition_rate=args.deposition_rate,
        max_steps=args.max_steps,
        seed=args.seed,
        n_snapshots=args.n_snapshots,
    )

    # Run experiment
    results = run_experiment(config)

    # Exit with appropriate code
    if results.validation_status.get("_overall_pass", False):
        logger.info("[OK] Experiment completed successfully")
        return 0
    else:
        logger.error("[FAIL] Experiment failed validation")
        return 1


if __name__ == "__main__":
    exit(main())
