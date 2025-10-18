"""
Run KMC simulations with trained policy.

This script runs simulations and analyzes the results.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis import calculate_fractal_dimension, calculate_roughness, fit_family_vicsek
from src.kmc.simulator import KMCSimulator
from src.settings import settings

# Setup logging
logger = settings.setup_logging()


def run_classical_kmc() -> None:
    """Run classical KMC simulation without RL."""
    logger.info("Running classical KMC simulation")

    simulator = KMCSimulator(
        lattice_size=(
            settings.kmc.lattice_size_x,
            settings.kmc.lattice_size_y,
            settings.kmc.lattice_size_z,
        ),
        temperature=settings.kmc.temperature,
        deposition_rate=settings.kmc.deposition_rate,
    )

    # Storage for analysis
    times = []
    roughnesses = []

    def snapshot_callback(sim: KMCSimulator) -> None:
        """Callback to record data."""
        height_profile = sim.lattice.get_height_profile()
        roughness = calculate_roughness(height_profile)

        times.append(sim.time)
        roughnesses.append(roughness)

        if sim.step % 1000 == 0:
            logger.info(f"Step {sim.step}, Time {sim.time:.2e}s, Roughness {roughness:.3f}")

    # Run simulation
    simulator.run(
        max_steps=settings.kmc.max_steps,
        max_time=settings.kmc.simulation_time,
        callback=snapshot_callback,
        snapshot_interval=100,
    )

    # Analysis
    height_profile = simulator.lattice.get_height_profile()
    final_roughness = calculate_roughness(height_profile)
    fractal_dim = calculate_fractal_dimension(height_profile)

    logger.info(f"Final roughness: {final_roughness:.3f}")
    logger.info(f"Fractal dimension: {fractal_dim:.3f}")

    # Fit Family-Vicsek
    if len(times) > 10:
        times_arr = np.array(times)
        roughnesses_arr = np.array(roughnesses)
        scaling = fit_family_vicsek(times_arr, roughnesses_arr, settings.kmc.lattice_size_x)
        logger.info(f"Scaling exponents: α={scaling['alpha']:.3f}, β={scaling['beta']:.3f}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.loglog(times_arr, roughnesses_arr, "o-", label="Simulation")
        plt.xlabel("Time (s)")
        plt.ylabel("Roughness W(L,t)")
        plt.title("Roughness Evolution - Classical KMC")
        plt.legend()
        plt.grid(True, alpha=0.3)

        results_dir = settings.paths.results_dir
        plt.savefig(results_dir / "roughness_classical.png", dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {results_dir / 'roughness_classical.png'}")
        plt.close()


def main() -> None:
    """Main execution."""
    logger.info("Starting simulation runs")

    # Run classical KMC
    run_classical_kmc()

    logger.info("Simulations completed")


if __name__ == "__main__":
    main()
