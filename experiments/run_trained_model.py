"""
Run simulations using a trained RL policy.

This script loads a trained PPO model and runs KMC simulations where the policy
selects the optimal n_swarm parameter at each step. Outputs are similar to
kmc_basic: roughness evolution, morphology snapshots, and final statistics.

Output: Results in experiments/results/swarm_predict/{timestamp}/

Usage:
    python experiments/run_trained_model.py --model path/to/model.zip
    python experiments/run_trained_model.py --model training_runs/stage1/stage1_final.zip --n-trials 10
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.roughness import calculate_roughness
from src.kmc.simulator import KMCSimulator
from src.rl.swarm_engine import SwarmEngine
from src.rl.swarm_policy import (
    create_adsorption_swarm_policy,
    create_desorption_swarm_policy,
    create_diffusion_swarm_policy,
    create_reaction_swarm_policy,
)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_simulation_with_policy(
    model: PPO,
    lattice_size: tuple[int, int, int],
    temperature: float,
    deposition_rate: float,
    max_steps: int,
    save_snapshots: bool = True,
    snapshot_interval: int | None = None,
) -> dict[str, any]:
    """
    Run a single KMC simulation using the trained RL policy.

    At each step, the policy selects which n_swarm to use (4, 8, 16, or 32).

    Args:
        model: Trained PPO model.
        lattice_size: (nx, ny, nz) lattice dimensions.
        temperature: Simulation temperature (K).
        deposition_rate: Deposition rate (ML/s).
        max_steps: Maximum number of KMC steps.
        save_snapshots: Whether to save lattice snapshots.
        snapshot_interval: Steps between snapshots (default: max_steps // 10).

    Returns:
        Dictionary with simulation results:
            - roughness: List of roughness values over time
            - coverage: List of coverage values over time
            - ti_fraction: List of Ti fraction over time
            - o_fraction: List of O fraction over time
            - snapshots: List of (step, heights) tuples if save_snapshots=True
            - actions: List of actions (n_swarm choices) taken by policy
            - final_stats: Dict with final statistics
    """
    if snapshot_interval is None:
        snapshot_interval = max(1, max_steps // 10)

    # Initialize simulator
    simulator = KMCSimulator(
        lattice_size=lattice_size,
        temperature=temperature,
        deposition_rate=deposition_rate,
    )

    # Initialize SwarmEngine
    diffusion_policy = create_diffusion_swarm_policy()
    adsorption_policy = create_adsorption_swarm_policy()
    desorption_policy = create_desorption_swarm_policy()
    reaction_policy = create_reaction_swarm_policy()

    swarm_engine = SwarmEngine(
        diffusion_policy=diffusion_policy,
        adsorption_policy=adsorption_policy,
        desorption_policy=desorption_policy,
        reaction_policy=reaction_policy,
        rate_calculator=simulator.rate_calculator,
        device="cpu",
    )

    # Tracking arrays
    roughness_history = []
    coverage_history = []
    ti_fraction_history = []
    o_fraction_history = []
    actions_taken = []
    snapshots = []

    logger.info(f"Starting RL-guided simulation: {max_steps} steps")
    logger.info(f"Lattice: {lattice_size}, T={temperature}K, deposition={deposition_rate}ML/s")

    for step in range(max_steps):
        # Get observation (same as in gym_environment)
        heights = simulator.lattice.get_height_profile()
        height_mean = float(np.mean(heights))
        height_std = float(np.std(heights))
        roughness = calculate_roughness(heights)

        nx, ny, _ = lattice_size
        total_surface_sites = nx * ny
        composition = simulator.lattice.get_composition()
        n_Ti = composition.get("Ti", 0)
        n_O = composition.get("O", 0)
        n_vacant = total_surface_sites - n_Ti - n_O
        coverage = (n_Ti + n_O) / total_surface_sites

        # Create observation array for policy
        obs = np.array(
            [height_mean, height_std, roughness, coverage, n_Ti, n_O, n_vacant],
            dtype=np.float32,
        )

        # Policy selects action (deterministic for inference)
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        # Map action to n_swarm: 0->4, 1->8, 2->16, 3->32
        n_swarm = 4 * (2**action)
        actions_taken.append(n_swarm)

        # Execute swarm step
        event, importance_weight = swarm_engine.run_step(
            simulator.lattice, n_swarm=n_swarm
        )

        if event is None:
            logger.warning(f"No valid events at step {step}, stopping simulation")
            break

        # Execute event in simulator
        simulator.execute_event(event)
        simulator.advance_time()
        simulator.step += 1

        # Record metrics
        roughness_history.append(roughness)
        coverage_history.append(coverage)
        ti_fraction_history.append(n_Ti / (n_Ti + n_O) if (n_Ti + n_O) > 0 else 0.0)
        o_fraction_history.append(n_O / (n_Ti + n_O) if (n_Ti + n_O) > 0 else 0.0)

        # Save snapshot
        if save_snapshots and (step % snapshot_interval == 0 or step == max_steps - 1):
            snapshots.append((step, heights.copy()))
            logger.debug(f"Snapshot saved at step {step}")

        # Log progress
        if (step + 1) % 100 == 0:
            logger.info(
                f"Step {step + 1}/{max_steps}: roughness={roughness:.3f}, "
                f"coverage={coverage:.3f}, n_swarm={n_swarm}"
            )

    # Final statistics
    final_heights = simulator.lattice.get_height_profile()
    final_roughness = calculate_roughness(final_heights)
    final_composition = simulator.lattice.get_composition()
    final_n_Ti = final_composition.get("Ti", 0)
    final_n_O = final_composition.get("O", 0)
    final_coverage = (final_n_Ti + final_n_O) / total_surface_sites
    final_ratio = final_n_Ti / final_n_O if final_n_O > 0 else 0.0

    final_stats = {
        "final_roughness": final_roughness,
        "final_coverage": final_coverage,
        "final_Ti": final_n_Ti,
        "final_O": final_n_O,
        "final_ratio": final_ratio,
        "total_steps": len(roughness_history),
        "mean_roughness": np.mean(roughness_history),
        "mean_coverage": np.mean(coverage_history),
    }

    logger.info("Simulation completed")
    logger.info(f"Final roughness: {final_roughness:.4f} nm")
    logger.info(f"Final coverage: {final_coverage:.4f}")
    logger.info(f"Final Ti/O ratio: {final_ratio:.4f}")
    logger.info(f"Mean n_swarm used: {np.mean(actions_taken):.2f}")

    return {
        "roughness": roughness_history,
        "coverage": coverage_history,
        "ti_fraction": ti_fraction_history,
        "o_fraction": o_fraction_history,
        "snapshots": snapshots,
        "actions": actions_taken,
        "final_stats": final_stats,
    }


def save_results(
    results: dict[str, any],
    output_dir: Path,
    trial_idx: int | None = None,
) -> None:
    """
    Save simulation results to disk (plots + data).

    Args:
        results: Results dictionary from run_simulation_with_policy.
        output_dir: Directory to save results.
        trial_idx: Optional trial index for multi-trial runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"trial_{trial_idx}_" if trial_idx is not None else ""

    # Save roughness evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(len(results["roughness"]))
    ax.plot(steps, results["roughness"], "o-", label="Roughness")
    ax.set_xlabel("KMC Step")
    ax.set_ylabel("Roughness (nm)")
    ax.set_title("Roughness Evolution (RL-Guided)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}roughness.png", dpi=150)
    plt.close()

    # Save coverage evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["coverage"], label="Coverage", color="blue")
    ax.set_xlabel("KMC Step")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage Evolution (RL-Guided)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}coverage.png", dpi=150)
    plt.close()

    # Save composition evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["ti_fraction"], label="Ti fraction", color="red")
    ax.plot(results["o_fraction"], label="O fraction", color="green")
    ax.axhline(2.0 / 3.0, color="red", linestyle="--", alpha=0.5, label="Ti target (2/3)")
    ax.axhline(1.0 / 3.0, color="green", linestyle="--", alpha=0.5, label="O target (1/3)")
    ax.set_xlabel("KMC Step")
    ax.set_ylabel("Fraction")
    ax.set_title("Composition Evolution (RL-Guided)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}composition.png", dpi=150)
    plt.close()

    # Save action distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_actions, counts = np.unique(results["actions"], return_counts=True)
    ax.bar(unique_actions, counts, color="purple", alpha=0.7)
    ax.set_xlabel("n_swarm")
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution (RL Policy Choices)")
    ax.set_xticks([4, 8, 16, 32])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}actions.png", dpi=150)
    plt.close()

    # Save snapshots
    if results["snapshots"]:
        snapshot_dir = output_dir / f"{prefix}snapshots"
        snapshot_dir.mkdir(exist_ok=True)

        for step, heights in results["snapshots"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(heights, cmap="viridis", origin="lower")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Height Map - Step {step}")
            plt.colorbar(im, ax=ax, label="Height (layers)")
            plt.tight_layout()
            plt.savefig(snapshot_dir / f"step_{step:06d}.png", dpi=150)
            plt.close()

        logger.info(f"Saved {len(results['snapshots'])} snapshots to {snapshot_dir}")

    # Save numerical data
    np.savez(
        output_dir / f"{prefix}data.npz",
        roughness=results["roughness"],
        coverage=results["coverage"],
        ti_fraction=results["ti_fraction"],
        o_fraction=results["o_fraction"],
        actions=results["actions"],
        **results["final_stats"],
    )

    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run KMC simulations using trained RL policy"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        nargs=3,
        default=[20, 20, 10],
        help="Lattice size (nx ny nz)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=600.0,
        help="Simulation temperature (K)",
    )
    parser.add_argument(
        "--deposition-rate",
        type=float,
        default=1.0,
        help="Deposition rate (ML/s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of KMC steps",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of independent trials to run",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Steps between snapshots (default: max_steps // 10)",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable snapshot saving",
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # Setup output directory with timestamp (same structure as validate_kmc_basic)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(__file__).parent / "results" / "swarm_predict" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load trained model
    logger.info(f"Loading trained model from {model_path}")
    try:
        model = PPO.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Run simulations
    logger.info("=" * 60)
    logger.info(f"Running {args.n_trials} trial(s) with RL policy")
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Save experiment configuration
    config_file = output_dir / "config.txt"
    with open(config_file, "w") as f:
        f.write("RL-Guided Simulation Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Lattice size: {args.lattice_size}\n")
        f.write(f"Temperature: {args.temperature} K\n")
        f.write(f"Deposition rate: {args.deposition_rate} ML/s\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"Number of trials: {args.n_trials}\n")
        f.write(f"Snapshot interval: {args.snapshot_interval}\n")
        f.write(f"Save snapshots: {not args.no_snapshots}\n")

    all_results = []

    for trial_idx in range(args.n_trials):
        logger.info(f"\nTrial {trial_idx + 1}/{args.n_trials}")
        logger.info("-" * 60)

        results = run_simulation_with_policy(
            model=model,
            lattice_size=tuple(args.lattice_size),
            temperature=args.temperature,
            deposition_rate=args.deposition_rate,
            max_steps=args.max_steps,
            save_snapshots=not args.no_snapshots,
            snapshot_interval=args.snapshot_interval,
        )

        save_results(
            results,
            output_dir,
            trial_idx=trial_idx if args.n_trials > 1 else None,
        )

        all_results.append(results)

    # Summary statistics across trials
    if args.n_trials > 1:
        logger.info("\n" + "=" * 60)
        logger.info("Summary across all trials")
        logger.info("=" * 60)

        final_roughnesses = [r["final_stats"]["final_roughness"] for r in all_results]
        final_coverages = [r["final_stats"]["final_coverage"] for r in all_results]
        final_ratios = [r["final_stats"]["final_ratio"] for r in all_results]

        logger.info(f"Final roughness: {np.mean(final_roughnesses):.4f} ± {np.std(final_roughnesses):.4f}")
        logger.info(f"Final coverage: {np.mean(final_coverages):.4f} ± {np.std(final_coverages):.4f}")
        logger.info(f"Final Ti/O ratio: {np.mean(final_ratios):.4f} ± {np.std(final_ratios):.4f}")

        # Save summary
        summary_file = output_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"RL-Guided Simulation Summary ({args.n_trials} trials)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Lattice: {args.lattice_size}\n")
            f.write(f"Temperature: {args.temperature} K\n")
            f.write(f"Deposition rate: {args.deposition_rate} ML/s\n")
            f.write(f"Max steps: {args.max_steps}\n\n")
            f.write("Results:\n")
            f.write(f"  Final roughness: {np.mean(final_roughnesses):.4f} ± {np.std(final_roughnesses):.4f} nm\n")
            f.write(f"  Final coverage: {np.mean(final_coverages):.4f} ± {np.std(final_coverages):.4f}\n")
            f.write(f"  Final Ti/O ratio: {np.mean(final_ratios):.4f} ± {np.std(final_ratios):.4f}\n")

        logger.info(f"Summary saved to {summary_file}")

    logger.info("\n" + "=" * 60)
    logger.info("All simulations completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
