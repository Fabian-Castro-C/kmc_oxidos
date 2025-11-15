"""
Inference script for trained agent-based SwarmThinkers models.

Loads trained PPO models and runs episodes to generate TiO2 thin films with visualization
and analysis of growth dynamics, morphology, and physical correctness.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.analysis.roughness import calculate_roughness
from src.settings import settings
from src.rl.agent_env import AgentBasedTiO2Env

# Setup logging
logger = settings.setup_logging()


def load_trained_model(model_path: Path, env: AgentBasedTiO2Env) -> PPO:
    """
    Load trained PPO model.

    Args:
        model_path: Path to saved model (.zip)
        env: Environment instance for model

    Returns:
        Loaded PPO model
    """
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)
    logger.info("Model loaded successfully")
    logger.info(f"Policy: {model.policy}")
    return model


def run_episode(
    model: PPO,
    env: AgentBasedTiO2Env,
    deterministic: bool = True,
    render: bool = False,
) -> dict[str, Any]:
    """
    Run one inference episode.

    Args:
        model: Trained PPO model
        env: Environment instance
        deterministic: Use deterministic policy
        render: Whether to render episode

    Returns:
        Episode results with rewards, info, and final state
    """
    logger.info("Starting inference episode...")
    logger.info(f"Max steps: {env.max_steps}, Deterministic: {deterministic}")

    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    step_rewards = []
    step_info = []

    done = False
    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        total_reward += reward
        step_count += 1
        step_rewards.append(reward)
        step_info.append(info)

        if render:
            env.render()

        if step_count % 10 == 0:
            logger.debug(
                f"Step {step_count}: reward={reward:.3f}, "
                f"coverage={info.get('coverage', 0):.3f}, "
                f"roughness={info.get('roughness', 0):.3f}"
            )

    logger.info(f"Episode completed: {step_count} steps, total_reward={total_reward:.3f}")
    logger.info(f"Final coverage: {info.get('coverage', 0):.3f}")
    logger.info(f"Final roughness: {info.get('roughness', 0):.3f}")

    # Get final lattice state
    final_lattice = env.lattice

    results = {
        "total_reward": total_reward,
        "steps": step_count,
        "step_rewards": step_rewards,
        "step_info": step_info,
        "final_info": info,
        "final_lattice": final_lattice,
    }

    return results


def analyze_growth(results: dict[str, Any], output_dir: Path) -> None:
    """
    Analyze and visualize growth results.

    Args:
        results: Episode results from run_episode
        output_dir: Directory to save analysis plots
    """
    logger.info("Analyzing growth results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    lattice = results["final_lattice"]
    step_info = results["step_info"]

    # Extract height field
    height_field = lattice.get_height_profile()
    logger.info(
        f"Final height stats: mean={height_field.mean():.2f}, "
        f"max={height_field.max():.2f}, std={height_field.std():.2f}"
    )

    # Calculate morphology metrics
    roughness = calculate_roughness(height_field)
    logger.info(f"Morphology: roughness={roughness:.3f}")

    # Extract species counts
    composition = lattice.get_composition()
    ti_count = composition.get(1, 0)  # SpeciesType TI
    o_count = composition.get(2, 0)  # SpeciesType O
    total = ti_count + o_count
    ti_fraction = ti_count / total if total > 0 else 0
    logger.info(f"Composition: Ti={ti_count} ({ti_fraction:.2%}), O={o_count}")

    # Plot 3D surface (simple implementation)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(range(height_field.shape[0]), range(height_field.shape[1]))
    surf = ax.plot_surface(X.T, Y.T, height_field, cmap="viridis", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title("Final TiO2 Surface (Agent-Based SwarmThinkers)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(output_dir / "surface_3d.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved 3D surface to {output_dir / 'surface_3d.png'}")
    plt.close(fig)

    # Plot height profile (cross-section)
    mid_y = height_field.shape[1] // 2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(height_field[:, mid_y], linewidth=2)
    ax.set_xlabel("X position")
    ax.set_ylabel("Height")
    ax.set_title("Height Profile (cross-section)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "height_profile.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved height profile to {output_dir / 'height_profile.png'}")
    plt.close(fig)

    # Plot reward evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    rewards = results["step_rewards"]
    ax.plot(rewards, linewidth=1.5, alpha=0.7, label="Step reward")
    ax.axhline(np.mean(rewards), color="r", linestyle="--", label=f"Mean={np.mean(rewards):.3f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_evolution.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved reward evolution to {output_dir / 'reward_evolution.png'}")
    plt.close(fig)

    # Plot metrics evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Coverage
    coverage = [info.get("coverage", 0) for info in step_info]
    axes[0, 0].plot(coverage, linewidth=1.5)
    axes[0, 0].set_title("Coverage Evolution")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Coverage")
    axes[0, 0].grid(True, alpha=0.3)

    # Roughness
    roughness_vals = [info.get("roughness", 0) for info in step_info]
    axes[0, 1].plot(roughness_vals, linewidth=1.5, color="orange")
    axes[0, 1].set_title("Roughness Evolution")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Roughness")
    axes[0, 1].grid(True, alpha=0.3)

    # ESS (if available)
    ess_vals = [info.get("ess", 0) for info in step_info]
    if any(ess_vals):
        axes[1, 0].plot(ess_vals, linewidth=1.5, color="green")
        axes[1, 0].set_title("ESS Evolution")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("ESS")
        axes[1, 0].grid(True, alpha=0.3)

    # Height evolution
    heights = [info.get("mean_height", 0) for info in step_info]
    axes[1, 1].plot(heights, linewidth=1.5, color="purple")
    axes[1, 1].set_title("Mean Height Evolution")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Mean Height")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "metrics_evolution.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved metrics evolution to {output_dir / 'metrics_evolution.png'}")
    plt.close(fig)

    logger.info("Analysis complete!")


def main() -> None:
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained agent-based SwarmThinkers models"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.zip)",
    )
    parser.add_argument(
        "--lattice-size",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="Lattice size (Lx Ly Lz)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum episode steps",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=128,
        help="Maximum number of agents (for padding)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=600.0,
        help="Growth temperature (K)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/inference"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (may be slow)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("AGENT-BASED SWARMTHINKERS INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Lattice: {args.lattice_size}")
    logger.info(f"Temperature: {args.temperature} K")
    logger.info(f"Episodes: {args.n_episodes}")

    # Create environment
    env = AgentBasedTiO2Env(
        lattice_size=tuple(args.lattice_size),
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_agents=args.max_agents,
        use_reweighting=True,
    )

    # Load model
    model = load_trained_model(args.model, env)

    # Run episodes
    for episode_idx in range(args.n_episodes):
        logger.info(f"\n--- Episode {episode_idx + 1}/{args.n_episodes} ---")

        # Run episode
        results = run_episode(
            model,
            env,
            deterministic=args.deterministic,
            render=args.render,
        )

        # Save results
        if args.n_episodes == 1:
            output_dir = args.output_dir
        else:
            output_dir = args.output_dir / f"episode_{episode_idx + 1}"

        # Analyze and visualize
        analyze_growth(results, output_dir)

    env.close()
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
