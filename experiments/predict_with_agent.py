"""
RL Agent Prediction Script for TiO2 Growth Simulation.

This script loads a trained PPO agent and runs predictions on a specified
lattice configuration at 600K, generating visualizations and metrics similar
to validate_kmc_basic.py.

Usage:
    python experiments/predict_with_agent.py --model path/to/model.pt --steps 500
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import SpeciesType
from src.rl.action_selection import select_action_gumbel_max
from src.rl.action_space import N_ACTIONS
from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.shared_policy import Actor, Critic
from src.settings import settings

# Setup logging
logger = settings.setup_logging()


class PredictionConfig:
    """Configuration for prediction run."""

    def __init__(
        self,
        model_path: str,
        lattice_size=(30, 30, 20),
        temperature=600.0,
        deposition_flux_ti=0.1,
        deposition_flux_o=0.2,
        max_steps=500,
        seed=42,
        n_snapshots=20,
    ):
        self.model_path = Path(model_path)
        self.lattice_size = lattice_size
        self.temperature = temperature
        self.deposition_flux_ti = deposition_flux_ti
        self.deposition_flux_o = deposition_flux_o
        self.max_steps = max_steps
        self.seed = seed
        self.n_snapshots = n_snapshots

    def to_dict(self):
        return {
            "model_path": str(self.model_path),
            "lattice_size": list(self.lattice_size),
            "temperature": self.temperature,
            "deposition_flux_ti": self.deposition_flux_ti,
            "deposition_flux_o": self.deposition_flux_o,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "n_snapshots": self.n_snapshots,
        }


class PredictionResults:
    """Container for prediction results and visualization."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(__file__).parent / "results" / "predictions" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Time series data
        self.steps = []
        self.roughnesses = []
        self.coverages = []
        self.rewards = []
        self.n_ti_list = []
        self.n_o_list = []
        self.n_agents_list = []

        # Action statistics
        self.action_counts = {
            "DEPOSIT_TI": 0,
            "DEPOSIT_O": 0,
            "DIFFUSE_X_POS": 0,
            "DIFFUSE_X_NEG": 0,
            "DIFFUSE_Y_POS": 0,
            "DIFFUSE_Y_NEG": 0,
            "DIFFUSE_Z_POS": 0,
            "DIFFUSE_Z_NEG": 0,
            "DESORB": 0,
        }

        # Snapshots for visualization
        self.height_profiles = []
        self.snapshot_steps = []

    def record_step(self, step: int, env: AgentBasedTiO2Env, reward: float, action_str: str):
        """Record metrics for current step."""
        self.steps.append(step)
        self.roughnesses.append(env._calculate_roughness())
        self.coverages.append(env._calculate_coverage())
        self.rewards.append(reward)
        self.n_ti_list.append(env._count_species(SpeciesType.TI))
        self.n_o_list.append(env._count_species(SpeciesType.O))
        self.n_agents_list.append(len(env.agents))

        # Count actions
        if action_str in self.action_counts:
            self.action_counts[action_str] += 1

        # Store snapshots at regular intervals
        if len(self.height_profiles) < self.config.n_snapshots:
            interval = max(1, self.config.max_steps // self.config.n_snapshots)
            if step % interval == 0 or step == self.config.max_steps - 1:
                self.height_profiles.append(env.lattice.get_height_profile())
                self.snapshot_steps.append(step)

    def save_results(self):
        """Save results to JSON."""
        results = {
            "config": self.config.to_dict(),
            "metrics": {
                "steps": self.steps,
                "roughnesses": self.roughnesses,
                "coverages": self.coverages,
                "rewards": self.rewards,
                "n_ti": self.n_ti_list,
                "n_o": self.n_o_list,
                "n_agents": self.n_agents_list,
            },
            "action_counts": self.action_counts,
            "final_metrics": {
                "final_roughness": self.roughnesses[-1] if self.roughnesses else 0.0,
                "final_coverage": self.coverages[-1] if self.coverages else 0.0,
                "total_reward": sum(self.rewards),
                "mean_reward": np.mean(self.rewards) if self.rewards else 0.0,
                "final_n_ti": self.n_ti_list[-1] if self.n_ti_list else 0,
                "final_n_o": self.n_o_list[-1] if self.n_o_list else 0,
            },
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {self.output_dir / 'results.json'}")

    def generate_plots(self, env: AgentBasedTiO2Env):
        """Generate all visualization plots."""
        logger.info("Generating plots...")

        # Plot 1: Roughness evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.steps, self.roughnesses, "b-", linewidth=2)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Roughness (nm)", fontsize=12)
        ax.set_title("Surface Roughness Evolution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_01_roughness.png", dpi=150)
        plt.close()

        # Plot 2: Coverage and composition
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Coverage
        ax1.plot(self.steps, self.coverages, "g-", linewidth=2)
        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Coverage", fontsize=12)
        ax1.set_title("Surface Coverage", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Composition
        ax2.plot(self.steps, self.n_ti_list, "r-", linewidth=2, label="Ti atoms")
        ax2.plot(self.steps, self.n_o_list, "b-", linewidth=2, label="O atoms")
        ax2.set_xlabel("Step", fontsize=12)
        ax2.set_ylabel("Number of Atoms", fontsize=12)
        ax2.set_title("Composition Evolution", fontsize=14, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_02_coverage_composition.png", dpi=150)
        plt.close()

        # Plot 3: Rewards
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.steps, self.rewards, "purple", linewidth=1, alpha=0.7)
        # Add moving average
        window = min(50, len(self.rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(self.rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                self.steps[window - 1 :],
                moving_avg,
                "r-",
                linewidth=2,
                label=f"Moving Avg ({window})",
            )
            ax.legend()
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Reward (eV)", fontsize=12)
        ax.set_title("Reward Evolution", fontsize=14, fontweight="bold")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_03_rewards.png", dpi=150)
        plt.close()

        # Plot 4: Action distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        actions = list(self.action_counts.keys())
        counts = list(self.action_counts.values())
        colors = [
            "#ff6b6b",
            "#4dabf7",
            "#51cf66",
            "#ffd43b",
            "#ff8787",
            "#74c0fc",
            "#8ce99a",
            "#ffd43b",
            "#ff6b6b",
        ]
        ax.bar(actions, counts, color=colors[: len(actions)])
        ax.set_xlabel("Action Type", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Action Distribution", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_04_action_distribution.png", dpi=150)
        plt.close()

        # Plot 5: Final height profile
        height_profile = env.lattice.get_height_profile()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(height_profile, cmap="viridis", interpolation="nearest")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title("Final Height Profile", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Height (layers)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "plot_05_height_profile.png", dpi=150)
        plt.close()

        # Generate snapshot frames
        self._generate_snapshot_frames()

        logger.info(f"All plots saved to {self.output_dir}")

    def _generate_snapshot_frames(self):
        """Generate individual frames for each snapshot."""
        logger.info(f"Generating {len(self.height_profiles)} snapshot frames...")

        snapshots_dir = self.output_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        # Find global min/max for consistent colorbar
        all_heights = np.concatenate([h.flatten() for h in self.height_profiles])
        vmin, vmax = float(all_heights.min()), float(all_heights.max())

        for _i, (height_profile, step) in enumerate(zip(self.height_profiles, self.snapshot_steps)):
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(
                height_profile, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax
            )
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_title(
                f"Step {step} | Roughness: {self.roughnesses[step]:.3f} nm",
                fontsize=14,
                fontweight="bold",
            )
            plt.colorbar(im, ax=ax, label="Height (layers)")
            plt.tight_layout()
            plt.savefig(snapshots_dir / f"snapshot_{step:06d}.png", dpi=150)
            plt.close()

        logger.info(f"Snapshots saved to {snapshots_dir}")


def load_model(model_path: Path, obs_dim: int, action_dim: int, global_obs_dim: int, device):
    """Load trained actor and critic models."""
    logger.info(f"Loading model from {model_path}")

    actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    critic = Critic(obs_dim=global_obs_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    actor.eval()
    critic.eval()

    logger.info("Model loaded successfully")
    return actor, critic


def run_prediction(config: PredictionConfig):
    """Run prediction with trained agent."""
    logger.info("=" * 80)
    logger.info("Starting RL Agent Prediction")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Lattice size: {config.lattice_size}")
    logger.info(f"Temperature: {config.temperature} K")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info("=" * 80)

    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    params = TiO2Parameters()
    env = AgentBasedTiO2Env(
        lattice_size=config.lattice_size,
        tio2_parameters=params,
        temperature=config.temperature,
        max_steps=config.max_steps,
        seed=config.seed,
    )

    # Calculate deposition logits
    n_sites = config.lattice_size[0] * config.lattice_size[1]
    deposition_logit_ti = torch.tensor(np.log(config.deposition_flux_ti * n_sites)).to(device)
    deposition_logit_o = torch.tensor(np.log(config.deposition_flux_o * n_sites)).to(device)

    # Load model
    obs_dim = env.single_agent_observation_space.shape[0]
    global_obs_dim = env.global_feature_space.shape[0]
    action_dim = N_ACTIONS

    actor, critic = load_model(config.model_path, obs_dim, action_dim, global_obs_dim, device)

    # Results container
    results = PredictionResults(config)

    # Run simulation
    logger.info("Running simulation...")
    obs, info = env.reset(seed=config.seed)
    start_time = time.time()

    for step in range(config.max_steps):
        if step % 50 == 0:
            logger.info(f"  Step {step}/{config.max_steps}...")

        # Get observations
        agent_obs = obs["agent_observations"]
        global_obs = obs["global_features"]

        if len(agent_obs) == 0:
            logger.warning(f"No agents at step {step}, ending simulation")
            break

        # Get action mask
        action_mask = env.get_action_mask()

        # Convert to tensors
        agent_obs_tensor = torch.tensor(np.array(agent_obs), dtype=torch.float32).to(device)
        _global_obs_tensor = torch.tensor(global_obs, dtype=torch.float32).unsqueeze(0).to(device)

        # Get logits from actor
        with torch.no_grad():
            agent_logits = actor(agent_obs_tensor)

        # Combine with deposition logits
        all_logits = torch.cat(
            [
                agent_logits.flatten(),
                deposition_logit_ti.unsqueeze(0),
                deposition_logit_o.unsqueeze(0),
            ]
        )

        # Create combined mask
        agent_mask_flat = torch.tensor(action_mask.flatten(), dtype=torch.bool).to(device)
        deposition_mask = torch.tensor([True, True], dtype=torch.bool).to(device)
        combined_mask = torch.cat([agent_mask_flat, deposition_mask])

        # Select action using Gumbel-Max
        selected_idx = select_action_gumbel_max(all_logits, combined_mask)

        # Determine action type
        n_agent_actions = len(agent_obs) * N_ACTIONS
        if selected_idx < n_agent_actions:
            agent_idx = selected_idx // N_ACTIONS
            action_idx = selected_idx % N_ACTIONS
            action = (agent_idx, action_idx)
            action_str = (
                f"DIFFUSE_{['X_POS', 'X_NEG', 'Y_POS', 'Y_NEG', 'Z_POS', 'Z_NEG'][action_idx]}"
                if action_idx < 6
                else "DESORB"
            )
        elif selected_idx == n_agent_actions:
            action = "DEPOSIT_TI"
            action_str = "DEPOSIT_TI"
        else:
            action = "DEPOSIT_O"
            action_str = "DEPOSIT_O"

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)

        # Record results
        results.record_step(step, env, reward, action_str)

        if terminated or truncated:
            break

    duration = time.time() - start_time
    sps = step / duration if duration > 0 else 0

    logger.info("=" * 80)
    logger.info(f"Simulation completed in {duration:.2f}s ({sps:.1f} steps/s)")
    logger.info(f"Final roughness: {results.roughnesses[-1]:.4f} nm")
    logger.info(f"Final coverage: {results.coverages[-1]:.4f}")
    logger.info(f"Total reward: {sum(results.rewards):.2f} eV")
    logger.info(f"Mean reward: {np.mean(results.rewards):.4f} eV")
    logger.info("=" * 80)

    # Save and visualize
    results.save_results()
    results.generate_plots(env)

    logger.info(f"All results saved to {results.output_dir}")
    logger.info("Prediction completed successfully!")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RL agent prediction for TiO2 growth")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument(
        "--lattice-size",
        type=int,
        nargs=3,
        default=[30, 30, 20],
        help="Lattice dimensions (nx ny nz)",
    )
    parser.add_argument(
        "--temperature", type=float, default=600.0, help="Temperature in Kelvin (default: 600K)"
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--snapshots", type=int, default=20, help="Number of snapshots to save")

    args = parser.parse_args()

    config = PredictionConfig(
        model_path=args.model,
        lattice_size=tuple(args.lattice_size),
        temperature=args.temperature,
        max_steps=args.steps,
        seed=args.seed,
        n_snapshots=args.snapshots,
    )

    try:
        run_prediction(config)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
