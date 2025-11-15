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
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fractal import calculate_fractal_dimension
from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import SpeciesType
from src.rl.action_selection import select_action_gumbel_max
from src.rl.action_space import N_ACTIONS
from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.shared_policy import Actor, Critic
from src.settings import settings

# Setup logging
logger = settings.setup_logging()

# --- Configuration ---
# NOTE: Deposition fluxes MUST match training configuration for consistency
CONFIG = {
    "torch_seed": 42,
    "lattice_size": (30, 30, 20),  # Can differ from training size
    "deposition_flux_ti": 0.1,  # MUST match training: Ti monolayers per second
    "deposition_flux_o": 0.2,   # MUST match training: O monolayers per second
    "temperature": 600.0,  # Temperature in Kelvin (should match training)
    "max_steps": 500,
    "n_snapshots": 20,
}


class PredictionResults:
    """Container for prediction results and visualization."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(__file__).parent / "results" / "predict" / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Time series data
        self.steps = []
        self.roughnesses = []
        self.coverages = []
        self.rewards = []
        self.n_ti_list = []
        self.n_o_list = []
        self.n_agents_list = []
        
        # Scaling analysis data
        self.fractal_dims = []
        self.alpha_list = []
        self.beta_list = []

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
        roughness = env._calculate_roughness()
        self.roughnesses.append(roughness)
        self.coverages.append(env._calculate_coverage())
        self.rewards.append(reward)
        self.n_ti_list.append(env._count_species(SpeciesType.TI))
        self.n_o_list.append(env._count_species(SpeciesType.O))
        self.n_agents_list.append(len(env.agents))

        # Count actions
        if action_str in self.action_counts:
            self.action_counts[action_str] += 1

        # Calculate fractal dimension at this step
        try:
            height_profile = env.lattice.get_height_profile()
            fractal_dim = float(calculate_fractal_dimension(height_profile))
            self.fractal_dims.append(fractal_dim)
        except Exception:
            self.fractal_dims.append(None)

        # Calculate scaling exponents if we have enough data
        if len(self.steps) >= 5 and len(self.roughnesses) >= 5:
            try:
                from src.analysis import fit_family_vicsek
                steps_array = np.array(self.steps, dtype=float)
                roughnesses_array = np.array(self.roughnesses)
                system_size = float(np.sqrt(CONFIG["lattice_size"][0] * CONFIG["lattice_size"][1]))
                scaling = fit_family_vicsek(steps_array, roughnesses_array, system_size)
                self.alpha_list.append(float(scaling["alpha"]))
                self.beta_list.append(float(scaling["beta"]))
            except Exception:
                self.alpha_list.append(None)
                self.beta_list.append(None)
        else:
            self.alpha_list.append(None)
            self.beta_list.append(None)

        # Store snapshots at regular intervals
        if len(self.height_profiles) < CONFIG["n_snapshots"]:
            interval = max(1, CONFIG["max_steps"] // CONFIG["n_snapshots"])
            if step % interval == 0 or step == CONFIG["max_steps"] - 1:
                self.height_profiles.append(env.lattice.get_height_profile())
                self.snapshot_steps.append(step)

    def save_results(self):
        """Save results to JSON and CSV formats."""
        # Get final values
        final_alpha = self.alpha_list[-1] if self.alpha_list and self.alpha_list[-1] is not None else None
        final_beta = self.beta_list[-1] if self.beta_list and self.beta_list[-1] is not None else None
        final_fractal = self.fractal_dims[-1] if self.fractal_dims and self.fractal_dims[-1] is not None else None

        results = {
            "config": {
                "model_path": str(self.model_path),
                "lattice_size": list(CONFIG["lattice_size"]),
                "temperature": CONFIG["temperature"],
                "deposition_flux_ti": CONFIG["deposition_flux_ti"],
                "deposition_flux_o": CONFIG["deposition_flux_o"],
                "max_steps": CONFIG["max_steps"],
                "seed": CONFIG["torch_seed"],
                "n_snapshots": CONFIG["n_snapshots"],
            },
            "metrics": {
                "steps": self.steps,
                "roughnesses": self.roughnesses,
                "coverages": self.coverages,
                "rewards": self.rewards,
                "n_ti": self.n_ti_list,
                "n_o": self.n_o_list,
                "n_agents": self.n_agents_list,
                "fractal_dimensions": self.fractal_dims,
                "alpha": self.alpha_list,
                "beta": self.beta_list,
            },
            "action_counts": self.action_counts,
            "final_metrics": {
                "final_roughness": self.roughnesses[-1] if self.roughnesses else 0.0,
                "final_coverage": self.coverages[-1] if self.coverages else 0.0,
                "total_reward": sum(self.rewards),
                "mean_reward": np.mean(self.rewards) if self.rewards else 0.0,
                "final_n_ti": self.n_ti_list[-1] if self.n_ti_list else 0,
                "final_n_o": self.n_o_list[-1] if self.n_o_list else 0,
                "final_fractal_dimension": final_fractal,
                "final_alpha": final_alpha,
                "final_beta": final_beta,
            },
        }

        # Save JSON
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save CSV for easy import into Origin/MATLAB/Excel
        df = pd.DataFrame({
            "step": self.steps,
            "roughness": self.roughnesses,
            "coverage": self.coverages,
            "reward": self.rewards,
            "n_ti": self.n_ti_list,
            "n_o": self.n_o_list,
            "n_agents": self.n_agents_list,
            "fractal_dimension": self.fractal_dims,
            "alpha": self.alpha_list,
            "beta": self.beta_list,
        })
        df.to_csv(self.output_dir / "timeseries.csv", index=False)

        logger.info(f"Results saved to {self.output_dir / 'results.json'}")
        logger.info(f"Time series data saved to {self.output_dir / 'timeseries.csv'}")

    def generate_plots(self, env: AgentBasedTiO2Env):
        """Generate all visualization plots in multiple formats."""
        logger.info("Generating plots...")

        # Plot 1: Roughness evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.steps, self.roughnesses, "b-", linewidth=2)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Roughness (Å)", fontsize=12)
        ax.set_title("Surface Roughness Evolution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save in multiple formats
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(self.output_dir / f"plot_01_roughness.{fmt}", dpi=150 if fmt == "png" else None)
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
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(self.output_dir / f"plot_02_coverage_composition.{fmt}", dpi=150 if fmt == "png" else None)
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
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(self.output_dir / f"plot_03_rewards.{fmt}", dpi=150 if fmt == "png" else None)
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
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(self.output_dir / f"plot_04_action_distribution.{fmt}", dpi=150 if fmt == "png" else None)
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
        for fmt in ["png", "svg", "pdf"]:
            plt.savefig(self.output_dir / f"plot_05_height_profile.{fmt}", dpi=150 if fmt == "png" else None)
        plt.close()

        # Plot 6: Scaling analysis (log-log)
        if len(self.steps) > 10:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.loglog(self.steps, self.roughnesses, "bo-", linewidth=2, markersize=4, label="Data")
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Roughness (Å)", fontsize=12)
            ax.set_title("Dynamic Scaling Analysis (Log-Log)", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend()
            plt.tight_layout()
            for fmt in ["png", "svg", "pdf"]:
                plt.savefig(self.output_dir / f"plot_06_scaling_loglog.{fmt}", dpi=150 if fmt == "png" else None)
            plt.close()

        # Plot 7: Scaling exponents evolution (α, β)
        if len(self.alpha_list) > 0:
            valid_indices = [i for i, a in enumerate(self.alpha_list) if a is not None]
            if valid_indices:
                valid_steps = [self.steps[i] for i in valid_indices]
                valid_alpha = [self.alpha_list[i] for i in valid_indices]
                valid_beta = [self.beta_list[i] for i in valid_indices]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

                # Alpha evolution
                ax1.plot(valid_steps, valid_alpha, "ro-", linewidth=2, markersize=4, label=r"$\alpha$ (roughness)")
                ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label=r"$\alpha=0.5$ (EW)")
                ax1.axhline(y=0.38, color='b', linestyle='--', alpha=0.3, label=r"$\alpha=0.38$ (KPZ)")
                ax1.set_xlabel("Step", fontsize=12)
                ax1.set_ylabel(r"$\alpha$", fontsize=14)
                ax1.set_title(r"Roughness Exponent ($\alpha$) Evolution", fontsize=14, fontweight="bold")
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Beta evolution
                ax2.plot(valid_steps, valid_beta, "bs-", linewidth=2, markersize=4, label=r"$\beta$ (growth)")
                ax2.axhline(y=0.25, color='k', linestyle='--', alpha=0.3, label=r"$\beta=0.25$ (EW)")
                ax2.axhline(y=0.33, color='r', linestyle='--', alpha=0.3, label=r"$\beta=0.33$ (KPZ)")
                ax2.set_xlabel("Step", fontsize=12)
                ax2.set_ylabel(r"$\beta$", fontsize=14)
                ax2.set_title(r"Growth Exponent ($\beta$) Evolution", fontsize=14, fontweight="bold")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                plt.tight_layout()
                for fmt in ["png", "svg", "pdf"]:
                    plt.savefig(self.output_dir / f"plot_07_scaling_exponents.{fmt}", dpi=150 if fmt == "png" else None)
                plt.close()

        # Plot 8: Fractal dimension evolution
        if len(self.fractal_dims) > 0:
            valid_indices = [i for i, f in enumerate(self.fractal_dims) if f is not None]
            if valid_indices:
                valid_steps = [self.steps[i] for i in valid_indices]
                valid_fractal = [self.fractal_dims[i] for i in valid_indices]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(valid_steps, valid_fractal, "go-", linewidth=2, markersize=4)
                ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.3, label="D=2.0 (flat)")
                ax.axhline(y=2.5, color='b', linestyle='--', alpha=0.3, label="D=2.5 (typical rough)")
                ax.set_xlabel("Step", fontsize=12)
                ax.set_ylabel("Fractal Dimension", fontsize=12)
                ax.set_title("Fractal Dimension Evolution", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                for fmt in ["png", "svg", "pdf"]:
                    plt.savefig(self.output_dir / f"plot_08_fractal_dimension.{fmt}", dpi=150 if fmt == "png" else None)
                plt.close()

        # Generate snapshot frames
        self._generate_snapshot_frames()

        logger.info(f"All plots saved to {self.output_dir} (PNG, SVG, PDF)")

    def _export_to_gsf(self, height_profile, output_path, lattice_constant, step, roughness, coverage):
        """
        Export height profile to GSF (GXSM Simple Field) format for Gwyddion.
        """
        ny, nx = height_profile.shape

        # Physical dimensions in nanometers
        xreal = nx * lattice_constant / 10.0  # Convert Angstrom to nm
        yreal = ny * lattice_constant / 10.0

        # Convert height from layers to physical units (Angstrom)
        height_angstrom = height_profile * lattice_constant

        # Write GSF file
        with open(output_path, 'wb') as f:
            # Write ASCII header
            header = f"""Gwyddion Simple Field 1.0
XRes = {nx}
YRes = {ny}
XReal = {xreal:.6e}
YReal = {yreal:.6e}
XOffset = 0.000000e+00
YOffset = 0.000000e+00
Title = RL Agent TiO2 Growth - Step {step}
XYUnits = nm
ZUnits = Angstrom
# Simulation metadata
# Step: {step}
# Roughness: {roughness:.6f} Angstrom
# Coverage: {coverage:.6f} ML
# Temperature: {CONFIG['temperature']} K
# Lattice constant: {lattice_constant} Angstrom
"""
            header_bytes = header.encode('ascii')
            f.write(header_bytes)

            # Padding to 4-byte alignment
            header_length = len(header_bytes)
            padding_length = 4 - (header_length % 4)
            f.write(b'\x00' * padding_length)

            # Write binary data (4-byte floats, little-endian)
            height_flat = height_angstrom.astype('<f4').tobytes()
            f.write(height_flat)

    def _generate_snapshot_frames(self):
        """Generate individual frames for movie creation."""
        logger.info(f"Generating {len(self.height_profiles)} snapshot frames...")

        snapshots_dir = self.output_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        # Create gwyddion directory for GSF files
        gwyddion_dir = self.output_dir / "gwyddion"
        gwyddion_dir.mkdir(exist_ok=True)

        # Lattice constant for TiO2 rutile
        lattice_constant = 4.59  # Angstroms

        # Find global min/max for consistent colorbar
        all_heights = np.concatenate([h.flatten() for h in self.height_profiles])
        vmin, vmax = float(all_heights.min()), float(all_heights.max())

        for i, height_profile in enumerate(self.height_profiles):
            step = self.snapshot_steps[i]
            # Find the index in the full data arrays corresponding to this step
            step_idx = self.steps.index(step) if step in self.steps else i
            roughness = self.roughnesses[step_idx] if step_idx < len(self.roughnesses) else 0.0
            coverage = self.coverages[step_idx] if step_idx < len(self.coverages) else 0.0

            # Export to GSF format for Gwyddion
            self._export_to_gsf(
                height_profile,
                gwyddion_dir / f"snapshot_{step:06d}.gsf",
                lattice_constant,
                step,
                roughness,
                coverage
            )

            # Create PNG snapshot
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(height_profile, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_xlabel("X", fontsize=11)
            ax.set_ylabel("Y", fontsize=11)
            ax.set_title(
                f"Step {step}: R={roughness:.2f}Å, θ={coverage:.3f}",
                fontsize=12,
                fontweight="bold",
            )
            plt.colorbar(im, ax=ax, label="Height (layers)")
            plt.tight_layout()
            plt.savefig(snapshots_dir / f"snapshot_{step:06d}.png", dpi=120)
            plt.close()

        logger.info(f"Snapshots saved to {snapshots_dir}")
        logger.info(f"GSF files (Gwyddion) saved to {gwyddion_dir}")


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


def run_prediction(model_path: str):
    """Run prediction with trained agent."""
    logger.info("=" * 80)
    logger.info("Starting RL Agent Prediction")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Lattice size: {CONFIG['lattice_size']}")
    logger.info(f"Temperature: {CONFIG['temperature']} K")
    logger.info(f"Deposition Flux (Ti): {CONFIG['deposition_flux_ti']} ML/s")
    logger.info(f"Deposition Flux (O): {CONFIG['deposition_flux_o']} ML/s")
    logger.info(f"Max steps: {CONFIG['max_steps']}")
    logger.info("=" * 80)

    # Setup
    torch.manual_seed(CONFIG["torch_seed"])
    np.random.seed(CONFIG["torch_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    params = TiO2Parameters()
    env = AgentBasedTiO2Env(
        lattice_size=CONFIG["lattice_size"],
        tio2_parameters=params,
        temperature=CONFIG["temperature"],
        max_steps=CONFIG["max_steps"],
        seed=CONFIG["torch_seed"],
    )

    # Calculate deposition logits (same formula as training)
    n_sites = CONFIG["lattice_size"][0] * CONFIG["lattice_size"][1]
    deposition_logit_ti = torch.tensor(np.log(CONFIG["deposition_flux_ti"] * n_sites)).to(device)
    deposition_logit_o = torch.tensor(np.log(CONFIG["deposition_flux_o"] * n_sites)).to(device)

    logger.info(f"Calculated Deposition Logit (Ti): {deposition_logit_ti.item():.4f}")
    logger.info(f"Calculated Deposition Logit (O): {deposition_logit_o.item():.4f}")

    # Load model
    obs_dim = env.single_agent_observation_space.shape[0]
    global_obs_dim = env.global_feature_space.shape[0]
    action_dim = N_ACTIONS

    actor, critic = load_model(model_path, obs_dim, action_dim, global_obs_dim, device)

    # Results container
    results = PredictionResults(model_path)

    # Run simulation
    logger.info("Running simulation...")
    obs, info = env.reset(seed=CONFIG["torch_seed"])
    start_time = time.time()

    for step in range(CONFIG["max_steps"]):
        if step % 50 == 0:
            logger.info(f"  Step {step}/{CONFIG['max_steps']}...")

        # Get observations
        agent_obs = obs["agent_observations"]
        global_obs = obs["global_features"]
        num_agents = len(agent_obs)

        # Decentralized Actor Action Selection (same logic as training)
        with torch.no_grad():
            if num_agents > 0:
                obs_tensor = torch.from_numpy(np.array(agent_obs)).to(device)
                diffusion_logits = actor(obs_tensor)  # [num_agents, num_actions]

                # Get and apply the action mask
                action_mask = env.get_action_mask()
                action_mask_tensor = torch.from_numpy(action_mask).to(device)
                diffusion_logits[~action_mask_tensor] = -1e9  # Mask out invalid actions

                # Flatten diffusion logits and combine with the fixed deposition logits
                all_possible_logits = torch.cat(
                    [
                        diffusion_logits.flatten(),
                        deposition_logit_ti.unsqueeze(0),
                        deposition_logit_o.unsqueeze(0),
                    ]
                )
            else:
                # No agents, so no diffusion actions to mask
                # Only deposition is possible
                all_possible_logits = torch.cat(
                    [deposition_logit_ti.unsqueeze(0), deposition_logit_o.unsqueeze(0)]
                )

        # Gumbel-Max for global action selection across all possibilities
        gumbel_action_idx, _log_prob = select_action_gumbel_max(all_possible_logits)

        # Deconstruct the chosen action
        diffusion_action_space_size = num_agents * action_dim
        if gumbel_action_idx < diffusion_action_space_size:
            # It's a diffusion action
            agent_idx = gumbel_action_idx // action_dim
            action_idx = gumbel_action_idx % action_dim
            action = (agent_idx, action_idx)
            action_str = (
                f"DIFFUSE_{['X_POS', 'X_NEG', 'Y_POS', 'Y_NEG', 'Z_POS', 'Z_NEG'][action_idx]}"
                if action_idx < 6
                else "DESORB"
            )
        elif gumbel_action_idx == diffusion_action_space_size:
            # It's a Ti deposition action
            action = "DEPOSIT_TI"
            action_str = "DEPOSIT_TI"
        else:
            # It's an O deposition action
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
        default=None,
        help="Lattice dimensions (nx ny nz). Default from CONFIG.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Temperature in Kelvin. Default from CONFIG."
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps. Default from CONFIG.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Default from CONFIG.")
    parser.add_argument("--snapshots", type=int, default=None, help="Number of snapshots to save. Default from CONFIG.")

    args = parser.parse_args()

    # Override CONFIG with command-line arguments if provided
    if args.lattice_size is not None:
        CONFIG["lattice_size"] = tuple(args.lattice_size)
    if args.temperature is not None:
        CONFIG["temperature"] = args.temperature
    if args.steps is not None:
        CONFIG["max_steps"] = args.steps
    if args.seed is not None:
        CONFIG["torch_seed"] = args.seed
    if args.snapshots is not None:
        CONFIG["n_snapshots"] = args.snapshots

    try:
        run_prediction(args.model)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
