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
from src.rl.action_space import N_ACTIONS, ActionType
from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.rate_calculator import ActionRateCalculator
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

        # Create subdirectories immediately
        self.snapshots_dir = self.output_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        self.gwyddion_dir = self.output_dir / "gwyddion"
        self.gwyddion_dir.mkdir(exist_ok=True)

        # Open CSV file for incremental writing
        self.csv_path = self.output_dir / "timeseries.csv"
        self.csv_file = open(self.csv_path, 'w', newline='', buffering=1)  # Line buffered
        self.csv_writer = None  # Will be initialized on first write

        # Keep minimal data in memory for final summary (last 100 points for plots)
        self.recent_steps = []
        self.recent_roughnesses = []
        self.recent_rewards = []
        self.max_recent = 100  # Keep last N points in memory

        # Counters for summary
        self.total_steps = 0
        self.sum_rewards = 0.0
        self.final_metrics = {}  # Will store last step metrics

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

        # Snapshot tracking
        self.snapshot_interval = None  # Will be set based on max_steps
        self.next_snapshot_step = 0
        self.snapshot_count = 0

        # Lattice constant for physical units
        self.lattice_constant = 4.59  # Angstroms

    def __del__(self):
        """Cleanup: ensure CSV file is closed."""
        if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
            self.csv_file.close()

    def record_step(self, step: int, env: AgentBasedTiO2Env, reward: float, action_str: str):
        """Record metrics for current step - writes immediately to CSV."""
        # Calculate metrics
        roughness = env._calculate_roughness()
        coverage = env._calculate_coverage()
        n_ti = env._count_species(SpeciesType.TI)
        n_o = env._count_species(SpeciesType.O)
        n_agents = len(env.agents)

        # Calculate fractal dimension
        height_profile = env.lattice.get_height_profile()
        try:
            fractal_dim = float(calculate_fractal_dimension(height_profile))
        except Exception:
            fractal_dim = None

        # Calculate scaling exponents if we have enough recent data
        alpha, beta = None, None
        if len(self.recent_steps) >= 5:
            try:
                from src.analysis import fit_family_vicsek
                steps_array = np.array(self.recent_steps, dtype=float)
                roughnesses_array = np.array(self.recent_roughnesses)
                system_size = float(np.sqrt(CONFIG["lattice_size"][0] * CONFIG["lattice_size"][1]))
                scaling = fit_family_vicsek(steps_array, roughnesses_array, system_size)
                alpha = float(scaling["alpha"])
                beta = float(scaling["beta"])
            except Exception:
                pass

        # Write to CSV immediately
        row = {
            "step": step,
            "roughness": roughness,
            "coverage": coverage,
            "reward": reward,
            "n_ti": n_ti,
            "n_o": n_o,
            "n_agents": n_agents,
            "fractal_dimension": fractal_dim if fractal_dim is not None else "",
            "alpha": alpha if alpha is not None else "",
            "beta": beta if beta is not None else "",
        }

        if self.csv_writer is None:
            # Initialize CSV writer with header
            import csv
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=row.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)

        # Update counters and final metrics
        self.total_steps = step + 1
        self.sum_rewards += reward
        self.final_metrics = {
            "final_roughness": roughness,
            "final_coverage": coverage,
            "final_n_ti": n_ti,
            "final_n_o": n_o,
            "final_n_agents": n_agents,
            "final_fractal_dimension": fractal_dim,
            "final_alpha": alpha,
            "final_beta": beta,
        }

        # Keep recent data in memory (rolling window)
        self.recent_steps.append(step)
        self.recent_roughnesses.append(roughness)
        self.recent_rewards.append(reward)
        if len(self.recent_steps) > self.max_recent:
            self.recent_steps.pop(0)
            self.recent_roughnesses.pop(0)
            self.recent_rewards.pop(0)

        # Count actions
        if action_str in self.action_counts:
            self.action_counts[action_str] += 1

        # Save snapshot if it's time (and GSF immediately)
        if self.snapshot_interval is None:
            self.snapshot_interval = max(1, CONFIG["max_steps"] // CONFIG["n_snapshots"])
            self.next_snapshot_step = 0

        if step >= self.next_snapshot_step and self.snapshot_count < CONFIG["n_snapshots"]:
            self._save_snapshot_immediately(step, height_profile, roughness, coverage)
            self.snapshot_count += 1
            self.next_snapshot_step += self.snapshot_interval

    def _save_snapshot_immediately(self, step: int, height_profile, roughness: float, coverage: float):
        """Save snapshot image and GSF file immediately."""
        # Save GSF file for Gwyddion
        self._export_to_gsf(
            height_profile,
            self.gwyddion_dir / f"snapshot_{step:06d}.gsf",
            self.lattice_constant,
            step,
            roughness,
            coverage
        )

        # Save PNG snapshot
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(height_profile, cmap="viridis", interpolation="nearest")
        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)
        ax.set_title(
            f"Step {step}: R={roughness:.2f}Å, θ={coverage:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        plt.colorbar(im, ax=ax, label="Height (layers)")
        plt.tight_layout()
        plt.savefig(self.snapshots_dir / f"snapshot_{step:06d}.png", dpi=120)
        plt.close()

    def save_final_summary(self):
        """Save final JSON summary after simulation completes."""
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()

        # Calculate summary statistics
        mean_reward = self.sum_rewards / self.total_steps if self.total_steps > 0 else 0.0

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
            "action_counts": self.action_counts,
            "final_metrics": {
                **self.final_metrics,
                "total_reward": self.sum_rewards,
                "mean_reward": mean_reward,
                "total_steps": self.total_steps,
            },
            "data_files": {
                "timeseries_csv": str(self.csv_path),
                "snapshots_dir": str(self.snapshots_dir),
                "gwyddion_dir": str(self.gwyddion_dir),
            }
        }

        # Save JSON summary
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Final summary saved to {self.output_dir / 'results.json'}")
        logger.info(f"Time series data: {self.csv_path}")
        logger.info("Snapshots saved incrementally during execution")

    def generate_plots(self, env: AgentBasedTiO2Env):
        """Generate all visualization plots in multiple formats - reads from saved CSV."""
        logger.info("Generating plots from saved data...")

        # Load data from CSV
        df = pd.read_csv(self.csv_path)
        steps = df["step"].values
        roughnesses = df["roughness"].values
        coverages = df["coverage"].values
        rewards = df["reward"].values
        n_ti_list = df["n_ti"].values
        n_o_list = df["n_o"].values
        n_agents_list = df["n_agents"].values  # noqa: F841 - may be used in future plots
        fractal_dims = df["fractal_dimension"].replace("", np.nan).values
        alpha_list = df["alpha"].replace("", np.nan).values
        beta_list = df["beta"].replace("", np.nan).values

        # Plot 1: Roughness evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, roughnesses, "b-", linewidth=2)
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
        ax1.plot(steps, coverages, "g-", linewidth=2)
        ax1.set_xlabel("Step", fontsize=12)
        ax1.set_ylabel("Coverage", fontsize=12)
        ax1.set_title("Surface Coverage", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Composition
        ax2.plot(steps, n_ti_list, "r-", linewidth=2, label="Ti atoms")
        ax2.plot(steps, n_o_list, "b-", linewidth=2, label="O atoms")
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
        ax.plot(steps, rewards, "purple", linewidth=1, alpha=0.7)
        # Add moving average
        window = min(50, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :],
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
        if len(steps) > 10:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.loglog(steps, roughnesses, "bo-", linewidth=2, markersize=4, label="Data")
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
        valid_mask = ~np.isnan(alpha_list)
        if valid_mask.any():
            valid_steps = steps[valid_mask]
            valid_alpha = alpha_list[valid_mask]
            valid_beta = beta_list[valid_mask]

            if len(valid_steps) > 0:
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
        fractal_mask = ~np.isnan(fractal_dims)
        if fractal_mask.any():
            valid_steps_f = steps[fractal_mask]
            valid_fractal = fractal_dims[fractal_mask]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(valid_steps_f, valid_fractal, "go-", linewidth=2, markersize=4)
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

        logger.info(f"All plots generated and saved to {self.output_dir}")

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


def load_model(model_path: Path, obs_dim: int, action_dim: int, global_obs_dim: int, device):
    """Load trained actor and critic models."""
    logger.info(f"Loading model from {model_path}")

    actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    # Updated Critic signature for Deep Sets
    critic = Critic(obs_dim=obs_dim, global_obs_dim=global_obs_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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

    # Initialize Rate Calculator for Physics-Guided RL
    rate_calculator = ActionRateCalculator(temperature=env.temperature)

    # Calculate Deposition Rate (R_dep)
    n_sites = CONFIG["lattice_size"][0] * CONFIG["lattice_size"][1]
    R_dep_ti = CONFIG["deposition_flux_ti"] * n_sites  # atoms/s
    R_dep_o = CONFIG["deposition_flux_o"] * n_sites    # atoms/s
    R_dep_total = R_dep_ti + R_dep_o

    logger.info(f"Deposition Rates: R_Ti={R_dep_ti:.2f}/s, R_O={R_dep_o:.2f}/s, R_total={R_dep_total:.2f}/s")
    logger.info("Using Physics-Based Competition (R_diff vs R_dep) and Action Reweighting")

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
        num_agents = len(agent_obs)

        # Decide between DEPOSITION (external event) or AGENT ACTION
        with torch.no_grad():
            # 1. Calculate Total Diffusion Rate (R_diff)
            current_diffusion_rates = []
            total_diff_rate = 0.0
            
            if num_agents > 0:
                # Calculate rates for all agents
                agent_rates = np.zeros((num_agents, N_ACTIONS), dtype=np.float32)
                
                for i, agent in enumerate(env.agents):
                    for act_idx in range(N_ACTIONS):
                        act_enum = ActionType(act_idx)
                        if act_enum in [ActionType.ADSORB_TI, ActionType.ADSORB_O, ActionType.REACT_TIO2]:
                            continue
                            
                        rate = rate_calculator.calculate_action_rate(agent, act_enum, env.lattice)
                        agent_rates[i, act_idx] = rate
                        total_diff_rate += rate
                
                current_diffusion_rates = agent_rates

            # 2. Calculate Event Probabilities
            R_total = R_dep_total + total_diff_rate
            p_deposit = R_dep_total / R_total if R_total > 0 else 1.0
            
            # Calculate physical time step (KMC residence time)
            dt = 1.0 / R_total if R_total > 0 else 0.0

            # 3. Select Event Type
            if np.random.random() < p_deposit:
                # DEPOSITION EVENT
                if np.random.random() < (R_dep_ti / R_dep_total):
                    action = "DEPOSIT_TI"
                    action_str = "DEPOSIT_TI"
                else:
                    action = "DEPOSIT_O"
                    action_str = "DEPOSIT_O"
            else:
                # AGENT ACTION (if agents exist)
                if num_agents > 0:
                    obs_tensor = torch.from_numpy(np.array(agent_obs)).to(device)
                    policy_logits = actor(obs_tensor)  # [num_agents, num_actions]

                    # --- REWEIGHTING: Combine Policy with Physics ---
                    # Logit(a) = Policy_Logit(a) + log(Rate(a))
                    log_rates = np.log(current_diffusion_rates + 1e-10)
                    log_rates_tensor = torch.from_numpy(log_rates).to(device)
                    
                    diffusion_logits = policy_logits + log_rates_tensor

                    # Get and apply the action mask
                    action_mask = env.get_action_mask()
                    action_mask_tensor = torch.from_numpy(action_mask).to(device)
                    diffusion_logits[~action_mask_tensor] = -1e9  # Mask out invalid actions

                    # Flatten diffusion logits for Gumbel-Max selection
                    all_possible_logits = diffusion_logits.flatten()

                    # Gumbel-Max for action selection
                    gumbel_action_idx, _log_prob = select_action_gumbel_max(all_possible_logits)

                    # Deconstruct the chosen action
                    agent_idx = gumbel_action_idx // action_dim
                    action_idx = gumbel_action_idx % action_dim
                    action = (agent_idx, action_idx)
                    action_str = (
                        f"DIFFUSE_{['X_POS', 'X_NEG', 'Y_POS', 'Y_NEG', 'Z_POS', 'Z_NEG'][action_idx]}"
                        if action_idx < 6
                        else "DESORB"
                    )
                else:
                    # No agents: force deposition to bootstrap
                    action = "DEPOSIT_TI" if np.random.random() < 0.5 else "DEPOSIT_O"
                    action_str = action

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action, dt=dt)

        # Record results
        results.record_step(step, env, reward, action_str)

        if terminated or truncated:
            break

    duration = time.time() - start_time
    sps = step / duration if duration > 0 else 0

    logger.info("=" * 80)
    logger.info(f"Simulation completed in {duration:.2f}s ({sps:.1f} steps/s)")
    logger.info(f"Final roughness: {results.final_metrics.get('final_roughness', 0):.4f} Å")
    logger.info(f"Final coverage: {results.final_metrics.get('final_coverage', 0):.4f}")
    logger.info(f"Total reward: {results.sum_rewards:.2f} eV")
    logger.info(f"Mean reward: {results.sum_rewards / results.total_steps:.4f} eV")
    logger.info("=" * 80)

    # Save final summary and generate plots from saved data
    results.save_final_summary()
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
    parser.add_argument(
        "--flux-ti", type=float, default=None, help="Ti deposition flux (ML/s). Default from CONFIG."
    )
    parser.add_argument(
        "--flux-o", type=float, default=None, help="O deposition flux (ML/s). Default from CONFIG."
    )

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
    if args.flux_ti is not None:
        CONFIG["deposition_flux_ti"] = args.flux_ti
    if args.flux_o is not None:
        CONFIG["deposition_flux_o"] = args.flux_o

    try:
        run_prediction(args.model)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
