"""
GPU-Accelerated SwarmThinkers Training Script.

This script replicates the logic of `train_scalable_agent.py` but uses the
`TensorTiO2Env` for massive parallel simulation on GPU. It maintains the
same Actor-Critic architecture, PPO hyperparameters, and logging format.
"""

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from src.kmc.lattice import SpeciesType
from src.rl.action_space import N_ACTIONS
from src.rl.shared_policy import Actor, Critic
from src.rl.tensor_env import TensorTiO2Env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Merge all config sections into one flat dict
    full_config = {}

    # Add top-level variables if they exist
    if hasattr(config_module, "ENV_CONFIG"):
        full_config.update(config_module.ENV_CONFIG)
    if hasattr(config_module, "TRAINING_CONFIG"):
        full_config.update(config_module.TRAINING_CONFIG)
    if hasattr(config_module, "PPO_CONFIG"):
        full_config.update(config_module.PPO_CONFIG)
    if hasattr(config_module, "COMPUTE_CONFIG"):
        full_config.update(config_module.COMPUTE_CONFIG)
    if hasattr(config_module, "FLUX_SCHEDULE_CONFIG"):
        full_config.update(config_module.FLUX_SCHEDULE_CONFIG)

    return full_config


def get_flux_for_update(update_num: int, config: dict) -> tuple[float, float]:
    """
    Get deposition flux for current update.
    """
    if config.get("enable_flux_schedule", False):
        stages = config.get("flux_stages", [])
        # Find the latest stage that applies
        current_stage = None
        for stage in stages:
            if update_num >= stage["at_update"]:
                current_stage = stage
            else:
                break

        if current_stage:
            return current_stage["flux_ti"], current_stage["flux_o"]

    return config["deposition_flux_ti"], config["deposition_flux_o"]


# Default Config from train_scalable_agent.py
DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "update_epochs": 6,
    "num_steps": 128,  # Steps per rollout
    "deposition_flux_ti": 1.0,
    "deposition_flux_o": 2.0,
    "lattice_size": (20, 20, 20),  # Default for GPU
    "total_timesteps": 100000,
    "num_envs": 64,
}


def train_gpu_swarm():
    parser = argparse.ArgumentParser(description="GPU SwarmThinkers Training")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=None, help="Total training steps (overrides config)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=None, help="Steps per rollout (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu) (overrides config)"
    )
    args = parser.parse_args()

    # 1. Start with Defaults
    current_config = DEFAULT_CONFIG.copy()

    # 2. Load Config File if provided
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        file_config = load_config(args.config)
        current_config.update(file_config)

    # 3. Override with CLI args if provided
    if args.num_envs is not None:
        current_config["num_envs"] = args.num_envs
    if args.total_timesteps is not None:
        current_config["total_timesteps"] = args.total_timesteps
    if args.num_steps is not None:
        current_config["num_steps"] = args.num_steps

    # Device handling
    if args.device:
        device_name = args.device
    elif "device" in current_config:
        device_name = current_config["device"]
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    logger.info(f"Training on {device}")

    # Extract parameters
    num_envs = current_config.get("num_envs", 64)
    total_timesteps = current_config.get("total_timesteps", 100000)
    lattice_size = current_config["lattice_size"]
    max_steps = current_config.get("max_steps_per_episode", 1000)

    # 1. Initialize Environment
    logger.info(f"Initializing {num_envs} parallel environments with max_steps={max_steps}...")
    env = TensorTiO2Env(
        num_envs=num_envs,
        lattice_size=lattice_size,
        device=device_name,
        max_steps=max_steps,
    )

    # 2. Initialize Networks
    obs_dim = 75  # 18 neighbors * 3 + 18 rel_z + 2 local + 1 abs_z
    global_obs_dim = 12  # Default in Critic

    actor = Actor(obs_dim=obs_dim, action_dim=N_ACTIONS).to(device)
    critic = Critic(obs_dim=obs_dim, global_obs_dim=global_obs_dim).to(device)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=current_config["learning_rate"]
    )

    # TensorBoard
    run_name = current_config.get(
        "run_name", f"gpu_swarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    writer = SummaryWriter(f"experiments/results/train/{run_name}")
    logger.info(f"Logging to experiments/results/train/{run_name}")

    # 3. Training Loop
    num_updates = total_timesteps // (num_envs * current_config["num_steps"])
    logger.info(f"Starting training: {num_updates} updates")

    start_time = time.time()
    global_step = 0
    best_reward = -float("inf")  # Track best reward

    # Deposition Accumulators (per environment)
    deposition_acc = torch.zeros(num_envs, device=device)

    # Initial Reset
    obs = env.reset()  # (Batch, 51, X, Y, Z)

    for update in range(1, num_updates + 1):
        # --- Flux Schedule ---
        flux_ti, flux_o = get_flux_for_update(update, current_config)
        n_sites = env.nx * env.ny
        R_dep_ti = flux_ti * n_sites
        R_dep_o = flux_o * n_sites
        R_dep_total = R_dep_ti + R_dep_o

        if update % 5 == 0:
            logger.info(
                f"\n--- Validation Update {update}/{num_updates}: Using flux Ti={flux_ti:.1f}, O={flux_o:.1f} ML/s ---"
            )
        else:
            logger.info(
                f"\n--- Training Update {update}/{num_updates}: Using flux Ti={flux_ti:.1f}, O={flux_o:.1f} ML/s ---"
            )

        logger.info(
            f"  Deposition Rates: R_Ti={R_dep_ti:.2f}/s, R_O={R_dep_o:.2f}/s, R_total={R_dep_total:.2f}/s"
        )
        logger.info("  Using Physics-Based Competition (R_diff vs R_dep) and Action Reweighting")

        # Storage for rollout
        batch_lattices = []  # Store state to recompute obs
        batch_actions = []  # Flat indices
        batch_logprobs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        batch_log_rates = []  # For reweighting

        # Logging stats for this update
        ep_rewards = []
        n_depositions = 0
        n_agent_actions = 0
        detailed_log_steps = []

        # --- Rollout Phase ---
        for step in range(current_config["num_steps"]):
            global_step += num_envs

            # Store Lattice State (int8, compact)
            batch_lattices.append(env.lattices.clone())

            # 1. Calculate Physics Rates
            base_rates = env.physics.calculate_diffusion_rates(env.lattices)
            total_diff_rates = base_rates.sum(dim=(1, 2, 3))  # (B,)
            R_diff_total = total_diff_rates * 6.0  # Approx upper bound

            # 2. Deposition vs Diffusion Competition
            R_total = R_dep_total + R_diff_total
            p_dep = R_dep_total / (R_total + 1e-10)
            p_dep = torch.clamp(p_dep, min=0.05)

            deposition_acc += p_dep
            should_deposit = deposition_acc >= 1.0
            deposition_acc[should_deposit] -= 1.0

            # 3. Prepare Observations
            B, C, X, Y, Z = obs.shape
            # (B, obs_dim, X, Y, Z) -> (B, XYZ, obs_dim)
            agent_obs = obs.permute(0, 2, 3, 4, 1).reshape(B, -1, obs_dim)

            # Global features: Coverage + Dummy
            coverage = (
                (env.lattices != SpeciesType.VACANT.value).float().mean(dim=(1, 2, 3)).unsqueeze(1)
            )  # (B, 1)
            global_features = torch.cat(
                [coverage, torch.zeros(B, 11, device=device)], dim=1
            )  # (B, 12)

            with torch.no_grad():
                # Actor: (B*XYZ, obs_dim) -> (B*XYZ, N_ACTIONS)
                flat_obs = agent_obs.reshape(-1, obs_dim)
                logits = actor(flat_obs)

                # Critic: (B, 12), (B, XYZ, obs_dim) -> (B, 1)
                values = critic(global_features, agent_obs)

            # 4. Action Selection (Physics-Guided)
            flat_base_rates = base_rates.view(-1, 1)
            flat_log_rates = torch.log(flat_base_rates + 1e-10)
            guided_logits = logits + flat_log_rates

            # Mask Vacant Sites
            is_occupied = (env.lattices != SpeciesType.VACANT.value).view(-1)
            guided_logits[~is_occupied] = -1e9

            # Sample Actions
            # Note: We don't need to sample for ALL agents, just the ones we might pick.
            # But for batch efficiency we compute logits for all.

            # 5. Select ONE action per environment
            flat_env_logits = guided_logits.view(B, -1)
            env_action_indices = torch.multinomial(
                torch.softmax(flat_env_logits, dim=1), num_samples=1
            ).squeeze(1)

            # Calculate logprob of the CHOSEN environment action
            env_probs = torch.softmax(flat_env_logits, dim=1)
            env_logprobs = torch.log(
                env_probs.gather(1, env_action_indices.unsqueeze(1)).squeeze(1)
            )

            # Decode index -> (x, y, z, action)
            N_A = N_ACTIONS
            temp = env_action_indices.clone()
            a = temp % N_A
            temp //= N_A
            z = temp % Z
            temp //= Z
            y = temp % Y
            x = temp // Y

            agent_actions = torch.stack([x, y, z, a], dim=1)

            # Create "final" rewards tensor
            step_rewards = torch.zeros(B, device=device)

            # Track action type for logging (0=Agent, 1=Deposition)
            # For the first env, we want to know exactly what happened
            log_action_type = None
            log_details = None

            # Capture Agent ID for logging (Env 0) BEFORE step moves it
            if not should_deposit[0]:
                lx, ly, lz = agent_actions[0, 0], agent_actions[0, 1], agent_actions[0, 2]
                log_agent_id = env.atom_ids[0, lx, ly, lz].item()

                act_idx = agent_actions[0, 3].item()
                # Action mapping from tensor_env.py: 0:X+, 1:X-, 2:Y+, 3:Y-, 4:Z+, 5:Z-, 6:DESORB
                act_names = ["X+", "X-", "Y+", "Y-", "Z+", "Z-", "DESORB"]
                act_name = act_names[act_idx] if act_idx < len(act_names) else f"UNKNOWN({act_idx})"

                log_action_type = "Agent Action"
                log_details = f"Agent #{log_agent_id} at ({lx},{ly},{lz}) -> {act_name}"

            # Execute Agent Step (for ALL, but we'll ignore/overwrite for depositing)
            next_obs, rewards, dones, _ = env.step(agent_actions)
            step_rewards += rewards
            n_agent_actions += (~should_deposit).sum().item()

            # Execute Deposition (Overwrite)
            if should_deposit.any():
                dep_indices = torch.nonzero(should_deposit).squeeze(1)
                n_dep = len(dep_indices)
                n_depositions += n_dep

                rand = torch.rand(n_dep, device=device)
                is_ti = rand < (R_dep_ti / R_dep_total)

                dep_x = torch.randint(0, X, (n_dep,), device=device)
                dep_y = torch.randint(0, Y, (n_dep,), device=device)

                # Find max Z (highest occupied index + 1)
                cols = env.lattices[dep_indices, dep_x, dep_y, :]
                is_occupied = cols != SpeciesType.VACANT.value

                # Create indices tensor [0, 1, ..., Z-1]
                z_indices = torch.arange(Z, device=device).expand_as(cols)

                # Mask indices where not occupied, then take max
                occupied_indices = torch.where(is_occupied, z_indices, torch.zeros_like(z_indices))
                max_z = occupied_indices.max(dim=1).values

                # If column is empty (only substrate at 0), max_z is 0. dep_z is 1.
                dep_z = max_z + 1

                # Filter out columns that are full (dep_z >= Z)
                valid_dep = dep_z < Z

                if valid_dep.any():
                    # Filter everything
                    dep_indices = dep_indices[valid_dep]
                    dep_x = dep_x[valid_dep]
                    dep_y = dep_y[valid_dep]
                    dep_z = dep_z[valid_dep]
                    is_ti = is_ti[valid_dep]

                    dep_coords = torch.stack([dep_x, dep_y, dep_z], dim=1)

                    if is_ti.any():
                        ti_idx = dep_indices[is_ti]
                        ti_coords = dep_coords[is_ti]
                        r_ti = env.deposit(ti_idx, SpeciesType.TI, ti_coords)
                        # step_rewards[ti_idx] = r_ti  # DISABLED: User requested training only on agent actions

                    if (~is_ti).any():
                        o_idx = dep_indices[~is_ti]
                        o_coords = dep_coords[~is_ti]
                        r_o = env.deposit(o_idx, SpeciesType.O, o_coords)
                        # step_rewards[o_idx] = r_o  # DISABLED: User requested training only on agent actions

                    # Check if first env was a deposition
                    if should_deposit[0]:
                        # Was it Ti or O?
                        # We need to check the specific choice for index 0
                        # Re-find the index in dep_indices (which is now filtered by valid_dep)
                        idx_in_dep = (dep_indices == 0).nonzero()
                        if len(idx_in_dep) > 0:
                            idx = idx_in_dep.item()
                            is_ti_0 = is_ti[idx]
                            species = "Ti" if is_ti_0 else "O"
                            # Get the ID of the newly deposited atom
                            new_id = env.atom_ids[0, dep_x[idx], dep_y[idx], dep_z[idx]].item()

                            log_action_type = "Deposition"
                            log_details = f"Deposit {species} #{new_id} at ({dep_x[idx]},{dep_y[idx]},{dep_z[idx]})"

            # Store data
            batch_actions.append(env_action_indices)
            batch_logprobs.append(env_logprobs)
            batch_values.append(values)
            batch_rewards.append(step_rewards)
            batch_dones.append(dones)
            batch_log_rates.append(flat_log_rates)  # Store for reweighting

            obs = next_obs
            ep_rewards.append(step_rewards.mean().item())

            # Log for first env
            num_steps = current_config["num_steps"]
            should_log_step = (
                step < 10
                or (step >= num_steps // 2 and step < num_steps // 2 + 10)
                or step >= num_steps - 10
            )
            if should_log_step and log_action_type is not None:
                # Count agents (non-vacant, non-substrate)
                # env.lattices is (B, X, Y, Z)
                # We only care about env 0
                lat0 = env.lattices[0]
                n_agents = (
                    ((lat0 != SpeciesType.VACANT.value) & (lat0 != SpeciesType.SUBSTRATE.value))
                    .sum()
                    .item()
                )

                detailed_log_steps.append(
                    {
                        "step": step,
                        "type": log_action_type,
                        "details": log_details,
                        "reward": step_rewards[0].item(),
                        "n_agents": n_agents,
                        "env_steps": env.steps[0].item(),
                        "env_max_steps": env.max_steps,
                        "done": dones[0].item(),
                    }
                )

        # Detailed Step Logging
        logger.info("--- Detailed Steps (Env 0) ---")

        printed_first = False
        printed_middle = False
        printed_last = False

        num_steps = current_config["num_steps"]

        for log in detailed_log_steps:
            step = log["step"]

            if step < 10 and not printed_first:
                logger.info("--- First 10 Steps ---")
                printed_first = True
            elif step >= num_steps // 2 and step < num_steps // 2 + 10 and not printed_middle:
                logger.info("--- Middle 10 Steps ---")
                printed_middle = True
            elif step >= num_steps - 10 and not printed_last:
                logger.info("--- Last 10 Steps ---")
                printed_last = True
            # Format similar to train_scalable_agent.py
            # [  0] DEPOSIT Ti → Reward: +2.000, Agents: 3
            # [  0] AGENT[ 0] DIFFUSE_Y_POS → Reward: +0.000, Agents: 3

            if log["type"] == "Deposition":
                # Parse details "Deposit Ti at (x,y,z)"
                parts = log["details"].split()
                species = parts[1]
                logger.info(
                    f"  [{log['step']:3d}] DEPOSIT {species:<2}            → Reward: {log['reward']:+6.3f}, Agents: {log['n_agents']:3d}, Steps: {log['env_steps']}/{log['env_max_steps']}, Done: {log['done']} | {log['details']}"
                )
            else:
                # Agent Action
                # details: "Agent at (x,y,z) -> ACTION_NAME"
                parts = log["details"].split(" -> ")
                action_name = parts[1]
                logger.info(
                    f"  [{log['step']:3d}] AGENT ACTION {action_name:<14} → Reward: {log['reward']:+6.3f}, Agents: {log['n_agents']:3d}, Steps: {log['env_steps']}/{log['env_max_steps']}, Done: {log['done']} | {log['details']}"
                )

        # --- Update Phase (PPO) ---
        b_lattices = torch.stack(batch_lattices)  # (T, B, X, Y, Z)
        b_actions = torch.stack(batch_actions)  # (T, B)
        b_logprobs = torch.stack(batch_logprobs)
        b_rewards = torch.stack(batch_rewards)
        b_dones = torch.stack(batch_dones)
        b_values = torch.stack(batch_values).squeeze(-1)  # (T, B)

        # Calculate Advantages
        with torch.no_grad():
            next_value = b_values[-1]  # Approx
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(current_config["num_steps"])):
                if t == current_config["num_steps"] - 1:
                    nextnonterminal = 1.0
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1].float()
                    nextvalues = b_values[t + 1]
                delta = (
                    b_rewards[t]
                    + current_config["gamma"] * nextvalues * nextnonterminal
                    - b_values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + current_config["gamma"]
                    * current_config["gae_lambda"]
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + b_values

        # Flatten
        b_lattices_flat = b_lattices.view(-1, X, Y, Z)
        b_actions_flat = b_actions.view(-1)
        b_logprobs_flat = b_logprobs.view(-1)
        b_advantages_flat = advantages.view(-1)
        b_returns_flat = returns.view(-1)
        b_values_flat = b_values.view(-1)

        # Optimization
        inds = np.arange(num_envs * current_config["num_steps"])
        for _epoch in range(current_config["update_epochs"]):
            np.random.shuffle(inds)
            # Reduced minibatch size to fit in VRAM (Swarm architecture = many agents per env)
            # 64 envs * 8000 agents = 512,000 samples per forward pass
            minibatch_size = current_config.get("minibatch_size", 64)

            for start in range(0, len(inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                # Reconstruct Observations
                mb_lattices = b_lattices_flat[mb_inds]

                # Hack: Use env to generate obs
                original_lattices = env.lattices
                env.lattices = mb_lattices
                env.num_envs = len(mb_inds)
                new_obs = env._get_observations()  # (MB, obs_dim, X, Y, Z)
                env.lattices = original_lattices
                env.num_envs = num_envs

                # Prepare inputs
                mb_agent_obs = new_obs.permute(0, 2, 3, 4, 1).reshape(len(mb_inds), -1, obs_dim)
                mb_coverage = (
                    (mb_lattices != SpeciesType.VACANT.value)
                    .float()
                    .mean(dim=(1, 2, 3))
                    .unsqueeze(1)
                )
                mb_global = torch.cat(
                    [mb_coverage, torch.zeros(len(mb_inds), 11, device=device)], dim=1
                )

                # Forward Pass
                new_values = critic(mb_global, mb_agent_obs).squeeze(-1)

                # Actor Logits
                flat_new_obs = mb_agent_obs.reshape(-1, obs_dim)
                new_logits = actor(flat_new_obs)

                # Re-calculate Log Rates (Expensive? We can approximate or recompute)
                # We need base_rates for the *stored* lattices.
                # env.physics.calculate_diffusion_rates is fast enough?
                mb_base_rates = env.physics.calculate_diffusion_rates(mb_lattices)
                mb_flat_log_rates = torch.log(mb_base_rates.view(-1, 1) + 1e-10)

                guided_logits = new_logits + mb_flat_log_rates

                # Mask
                is_occupied = (mb_lattices != SpeciesType.VACANT.value).view(-1)
                guided_logits[~is_occupied] = -1e9

                # Calculate LogProb of TAKEN action
                # We need to map the taken action index (which was for the WHOLE lattice)
                # to the new logits.
                # The logits are (MB*XYZ, N_A).
                # The taken action index `b_actions_flat` is in [0, XYZ*N_A).
                # This matches `flat_env_logits` which is (MB, XYZ*N_A).

                flat_env_logits = guided_logits.view(len(mb_inds), -1)
                new_probs = torch.softmax(flat_env_logits, dim=1)

                mb_actions = b_actions_flat[mb_inds]
                new_logprobs = torch.log(new_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1))

                # Losses
                mb_adv = b_advantages_flat[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                logratio = new_logprobs - b_logprobs_flat[mb_inds]
                ratio = logratio.exp()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - current_config["clip_coef"], 1 + current_config["clip_coef"]
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - b_returns_flat[mb_inds]) ** 2).mean()

                entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=1).mean()

                loss = (
                    pg_loss
                    - current_config["ent_coef"] * entropy
                    + current_config["vf_coef"] * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    current_config["max_grad_norm"],
                )
                optimizer.step()

        # Logging
        elapsed = time.time() - start_time
        fps = int((update) * num_envs * current_config["num_steps"] / elapsed)
        mean_reward = b_rewards.mean().item()

        logger.info(
            f"Update {update}/{num_updates} | FPS: {fps} | Loss: {loss.item():.4f} | Reward: {mean_reward:.4f}"
        )

        # Rollout Summary
        total_steps = num_envs * current_config["num_steps"]
        dep_pct = 100.0 * n_depositions / total_steps
        agent_pct = 100.0 * n_agent_actions / total_steps
        logger.info(
            f"  Rollout summary: {n_depositions} depositions ({dep_pct:.1f}%), {n_agent_actions} agent actions ({agent_pct:.1f}%)"
        )

        writer.add_scalar("charts/fps", fps, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/reward", mean_reward, global_step)

        # Save Checkpoint
        if update % 50 == 0:
            torch.save(
                {"actor": actor.state_dict(), "critic": critic.state_dict(), "update": update},
                f"experiments/results/train/{run_name}/checkpoint_{update}.pt",
            )

        # Save Best Model
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "update": update,
                    "reward": best_reward,
                },
                f"experiments/results/train/{run_name}/best_model.pt",
            )
            logger.info(f"  New best model saved with reward: {best_reward:.4f}")

    writer.close()


if __name__ == "__main__":
    train_gpu_swarm()
