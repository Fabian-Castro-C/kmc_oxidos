"""
GPU-Accelerated SwarmThinkers Training Script.

This script replicates the logic of `train_scalable_agent.py` but uses the
`TensorTiO2Env` for massive parallel simulation on GPU. It maintains the
same Actor-Critic architecture, PPO hyperparameters, and logging format.
"""

import argparse
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
}


def train_gpu_swarm():
    parser = argparse.ArgumentParser(description="GPU SwarmThinkers Training")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # 1. Initialize Environment
    logger.info(f"Initializing {args.num_envs} parallel environments...")
    env = TensorTiO2Env(
        num_envs=args.num_envs, lattice_size=DEFAULT_CONFIG["lattice_size"], device=args.device
    )

    # 2. Initialize Networks
    obs_dim = 51
    global_obs_dim = 12  # Default in Critic

    actor = Actor(obs_dim=obs_dim, action_dim=N_ACTIONS).to(device)
    critic = Critic(obs_dim=obs_dim, global_obs_dim=global_obs_dim).to(device)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=DEFAULT_CONFIG["learning_rate"]
    )

    # TensorBoard
    run_name = f"gpu_swarm_{int(time.time())}"
    writer = SummaryWriter(f"experiments/results/train/{run_name}")
    logger.info(f"Logging to experiments/results/train/{run_name}")

    # 3. Training Loop
    num_updates = args.total_timesteps // (args.num_envs * DEFAULT_CONFIG["num_steps"])
    logger.info(f"Starting training: {num_updates} updates")

    start_time = time.time()
    global_step = 0

    # Deposition Accumulators (per environment)
    deposition_acc = torch.zeros(args.num_envs, device=device)

    # Flux parameters
    flux_ti = DEFAULT_CONFIG["deposition_flux_ti"]
    flux_o = DEFAULT_CONFIG["deposition_flux_o"]
    n_sites = env.nx * env.ny
    R_dep_ti = flux_ti * n_sites
    R_dep_o = flux_o * n_sites
    R_dep_total = R_dep_ti + R_dep_o

    # Initial Reset
    obs = env.reset()  # (Batch, 51, X, Y, Z)

    for update in range(1, num_updates + 1):
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

        # --- Rollout Phase ---
        for step in range(DEFAULT_CONFIG["num_steps"]):
            global_step += args.num_envs

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
            # (B, 51, X, Y, Z) -> (B, XYZ, 51)
            agent_obs = obs.permute(0, 2, 3, 4, 1).reshape(B, -1, 51)

            # Global features: Coverage + Dummy
            coverage = (
                (env.lattices != SpeciesType.VACANT.value).float().mean(dim=(1, 2, 3)).unsqueeze(1)
            )  # (B, 1)
            global_features = torch.cat(
                [coverage, torch.zeros(B, 11, device=device)], dim=1
            )  # (B, 12)

            with torch.no_grad():
                # Actor: (B*XYZ, 51) -> (B*XYZ, N_ACTIONS)
                flat_obs = agent_obs.reshape(-1, 51)
                logits = actor(flat_obs)

                # Critic: (B, 12), (B, XYZ, 51) -> (B, 1)
                values = critic(global_features, agent_obs)

            # 4. Action Selection (Physics-Guided)
            flat_base_rates = base_rates.view(-1, 1)
            flat_log_rates = torch.log(flat_base_rates + 1e-10)
            guided_logits = logits + flat_log_rates

            # Mask Vacant Sites
            is_occupied = (env.lattices != SpeciesType.VACANT.value).view(-1)
            guided_logits[~is_occupied] = -1e9

            # Sample Actions
            probs = torch.softmax(guided_logits, dim=1)
            action_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B*XYZ,)

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

                # Find max Z
                cols = env.lattices[dep_indices, dep_x, dep_y, :]
                heights = (cols != SpeciesType.VACANT.value).sum(dim=1)
                dep_z = torch.clamp(heights, max=Z - 1)

                dep_coords = torch.stack([dep_x, dep_y, dep_z], dim=1)

                if is_ti.any():
                    ti_idx = dep_indices[is_ti]
                    ti_coords = dep_coords[is_ti]
                    r_ti = env.deposit(ti_idx, SpeciesType.TI, ti_coords)
                    step_rewards[ti_idx] = r_ti  # Overwrite reward

                if (~is_ti).any():
                    o_idx = dep_indices[~is_ti]
                    o_coords = dep_coords[~is_ti]
                    r_o = env.deposit(o_idx, SpeciesType.O, o_coords)
                    step_rewards[o_idx] = r_o  # Overwrite reward

            # Store data
            batch_actions.append(env_action_indices)
            batch_logprobs.append(env_logprobs)
            batch_values.append(values)
            batch_rewards.append(step_rewards)
            batch_dones.append(dones)
            batch_log_rates.append(flat_log_rates)  # Store for reweighting

            obs = next_obs
            ep_rewards.append(step_rewards.mean().item())

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
            for t in reversed(range(DEFAULT_CONFIG["num_steps"])):
                if t == DEFAULT_CONFIG["num_steps"] - 1:
                    nextnonterminal = 1.0
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1].float()
                    nextvalues = b_values[t + 1]
                delta = (
                    b_rewards[t]
                    + DEFAULT_CONFIG["gamma"] * nextvalues * nextnonterminal
                    - b_values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + DEFAULT_CONFIG["gamma"]
                    * DEFAULT_CONFIG["gae_lambda"]
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
        inds = np.arange(args.num_envs * DEFAULT_CONFIG["num_steps"])
        for epoch in range(DEFAULT_CONFIG["update_epochs"]):
            np.random.shuffle(inds)
            minibatch_size = 256  # Adjust based on VRAM

            for start in range(0, len(inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                # Reconstruct Observations
                mb_lattices = b_lattices_flat[mb_inds]

                # Hack: Use env to generate obs
                original_lattices = env.lattices
                env.lattices = mb_lattices
                env.num_envs = len(mb_inds)
                new_obs = env._get_observations()  # (MB, 51, X, Y, Z)
                env.lattices = original_lattices
                env.num_envs = args.num_envs

                # Prepare inputs
                mb_agent_obs = new_obs.permute(0, 2, 3, 4, 1).reshape(len(mb_inds), -1, 51)
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
                flat_new_obs = mb_agent_obs.reshape(-1, 51)
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
                    ratio, 1 - DEFAULT_CONFIG["clip_coef"], 1 + DEFAULT_CONFIG["clip_coef"]
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - b_returns_flat[mb_inds]) ** 2).mean()

                entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=1).mean()

                loss = (
                    pg_loss
                    - DEFAULT_CONFIG["ent_coef"] * entropy
                    + DEFAULT_CONFIG["vf_coef"] * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    DEFAULT_CONFIG["max_grad_norm"],
                )
                optimizer.step()

        # Logging
        elapsed = time.time() - start_time
        fps = int((update) * args.num_envs * DEFAULT_CONFIG["num_steps"] / elapsed)
        mean_reward = b_rewards.mean().item()

        logger.info(
            f"Update {update}/{num_updates} | FPS: {fps} | Loss: {loss.item():.4f} | Reward: {mean_reward:.4f}"
        )

        # Detailed Step Logging (First 30 steps)
        logger.info("--- First 30 Steps ---")
        for i in range(min(30, DEFAULT_CONFIG["num_steps"])):
            # We need to know if it was Deposition or Agent Action
            # We can infer from reward? Or just log what we have.
            # We didn't store action type explicitly.
            # But we can log the reward.
            r = b_rewards[i, 0].item()  # Log for first env
            logger.info(f"  [{i:3d}] Reward: {r:+.4f}")

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

    writer.close()


if __name__ == "__main__":
    train_gpu_swarm()
