"""
Custom PPO Training Script for Scalable Agent-Based TiO2 Growth.

This script implements a custom PPO training loop to handle the dynamic
observation and action spaces of the scalable `AgentBasedTiO2Env`. It uses
the Actor-Critic architecture defined in `src/rl/shared_policy.py`, which
is based on the SwarmThinkers paper.

The training follows the Centralized Training, Decentralized Execution (CTDE)
paradigm:
- The Critic is centralized: It receives an aggregated global state observation
  to estimate the state value.
- The Actor is decentralized: It is shared among all agents and acts based on
  purely local observations.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.data.tio2_parameters import TiO2Parameters
from src.rl.action_selection import select_action_gumbel_max
from src.rl.action_space import N_ACTIONS
from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.shared_policy import Actor, Critic

# --- Configuration ---
# Hyperparameters (placeholders, to be tuned)
CONFIG = {
    "project_name": "Scalable_TiO2_PPO",
    "run_name": f"run_{int(time.time())}",
    "torch_seed": 42,
    "lattice_size": (5, 5, 8),
    "deposition_flux_ti": 0.1,  # Ti monolayers per second
    "deposition_flux_o": 0.2,  # O monolayers per second (often higher in practice)
    "total_timesteps": 512,  # Short run for debugging
    "num_steps": 128,  # Number of steps to run for each environment per update
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "adam_eps": 1e-5,
    "target_kl": None,
    "update_epochs": 10,
    "results_path": Path("experiments/results"),
}


def main() -> None:
    """Main training function."""
    # Setup
    torch.manual_seed(CONFIG["torch_seed"])
    np.random.seed(CONFIG["torch_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a specific directory for this run
    run_dir = CONFIG["results_path"] / CONFIG["run_name"]
    log_dir = run_dir / "logs"
    model_dir = run_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    # Environment & Deposition Logit
    params = TiO2Parameters()
    env = AgentBasedTiO2Env(
        lattice_size=CONFIG["lattice_size"],
        tio2_parameters=params,
        max_steps=CONFIG["num_steps"],
    )
    n_sites = CONFIG["lattice_size"][0] * CONFIG["lattice_size"][1]
    deposition_logit_ti = torch.tensor(
        np.log(CONFIG["deposition_flux_ti"] * n_sites)
    ).to(device)
    deposition_logit_o = torch.tensor(
        np.log(CONFIG["deposition_flux_o"] * n_sites)
    ).to(device)

    # Actor-Critic Models
    # The Actor acts on local observations, Critic on global features
    obs_dim = env.single_agent_observation_space.shape[0]
    global_obs_dim = env.global_feature_space.shape[0]
    action_dim = N_ACTIONS
    actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    critic = Critic(obs_dim=global_obs_dim).to(device)

    # Optimizer - Note: deposition_logit is NOT included here as it's a fixed parameter
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=CONFIG["learning_rate"],
        eps=CONFIG["adam_eps"],
    )

    print("Starting training...")
    print(f"Device: {device}")
    print(f"Deposition Flux (Ti): {CONFIG['deposition_flux_ti']} ML/s")
    print(f"Deposition Flux (O): {CONFIG['deposition_flux_o']} ML/s")
    print(f"Calculated Deposition Logit (Ti): {deposition_logit_ti.item():.4f}")
    print(f"Calculated Deposition Logit (O): {deposition_logit_o.item():.4f}")
    print(f"Actor Params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic Params: {sum(p.numel() for p in critic.parameters()):,}")

    # --- PPO Training Loop ---
    global_step = 0
    start_time = time.time()
    num_updates = CONFIG["total_timesteps"] // CONFIG["num_steps"]

    # Rollout storage
    # Note: These are lists because the number of agents varies per step
    all_obs = []  # List of (full_obs_dict)
    all_actions = []  # List of (agent_idx, action_idx) tuples
    all_logprobs = []  # List of tensors
    all_rewards = []  # List of floats
    all_dones = []  # List of bools
    all_values = []  # List of tensors
    all_action_masks = [] # List of action masks

    next_obs, _ = env.reset()
    agent_obs = next_obs["agent_observations"]
    global_obs = next_obs["global_features"]
    next_done = torch.zeros(1).to(device)

    for update in range(1, num_updates + 1):
        print(f"\n--- Starting Update {update}/{num_updates} ---")
        # --- Rollout Collection Phase ---
        print("Collecting rollouts...")
        for _step in range(CONFIG["num_steps"]):
            if _step > 0 and _step % 256 == 0:
                print(f"  Rollout step {_step}/{CONFIG['num_steps']}...")
            global_step += 1
            all_dones.append(next_done)
            all_obs.append({"agent_observations": agent_obs, "global_features": global_obs})

            # Get agent observations from the environment state
            num_agents = len(agent_obs)

            # Centralized Critic Value
            if num_agents > 0:
                value = critic(torch.from_numpy(global_obs).unsqueeze(0).to(device))
            else:
                # If no agents, the value is estimated from a zero-vector observation
                value = critic(torch.zeros(1, global_obs_dim).to(device))
            all_values.append(value)

            # Decentralized Actor Action Selection
            with torch.no_grad():
                if num_agents > 0:
                    obs_tensor = torch.from_numpy(np.array(agent_obs)).to(device)
                    diffusion_logits = actor(obs_tensor)  # [num_agents, num_actions]

                    # Get, save, and apply the action mask
                    action_mask = env.get_action_mask()
                    all_action_masks.append(action_mask)
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
                    all_action_masks.append(np.zeros((0, action_dim), dtype=bool))
                    # Only deposition is possible
                    all_possible_logits = torch.cat(
                        [deposition_logit_ti.unsqueeze(0), deposition_logit_o.unsqueeze(0)]
                    )

            # Gumbel-Max for global action selection across all possibilities
            gumbel_action_idx, log_prob = select_action_gumbel_max(all_possible_logits)
            all_logprobs.append(log_prob)

            # Deconstruct the chosen action
            diffusion_action_space_size = num_agents * action_dim
            if gumbel_action_idx < diffusion_action_space_size:
                # It's a diffusion action
                agent_idx = gumbel_action_idx // action_dim
                action_idx = gumbel_action_idx % action_dim
                action = (agent_idx, action_idx)
            elif gumbel_action_idx == diffusion_action_space_size:
                # It's a Ti deposition action
                action = "DEPOSIT_TI"
            else:
                # It's an O deposition action
                action = "DEPOSIT_O"

            all_actions.append(action)


            # Execute action in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            all_rewards.append(reward)
            agent_obs = next_obs["agent_observations"]
            global_obs = next_obs["global_features"]
            next_done = torch.tensor(1.0 if done else 0.0).to(device)

        print("Rollout collection finished.")
        # --- GAE and Advantage Calculation ---
        print("Calculating GAE and advantages...")
        with torch.no_grad():
            # Get value of the last state
            if len(agent_obs) > 0:
                next_value = critic(torch.from_numpy(global_obs).unsqueeze(0).to(device)).reshape(1, -1)
            else:
                next_value = critic(torch.zeros(1, global_obs_dim).to(device)).reshape(1, -1)

            advantages = torch.zeros(len(all_rewards)).to(device)
            last_gae_lam = 0
            for t in reversed(range(len(all_rewards))):
                if t == len(all_rewards) - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - all_dones[t + 1]
                    nextvalues = all_values[t + 1]

                delta = (
                    all_rewards[t] + CONFIG["gamma"] * nextvalues * nextnonterminal - all_values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * nextnonterminal * last_gae_lam
                )
            returns = advantages + torch.cat(all_values).squeeze()
        print("GAE calculation finished.")

        # --- PPO Update Phase ---
        print("Starting PPO update phase...")
        # Flatten the batch
        b_logprobs = torch.stack(all_logprobs).to(device)

        # Optimizing the policy and value network
        for _epoch in range(CONFIG["update_epochs"]):
            print(f"  PPO Epoch {_epoch + 1}/{CONFIG['update_epochs']}")
            # This is a simplified loop that processes one agent at a time.
            # A more advanced implementation would use minibatches, but that is
            # complex with variable numbers of agents.
            for i in range(len(all_obs)):
                if i > 0 and i % 256 == 0:
                    print(f"    Processing batch item {i}/{len(all_obs)}...")
                current_agent_obs = all_obs[i]["agent_observations"]
                current_global_obs = all_obs[i]["global_features"]
                num_agents = len(current_agent_obs)


                # Determine which action was taken
                taken_action = all_actions[i]
                is_deposition = isinstance(taken_action, str)

                # We only update the policy for diffusion actions, as deposition is fixed
                if not is_deposition and num_agents > 0:
                    # Recalculate log_probs, entropy, and values
                    obs_tensor = torch.from_numpy(np.array(current_agent_obs)).to(device)
                    diffusion_logits = actor(obs_tensor)

                    # Apply the saved mask for this specific step
                    action_mask = torch.from_numpy(all_action_masks[i]).to(device)
                    if action_mask.shape[0] == diffusion_logits.shape[0]:
                        diffusion_logits[~action_mask] = -1e9
                    else:
                        # This can happen on the last step of an episode if the number of agents changes.
                        # We can skip this update as it's a minor edge case.
                        continue

                    all_possible_logits = torch.cat(
                        [
                            diffusion_logits.flatten(),
                            deposition_logit_ti.unsqueeze(0),
                            deposition_logit_o.unsqueeze(0),
                        ]
                    )
                    dist = torch.distributions.Categorical(logits=all_possible_logits)

                    # Find the index of the action taken in the flattened logit tensor
                    _agent_idx, action_idx = taken_action
                    gumbel_action_idx = _agent_idx * action_dim + action_idx
                    new_logprob = dist.log_prob(
                        torch.tensor(gumbel_action_idx).to(device)
                    )
                    entropy = dist.entropy().mean()

                    new_value = critic(torch.from_numpy(current_global_obs).unsqueeze(0).to(device))

                    # Policy loss
                    logratio = new_logprob - b_logprobs[i]
                    ratio = logratio.exp()

                    adv = advantages[i]
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - CONFIG["clip_coef"], 1 + CONFIG["clip_coef"]
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2)

                    # Value loss
                    v_loss = 0.5 * ((new_value - returns[i]) ** 2).mean()

                    # Total loss
                    entropy_loss = entropy * CONFIG["ent_coef"]
                    loss = pg_loss - entropy_loss + v_loss * CONFIG["vf_coef"]

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()),
                        CONFIG["max_grad_norm"],
                    )
                    optimizer.step()

        print("PPO update phase finished.")

        # Logging
        sps = int(global_step / (time.time() - start_time))
        mean_reward = np.mean(all_rewards) if all_rewards else 0.0
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/sps", sps, global_step)
        if "v_loss" in locals():
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/mean_reward", mean_reward, global_step)

        # Log failure reasons for a few steps to debug
        if update == 1 and env.step_info:
            print("\n--- Sample of Action Outcomes (Update 1) ---")
            for i in range(min(20, len(env.step_info))):  # Log first 20 steps
                info = env.step_info[i]
                action_str = (
                    info["executed_action"]
                    if isinstance(info["executed_action"], str)
                    else f"Agent {info['executed_action'][0]}, Action {info['executed_action'][1]}"
                )
                if not info["success"]:
                    print(
                        f"Step {i}: Action {action_str} failed. Reason: {info['failure_reason']}"
                    )
                else:
                    print(
                        f"Step {i}: Action {action_str} succeeded. Reward: {info['reward']:.4f}"
                    )

        # Log for last update to see what's happening
        if update == num_updates and env.step_info:
            print(f"\n--- Sample of Action Outcomes (Update {update}) ---")
            for i in range(min(20, len(env.step_info))):  # Log first 20 steps
                info = env.step_info[i]
                action_str = (
                    info["executed_action"]
                    if isinstance(info["executed_action"], str)
                    else f"Agent {info['executed_action'][0]}, Action {info['executed_action'][1]}"
                )
                if not info["success"]:
                    print(
                        f"Step {i}: Action {action_str} failed. Reason: {info['failure_reason']}"
                    )
                else:
                    print(
                        f"Step {i}: Action {action_str} succeeded. Reward: {info['reward']:.4f}"
                    )

        print(
            f"Update {update}/{num_updates} | SPS: {sps} | Mean Reward: {mean_reward:.4f}"
        )

        # Clear rollout storage
        all_obs, all_actions, all_logprobs, all_rewards, all_dones, all_values, all_action_masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        env.step_info.clear()  # Clear step info for next update

    # Save final model
    model_path = model_dir / "final_model.pt"
    torch.save(
        {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()},
        model_path,
    )
    print(f"Training finished. Model saved to {model_path}")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
