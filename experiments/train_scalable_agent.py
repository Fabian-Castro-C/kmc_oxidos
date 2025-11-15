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
from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.shared_policy import Actor, Critic

# --- Configuration ---
# Hyperparameters (placeholders, to be tuned)
CONFIG = {
    "project_name": "Scalable_TiO2_PPO",
    "run_name": f"run_{int(time.time())}",
    "torch_seed": 42,
    "lattice_size": (5, 5, 8),
    "total_timesteps": 1_000_000,
    "num_steps": 2048,  # Number of steps to run for each environment per update
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
    "save_path": Path("models/"),
}


def main() -> None:
    """Main training function."""
    # Setup
    torch.manual_seed(CONFIG["torch_seed"])
    np.random.seed(CONFIG["torch_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(f"runs/{CONFIG['run_name']}")
    CONFIG["save_path"].mkdir(parents=True, exist_ok=True)

    # Environment
    params = TiO2Parameters()
    env = AgentBasedTiO2Env(
        lattice_size=CONFIG["lattice_size"],
        tio2_parameters=params,
        max_steps=CONFIG["num_steps"],
    )

    # Actor-Critic Models
    # The Actor acts on local observations, Critic on global (mean) observations
    obs_dim = env.observation_space["agents"].shape[1]
    action_dim = env.action_space.n
    actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    critic = Critic(obs_dim=obs_dim).to(device)

    # Optimizer
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=CONFIG["learning_rate"],
        eps=CONFIG["adam_eps"],
    )

    print("Starting training...")
    print(f"Device: {device}")
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

    next_obs, _ = env.reset()
    next_done = False

    for update in range(1, num_updates + 1):
        # --- Rollout Collection Phase ---
        for _step in range(CONFIG["num_steps"]):
            global_step += 1
            all_dones.append(next_done)
            all_obs.append(next_obs)

            # Get agent observations from the environment state
            agent_obs = next_obs["agents"]
            num_agents = agent_obs.shape[0]

            if num_agents > 0:
                # Centralized Critic Value
                # Aggregate agent observations to form a global state view (mean)
                global_obs = torch.from_numpy(agent_obs).mean(dim=0, keepdim=True).to(device)
                value = critic(global_obs)
                all_values.append(value)

                # Decentralized Actor Action
                obs_tensor = torch.from_numpy(agent_obs).to(device)
                with torch.no_grad():
                    logits = actor(obs_tensor)

                # Gumbel-Max for global action selection
                gumbel_action_idx, agent_idx = select_action_gumbel_max(logits.cpu().numpy())
                action = (agent_idx, gumbel_action_idx)

                # Calculate log probability of the chosen action
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(gumbel_action_idx).to(device))[agent_idx]
                all_logprobs.append(log_prob)

            else:
                # No agents, only deposition is possible
                action = (0, 9)  # (agent_idx=0, action_idx=9 for deposition)
                all_values.append(torch.tensor([0.0]).to(device))  # Placeholder value
                all_logprobs.append(torch.tensor(0.0).to(device))  # Placeholder logprob

            all_actions.append(action)

            # Execute action in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            all_rewards.append(reward)
            next_done = done

        # --- GAE and Advantage Calculation ---
        with torch.no_grad():
            # Get value of the last state
            next_agent_obs = next_obs["agents"]
            if next_agent_obs.shape[0] > 0:
                next_global_obs = (
                    torch.from_numpy(next_agent_obs).mean(dim=0, keepdim=True).to(device)
                )
                next_value = critic(next_global_obs).reshape(1, -1)
            else:
                next_value = torch.zeros(1, 1).to(device)

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

        # --- PPO Update Phase ---
        # Flatten the batch
        b_logprobs = torch.stack(all_logprobs).to(device)

        # Optimizing the policy and value network
        for _epoch in range(CONFIG["update_epochs"]):
            # This is a simplified loop that processes one agent at a time.
            # A more advanced implementation would use minibatches, but that is
            # complex with variable numbers of agents.
            for i in range(len(all_obs)):
                agent_obs = all_obs[i]["agents"]
                if agent_obs.shape[0] > 0:
                    # Recalculate log_probs, entropy, and values
                    obs_tensor = torch.from_numpy(agent_obs).to(device)
                    logits = actor(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)

                    agent_idx, action_idx = all_actions[i]
                    new_logprob = dist.log_prob(torch.tensor(action_idx).to(device))[agent_idx]
                    entropy = dist.entropy().mean()

                    global_obs = torch.from_numpy(agent_obs).mean(dim=0, keepdim=True).to(device)
                    new_value = critic(global_obs)

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

        # Logging
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/mean_reward", np.mean(all_rewards), global_step)

        # Clear rollout storage
        all_obs, all_actions, all_logprobs, all_rewards, all_dones, all_values = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        print(
            f"Update {update}/{num_updates} | SPS: {sps} | Mean Reward: {np.mean(all_rewards):.4f}"
        )

    # Save final model
    model_path = CONFIG["save_path"] / f"{CONFIG['run_name']}_final.pt"
    torch.save(
        {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()},
        model_path,
    )
    print(f"Training finished. Model saved to {model_path}")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
