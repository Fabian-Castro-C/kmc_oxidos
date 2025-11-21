"""
GPU-Accelerated SwarmThinkers Training Script.

This script replicates the logic of `train_scalable_agent.py` but uses the
`TensorTiO2Env` for massive parallel simulation on GPU. It maintains the
same Actor-Critic architecture and PPO hyperparameters.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.tensor_env import TensorTiO2Env
from src.rl.shared_policy import Actor
from src.rl.action_space import N_ACTIONS

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
}

class SimpleCritic(nn.Module):
    """
    Simple Critic for GPU training.
    Takes local observations and outputs value.
    Same architecture as Actor.
    """
    def __init__(self, obs_dim):
        super().__init__()
        self.net = Actor(obs_dim=obs_dim, action_dim=1)
        
    def forward(self, x):
        return self.net(x)

def train_gpu_swarm():
    parser = argparse.ArgumentParser(description="GPU SwarmThinkers Training")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Initialize Environment
    env = TensorTiO2Env(num_envs=args.num_envs, device=args.device)
    obs = env.reset() # (Batch, 51, X, Y, Z)
    
    # 2. Initialize Networks
    # Observation dim is 51 (from observations.py)
    actor = Actor(obs_dim=51, action_dim=N_ACTIONS).to(device)
    critic = SimpleCritic(obs_dim=51).to(device)
    
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=DEFAULT_CONFIG["learning_rate"])
    
    # TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f"runs/gpu_swarm_{int(time.time())}")
    
    # 3. Training Loop
    num_updates = args.total_timesteps // (args.num_envs * DEFAULT_CONFIG["num_steps"])
    print(f"Starting training: {num_updates} updates")
    
    start_time = time.time()
    global_step = 0
    
    for update in range(num_updates):
        # Storage for rollout
        # We store states (lattices) to save memory, and re-compute obs
        batch_lattices = []
        batch_actions = [] # Flat indices
        batch_logprobs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        
        # --- Rollout Phase ---
        for step in range(DEFAULT_CONFIG["num_steps"]):
            global_step += 1
            
            # Store current state (clone to avoid reference issues)
            batch_lattices.append(env.lattices.clone())
            
            # 1. Prepare Observation
            # (B, 51, X, Y, Z)
            obs = env._get_observations() 
            B, C, X, Y, Z = obs.shape
            flat_obs = obs.permute(0, 2, 3, 4, 1).reshape(-1, 51)
            
            with torch.no_grad():
                logits = actor(flat_obs) # (B*X*Y*Z, N_ACTIONS)
                # Critic value: We need a single scalar per environment for PPO?
                # Or per-agent?
                # If we treat the whole lattice update as one step, we need one value per env.
                # Let's aggregate the local values (e.g. mean or sum)
                local_values = critic(flat_obs) # (B*X*Y*Z, 1)
                # Reshape to (B, -1) and take mean as "Global Value" estimate
                env_values = local_values.view(B, -1).mean(dim=1)
            
            # 2. Action Selection
            logits_grid = logits.view(B, X, Y, Z, N_ACTIONS)
            flat_logits = logits_grid.view(B, -1)
            
            # Masking: We should mask vacant sites?
            # For now, let's trust the policy learns to not pick them (or env ignores)
            
            probs = torch.softmax(flat_logits, dim=1)
            action_indices = torch.multinomial(probs, num_samples=1).squeeze(1) # (B,)
            
            # Calculate logprob of selected actions
            logprobs = torch.log(probs.gather(1, action_indices.unsqueeze(1)).squeeze(1))
            
            # Decode action index -> (x, y, z, action_type)
            N_A = N_ACTIONS
            temp_idx = action_indices.clone()
            a = temp_idx % N_A
            temp_idx = torch.div(temp_idx, N_A, rounding_mode='floor')
            z = temp_idx % Z
            temp_idx = torch.div(temp_idx, Z, rounding_mode='floor')
            y = temp_idx % Y
            temp_idx = torch.div(temp_idx, Y, rounding_mode='floor')
            x = temp_idx
            
            decoded_actions = torch.stack([x, y, z, a], dim=1) # (B, 4)
            
            next_obs_unused, rewards, dones, _ = env.step(decoded_actions)
            
            # Store data
            batch_actions.append(action_indices)
            batch_logprobs.append(logprobs)
            batch_values.append(env_values)
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            
        # --- Update Phase (PPO) ---
        # Convert lists to tensors
        # (Num_steps, B, ...)
        b_lattices = torch.stack(batch_lattices) 
        b_actions = torch.stack(batch_actions)
        b_logprobs = torch.stack(batch_logprobs)
        b_rewards = torch.stack(batch_rewards)
        b_dones = torch.stack(batch_dones)
        b_values = torch.stack(batch_values)
        
        # Calculate Advantages (GAE)
        with torch.no_grad():
            # Bootstrap value
            last_obs = env._get_observations()
            flat_last_obs = last_obs.permute(0, 2, 3, 4, 1).reshape(-1, 51)
            last_local_values = critic(flat_last_obs)
            next_value = last_local_values.view(B, -1).mean(dim=1)
            
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(DEFAULT_CONFIG["num_steps"])):
                if t == DEFAULT_CONFIG["num_steps"] - 1:
                    nextnonterminal = 1.0 - 0.0 # We don't have next_done for last step easily here
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1].float()
                    nextvalues = b_values[t + 1]
                
                delta = b_rewards[t] + DEFAULT_CONFIG["gamma"] * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + DEFAULT_CONFIG["gamma"] * DEFAULT_CONFIG["gae_lambda"] * nextnonterminal * lastgaelam
            
            returns = advantages + b_values

        # Flatten batch
        # (Num_steps * B, ...)
        b_lattices_flat = b_lattices.view(-1, X, Y, Z)
        b_actions_flat = b_actions.view(-1)
        b_logprobs_flat = b_logprobs.view(-1)
        b_advantages_flat = advantages.view(-1)
        b_returns_flat = returns.view(-1)
        b_values_flat = b_values.view(-1)
        
        # Optimization Epochs
        inds = np.arange(args.num_envs * DEFAULT_CONFIG["num_steps"])
        for epoch in range(DEFAULT_CONFIG["update_epochs"]):
            np.random.shuffle(inds)
            # Mini-batch size? Let's do full batch for simplicity or chunks
            # Memory might be tight if we process all at once.
            # Let's process in chunks of 64 envs * 16 steps = 1024
            minibatch_size = 256
            
            for start in range(0, len(inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]
                
                # Re-evaluate observations
                # We need to reconstruct the environment state to get observations
                # This is tricky because `_get_observations` uses `self.lattices`
                # We can temporarily set `env.lattices` or make `_get_observations` static/functional
                
                # Functional approach:
                # We need to extract the slice of lattices
                mb_lattices = b_lattices_flat[mb_inds] # (MB, X, Y, Z)
                
                # We can hack `env` to use this batch
                # Save original
                original_lattices = env.lattices
                env.lattices = mb_lattices
                env.num_envs = len(mb_inds) # Temporarily resize
                
                new_obs = env._get_observations() # (MB, 51, X, Y, Z)
                
                # Restore env
                env.lattices = original_lattices
                env.num_envs = args.num_envs
                
                # Forward Pass
                flat_new_obs = new_obs.permute(0, 2, 3, 4, 1).reshape(-1, 51)
                new_logits = actor(flat_new_obs)
                new_local_values = critic(flat_new_obs)
                new_values = new_local_values.view(len(mb_inds), -1).mean(dim=1)
                
                # Calculate LogProbs
                new_logits_grid = new_logits.view(len(mb_inds), X, Y, Z, N_ACTIONS)
                new_flat_logits = new_logits_grid.view(len(mb_inds), -1)
                new_probs = torch.softmax(new_flat_logits, dim=1)
                
                mb_actions = b_actions_flat[mb_inds]
                new_logprobs = torch.log(new_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1))
                
                # Policy Loss
                mb_advantages = b_advantages_flat[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                logratio = new_logprobs - b_logprobs_flat[mb_inds]
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - DEFAULT_CONFIG["clip_coef"], 1 + DEFAULT_CONFIG["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                v_loss = 0.5 * ((new_values - b_returns_flat[mb_inds]) ** 2).mean()
                
                # Entropy Loss
                entropy = -(new_probs * torch.log(new_probs + 1e-10)).sum(dim=1).mean()
                
                loss = pg_loss - DEFAULT_CONFIG["ent_coef"] * entropy + DEFAULT_CONFIG["vf_coef"] * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), DEFAULT_CONFIG["max_grad_norm"])
                optimizer.step()

        # Logging
        if update % 1 == 0:
            elapsed = time.time() - start_time
            fps = int((update+1)*args.num_envs*DEFAULT_CONFIG['num_steps'] / elapsed)
            print(f"Update {update}/{num_updates} | FPS: {fps} | Loss: {loss.item():.4f} | Reward: {b_rewards.mean().item():.4f}")
            writer.add_scalar("charts/fps", fps, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.item(), global_step)
            writer.add_scalar("charts/reward", b_rewards.mean().item(), global_step)

    writer.close()

if __name__ == "__main__":
    train_gpu_swarm()
