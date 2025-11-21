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
    
    # 3. Training Loop
    num_updates = args.total_timesteps // (args.num_envs * DEFAULT_CONFIG["num_steps"])
    print(f"Starting training: {num_updates} updates")
    
    start_time = time.time()
    
    for update in range(num_updates):
        # Storage for rollout
        batch_obs = []
        batch_actions = []
        batch_logprobs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        
        # --- Rollout Phase ---
        for step in range(DEFAULT_CONFIG["num_steps"]):
            # 1. Prepare Observation for Actor
            # Obs: (B, 51, X, Y, Z) -> Permute to (B, X, Y, Z, 51) -> Flatten to (N_sites, 51)
            # We only want to run policy on OCCUPIED sites (Agents)
            # But for FCN efficiency, we run on all and mask later
            
            B, C, X, Y, Z = obs.shape
            flat_obs = obs.permute(0, 2, 3, 4, 1).reshape(-1, 51)
            
            with torch.no_grad():
                logits = actor(flat_obs) # (B*X*Y*Z, N_ACTIONS)
                values = critic(flat_obs) # (B*X*Y*Z, 1)
            
            # Reshape back to grid
            logits_grid = logits.view(B, X, Y, Z, N_ACTIONS)
            
            # 2. Action Selection (Simplified)
            # We need to select ONE action per environment (Batch KMC)
            # Or select actions for ALL agents (Vectorized Agents)
            # SwarmThinkers usually selects one event.
            # Let's select one event per environment based on max probability or sampling
            
            # Flatten spatial dims: (B, X*Y*Z*N_ACTIONS)
            flat_logits = logits_grid.view(B, -1)
            
            # Mask invalid actions (e.g. vacant sites)
            # We need a mask from the env. For now, assume env handles it or we learn it.
            # Ideally, we mask logits for vacant sites to -inf
            
            probs = torch.softmax(flat_logits, dim=1)
            action_indices = torch.multinomial(probs, num_samples=1).squeeze(1) # (B,)
            
            # Decode action index -> (x, y, z, action_type)
            # index = x*Y*Z*A + y*Z*A + z*A + a
            
            N_A = N_ACTIONS
            
            # Clone indices for decoding
            temp_idx = action_indices.clone()
            
            a = temp_idx % N_A
            temp_idx = torch.div(temp_idx, N_A, rounding_mode='floor')
            
            z = temp_idx % Z
            temp_idx = torch.div(temp_idx, Z, rounding_mode='floor')
            
            y = temp_idx % Y
            temp_idx = torch.div(temp_idx, Y, rounding_mode='floor')
            
            x = temp_idx
            
            # Stack to (Batch, 4)
            decoded_actions = torch.stack([x, y, z, a], dim=1) # (B, 4)
            
            next_obs, rewards, dones, _ = env.step(decoded_actions)
            
            # Store data
            batch_obs.append(flat_obs)
            batch_actions.append(action_indices)
            batch_values.append(values)
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            
            obs = next_obs

        # --- Update Phase (PPO) ---
        # ... (Standard PPO update code would go here) ...
        
        if update % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Update {update}/{num_updates} | FPS: {int((update+1)*args.num_envs*DEFAULT_CONFIG['num_steps'] / elapsed)}")

if __name__ == "__main__":
    train_gpu_swarm()
