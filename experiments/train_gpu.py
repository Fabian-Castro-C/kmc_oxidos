"""
Training script for GPU-Accelerated RL.

This script demonstrates how to use the TensorTiO2Env to train
agents at massive speeds using batched GPU operations.
"""

import argparse
import time
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.tensor_env import TensorTiO2Env

def train_gpu_accelerated():
    parser = argparse.ArgumentParser(description="GPU Accelerated Training")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to simulate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    args = parser.parse_args()

    num_envs = args.num_envs
    steps = args.steps
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"{'='*80}")
    print("GPU ACCELERATED TRAINING")
    print(f"Environments: {num_envs}")
    print(f"Device: {device.upper()}")
    print(f"{'='*80}")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # 1. Initialize Vectorized Environment
    print("\n[1] Initializing Tensor Environments...")
    env = TensorTiO2Env(num_envs=num_envs, device=device)
    obs = env.reset()
    print(f"    Batch Shape: {obs.shape}")
    if device == "cuda":
        print(f"    Memory Used: {torch.cuda.memory_allocated()/1e6:.2f} MB")

    # 2. Training Loop (Simulation)
    print(f"\n[2] Running Training Loop ({steps} steps)...")
    start_time = time.time()
    
    total_transitions = 0
    
    for step in range(steps):
        # Dummy actions (random)
        actions = torch.randint(0, 9, (num_envs, 10), device=device)
        
        # Step environment
        next_obs, rewards, dones, info = env.step(actions)
        
        total_transitions += num_envs
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tps = total_transitions / elapsed
            print(f"    Step {step}: {tps:.0f} transitions/sec")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Training Finished!")
    print(f"Total Transitions: {total_transitions:,}")
    print(f"Throughput: {total_transitions/total_time:.0f} transitions/sec")
    print(f"{'='*80}")

if __name__ == "__main__":
    train_gpu_accelerated()
