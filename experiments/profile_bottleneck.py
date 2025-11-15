"""
Quick GPU profiling script to identify bottlenecks.
"""

import time

import numpy as np
import torch

from src.data.tio2_parameters import TiO2Parameters
from src.rl.action_space import N_ACTIONS
from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.swarm_policy import Actor, Critic

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Create environment
params = TiO2Parameters()
env = AgentBasedTiO2Env(
    lattice_size=(10, 10, 20),
    tio2_parameters=params,
    max_steps=256,
)

# Create models
obs_dim = env.single_agent_observation_space.shape[0]
global_obs_dim = env.global_feature_space.shape[0]
action_dim = N_ACTIONS

actor = Actor(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[256, 256]).to(device)
critic = Critic(obs_dim=global_obs_dim, hidden_dims=[256, 256]).to(device)

# Benchmark environment steps
print("\n" + "=" * 60)
print("Benchmarking ENVIRONMENT STEP (CPU-bound)")
print("=" * 60)
obs, _ = env.reset()
n_steps = 100
start_time = time.time()
for _ in range(n_steps):
    # Random action
    if len(obs["agent_observations"]) > 0:
        agent_idx = np.random.randint(0, len(obs["agent_observations"]))
        action_idx = np.random.randint(0, action_dim)
        action = (agent_idx, action_idx)
    else:
        action = "deposition_Ti"
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
env_time = time.time() - start_time
env_sps = n_steps / env_time
print(f"Environment Steps: {n_steps}")
print(f"Time: {env_time:.2f}s")
print(f"SPS: {env_sps:.1f}")

# Benchmark actor forward pass
print("\n" + "=" * 60)
print("Benchmarking ACTOR FORWARD PASS (GPU)")
print("=" * 60)
obs_batch = torch.randn(100, obs_dim, device=device)
n_forward = 1000
start_time = time.time()
with torch.no_grad():
    for _ in range(n_forward):
        logits = actor(obs_batch)
actor_time = time.time() - start_time
actor_fps = n_forward / actor_time
print(f"Forward Passes: {n_forward}")
print("Batch Size: 100")
print(f"Time: {actor_time:.2f}s")
print(f"FPS: {actor_fps:.1f}")
print(f"Throughput: {actor_fps * 100:.0f} samples/sec")

# Benchmark critic forward pass
print("\n" + "=" * 60)
print("Benchmarking CRITIC FORWARD PASS (GPU)")
print("=" * 60)
global_obs_batch = torch.randn(100, global_obs_dim, device=device)
start_time = time.time()
with torch.no_grad():
    for _ in range(n_forward):
        values = critic(global_obs_batch)
critic_time = time.time() - start_time
critic_fps = n_forward / critic_time
print(f"Forward Passes: {n_forward}")
print("Batch Size: 100")
print(f"Time: {critic_time:.2f}s")
print(f"FPS: {critic_fps:.1f}")
print(f"Throughput: {critic_fps * 100:.0f} samples/sec")

# Summary
print("\n" + "=" * 60)
print("BOTTLENECK ANALYSIS")
print("=" * 60)
total_time_per_step = 1.0 / env_sps
actor_time_per_step = 0.001 / actor_fps  # Assuming 1 call per step
critic_time_per_step = 0.001 / critic_fps
print(
    f"Environment: {total_time_per_step * 1000:.2f} ms/step ({total_time_per_step / total_time_per_step * 100:.0f}%)"
)
print(
    f"Actor: {actor_time_per_step * 1000:.2f} ms/step ({actor_time_per_step / total_time_per_step * 100:.1f}%)"
)
print(
    f"Critic: {critic_time_per_step * 1000:.2f} ms/step ({critic_time_per_step / total_time_per_step * 100:.1f}%)"
)
print()
print("CONCLUSION: Environment step is the bottleneck (CPU-bound)")
print("GPU is underutilized. Need to:")
print("  1. Use parallel environments (8-16 envs)")
print("  2. Profile and optimize KMC code")
print("  3. Consider Numba/Cython for hot paths")
