# GPU-Accelerated SwarmThinkers Training

This directory contains the implementation of the GPU-accelerated training pipeline for the SwarmThinkers agent.

## Overview

The GPU implementation replaces the CPU-bound `AgentBasedTiO2Env` with `TensorTiO2Env`, which runs the entire simulation (physics, state updates, observation generation, and reward calculation) on the GPU using PyTorch tensors. This eliminates the CPU-GPU data transfer bottleneck and allows for massive parallelization.

## Key Components

*   **`src/rl/tensor_env.py`**: The vectorized environment.
    *   **State**: `(Batch, X, Y, Z)` int8 tensor.
    *   **Physics**: Uses 3D convolutions for neighbor counting and rate calculation.
    *   **Rewards**: Calculates Grand Potential ($\Omega = E - \mu N$) changes on GPU.
*   **`experiments/train_gpu_swarm.py`**: The training script.
    *   Implements a PPO loop tailored for the tensor environment.
    *   Uses a shared Actor-Critic architecture.

## Usage

To run the training on a GPU (e.g., RunPod):

```bash
uv run python experiments/train_gpu_swarm.py --num_envs 128 --total_timesteps 1000000 --device cuda
```

### Parameters

*   `--num_envs`: Number of parallel environments. Higher is better for GPU utilization (try 128, 256, 512).
*   `--total_timesteps`: Total number of agent steps to train.
*   `--device`: `cuda` for GPU, `cpu` for debugging.

## Performance

On an NVIDIA A40, you should expect significantly higher throughput compared to the CPU implementation, especially with large `num_envs`.

## Notes

*   The reward function is $r_t = -\Delta \Omega / 5.0$, favoring thermodynamic stability.
*   The environment currently supports Diffusion and Desorption actions. Deposition is handled implicitly or can be added as a global event.
