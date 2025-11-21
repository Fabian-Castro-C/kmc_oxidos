"""
Massive Scale Prediction Script (GPU Accelerated).

This script runs the trained RL agent on a massive lattice (e.g., 1000x1000x1000)
using the new TensorLattice and TensorRateCalculator infrastructure.

It demonstrates the capability to simulate billions of sites by leveraging
GPU memory and parallel convolution operations.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import SpeciesType
from src.kmc.tensor_lattice import TensorLattice
from src.kmc.tensor_rates import TensorRateCalculator


def run_massive_simulation(lattice_size, steps, device_name="cuda"):
    print(f"{'=' * 80}")
    print(f"MASSIVE SCALE SIMULATION: {lattice_size}")
    print(f"Target: {np.prod(lattice_size):,} sites")
    print(f"{'=' * 80}")

    if not torch.cuda.is_available():
        print("CRITICAL WARNING: CUDA not available. This will be extremely slow on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Print VRAM info
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"VRAM: {a / 1e9:.2f}GB allocated / {r / 1e9:.2f}GB reserved / {t / 1e9:.2f}GB total")

    # 1. Initialize Tensor Lattice
    print("\n[1] Initializing Tensor Lattice...")
    t0 = time.time()
    lattice = TensorLattice(lattice_size, device)
    print(f"    Done in {time.time() - t0:.4f}s")

    # 2. Initialize Physics Engine
    print("\n[2] Initializing Physics Engine (TensorRates)...")
    params = TiO2Parameters()
    physics = TensorRateCalculator(params, temperature=600.0, device=device)

    # 3. Load RL Agent
    print("\n[3] Loading RL Agent...")
    # Note: We need to adapt the agent to work with tensors directly in the future.
    # For this demo, we will simulate the "Physics-Only" mode (KMC) to prove performance,
    # as the RL agent expects observation vectors that are hard to generate for 1 billion atoms
    # without a fully tensorized observation builder (which is the next step).
    print("    (Running in High-Performance KMC Mode for Benchmark)")

    # 4. Simulation Loop
    print(f"\n[4] Starting Simulation ({steps} steps)...")

    start_time = time.time()

    # Pre-allocate tensors for performance
    # flux_ti = 5.0  # ML/s (Unused in this demo)
    # flux_o = 10.0  # ML/s (Unused in this demo)
    # surface_area = lattice_size[0] * lattice_size[1] (Unused in this demo)

    for step in range(steps):
        # A. Calculate Rates (The heavy lifting)
        # This computes diffusion rates for ALL atoms in parallel
        rates = physics.calculate_diffusion_rates(lattice.state)
        total_diff_rate = physics.get_total_system_rate(rates)

        # B. Deposition Event (Simplified for demo)
        # In a real run, we'd use KMC time selection.
        # Here we just deposit atoms to fill the lattice and make the calculation harder

        # Deposit 1000 atoms per step to stress test
        n_deposit = 1000

        # Random positions
        xs = torch.randint(0, lattice_size[0], (n_deposit,), device=device)
        ys = torch.randint(0, lattice_size[1], (n_deposit,), device=device)

        # Update height map (CPU sync required for logic in this simple version,
        # but fully GPU version would use scatter_add)
        # For benchmark speed, we just update the state tensor directly at z=1 for now
        # to simulate occupancy without complex collision logic
        lattice.state[xs, ys, 1] = SpeciesType.TI.value

        if step % 10 == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed
            print(
                f"    Step {step}: {sps:.2f} steps/sec | Active Atoms: {torch.sum(lattice.get_occupancy_mask()).item()} | Rate: {total_diff_rate:.2e} Hz"
            )

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("Simulation Finished!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Speed: {steps / total_time:.2f} steps/sec")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000, help="Lattice side length (cubic)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    args = parser.parse_args()

    run_massive_simulation(lattice_size=(args.size, args.size, args.size), steps=args.steps)
