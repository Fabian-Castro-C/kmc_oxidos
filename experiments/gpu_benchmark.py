"""
GPU Benchmark for Tensorized KMC (Proof of Concept).

This script demonstrates the feasibility of running massive lattice simulations
(e.g., 500^3 or 1000^3) entirely on the GPU using PyTorch tensors and convolutions.

It compares:
1. Memory usage (Tensor vs Object-based estimation)
2. Neighbor counting speed (Conv3d vs CPU iteration estimation)
"""

import time
import sys
import torch
import torch.nn.functional as F
import psutil
import numpy as np

def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def benchmark_gpu_lattice(size=(500, 500, 500), device='cuda'):
    print(f"\n{'='*60}")
    print(f"Benchmarking TensorLattice on {device.upper()}")
    print(f"Lattice Size: {size} = {np.prod(size):,} sites")
    print(f"{'='*60}")

    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA not available! Falling back to CPU for tensor operations.")
        device = 'cpu'

    try:
        # 1. MEMORY TEST
        print("\n[1] Memory Allocation Test...")
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated() if device == 'cuda' else 0
        
        # Create lattice: 0=Vacant, 1=Ti, 2=O (int8 is enough)
        # We use a random initialization to simulate a filled lattice
        t0 = time.time()
        lattice = torch.randint(0, 3, size, dtype=torch.int8, device=device)
        t1 = time.time()
        
        end_mem = torch.cuda.memory_allocated() if device == 'cuda' else 0
        mem_used_mb = (end_mem - start_mem) / 1024 / 1024
        
        print(f"   -> Allocation Time: {t1 - t0:.4f} s")
        print(f"   -> VRAM Used: {mem_used_mb:.2f} MB")
        print(f"   -> Theoretical RAM for Objects (Python): ~{np.prod(size) * 100 / 1024 / 1024 / 1024:.2f} GB")
        
        # 2. COMPUTE TEST (Neighbor Counting via Convolution)
        print("\n[2] Neighbor Counting Speed Test (Physics Kernel)...")
        
        # Define a 3x3x3 kernel that sums neighbors (excluding center)
        # Shape: [out_channels, in_channels, d, h, w]
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=device)
        kernel[0, 0, 1, 1, 1] = 0  # Don't count self
        
        # We need float for conv3d, but we can cast on the fly or keep a float copy
        # For this benchmark, we'll cast to simulate the cost
        lattice_float = lattice.float().unsqueeze(0).unsqueeze(0) # Add batch and channel dims
        
        # Warmup
        _ = F.conv3d(lattice_float, kernel, padding=1)
        torch.cuda.synchronize() if device == 'cuda' else None
        
        # Benchmark
        iterations = 10
        t0 = time.time()
        for _ in range(iterations):
            coordination = F.conv3d(lattice_float, kernel, padding=1)
        torch.cuda.synchronize() if device == 'cuda' else None
        t1 = time.time()
        
        avg_time = (t1 - t0) / iterations
        sites_per_sec = np.prod(size) / avg_time
        
        print(f"   -> Avg Compute Time (Conv3d): {avg_time*1000:.2f} ms")
        print(f"   -> Throughput: {sites_per_sec/1e6:.2f} Million sites/sec")
        print(f"   -> Equivalent to checking {sites_per_sec/1e9:.2f} Billion neighbors/sec")
        
        return True

    except torch.cuda.OutOfMemoryError:
        print("\n[!] OOM Error: GPU ran out of memory!")
        return False
    except Exception as e:
        print(f"\n[!] Error: {e}")
        return False

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test 1: "Small" 100^3 (1 Million sites) - Current CPU limit
    benchmark_gpu_lattice((100, 100, 100))
    
    # Test 2: Medium 500^3 (125 Million sites)
    benchmark_gpu_lattice((500, 500, 500))
    
    # Test 3: Massive 1000^3 (1 Billion sites) - The goal
    benchmark_gpu_lattice((1000, 1000, 1000))
