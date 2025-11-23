"""
Massive Scale Agent Prediction Script (GPU Accelerated).

This script runs the trained RL agent on a massive lattice (e.g., 1000x1000x30)
using the TensorTiO2Env infrastructure. It is designed to observe island formation
and large-scale growth patterns.

Usage:
    python experiments/predict_massive_agent.py --model path/to/model.pt --size 1000 --steps 10000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fractal import calculate_fractal_dimension
from src.analysis.roughness import fit_family_vicsek
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


def export_to_gsf(height_profile, output_path, lattice_constant, step, roughness, coverage):
    """
    Export height profile to GSF (GXSM Simple Field) format for Gwyddion.
    """
    ny, nx = height_profile.shape

    # Physical dimensions in nanometers
    xreal = nx * lattice_constant / 10.0  # Convert Angstrom to nm
    yreal = ny * lattice_constant / 10.0

    # Convert height from layers to physical units (Angstrom)
    height_angstrom = height_profile * lattice_constant

    # Write GSF file
    with open(output_path, "wb") as f:
        # Write ASCII header
        header = f"""Gwyddion Simple Field 1.0
XRes = {nx}
YRes = {ny}
XReal = {xreal:.6e}
YReal = {yreal:.6e}
XOffset = 0.000000e+00
YOffset = 0.000000e+00
Title = RL Agent TiO2 Growth - Step {step}
XYUnits = nm
ZUnits = Angstrom
# Simulation metadata
# Step: {step}
# Roughness: {roughness:.6f} Angstrom
# Coverage: {coverage:.6f} ML
# Lattice constant: {lattice_constant} Angstrom
"""
        header_bytes = header.encode("ascii")
        f.write(header_bytes)

        # Padding to 4-byte alignment
        header_length = len(header_bytes)
        padding_length = 4 - (header_length % 4)
        f.write(b"\x00" * padding_length)

        # Write binary data (4-byte floats, little-endian)
        height_flat = height_angstrom.astype("<f4").tobytes()
        f.write(height_flat)


def run_massive_prediction(
    model_path: str,
    lattice_size_xy: int,
    lattice_size_z: int,
    steps: int,
    snapshot_interval: int,
    device_name: str = "cuda",
):
    # --- Setup ---
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU (will be slow).")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(__file__).parent / "results" / "massive" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    gwyddion_dir = output_dir / "gwyddion"
    gwyddion_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # --- Initialize Environment ---
    # We use a single environment (num_envs=1) but with a huge lattice
    logger.info(
        f"Initializing TensorTiO2Env ({lattice_size_xy}x{lattice_size_xy}x{lattice_size_z})..."
    )

    # Check VRAM requirements roughly
    # Lattice: X*Y*Z bytes
    # Logits: X*Y*Z * 7 * 4 bytes
    total_sites = lattice_size_xy * lattice_size_xy * lattice_size_z
    est_vram_gb = (total_sites * (1 + 28)) / 1e9  # 1 byte state + 28 bytes logits (float32)
    logger.info(f"Estimated VRAM usage for tensors: ~{est_vram_gb:.2f} GB")

    env = TensorTiO2Env(
        num_envs=1,
        lattice_size=(lattice_size_xy, lattice_size_xy, lattice_size_z),
        device=device,
    )
    # Manually set fluxes (since they are not in __init__)
    # Use very low flux to allow diffusion (similar to training)
    # Training default was 1.0/2.0, but for massive scale we need to be careful.
    # If we want to see island formation, diffusion must be faster than deposition.
    env.flux_ti = 0.2
    env.flux_o = 0.4

    # Increase temperature to boost diffusion rates
    # Training used 600K. Let's try 1000K to see if diffusion activates.
    # Rate = v0 * exp(-Ea / kT)
    # Higher T -> Higher Rate
    env.physics.kT = env.params.k_boltzmann * 600.0

    # Debug logging for rates
    logger.info(f"Flux Ti: {env.flux_ti}, Flux O: {env.flux_o}, T: 600K")

    # --- Load Agent ---
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize networks
    # TensorTiO2Env produces obs of size 75 (18*3 neighbors + 18 rel_z + 2 counts + 1 abs_z)
    obs_dim = 75
    actor = Actor(obs_dim, N_ACTIONS).to(device)
    critic = Critic(obs_dim).to(device)

    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    actor.eval()
    critic.eval()

    logger.info("Model loaded successfully.")

    # --- Simulation Loop ---
    logger.info(f"Starting simulation for {steps} steps...")
    start_time = time.time()

    # Metrics
    total_depositions = 0
    total_moves = 0

    # Data for plotting
    step_list = []
    roughness_list = []
    coverage_list = []
    n_ti_list = []
    n_o_list = []
    fractal_dim_list = []
    alpha_list = []
    beta_list = []

    # Rolling window for scaling exponents
    recent_steps = []
    recent_roughnesses = []
    max_recent = 100

    # Action tracking
    action_counts = {"DEPOSIT_TI": 0, "DEPOSIT_O": 0, "DIFFUSE": 0, "DESORB": 0}

    # Accumulator for deposition (from train_gpu_swarm.py)
    deposition_acc = 0.0

    # Initial snapshot
    # np.save(snapshots_dir / "lattice_000000.npy", env.lattices.cpu().numpy())

    for step in range(1, steps + 1):
        # 1. Calculate Rates First (Physics)
        # We do this BEFORE inference to skip inference if we choose deposition.
        B, X, Y, Z = env.lattices.shape

        base_rates = env.physics.calculate_diffusion_rates(env.lattices)
        total_diff_rate = base_rates.sum()

        # Deposition rates
        n_sites = X * Y
        R_dep_ti = env.flux_ti * n_sites
        R_dep_o = env.flux_o * n_sites
        R_dep_total = R_dep_ti + R_dep_o

        # KMC Selection: Deposition vs Diffusion
        # Use the same logic as train_gpu_swarm.py
        # R_diff_total is the sum of all diffusion rates in the lattice
        # Multiply by 6.0 as an approximate upper bound for total possible moves (isotropic assumption)
        R_diff_total = total_diff_rate * 6.0

        R_total = R_dep_total + R_diff_total
        p_dep = R_dep_total / (R_total + 1e-10)

        # --- VISUALIZATION FIX: Minimum Deposition Probability ---
        # Ensure at least 5% of events are depositions to guarantee growth visualization
        # Otherwise we just watch diffusion for millions of steps
        # This matches the logic in train_gpu_swarm.py: p_dep = torch.clamp(p_dep, min=0.05)
        p_dep = max(p_dep, 0.05)

        if step % 100 == 0:
            logger.info(
                f"Rates - Dep: {R_dep_total:.2e}, Diff: {total_diff_rate:.2e}, p_dep: {p_dep:.4f}, Acc: {deposition_acc:.2f}"
            )

        # Update accumulator
        deposition_acc += p_dep
        should_deposit = deposition_acc >= 1.0

        if should_deposit:
            # Decrement accumulator
            deposition_acc -= 1.0

            # --- DEPOSITION EVENT ---
            is_ti = torch.rand(1, device=device) < (R_dep_ti / R_dep_total)

            # Pick random column
            dep_x = torch.randint(0, X, (1,), device=device)
            dep_y = torch.randint(0, Y, (1,), device=device)

            # Find height
            col = env.lattices[0, dep_x, dep_y, :].squeeze()
            is_occ = col != SpeciesType.VACANT.value
            if not is_occ.any():
                dep_z = 1  # 0 is substrate
            else:
                # Last occupied index
                last_occ = torch.where(is_occ)[0].max()
                dep_z = last_occ + 1

            if dep_z < Z:
                species = SpeciesType.TI if is_ti else SpeciesType.O
                env.lattices[0, dep_x, dep_y, dep_z] = species.value
                total_depositions += 1
                if is_ti:
                    action_counts["DEPOSIT_TI"] += 1
                else:
                    action_counts["DEPOSIT_O"] += 1
        else:
            # --- DIFFUSION EVENT (Agent Action) ---
            # Only NOW do we run the expensive inference

            # 1. Get Observations & Inference (Chunked to avoid OOM)
            logits_list = []
            x_chunk_size = 50
            num_neighbors = len(env.neighbor_offsets)

            # Pre-compute constant tensors for chunks
            z_coords = torch.arange(Z, device=device, dtype=torch.float32)
            obs_abs_z_template = z_coords.view(1, 1, 1, 1, Z).expand(B, 1, x_chunk_size, Y, Z)

            obs_rel_z_template = torch.zeros((B, num_neighbors, x_chunk_size, Y, Z), device=device)
            for i, val in enumerate(env.relative_z_values):
                obs_rel_z_template[:, i, :, :, :] = val

            for x_start in range(0, X, x_chunk_size):
                x_end = min(x_start + x_chunk_size, X)
                current_chunk_width = x_end - x_start

                if current_chunk_width != x_chunk_size:
                    this_obs_abs_z = obs_abs_z_template[:, :, :current_chunk_width, :, :]
                    this_obs_rel_z = obs_rel_z_template[:, :, :current_chunk_width, :, :]
                else:
                    this_obs_abs_z = obs_abs_z_template
                    this_obs_rel_z = obs_rel_z_template

                indices = torch.arange(x_start - 1, x_end + 1, device=device) % X
                lattice_slice = env.lattices[:, indices, :, :]

                one_hot = torch.zeros(
                    (B, 4, lattice_slice.shape[1], Y, Z), device=device, dtype=torch.float32
                )
                one_hot.scatter_(1, lattice_slice.unsqueeze(1).long(), 1.0)

                obs_neighbors = torch.empty(
                    (B, num_neighbors * 3, current_chunk_width, Y, Z),
                    device=device,
                    dtype=torch.float32,
                )

                for i, (dx, dy, dz) in enumerate(env.neighbor_offsets):
                    if dx == 1:
                        sliced = one_hot[:, :, 2 : 2 + current_chunk_width, :, :]
                    elif dx == -1:
                        sliced = one_hot[:, :, 0:current_chunk_width, :, :]
                    else:
                        sliced = one_hot[:, :, 1 : 1 + current_chunk_width, :, :]

                    if dy != 0 or dz != 0:
                        sliced = torch.roll(sliced, shifts=(-dy, -dz), dims=(3, 4))

                    obs_neighbors[:, i * 3 : (i + 1) * 3] = sliced[:, 0:3]

                ti_indices = [i * 3 + 1 for i in range(num_neighbors)]
                o_indices = [i * 3 + 2 for i in range(num_neighbors)]
                n_ti = obs_neighbors[:, ti_indices].sum(dim=1, keepdim=True)
                n_o = obs_neighbors[:, o_indices].sum(dim=1, keepdim=True)

                full_obs_chunk = torch.cat(
                    [obs_neighbors, this_obs_rel_z, n_ti, n_o, this_obs_abs_z], dim=1
                )
                flat_obs_chunk = full_obs_chunk.permute(0, 2, 3, 4, 1).reshape(-1, obs_dim)

                with torch.no_grad():
                    chunk_logits = actor(flat_obs_chunk)
                    logits_list.append(chunk_logits)

                del full_obs_chunk, flat_obs_chunk, obs_neighbors, one_hot, lattice_slice

            logits = torch.cat(logits_list, dim=0)

            # Physics masking/reweighting
            flat_base_rates = base_rates.view(-1, 1)
            flat_log_rates = torch.log(flat_base_rates + 1e-10)

            guided_logits = logits + flat_log_rates

            # Mask Vacant AND Substrate (Fixed)
            is_mobile = (env.lattices != SpeciesType.VACANT.value) & (
                env.lattices != SpeciesType.SUBSTRATE.value
            )
            is_mobile = is_mobile.view(-1)
            guided_logits[~is_mobile] = -1e9

            # Hierarchical Sampling
            flat_logits = guided_logits.view(-1)
            n_total = flat_logits.numel()
            block_size = 10_000_000
            max_logit = flat_logits.max()
            weights = torch.exp(flat_logits - max_logit)

            num_blocks = (n_total + block_size - 1) // block_size
            block_sums = torch.zeros(num_blocks, device=device)

            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, n_total)
                block_sums[i] = weights[start:end].sum()

            block_idx = torch.multinomial(block_sums, 1).item()

            start = block_idx * block_size
            end = min(start + block_size, n_total)
            block_weights = weights[start:end]

            local_idx = torch.multinomial(block_weights, 1).item()
            global_idx = start + local_idx

            env_action_indices = torch.tensor([global_idx], device=device)

            # Decode index
            N_A = N_ACTIONS
            temp = env_action_indices.clone()
            a = temp % N_A
            temp //= N_A
            z = temp % Z
            temp //= Z
            y = temp % Y
            x = temp // Y

            agent_actions = torch.stack([x, y, z, a], dim=1)

            env.step(agent_actions)
            total_moves += 1

            act_type = agent_actions[0, 3].item()
            if act_type == 6:  # DESORB
                action_counts["DESORB"] += 1
            else:
                action_counts["DIFFUSE"] += 1

            # logger.info(f"Step {step}: Agent moved")

        # 6. Snapshot & Metrics
        if step % snapshot_interval == 0:
            # Calculate Metrics
            # Height Profile
            # (1, X, Y, Z) -> (X, Y)
            is_occupied = env.lattices[0] != SpeciesType.VACANT.value  # (X, Y, Z)

            # Create Z indices tensor
            z_indices = torch.arange(Z, device=device).view(1, 1, Z).expand(X, Y, Z)

            # Mask unoccupied
            occupied_z = torch.where(is_occupied, z_indices, torch.zeros_like(z_indices))

            # Max Z per column
            height_map = occupied_z.max(dim=2).values.float()  # (X, Y)

            # Roughness (RMS)
            mean_h = height_map.mean()
            sq_diff = (height_map - mean_h) ** 2
            rms = torch.sqrt(sq_diff.mean()).item()

            # Coverage (Total atoms / Surface Area)
            # Or coverage of first layer? Usually total atoms deposited / sites
            n_atoms = is_occupied.sum().item() - (X * Y)  # Subtract substrate
            coverage = n_atoms / (X * Y)

            # Composition
            n_ti = (env.lattices == SpeciesType.TI.value).sum().item()
            n_o = (env.lattices == SpeciesType.O.value).sum().item()

            # Fractal Dimension (CPU)
            h_np = height_map.cpu().numpy()
            try:
                fractal_dim = calculate_fractal_dimension(h_np)
            except Exception:
                fractal_dim = np.nan

            # Scaling Exponents (Alpha, Beta)
            # Update rolling window
            recent_steps.append(step)
            recent_roughnesses.append(rms)
            if len(recent_steps) > max_recent:
                recent_steps.pop(0)
                recent_roughnesses.pop(0)

            alpha, beta = np.nan, np.nan
            if len(recent_steps) >= 10:
                try:
                    steps_array = np.array(recent_steps, dtype=float)
                    roughnesses_array = np.array(recent_roughnesses)
                    system_size = float(np.sqrt(lattice_size_xy * lattice_size_xy))
                    scaling = fit_family_vicsek(steps_array, roughnesses_array, system_size)
                    alpha = float(scaling["alpha"])
                    beta = float(scaling["beta"])
                except Exception:
                    pass

            # Store
            step_list.append(step)
            roughness_list.append(rms)
            coverage_list.append(coverage)
            n_ti_list.append(n_ti)
            n_o_list.append(n_o)
            fractal_dim_list.append(fractal_dim)
            alpha_list.append(alpha)
            beta_list.append(beta)

            logger.info(
                f"Step {step}/{steps} | Dep: {total_depositions}, Mov: {total_moves} | Rq: {rms:.4f}, Cov: {coverage:.4f}, Df: {fractal_dim:.3f}, a: {alpha:.2f}, b: {beta:.2f}"
            )

            # Save compressed numpy array
            save_path = snapshots_dir / f"lattice_{step:06d}.npy"
            np.save(save_path, env.lattices.cpu().numpy())

            # Save GSF for Gwyddion
            gsf_path = gwyddion_dir / f"snapshot_{step:06d}.gsf"
            export_to_gsf(
                height_profile=h_np,
                output_path=gsf_path,
                lattice_constant=env.params.lattice_constant_a,
                step=step,
                roughness=rms * env.params.lattice_constant_a,  # Convert to Angstrom
                coverage=coverage,
            )

            # Save PNG snapshot
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(h_np, cmap="viridis", interpolation="nearest")
            ax.set_xlabel("X", fontsize=11)
            ax.set_ylabel("Y", fontsize=11)
            ax.set_title(
                f"Step {step}: R={rms * env.params.lattice_constant_a:.2f}Å, θ={coverage:.3f}",
                fontsize=12,
                fontweight="bold",
            )
            plt.colorbar(im, ax=ax, label="Height (layers)")
            plt.tight_layout()
            plt.savefig(snapshots_dir / f"snapshot_{step:06d}.png", dpi=120)
            plt.close()

    total_time = time.time() - start_time
    logger.info(f"Simulation finished in {total_time:.2f}s")
    logger.info(f"Average speed: {steps / total_time:.2f} steps/s")

    # --- Plotting ---
    logger.info("Generating plots...")

    # Plot 1: Roughness
    plt.figure(figsize=(10, 6))
    plt.plot(step_list, roughness_list, "b-", label="RMS Roughness")
    plt.xlabel("Step")
    plt.ylabel("Roughness (Layers)")
    plt.title("Surface Roughness Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "roughness.png")
    plt.close()

    # Plot 2: Coverage
    plt.figure(figsize=(10, 6))
    plt.plot(step_list, coverage_list, "g-", label="Coverage (ML)")
    plt.xlabel("Step")
    plt.ylabel("Coverage (ML)")
    plt.title("Coverage Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "coverage.png")
    plt.close()

    # Plot 3: Composition
    plt.figure(figsize=(10, 6))
    plt.plot(step_list, n_ti_list, "r-", label="Ti Atoms")
    plt.plot(step_list, n_o_list, "b-", label="O Atoms")
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title("Composition Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "composition.png")
    plt.close()

    # Plot 4: Action Distribution
    plt.figure(figsize=(10, 6))
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    plt.bar(actions, counts, color=["red", "blue", "green", "orange"])
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.title("Action Distribution")
    plt.savefig(output_dir / "action_distribution.png")
    plt.close()

    # Plot 5: Final Height Profile
    # We need to re-calculate height map from final state
    is_occupied = env.lattices[0] != SpeciesType.VACANT.value
    z_indices = (
        torch.arange(lattice_size_z, device=device)
        .view(1, 1, lattice_size_z)
        .expand(lattice_size_xy, lattice_size_xy, lattice_size_z)
    )
    occupied_z = torch.where(is_occupied, z_indices, torch.zeros_like(z_indices))
    height_map = occupied_z.max(dim=2).values.float().cpu().numpy()

    plt.figure(figsize=(8, 7))
    plt.imshow(height_map, cmap="viridis", interpolation="nearest")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Final Height Profile")
    plt.colorbar(label="Height (layers)")
    plt.savefig(output_dir / "height_profile.png")
    plt.close()

    # Plot 6: Scaling Analysis (Log-Log)
    if len(step_list) > 10:
        plt.figure(figsize=(10, 6))
        plt.loglog(step_list, roughness_list, "bo-", label="RMS Roughness")
        plt.xlabel("Step")
        plt.ylabel("Roughness (Layers)")
        plt.title("Dynamic Scaling Analysis (Log-Log)")
        plt.grid(True, alpha=0.3, which="both")
        plt.legend()
        plt.savefig(output_dir / "scaling_loglog.png")
        plt.close()

    # Plot 8: Fractal Dimension Evolution
    # Filter NaNs
    valid_steps = [s for s, f in zip(step_list, fractal_dim_list) if not np.isnan(f)]
    valid_fractal = [f for f in fractal_dim_list if not np.isnan(f)]

    if len(valid_steps) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_steps, valid_fractal, "go-", label="Fractal Dimension")
        plt.axhline(y=2.0, color="k", linestyle="--", alpha=0.3, label="D=2.0 (flat)")
        plt.axhline(y=2.5, color="b", linestyle="--", alpha=0.3, label="D=2.5 (rough)")
        plt.xlabel("Step")
        plt.ylabel("Fractal Dimension")
        plt.title("Fractal Dimension Evolution")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "fractal_dimension.png")
        plt.close()

    # Plot 9: Scaling Exponents Evolution (Alpha, Beta)
    valid_mask = ~np.isnan(alpha_list)
    if valid_mask.any():
        valid_steps_sc = np.array(step_list)[valid_mask]
        valid_alpha = np.array(alpha_list)[valid_mask]
        valid_beta = np.array(beta_list)[valid_mask]

        if len(valid_steps_sc) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

            # Alpha evolution
            ax1.plot(
                valid_steps_sc,
                valid_alpha,
                "ro-",
                linewidth=2,
                markersize=4,
                label=r"$\alpha$ (roughness)",
            )
            ax1.axhline(y=0.5, color="k", linestyle="--", alpha=0.3, label=r"$\alpha=0.5$ (EW)")
            ax1.axhline(y=0.38, color="b", linestyle="--", alpha=0.3, label=r"$\alpha=0.38$ (KPZ)")
            ax1.set_xlabel("Step", fontsize=12)
            ax1.set_ylabel(r"$\alpha$", fontsize=14)
            ax1.set_title(
                r"Roughness Exponent ($\alpha$) Evolution", fontsize=14, fontweight="bold"
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Beta evolution
            ax2.plot(
                valid_steps_sc,
                valid_beta,
                "bs-",
                linewidth=2,
                markersize=4,
                label=r"$\beta$ (growth)",
            )
            ax2.axhline(y=0.25, color="k", linestyle="--", alpha=0.3, label=r"$\beta=0.25$ (EW)")
            ax2.axhline(y=0.33, color="r", linestyle="--", alpha=0.3, label=r"$\beta=0.33$ (KPZ)")
            ax2.set_xlabel("Step", fontsize=12)
            ax2.set_ylabel(r"$\beta$", fontsize=14)
            ax2.set_title(r"Growth Exponent ($\beta$) Evolution", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(output_dir / "scaling_exponents.png")
            plt.close()

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Massive Scale Agent Prediction")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument(
        "--size", type=int, default=200, help="XY Lattice size (e.g. 200 for 200x200)"
    )
    parser.add_argument("--height", type=int, default=30, help="Z Lattice height")
    parser.add_argument("--steps", type=int, default=10000, help="Number of simulation steps")
    parser.add_argument("--snapshot", type=int, default=1000, help="Snapshot interval")

    args = parser.parse_args()

    run_massive_prediction(
        model_path=args.model,
        lattice_size_xy=args.size,
        lattice_size_z=args.height,
        steps=args.steps,
        snapshot_interval=args.snapshot,
    )
