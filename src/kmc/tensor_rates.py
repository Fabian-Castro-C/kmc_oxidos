"""
Tensor-based Rate Calculator using 3D Convolutions.

This module implements the physics engine entirely on the GPU.
It uses 3D convolutions to count neighbors for millions of atoms in parallel,
replacing the slow object-based iteration.
"""

import torch
import torch.nn.functional as F

from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import SpeciesType


class TensorRateCalculator:
    def __init__(self, params: TiO2Parameters, temperature: float, device: torch.device):
        self.params = params
        self.device = device
        self.kT = params.k_boltzmann * temperature

        # --- Convolution Kernels for Neighbor Counting ---
        # Shape: (out_channels, in_channels, D, H, W)
        # We use 3x3x3 kernels to check nearest neighbors

        # Kernel 1: Count ALL occupied neighbors (Coordination Number)
        self.kernel_coordination = torch.zeros((1, 1, 3, 3, 3), device=device)

        # Nearest Neighbors (Up, Down, Left, Right, Front, Back)
        neighbors = [
            (1, 1, 2),
            (1, 1, 0),  # Z+1, Z-1
            (1, 2, 1),
            (1, 0, 1),  # Y+1, Y-1
            (2, 1, 1),
            (0, 1, 1),  # X+1, X-1
        ]
        for idx in neighbors:
            self.kernel_coordination[0, 0, idx[0], idx[1], idx[2]] = 1.0

        # Pre-calculate rate constants (Arrhenius)
        # Rate = v0 * exp(-Ea / kT)
        # We can vectorize this: Rate = v0 * exp(-(E_base + n*E_bond) / kT)

        self.nu0 = 1e13  # Attempt frequency (Hz)

        # Energy parameters
        self.E_diff_ti = params.ea_diff_ti
        self.E_diff_o = params.ea_diff_o
        self.E_bond = params.bond_energy_ti_o

    def calculate_diffusion_rates(self, lattice_state: torch.Tensor):
        """
        Calculate diffusion rates for ALL atoms in the lattice simultaneously.

        Args:
            lattice_state: (nx, ny, nz) OR (batch, nx, ny, nz) int8 tensor

        Returns:
            rates: Same shape as input, float32 tensor of total diffusion rates
        """
        # 1. Prepare input for convolution
        # We treat any non-vacant site as "occupied" (1.0)
        is_batch = lattice_state.ndim == 4

        if is_batch:
            # Input: (Batch, X, Y, Z) -> (Batch, 1, X, Y, Z)
            occupancy = (lattice_state != SpeciesType.VACANT.value).float().unsqueeze(1)
        else:
            # Input: (X, Y, Z) -> (1, 1, X, Y, Z)
            occupancy = (
                (lattice_state != SpeciesType.VACANT.value).float().unsqueeze(0).unsqueeze(0)
            )

        # 2. Convolve to get coordination number for every site
        # We need Periodic Boundary Conditions (PBC) for X and Y, but NOT for Z.
        # F.conv3d(padding='circular') applies to all dims, which is wrong for Z.
        # So we manually pad:
        # X, Y: Circular padding (wrap around)
        # Z: Zero padding (open boundary)
        
        # Pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # Corresponds to (Z, Z, Y, Y, X, X) in PyTorch 5D tensor (Batch, Channel, X, Y, Z)
        # Wait, PyTorch pad is (last_dim_left, last_dim_right, 2nd_last_left, ...)
        # Our tensor is (Batch, 1, X, Y, Z)
        # So dims are: Z (last), Y, X
        # Pad: (pad_z_left, pad_z_right, pad_y_left, pad_y_right, pad_x_left, pad_x_right)
        
        padded_occupancy = F.pad(occupancy, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        
        # Now fix X and Y to be circular
        # We can't mix modes in F.pad easily.
        # Let's use circular pad for X,Y first, then zero pad Z.
        
        # Input: (Batch, 1, X, Y, Z)
        # Pad X and Y (circular)
        # Dims to pad: Y (2nd last), X (3rd last). Z is last.
        # pad args: (z_l, z_r, y_l, y_r, x_l, x_r)
        
        # Step 1: Circular pad X and Y
        # We treat Z as "channel" for 2D padding? No, it's 3D.
        # Let's slice and concat manually or use F.pad with 'circular' on a permuted tensor?
        # Easier: Just use F.pad with 'circular' for the whole thing, then zero-out the Z-padding?
        # No, that wraps Z top to bottom.
        
        # Manual Circular Padding for X and Y:
        # Append last slice to front, first slice to back.
        
        # X axis (dim 2):
        x_pad_pre = occupancy[:, :, -1:, :, :]
        x_pad_post = occupancy[:, :, :1, :, :]
        occ_pad_x = torch.cat([x_pad_pre, occupancy, x_pad_post], dim=2)
        
        # Y axis (dim 3):
        y_pad_pre = occ_pad_x[:, :, :, -1:, :]
        y_pad_post = occ_pad_x[:, :, :, :1, :]
        occ_pad_xy = torch.cat([y_pad_pre, occ_pad_x, y_pad_post], dim=3)
        
        # Z axis (dim 4): Zero padding (default for boundaries)
        # We need to pad Z with zeros to handle the kernel at z=0 and z=max
        z_pad = torch.zeros_like(occ_pad_xy[:, :, :, :, :1])
        occ_pad_xyz = torch.cat([z_pad, occ_pad_xy, z_pad], dim=4)
        
        # Now Convolve with padding=0 (valid)
        coordination_map = F.conv3d(occ_pad_xyz, self.kernel_coordination, padding=0)
        
        # Remove channel dim
        coordination_map = coordination_map.squeeze(1) if is_batch else coordination_map.squeeze()

        # 3. Calculate Activation Energy for every site
        # Use species-specific base barriers
        base_energies = torch.zeros_like(lattice_state, dtype=torch.float32)
        base_energies[lattice_state == SpeciesType.TI.value] = self.E_diff_ti
        base_energies[lattice_state == SpeciesType.O.value] = self.E_diff_o
        # Substrate/Vacant dummy values (masked later)
        base_energies[lattice_state == SpeciesType.SUBSTRATE.value] = 10.0
        base_energies[lattice_state == SpeciesType.VACANT.value] = 10.0

        # Bond Counting Model (Additive)
        # Ea = E_diff + N * E_bond_lateral
        # We use a moderate lateral bond energy (0.3 eV) to allow nucleation without freezing.
        E_bond_lateral = 0.05
        activation_energies = base_energies + (coordination_map * E_bond_lateral)

        # 4. Calculate Rates (Arrhenius)
        # Rate = nu0 * exp(-Ea / kT)
        rates = self.nu0 * torch.exp(-activation_energies / self.kT)

        # 5. Mask out vacant sites (vacant sites don't diffuse)
        # We only want rates for actual atoms
        atom_mask = lattice_state != SpeciesType.VACANT.value
        rates = rates * atom_mask.float()

        # Replace NaNs with 0.0 (just in case)
        rates = torch.nan_to_num(rates, nan=0.0)

        return rates

    def get_total_system_rate(self, rates: torch.Tensor):
        """Sum all rates to get KMC total rate."""
        return torch.sum(rates)
