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

        # Kernel 2: Barrier calculation (Include Z-1 to match CPU logic)
        self.kernel_barrier = torch.zeros((1, 1, 3, 3, 3), device=device)
        barrier_neighbors = [
            (1, 1, 2),  # Z+1 (Buried -> High barrier)
            (1, 1, 0),  # Z-1 (Substrate/Support -> Increases barrier)
            (1, 2, 1),  # Y+1
            (1, 0, 1),  # Y-1
            (2, 1, 1),  # X+1
            (0, 1, 1),  # X-1
        ]
        for idx in barrier_neighbors:
            self.kernel_barrier[0, 0, idx[0], idx[1], idx[2]] = 1.0

        # Pre-calculate rate constants (Arrhenius)
        # Rate = v0 * exp(-Ea / kT)
        # We can vectorize this: Rate = v0 * exp(-(E_base + n*E_bond) / kT)

        self.nu0 = 1e13  # Attempt frequency (Hz)

        # Energy parameters
        self.E_diff_base = params.ea_diff_ti  # Base diffusion barrier
        self.E_bond = params.bond_energy_ti_o  # Bond energy contribution

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
        # Padding=1 ensures we handle boundaries (requires careful PBC handling later)
        # For now, zero-padding is used (open boundaries)
        # Use kernel_barrier instead of kernel_coordination for Ea calculation
        coordination_map = F.conv3d(occupancy, self.kernel_barrier, padding=1)

        # Remove channel dim
        coordination_map = coordination_map.squeeze(1) if is_batch else coordination_map.squeeze()

        # 3. Calculate Activation Energy for every site
        # MATCH CPU LOGIC (src/kmc/rates.py):
        # effective_ea = activation_energy * (1.0 + 0.5 * coordination_factor)
        # where coordination_factor = coordination / 6.0
        
        # coordination_map contains N (number of neighbors)
        coordination_factor = coordination_map / 6.0
        
        # We use E_diff_base as the base activation energy
        activation_energies = self.E_diff_base * (1.0 + 0.5 * coordination_factor)

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
