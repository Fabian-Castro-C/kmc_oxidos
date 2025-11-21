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
        self.E_diff_base = params.e_diff_ti  # Base diffusion barrier
        self.E_bond = params.bond_energy_ti_o  # Bond energy contribution

    def calculate_diffusion_rates(self, lattice_state: torch.Tensor):
        """
        Calculate diffusion rates for ALL atoms in the lattice simultaneously.

        Args:
            lattice_state: (nx, ny, nz) int8 tensor

        Returns:
            rates: (nx, ny, nz) float32 tensor of total diffusion rates
        """
        # 1. Prepare input for convolution
        # Convert to float and add Batch/Channel dims: (1, 1, nx, ny, nz)
        # We treat any non-vacant site as "occupied" (1.0)
        occupancy = (lattice_state != SpeciesType.VACANT.value).float().unsqueeze(0).unsqueeze(0)

        # 2. Convolve to get coordination number for every site
        # Padding=1 ensures we handle boundaries (requires careful PBC handling later)
        # For now, zero-padding is used (open boundaries)
        coordination_map = F.conv3d(occupancy, self.kernel_coordination, padding=1)

        # Remove batch/channel dims
        coordination_map = coordination_map.squeeze()

        # 3. Calculate Activation Energy for every site
        # Ea = E_base + (Coordination * E_bond_contribution)
        # Note: This is a simplified model. Real KMC distinguishes Ti-O vs Ti-Ti bonds.
        # But for the benchmark/demo, this captures the computational complexity.
        activation_energies = self.E_diff_base + (coordination_map * self.E_bond)

        # 4. Calculate Rates (Arrhenius)
        # Rate = nu0 * exp(-Ea / kT)
        rates = self.nu0 * torch.exp(-activation_energies / self.kT)

        # 5. Mask out vacant sites (vacant sites don't diffuse)
        # We only want rates for actual atoms
        atom_mask = lattice_state != SpeciesType.VACANT.value
        rates = rates * atom_mask.float()

        return rates

    def get_total_system_rate(self, rates: torch.Tensor):
        """Sum all rates to get KMC total rate."""
        return torch.sum(rates)
