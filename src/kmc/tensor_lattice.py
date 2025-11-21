"""
Tensor-based Lattice implementation for GPU-accelerated KMC.

This class replaces the object-based 'Lattice' with a PyTorch tensor implementation.
It stores the entire system state in VRAM (int8), allowing for massive scalability.
"""

import torch

from src.kmc.lattice import SpeciesType


class TensorLattice:
    def __init__(
        self, size: tuple[int, int, int], device: torch.device, lattice_constant: float = 4.59
    ):
        """
        Initialize a GPU-accelerated lattice.

        Args:
            size: (nx, ny, nz) dimensions
            device: torch device (cuda/cpu)
            lattice_constant: Lattice parameter in Angstroms
        """
        self.nx, self.ny, self.nz = size
        self.device = device
        self.lattice_constant = lattice_constant

        # State Tensor: (nx, ny, nz)
        # 0 = VACANT
        # 1 = TI
        # 2 = O
        # 3 = SUBSTRATE
        self.state = torch.zeros(size, dtype=torch.int8, device=device)

        # Initialize Substrate at z=0
        self.state[:, :, 0] = SpeciesType.SUBSTRATE.value

        # Cache for height map (nx, ny)
        self.height_map = torch.zeros((self.nx, self.ny), dtype=torch.int16, device=device)

    def reset(self):
        """Reset lattice to initial state."""
        self.state.fill_(SpeciesType.VACANT.value)
        self.state[:, :, 0] = SpeciesType.SUBSTRATE.value
        self.height_map.fill_(0)

    def update_height_map(self):
        """
        Recalculate height map efficiently using GPU operations.
        Finds the highest occupied index z for each (x, y).
        """
        # Create mask of occupied sites (1 if occupied, 0 if vacant)
        occupied = (self.state != SpeciesType.VACANT.value).int()

        # Create a z-index tensor [0, 1, 2, ..., nz-1]
        z_indices = torch.arange(self.nz, device=self.device).view(1, 1, -1)

        # Multiply occupied mask by z-indices.
        # Vacant sites become 0. Occupied sites keep their z-index.
        occupied_indices = occupied * z_indices

        # Max over z-dimension gives the height
        self.height_map = torch.max(occupied_indices, dim=2).values.to(torch.int16)

    def get_occupancy_mask(self) -> torch.Tensor:
        """Return boolean mask of occupied sites."""
        return self.state != SpeciesType.VACANT.value

    def deposit_atom(self, x: int, y: int, species: SpeciesType):
        """
        Deposit an atom at (x, y) on top of the surface.
        """
        # Get current height at (x, y)
        z = int(self.height_map[x, y].item())

        # Deposit at z + 1 (if within bounds)
        if z + 1 < self.nz:
            self.state[x, y, z + 1] = species.value
            self.height_map[x, y] = z + 1
            return True
        return False

    def batch_deposit(self, positions: torch.Tensor, species_vals: torch.Tensor):
        """
        Parallel deposition for multiple atoms.
        Args:
            positions: (N, 2) tensor of (x, y) coordinates
            species_vals: (N,) tensor of species values
        """
        # This would be used for massive parallel updates
        # Implementation requires handling collisions (atomicAdd or scatter_reduce)
        pass
