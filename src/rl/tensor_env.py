"""
GPU-Accelerated RL Environment for TiO2 Growth.

This environment replaces the CPU-bound 'AgentBasedTiO2Env' with a fully tensorized
implementation that runs entirely on the GPU. It enables massive parallel training
(vectorized environments) without CPU-GPU data transfer bottlenecks.

Key Features:
- State: TensorLattice (int8 tensor in VRAM)
- Physics: TensorRateCalculator (Conv3d in VRAM)
- Observations: Generated directly as tensors on GPU
- Rewards: Computed in batch on GPU
"""

import numpy as np
import torch
from gymnasium import spaces

from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import SpeciesType
from src.kmc.tensor_rates import TensorRateCalculator
from src.rl.action_space import N_ACTIONS


class TensorTiO2Env:
    def __init__(
        self,
        num_envs: int = 1,
        lattice_size: tuple[int, int, int] = (20, 20, 20),
        device: str = "cuda",
        max_steps: int = 1000,
    ):
        """
        Initialize a batch of GPU environments.

        Args:
            num_envs: Number of parallel environments to simulate
            lattice_size: Dimensions of each lattice
            device: 'cuda' or 'cpu'
        """
        self.num_envs = num_envs
        self.nx, self.ny, self.nz = lattice_size
        self.device = torch.device(device)
        self.max_steps = max_steps

        # Physics Parameters
        self.params = TiO2Parameters()
        self.physics = TensorRateCalculator(self.params, temperature=600.0, device=self.device)

        # State Tensors (Batch, X, Y, Z)
        # We stack num_envs lattices into a 4D tensor
        self.lattices = torch.zeros(
            (num_envs, self.nx, self.ny, self.nz), dtype=torch.int8, device=self.device
        )

        # Initialize Substrate (z=0)
        self.lattices[:, :, :, 0] = SpeciesType.SUBSTRATE.value

        # Step counters
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Gym Spaces (defined for a single environment)
        # Observation: The lattice state (SpeciesType)
        self.observation_space = spaces.Box(
            low=0, high=len(SpeciesType), shape=(self.nx, self.ny, self.nz), dtype=np.int8
        )

        # Action: Discrete actions (Move Up, Down, Left, Right, Forward, Backward, etc.)
        # We treat the entire lattice as one "agent" with N_sites * N_ACTIONS possibilities
        # But for efficiency, we'll handle action selection externally and pass
        # (batch_size, 4) tensor: [batch_idx, x, y, z, action_enum]
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Pre-compute observation kernels
        self._init_observation_kernels()

        # Track previous Grand Potential for reward calculation
        self.prev_omega = torch.zeros(num_envs, device=self.device)

    def _init_observation_kernels(self):
        """Initialize convolution kernels for observation generation."""
        # 1. Neighbor Kernels (for 1-hot neighbors)
        # We have 6 nearest neighbors in the lattice:
        # (x+1), (x-1), (y+1), (y-1), (z+1), (z-1)
        # But observation space expects 12 neighbors. We fill the first 6.

        self.neighbor_offsets = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        # Remaining 6 are dummy (0,0,0) or handled by padding

        # 2. Relative Height Kernels
        # For the 6 neighbors, relative Z is: 0, 0, 0, 0, +1, -1
        self.relative_z_values = torch.tensor(
            [0, 0, 0, 0, 1, -1] + [0] * 6, dtype=torch.float32, device=self.device
        )

    def reset(self):
        """Reset all environments."""
        self.lattices.fill_(SpeciesType.VACANT.value)
        self.lattices[:, :, :, 0] = SpeciesType.SUBSTRATE.value
        self.steps.fill_(0)

        # Reset Omega
        self.prev_omega = self._calculate_grand_potential()

        return self._get_observations()

    def _calculate_grand_potential(self):
        """
        Calculate Grand Potential (Omega = E - mu*N) for all environments.
        Returns: (Batch,) tensor of Omega values in eV.
        """
        # 1. Count Species (N_Ti, N_O)
        # lattice: (B, X, Y, Z)
        n_ti = (self.lattices == SpeciesType.TI.value).sum(dim=(1, 2, 3)).float()
        n_o = (self.lattices == SpeciesType.O.value).sum(dim=(1, 2, 3)).float()

        # 2. Calculate Total Energy (Sum of Bonds)
        # One-hot: (B, 4, X, Y, Z)
        one_hot = torch.zeros((self.num_envs, 4, self.nx, self.ny, self.nz), device=self.device)
        one_hot.scatter_(1, self.lattices.unsqueeze(1).long(), 1.0)

        # Channels: 1=Ti, 2=O, 3=Substrate
        ti = one_hot[:, 1:2]
        o = one_hot[:, 2:3]
        sub = one_hot[:, 3:4]

        # Neighbor kernel (sum of 6 neighbors)
        k = self.physics.kernel_coordination  # (1, 1, 3, 3, 3)

        # Convolve to find neighbors
        # Padding=1 handles boundaries (open)
        ti_neighbors = torch.nn.functional.conv3d(ti, k, padding=1)
        o_neighbors = torch.nn.functional.conv3d(o, k, padding=1)
        sub_neighbors = torch.nn.functional.conv3d(sub, k, padding=1)

        # Calculate Bond Energies
        # Ti-Ti: Sum(Ti * Ti_neighbors) / 2
        e_ti_ti = (ti * ti_neighbors).sum(dim=(1, 2, 3, 4)) * 0.5 * self.params.bond_energy_ti_ti

        # O-O: Sum(O * O_neighbors) / 2
        e_o_o = (o * o_neighbors).sum(dim=(1, 2, 3, 4)) * 0.5 * self.params.bond_energy_o_o

        # Ti-O: Sum(Ti * O_neighbors) (No 0.5 because distinct sets)
        e_ti_o = (ti * o_neighbors).sum(dim=(1, 2, 3, 4)) * self.params.bond_energy_ti_o

        # Substrate Interactions (Assume similar to Ti-O for adhesion)
        # Ti-Substrate
        e_ti_sub = (ti * sub_neighbors).sum(dim=(1, 2, 3, 4)) * self.params.bond_energy_ti_o
        # O-Substrate
        e_o_sub = (o * sub_neighbors).sum(dim=(1, 2, 3, 4)) * self.params.bond_energy_ti_o

        total_energy = e_ti_ti + e_o_o + e_ti_o + e_ti_sub + e_o_sub

        # Grand Potential: E - mu*N
        omega = total_energy - (self.params.mu_ti * n_ti) - (self.params.mu_o * n_o)

        return omega

    def _get_observations(self):
        """
        Generate observations directly on GPU.
        Matches the shape (Batch, 51, X, Y, Z).
        """
        B, X, Y, Z = self.lattices.shape

        # 1. One-Hot Encoding of Species
        # (Batch, 4, X, Y, Z) -> Vacant, Ti, O, Substrate
        # We use scatter to create one-hot
        one_hot = torch.zeros((B, 4, X, Y, Z), device=self.device, dtype=torch.float32)
        one_hot.scatter_(1, self.lattices.unsqueeze(1).long(), 1.0)

        # 2. Construct Neighbor Features (Channels 0-35)
        # We need to shift the one-hot tensor for each neighbor direction
        # Neighbors 1-6: Real neighbors
        # Neighbors 7-12: Zeros (since we only have 6 NN)

        neighbor_feats = []

        # Helper to shift tensor
        def shift(t, dx, dy, dz):
            return torch.roll(t, shifts=(-dx, -dy, -dz), dims=(2, 3, 4))

        for dx, dy, dz in self.neighbor_offsets:
            # Get one-hot of neighbor (Vacant, Ti, O) - Ignore Substrate for obs?
            # observations.py uses [VACANT, TI, O] -> indices 0, 1, 2
            shifted = shift(one_hot, dx, dy, dz)
            neighbor_feats.append(shifted[:, 0:3])  # Take first 3 channels

        # Pad with zeros for neighbors 7-12
        for _ in range(6):
            neighbor_feats.append(torch.zeros((B, 3, X, Y, Z), device=self.device))

        # Stack: (Batch, 36, X, Y, Z)
        obs_neighbors = torch.cat(neighbor_feats, dim=1)

        # 3. Relative Heights (Channels 36-47)
        # These are constant planes for a grid, except boundaries
        # (Batch, 12, X, Y, Z)
        obs_rel_z = torch.zeros((B, 12, X, Y, Z), device=self.device)
        for i, val in enumerate(self.relative_z_values):
            obs_rel_z[:, i, :, :, :] = val

        # 4. Local Composition (Channels 48-49)
        # Count Ti (idx 1) and O (idx 2) in 1st shell
        # We can sum the neighbor features we just extracted
        # Ti is channel 1 in each block of 3
        # O is channel 2 in each block of 3

        # Extract Ti channels: indices 1, 4, 7, ...
        ti_indices = [i * 3 + 1 for i in range(6)]
        o_indices = [i * 3 + 2 for i in range(6)]

        n_ti = obs_neighbors[:, ti_indices].sum(dim=1, keepdim=True)
        n_o = obs_neighbors[:, o_indices].sum(dim=1, keepdim=True)

        # 5. Absolute Z (Channel 50)
        z_coords = torch.arange(Z, device=self.device, dtype=torch.float32)
        obs_abs_z = z_coords.view(1, 1, 1, 1, Z).expand(B, 1, X, Y, Z)

        # Concatenate all
        # 36 + 12 + 2 + 1 = 51
        full_obs = torch.cat([obs_neighbors, obs_rel_z, n_ti, n_o, obs_abs_z], dim=1)

        return full_obs

    def step(self, actions):
        """
        Execute ONE action per environment.

        Args:
            actions: (Batch, 4) tensor of [x, y, z, action_enum]
        """
        # actions: (B, 4) -> x, y, z, action_type
        B = self.num_envs

        x = actions[:, 0]
        y = actions[:, 1]
        z = actions[:, 2]
        a = actions[:, 3]

        # Batch indices
        b_idx = torch.arange(B, device=self.device)

        # Get current species at target
        # We need to handle bounds checking if the policy is bad
        # But assuming valid indices for now

        # Action Logic
        # 0-5: Diffusion (Neighbor swap)
        # 6: Desorption (Set to Vacant)

        # Define neighbor offsets for actions 0-5
        # 0: X+1, 1: X-1, 2: Y+1, 3: Y-1, 4: Z+1, 5: Z-1
        dx = torch.zeros(B, dtype=torch.long, device=self.device)
        dy = torch.zeros(B, dtype=torch.long, device=self.device)
        dz = torch.zeros(B, dtype=torch.long, device=self.device)

        # Mask for diffusion actions
        is_diff = a < 6

        # Map action to delta (simplified, not efficient but clear)
        dx = torch.where(a == 0, torch.tensor(1, device=self.device), dx)
        dx = torch.where(a == 1, torch.tensor(-1, device=self.device), dx)
        dy = torch.where(a == 2, torch.tensor(1, device=self.device), dy)
        dy = torch.where(a == 3, torch.tensor(-1, device=self.device), dy)
        dz = torch.where(a == 4, torch.tensor(1, device=self.device), dz)
        dz = torch.where(a == 5, torch.tensor(-1, device=self.device), dz)

        # Target coordinates
        tx = (x + dx) % self.nx  # PBC
        ty = (y + dy) % self.ny  # PBC
        tz = z + dz  # No PBC in Z

        # Check bounds for Z
        valid_z = (tz >= 0) & (tz < self.nz)

        # Execute Diffusion
        # Swap (x,y,z) with (tx,ty,tz)
        # Only if valid_z and is_diff
        mask_move = is_diff & valid_z

        if mask_move.any():
            # Filter indices
            mb = b_idx[mask_move]
            mx, my, mz = x[mask_move], y[mask_move], z[mask_move]
            mtx, mty, mtz = tx[mask_move], ty[mask_move], tz[mask_move]

            # Get values
            src_val = self.lattices[mb, mx, my, mz]
            dst_val = self.lattices[mb, mtx, mty, mtz]

            # Swap
            self.lattices[mb, mx, my, mz] = dst_val
            self.lattices[mb, mtx, mty, mtz] = src_val

        # Execute Desorption (Action 6)
        mask_desorb = a == 6
        if mask_desorb.any():
            self.lattices[b_idx[mask_desorb], x[mask_desorb], y[mask_desorb], z[mask_desorb]] = (
                SpeciesType.VACANT.value
            )

        # Calculate Rewards
        current_omega = self._calculate_grand_potential()
        delta_omega = current_omega - self.prev_omega

        # Reward = -DeltaOmega (favor stability)
        # Scaled by 5.0 as in SwarmThinkers
        rewards = -delta_omega / 5.0

        self.prev_omega = current_omega

        self.steps += 1
        terminated = self.steps >= self.max_steps

        if terminated.any():
            self.reset_envs(torch.where(terminated)[0])

        return self._get_observations(), rewards, terminated, {}

    def reset_envs(self, env_indices):
        """Reset specific environments."""
        self.lattices[env_indices].fill_(SpeciesType.VACANT.value)
        self.lattices[env_indices, :, :, 0] = SpeciesType.SUBSTRATE.value
        self.steps[env_indices] = 0

        # Reset Omega for these envs
        # We need to recalculate for these specific indices
        # But _calculate_grand_potential returns all.
        # Efficient way: just update the slice
        # Or simpler: just recalculate all (expensive but safe)
        # Or better: implement _calculate_grand_potential to accept indices?
        # For now, let's just recalculate all and slice, or assume reset state is known.
        # Reset state (empty) has Omega = 0 (if we ignore substrate energy or it cancels out)
        # Actually, substrate has energy? No, we only count bonds involving deposited atoms.
        # So empty lattice has E=0, N=0 -> Omega=0.
        self.prev_omega[env_indices] = 0.0

    def deposit(self, env_indices, species_type, coords):
        """
        Execute deposition for specific environments.

        Args:
            env_indices: Tensor of environment indices to deposit in.
            species_type: SpeciesType.TI or SpeciesType.O
            coords: (N, 3) tensor of (x, y, z) coordinates for each env in env_indices.

        Returns:
            rewards: Tensor of rewards for these depositions.
        """
        # Update Lattice
        # coords: (N, 3) -> x, y, z
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # Set species
        # We need to use advanced indexing
        # self.lattices[env_indices, x, y, z] = species_type.value
        self.lattices.index_put_(
            (env_indices, x, y, z),
            torch.tensor(species_type.value, device=self.device, dtype=torch.int8),
        )

        # Calculate Rewards (Change in Grand Potential)
        # We only need to recalculate Omega for the affected environments
        # But _calculate_grand_potential computes for ALL envs.
        # For efficiency, we might want to compute delta locally, but for now re-computing all is safer/easier.

        new_omega = self._calculate_grand_potential()

        # Reward = -(Omega_new - Omega_old) / kT (or scaled)
        # We use the same scaling as CPU: -delta_omega / 5.0
        delta_omega = new_omega - self.prev_omega
        rewards = -delta_omega / 5.0

        # Update prev_omega
        self.prev_omega = new_omega

        # Return rewards for the affected envs
        return rewards[env_indices]
