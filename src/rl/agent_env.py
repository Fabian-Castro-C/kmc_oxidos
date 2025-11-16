"""
Agent-based Gymnasium environment for SwarmThinkers.

This environment implements the faithful SwarmThinkers architecture:
- Per-particle agents with local observations
- Shared policy network
- Global softmax event selection
- Physical rate reweighting

Observation: Variable [N_agents, 58] with padding
Action: Single integer selecting agent×action pair
Reward: Morphology-based (roughness, coverage, Ti:O ratio)
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from src.analysis.roughness import calculate_roughness
from src.data.tio2_parameters import TiO2Parameters
from src.kmc.lattice import Lattice, SpeciesType
from src.rl.action_space import N_ACTIONS, ActionType, create_action_mask
from src.rl.particle_agent import create_agents_from_lattice
from src.rl.rate_calculator import ActionRateCalculator

from .energy_calculator import SystemEnergyCalculator

logger = logging.getLogger(__name__)


class AgentBasedTiO2Env(gym.Env):  # type: ignore[misc]
    """
    Agent-based Gymnasium environment for SwarmThinkers training.

    This version is refactored for true scalability, removing the `max_agents`
    constraint. The environment adapts to the lattice size, allowing models
    to be trained on small lattices and run on large ones.

    Observation Space:
        A dictionary containing the observations for all active agents and
        global system features. This is handled by a custom training loop,
        not fed directly into a standard SB3 model.
        - "agent_observations": List of local observations.
        - "global_features": Numpy array of global features.

    Action Space:
        A `MultiDiscrete` space representing `[agent_index, action_index]`.
        The number of agents is dynamic.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lattice_size: tuple[int, int, int] = (8, 8, 50),
        tio2_parameters: TiO2Parameters | None = None,
        temperature: float | None = None,
        temperature_range: tuple[float, float] | None = None,
        deposition_rate: float = 1.0,
        max_steps: int = 1000,
        use_reweighting: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the scalable agent-based environment.

        Args:
            lattice_size: Lattice dimensions (nx, ny, nz).
            tio2_parameters: Dataclass with physical parameters.
            temperature: Fixed temperature (K). Mutually exclusive with temperature_range.
            temperature_range: Temperature range (T_min, T_max) for curriculum.
            deposition_rate: Deposition rate (ML/s).
            max_steps: Maximum steps per episode.
            use_reweighting: Use physical rate reweighting (default True).
            seed: Random seed.
        """
        super().__init__()

        # Temperature handling
        if temperature is not None and temperature_range is not None:
            raise ValueError("Specify either temperature or temperature_range, not both")
        if temperature is None and temperature_range is None:
            temperature = 600.0  # Default

        self.lattice_size = lattice_size
        self.tio2_params = tio2_parameters if tio2_parameters else TiO2Parameters()
        self.temperature_fixed = temperature
        self.temperature_range = temperature_range
        self.temperature = (
            temperature
            if temperature is not None
            else (temperature_range[0] if temperature_range else 600.0)
        )
        self.deposition_rate = deposition_rate
        self.max_steps = max_steps
        self.use_reweighting = use_reweighting

        # Random seed
        if seed is not None:
            np.random.seed(seed)

        # The observation space is a dictionary containing observations for all agents
        # and global system features. Since the number of agents is dynamic,
        # we define the agent observation space and will handle the list of
        # observations in the custom training loop.
        self.single_agent_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(58,), dtype=np.float32
        )
        self.global_feature_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "agent_observations": self.single_agent_observation_space,  # Placeholder for shape
                "global_features": self.global_feature_space,
            }
        )

        # The action space is now decoupled from a fixed max_agents value.
        # The training loop will select between agent actions and global actions.
        self.action_space = spaces.Discrete(N_ACTIONS)  # For a single agent

        # Initialize components (will be (re)created in reset())
        self.lattice: Lattice = Lattice(size=self.lattice_size)
        self.rate_calculator: ActionRateCalculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )
        self.energy_calculator: SystemEnergyCalculator = SystemEnergyCalculator(
            params=self.tio2_params
        )
        self.agents: list = []
        self.site_to_agent_map: dict[int, int] = {}  # Maps site_idx to agent index in self.agents

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info: dict[str, Any] = {}
        self.prev_omega: float = 0.0  # Track grand potential for reward calculation
        self.step_info: list[dict[str, Any]] = []  # Store info for logging

        # Cached metrics (updated only at episode end)
        self._cached_roughness: float | None = None
        self._cached_coverage: float | None = None

        # Cached species counts (updated incrementally)
        self._species_counts = {SpeciesType.TI: 0, SpeciesType.O: 0, SpeciesType.VACANT: 0}

        # Track valid deposition sites for O(1) access
        self._valid_deposition_sites: list[int] = []

        # Anti-loop mechanism: track recent states
        self._recent_actions: list[tuple] = []  # Last N actions
        self._action_history_size = 10  # Track last 10 actions
        self._loop_penalty = 1.0  # Penalty for detected loops
        self._step_penalty = 0.005  # Small penalty per step to encourage efficiency
        self._action_type_history: list[
            str
        ] = []  # Track action types for DEPOSIT->DESORB detection

        # Observation caching: cache agent observations to avoid recomputation
        self._observation_cache: dict[
            int, npt.NDArray[np.float32]
        ] = {}  # site_idx -> observation vector
        self._dirty_observations: set[int] = set()  # Set of site_idx that need recomputation

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Can contain 'lattice_size' to dynamically change it.

        Returns:
            observation: Initial observation dict
            info: Episode information dict
        """
        if seed is not None:
            np.random.seed(seed)

        # Allow dynamic resizing of the lattice on reset
        if options and "lattice_size" in options:
            self.lattice_size = options["lattice_size"]

        # Sample temperature if using range
        if self.temperature_range is not None:
            self.temperature = np.random.uniform(
                self.temperature_range[0], self.temperature_range[1]
            )
        else:
            self.temperature = (
                self.temperature_fixed if self.temperature_fixed is not None else 600.0
            )

        # Initialize lattice
        self.lattice = Lattice(size=self.lattice_size)

        # The action space is for a single agent; the training loop handles the rest.
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Initialize rate and energy calculators with current temperature
        self.rate_calculator = ActionRateCalculator(
            temperature=self.temperature,
            deposition_rate=self.deposition_rate,
        )
        self.energy_calculator = SystemEnergyCalculator(params=self.tio2_params)

        # Create initial agents (surface sites) and build the map
        self._rebuild_agents()

        # Reset episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.episode_info = {
            "episode_length": 0,
            "episode_reward": 0.0,
            "final_roughness": 0.0,
            "final_coverage": 0.0,
            "n_ti": 0,
            "n_o": 0,
        }

        # Calculate initial grand potential for reward computation
        self.prev_omega = self.energy_calculator.calculate_grand_potential(self.lattice)

        # Initialize cached species counts
        self._update_species_counts_full()

        # Initialize valid deposition sites
        self._update_deposition_sites()

        # Clear cached metrics (will be computed at episode end)
        self._cached_roughness = None
        self._cached_coverage = None

        # Clear action history for loop detection
        self._recent_actions = []
        self._action_type_history = []

        # Clear observation cache
        self._observation_cache.clear()
        self._dirty_observations.clear()

        observation = self._get_observation()
        info = {"step": 0, "n_agents": len(self.agents)}

        return observation, info

    def step(self, action: tuple[int, int] | str) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one time step within the environment.

        The action can be one of two things:
        1. A tuple (agent_idx, action_idx) for an agent-driven event.
        2. A string "DEPOSIT_TI" or "DEPOSIT_O" for a global deposition event.
        """
        self.step_count += 1

        is_deposition_action = isinstance(action, str)

        if is_deposition_action:
            if action == "DEPOSIT_TI":
                success, reason = self._execute_deposition(species=SpeciesType.TI)
            elif action == "DEPOSIT_O":
                success, reason = self._execute_deposition(species=SpeciesType.O)
            else:
                success = False
                reason = f"Unknown deposition action: {action}"
        else:
            agent_idx, action_idx = action

            # Validate agent index
            if agent_idx >= len(self.agents):
                reward = -0.01  # Small penalty for invalid agent
                terminated = False
                truncated = self.step_count >= self.max_steps
                reason = f"Agent index {agent_idx} out of bounds ({len(self.agents)} agents)."
                info = {
                    "failure_reason": reason,
                    "step": self.step_count,
                    "n_agents": len(self.agents),
                }
                return self._get_observation(), reward, terminated, truncated, info

            _agent = self.agents[agent_idx]
            _action_enum = ActionType(action_idx)
            success, reason = self._execute_agent_action(agent_idx, action_idx)

        # Track action for loop detection
        if is_deposition_action:
            action_signature = ("DEPOSIT", action)
            self._action_type_history.append("DEPOSIT")
        else:
            # Track (agent_site, action_type) to detect position-action loops
            agent = self.agents[agent_idx] if agent_idx < len(self.agents) else None
            action_signature = (agent.site_idx if agent else -1, action_idx)
            # Track if this is a DESORB action
            if success and ActionType(action_idx) == ActionType.DESORB:
                self._action_type_history.append("DESORB")
            else:
                self._action_type_history.append("OTHER")

        # Keep action type history limited
        if len(self._action_type_history) > 10:
            self._action_type_history.pop(0)

        self._recent_actions.append(action_signature)
        if len(self._recent_actions) > self._action_history_size:
            self._recent_actions.pop(0)

        # Determine if this was an exploration action (DIFFUSE or DESORB)
        action_was_exploration = False
        if success and not is_deposition_action:
            action_type_enum = ActionType(action_idx)
            action_was_exploration = action_type_enum in [
                ActionType.DESORB,
                ActionType.DIFFUSE_X_POS,
                ActionType.DIFFUSE_X_NEG,
                ActionType.DIFFUSE_Y_POS,
                ActionType.DIFFUSE_Y_NEG,
                ActionType.DIFFUSE_Z_POS,
                ActionType.DIFFUSE_Z_NEG,
            ]

        # Calculate reward based on change in grand potential
        reward = self._calculate_reward(action_was_exploration=action_was_exploration)

        # Detect and penalize loops
        if self._detect_action_loop():
            reward -= self._loop_penalty

        if not success:
            # If the action was not successful, the grand potential will not have changed.
            # We apply a small penalty to discourage the agent from choosing invalid actions.
            reward = -0.01
        self.total_reward += reward

        # Check termination conditions
        terminated = False  # Define a success condition, e.g., target thickness
        truncated = self.step_count >= self.max_steps

        # --- STRUCTURAL METRICS LOGGING (no reward impact) ---
        # Compute structural metrics every N steps for monitoring
        structural_metrics = {}
        if self.step_count % 100 == 0 or terminated or truncated:
            # Calculate key structural metrics
            n_ti = self._count_species(SpeciesType.TI)
            n_o = self._count_species(SpeciesType.O)
            ti_o_ratio = n_ti / (n_o / 2.0) if n_o > 0 else 0.0

            # Count bonds (already computed for critic features)
            ti_ti_bonds, ti_o_bonds, o_o_bonds = self._count_bonds()
            total_bonds = ti_ti_bonds + ti_o_bonds + o_o_bonds
            ti_o_fraction = ti_o_bonds / total_bonds if total_bonds > 0 else 0.0

            # Average coordination from global features
            global_features = self._get_observation()["global_features"]
            avg_coordination = global_features[3]  # Index 3 is avg_coord in _get_observation

            structural_metrics = {
                "ti_o_ratio": ti_o_ratio,
                "avg_coordination": avg_coordination,
                "ti_o_bonds_fraction": ti_o_fraction,
                "total_bonds": total_bonds,
            }

        # Prepare info dict
        if is_deposition_action:
            executed_action = action  # This will be "DEPOSIT_TI" or "DEPOSIT_O"
        else:
            action_name = str(ActionType(action_idx).name)
            executed_action = (agent_idx, action_name)

        # Use cached values during episode, only compute at end
        if terminated or truncated:
            # Compute final metrics once
            roughness = self._calculate_roughness_uncached()
            coverage = self._calculate_coverage_uncached()
            self._cached_roughness = roughness
            self._cached_coverage = coverage
        else:
            # During episode, use dummy values (not used for training)
            roughness = 0.0
            coverage = 0.0

        info = {
            "step": self.step_count,
            "roughness": roughness,
            "coverage": coverage,
            "n_agents": len(self.agents),
            "executed_action": executed_action,
            "reward": reward,
            "success": success,
            "failure_reason": reason,
            **structural_metrics,  # Add structural metrics if computed
        }
        self.step_info.append(info)

        # At the end of an episode, populate final stats
        if terminated or truncated:
            self.episode_info.update(
                {
                    "episode_length": self.step_count,
                    "episode_reward": self.total_reward,
                    "final_roughness": info["roughness"],
                    "final_coverage": info["coverage"],
                    "n_ti": self._count_species(SpeciesType.TI),
                    "n_o": self._count_species(SpeciesType.O),
                }
            )
            info.update(self.episode_info)

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for all valid actions for the current agents.

        Returns:
            A numpy array of shape (n_agents, N_ACTIONS) where True indicates a
            valid action.
        """
        return create_action_mask(self.agents, self.lattice.size, self.lattice)

    def _rebuild_agents(self) -> None:
        """
        Rebuild the full list of active agents and the site-to-agent map.
        Only called at reset, not during step execution.
        """
        self.agents = create_agents_from_lattice(self.lattice)
        self.site_to_agent_map = {agent.site_idx: idx for idx, agent in enumerate(self.agents)}

    def _add_agent(self, site_idx: int) -> None:
        """
        Add a new agent for a given site.

        Args:
            site_idx: Index of the site that became an agent.
        """
        from src.rl.particle_agent import ParticleAgent

        new_agent = ParticleAgent(site_idx=site_idx, lattice=self.lattice)
        agent_idx = len(self.agents)
        self.agents.append(new_agent)
        self.site_to_agent_map[site_idx] = agent_idx

    def _remove_agent(self, site_idx: int) -> None:
        """
        Remove an agent at a given site.

        Args:
            site_idx: Index of the site that is no longer an agent.
        """
        if site_idx not in self.site_to_agent_map:
            return

        agent_idx = self.site_to_agent_map[site_idx]

        # Remove from agents list (swap with last element for O(1) removal)
        last_agent = self.agents[-1]
        if agent_idx < len(self.agents) - 1:
            self.agents[agent_idx] = last_agent
            self.site_to_agent_map[last_agent.site_idx] = agent_idx

        self.agents.pop()
        del self.site_to_agent_map[site_idx]

    def _update_agent_at_site(self, site_idx: int) -> None:
        """
        Update or create/remove agent at a specific site based on current lattice state.

        Args:
            site_idx: Index of the site to update.
        """
        site = self.lattice.sites[site_idx]
        should_be_agent = site.is_occupied()
        is_agent = site_idx in self.site_to_agent_map

        if should_be_agent and not is_agent:
            self._add_agent(site_idx)
        elif not should_be_agent and is_agent:
            self._remove_agent(site_idx)

    def _update_affected_agents(self, affected_site_indices: list[int]) -> None:
        """
        Update agents for affected sites and their neighbors.
        This is called after deposition, desorption, or diffusion events.

        Args:
            affected_site_indices: Indices of sites that changed.
        """
        sites_to_check = set(affected_site_indices)

        # Add neighbors of affected sites
        for site_idx in affected_site_indices:
            site = self.lattice.sites[site_idx]
            sites_to_check.update(site.neighbors)

        # Mark all affected sites as having dirty observations
        self._dirty_observations.update(sites_to_check)

        # Update each affected site
        for site_idx in sites_to_check:
            self._update_agent_at_site(site_idx)

    def _execute_deposition(self, species: SpeciesType) -> tuple[bool, str]:
        """
        Executes a global deposition event for a given species.
        Tries random columns until finding an available spot or exhausting all options.
        """
        nx, ny, nz = self.lattice.size

        # Create list of all columns and shuffle them
        all_columns = [(x, y) for x in range(nx) for y in range(ny)]
        np.random.shuffle(all_columns)

        # Try each column until we find one with space
        for x, y in all_columns:
            # Search upward from z=1 (skip z=0, it's the substrate) to find first vacant site with support
            for z in range(1, nz):  # Start from z=1, not z=0
                site_idx = x + y * nx + z * nx * ny
                site = self.lattice.sites[site_idx]

                if site.species == SpeciesType.VACANT:
                    # Check if site below is occupied (provides support)
                    # Support can be SUBSTRATE, TI, or O (anything except VACANT)
                    site_below_idx = x + y * nx + (z - 1) * nx * ny
                    site_below = self.lattice.sites[site_below_idx]

                    if site_below.species != SpeciesType.VACANT:
                        # Found valid deposition site with support (below is SUBSTRATE, TI, or O)
                        success, reason = self.lattice.deposit_atom(site_idx, species)
                        if success:
                            # Update only affected agents incrementally
                            self._update_affected_agents([site_idx])
                        return success, reason
                    # This vacant site doesn't have support, check next z level

        # All columns exhausted - lattice is completely full
        return False, "Deposition failed: all columns are full"

    def _execute_agent_action(self, agent_idx: int, action_idx: int) -> tuple[bool, str]:
        """
        Execute the selected event on the lattice for a specific agent.
        This was previously the _execute_event method.
        """
        agent = self.agents[agent_idx]

        # Diffusion actions
        if action_idx in [
            ActionType.DIFFUSE_X_POS.value,
            ActionType.DIFFUSE_X_NEG.value,
            ActionType.DIFFUSE_Y_POS.value,
            ActionType.DIFFUSE_Y_NEG.value,
            ActionType.DIFFUSE_Z_POS.value,
            ActionType.DIFFUSE_Z_NEG.value,
        ]:
            action_enum = ActionType(action_idx)
            target_site = agent.get_neighbor_site(action_enum, self.lattice.size)

            if target_site is None:
                return False, f"Diffusion failed: invalid move for action {action_enum.name}"

            from_site_idx = agent.site_idx
            success, reason = self.lattice.diffuse_atom(from_site_idx, target_site)
            if success:
                # Update only affected agents incrementally (both source and destination)
                self._update_affected_agents([from_site_idx, target_site])

                # Update valid deposition sites
                self._update_deposition_sites_incremental(from_site_idx)
                self._update_deposition_sites_incremental(target_site)

        elif action_idx == ActionType.DESORB.value:
            # Desorption action
            from_site_idx = agent.site_idx
            old_species = self.lattice.sites[from_site_idx].species
            success, reason = self.lattice.desorb_atom(from_site_idx)
            if success:
                # Update species count
                if old_species in self._species_counts:
                    self._species_counts[old_species] -= 1
                    self._species_counts[SpeciesType.VACANT] += 1

                # Update only affected agents incrementally
                self._update_affected_agents([from_site_idx])

                # Update valid deposition sites
                self._update_deposition_sites_incremental(from_site_idx)
        else:
            return False, f"Unknown action index: {action_idx}"

        return success, reason

    def _get_observation(self) -> dict[str, Any]:
        """
        Get current observation. Optimized with caching and pre-allocation.

        Returns:
            A dictionary containing a list of agent observations and global features.
            This structure is meant for a custom training loop that can handle
            variable numbers of agents per step.
        """
        num_agents = len(self.agents)

        # Pre-allocate numpy array for agent observations
        # Shape: (num_agents, 58) where 58 is the observation vector size
        agent_observations = np.zeros((num_agents, 58), dtype=np.float32)

        # Populate the array using cache when possible
        for i, agent in enumerate(self.agents):
            site_idx = agent.site_idx

            # Check if observation is cached and not dirty
            if site_idx in self._observation_cache and site_idx not in self._dirty_observations:
                # Use cached observation
                agent_observations[i] = self._observation_cache[site_idx]
            else:
                # Compute and cache new observation
                obs_vector = agent.observe().to_vector()
                agent_observations[i] = obs_vector
                self._observation_cache[site_idx] = obs_vector
                # Remove from dirty set
                self._dirty_observations.discard(site_idx)

        # Compute thermodynamic and structural features for the critic
        total_energy = self.energy_calculator.calculate_system_energy(self.lattice)
        num_ti = self._species_counts[SpeciesType.TI]
        num_o = self._species_counts[SpeciesType.O]
        num_atoms = num_ti + num_o

        # Count bonds
        num_ti_ti_bonds, num_ti_o_bonds, num_o_o_bonds = self._count_bonds()
        total_bonds = num_ti_ti_bonds + num_ti_o_bonds + num_o_o_bonds

        # Grand potential (Ω = E - μN)
        mu_ti = self.tio2_params.mu_ti
        mu_o = self.tio2_params.mu_o
        grand_potential = total_energy - (mu_ti * num_ti + mu_o * num_o)

        # Average coordination (bonds per atom)
        avg_coordination = (total_bonds / num_atoms) if num_atoms > 0 else 0.0

        # Bond density (bonds per lattice site)
        total_sites = np.prod(self.lattice_size)
        bond_density = total_bonds / total_sites

        # Global features with rich thermodynamic information
        global_features = np.array(
            [
                self._calculate_mean_height(),
                self._calculate_height_std(),
                self._species_counts[SpeciesType.TI],
                self._species_counts[SpeciesType.O],
                self._species_counts[SpeciesType.VACANT],
                total_energy,  # NEW: System energy
                grand_potential,  # NEW: Grand potential Ω
                float(total_bonds),  # NEW: Total number of bonds
                avg_coordination,  # NEW: Average coordination number
                bond_density,  # NEW: Bond density (structural compactness)
                float(num_ti_o_bonds),  # NEW: Ti-O bonds (most important for TiO2)
                float(num_atoms),  # NEW: Total atoms
            ],
            dtype=np.float32,
        )

        return {
            "agent_observations": agent_observations,
            "global_features": global_features,
        }

    def _calculate_reward(
        self,
        action_was_exploration: bool = False,
    ) -> float:
        """
        Calculate SwarmThinkers reward for an open system: r_t = -ΔΩ + bonuses.

        A negative change in grand potential (system becomes more stable)
        results in a positive reward.

        Args:
            action_was_exploration: True if action was DIFFUSE or DESORB
            base_reward_before_scaling: Unscaled -ΔΩ value (for bonus threshold check)

        Returns:
            Reward r_t = -ΔΩ (eV) + exploration bonus, scaled to reduce variance
        """
        # Calculate current grand potential
        current_omega = self.energy_calculator.calculate_grand_potential(self.lattice)

        # Grand potential change
        delta_omega = current_omega - self.prev_omega

        # Reward = -ΔΩ (favor stability) - step_penalty (encourage efficiency)
        reward = -delta_omega - self._step_penalty

        # --- REWARD SHAPING: Exploration Bonus ---
        # Compensate thermodynamic penalty for DIFFUSE/DESORB to allow exploration
        # Only apply if the unscaled reward is not catastrophically negative
        exploration_bonus = 0.0
        if action_was_exploration and reward > -2.0:  # Threshold before scaling
            exploration_bonus = 0.4  # Small bonus to offset typical diffusion/desorption cost
            reward += exploration_bonus

        # Scale reward to reduce variance (rewards range from ~-10 to +13 eV)
        # Scaling by 5.0 brings them to ~-2 to +2.6 range, easier for RL to learn
        reward = reward / 5.0

        # Anti-repetition penalty: discourage immediate DEPOSIT→DESORB cycles
        # This is reward shaping, NOT physics violation - thermodynamics still intact
        if len(self._action_type_history) >= 2 and (
            self._action_type_history[-2] == "DEPOSIT" and self._action_type_history[-1] == "DESORB"
        ):
            reward -= 0.1  # Scaled penalty to match scaled rewards

        # Update previous grand potential for next step
        self.prev_omega = current_omega

        return float(reward)

    def _calculate_roughness_uncached(self) -> float:
        """Calculate surface roughness (expensive, use sparingly)."""
        if self.lattice is None:
            return 0.0
        try:
            height_profile = self.lattice.get_height_profile()
            roughness = calculate_roughness(height_profile)
            return float(roughness)
        except Exception:
            return 0.0

    def _calculate_roughness(self) -> float:
        """Get cached roughness or compute if needed."""
        if self._cached_roughness is None:
            self._cached_roughness = self._calculate_roughness_uncached()
        return self._cached_roughness

    def _calculate_coverage_uncached(self) -> float:
        """Calculate surface coverage fraction (expensive, use sparingly)."""
        if self.lattice is None:
            return 0.0
        n_atoms = self._species_counts[SpeciesType.TI] + self._species_counts[SpeciesType.O]
        nx, ny, _ = self.lattice_size
        total_surface_sites = nx * ny
        return n_atoms / total_surface_sites

    def _calculate_coverage(self) -> float:
        """Get cached coverage or compute if needed."""
        if self._cached_coverage is None:
            self._cached_coverage = self._calculate_coverage_uncached()
        return self._cached_coverage

    def _calculate_mean_height(self) -> float:
        """Calculate mean surface height."""
        if self.lattice is None:
            return 0.0
        heights = [site.position[2] for site in self.lattice.sites if site.is_occupied()]
        return float(np.mean(heights)) if heights else 0.0

    def _calculate_height_std(self) -> float:
        """Calculate height standard deviation."""
        if self.lattice is None:
            return 0.0
        heights = [site.position[2] for site in self.lattice.sites if site.is_occupied()]
        return float(np.std(heights)) if heights else 0.0

    def _count_species(self, species: SpeciesType) -> int:
        """Count atoms of given species (cached, O(1))."""
        return self._species_counts.get(species, 0)

    def _count_bonds(self) -> tuple[int, int, int]:
        """
        Count the number of bonds in the system.

        Returns:
            Tuple of (Ti-Ti bonds, Ti-O bonds, O-O bonds)
        """
        ti_ti_bonds = 0
        ti_o_bonds = 0
        o_o_bonds = 0

        # Count each bond once by only looking at neighbors with higher indices
        for i, site in enumerate(self.lattice.sites):
            if site.species == SpeciesType.VACANT:
                continue

            # Check all neighbors using the site's neighbors list
            for neighbor_idx in site.neighbors:
                # Only count each bond once (when i < neighbor_idx)
                if neighbor_idx <= i:
                    continue

                neighbor = self.lattice.sites[neighbor_idx]
                if neighbor.species == SpeciesType.VACANT:
                    continue

                # Classify the bond
                if site.species == SpeciesType.TI:
                    if neighbor.species == SpeciesType.TI:
                        ti_ti_bonds += 1
                    elif neighbor.species == SpeciesType.O:
                        ti_o_bonds += 1
                elif site.species == SpeciesType.O and neighbor.species == SpeciesType.O:
                    o_o_bonds += 1

        return ti_ti_bonds, ti_o_bonds, o_o_bonds

    def _update_species_counts_full(self) -> None:
        """Full recount of all species (only at reset)."""
        self._species_counts = {SpeciesType.TI: 0, SpeciesType.O: 0, SpeciesType.VACANT: 0}
        for site in self.lattice.sites:
            if site.species in self._species_counts:
                self._species_counts[site.species] += 1

    def _update_deposition_sites(self) -> None:
        """Full rebuild of valid deposition sites (only at reset)."""
        nx, ny, nz = self.lattice.size
        self._valid_deposition_sites = []

        for x in range(nx):
            for y in range(ny):
                for z in range(1, nz):  # Skip substrate at z=0
                    site_idx = x + y * nx + z * nx * ny
                    site = self.lattice.sites[site_idx]

                    if site.species == SpeciesType.VACANT:
                        site_below_idx = x + y * nx + (z - 1) * nx * ny
                        site_below = self.lattice.sites[site_below_idx]

                        if site_below.species != SpeciesType.VACANT:
                            self._valid_deposition_sites.append(site_idx)
                            break  # Only need first valid site per column

    def _update_deposition_sites_incremental(self, affected_site_idx: int) -> None:
        """Update valid deposition sites incrementally after a change."""
        nx, ny, nz = self.lattice.size
        x, y, z = self.lattice.sites[affected_site_idx].position

        # Remove all sites from this column from valid list
        self._valid_deposition_sites = [
            idx
            for idx in self._valid_deposition_sites
            if self.lattice.sites[idx].position[0] != x or self.lattice.sites[idx].position[1] != y
        ]

        # Find new valid site in this column
        for z_check in range(1, nz):
            site_idx = x + y * nx + z_check * nx * ny
            site = self.lattice.sites[site_idx]

            if site.species == SpeciesType.VACANT:
                site_below_idx = x + y * nx + (z_check - 1) * nx * ny
                site_below = self.lattice.sites[site_below_idx]

                if site_below.species != SpeciesType.VACANT:
                    self._valid_deposition_sites.append(site_idx)
                    break  # Only add first valid site per column

    def _detect_action_loop(self) -> bool:
        """
        Detect if the agent is stuck in a loop (e.g., moving back and forth).

        Returns:
            True if a loop is detected in recent actions.
        """
        if len(self._recent_actions) < 2:
            return False

        # Filter only agent actions (exclude deposits)
        agent_actions = [a for a in self._recent_actions if a[0] != "DEPOSIT"]

        if len(agent_actions) < 2:
            return False

        # Check for immediate reversible diffusion: action N and N+1 are opposites
        # This catches: DIFFUSE_X_POS immediately followed by DIFFUSE_X_NEG
        last_2 = agent_actions[-2:]
        site_0, action_0 = last_2[0]
        site_1, action_1 = last_2[1]

        # Opposite diffusion pairs (value difference of 1)
        opposite_pairs = [
            (0, 1),
            (1, 0),  # X_POS <-> X_NEG
            (2, 3),
            (3, 2),  # Y_POS <-> Y_NEG
            (4, 5),
            (5, 4),  # Z_POS <-> Z_NEG
        ]

        if (action_0, action_1) in opposite_pairs:
            return True

        # Check for 2-step loops: A -> B -> A -> B (same site, same action sequence)
        if len(agent_actions) >= 4:
            last_4 = agent_actions[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                return True

        # Check for 3-action loops: A -> B -> C -> A -> B -> C
        if len(agent_actions) >= 6:
            last_6 = agent_actions[-6:]
            if last_6[0] == last_6[3] and last_6[1] == last_6[4] and last_6[2] == last_6[5]:
                return True

        return False

    def render(self) -> None:
        """Render environment (text output)."""
        if self.lattice is None:
            print("Environment not initialized")
            return

        print(f"\nStep: {self.step_count}/{self.max_steps}")
        print(f"Agents: {len(self.agents)}")
        print(f"Roughness: {self._calculate_roughness():.3f} nm")
        print(f"Coverage: {self._calculate_coverage():.3f}")
        print(f"Ti: {self._count_species(SpeciesType.TI)}, O: {self._count_species(SpeciesType.O)}")

    def close(self) -> None:
        """Clean up resources."""
        pass
