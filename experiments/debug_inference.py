"""Debug inference to understand why no atoms are deposited."""

import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kmc.lattice import SpeciesType
from src.rl.action_selection import select_action_gumbel_max
from src.rl.action_space import N_ACTIONS, ActionType
from src.rl.agent_env import AgentBasedTiO2Env


def debug_inference():
    """Run step-by-step inference with debugging."""

    # Create environment with a larger lattice for inference
    inference_lattice_size = (10, 10, 8)
    env = AgentBasedTiO2Env(
        lattice_size=inference_lattice_size,
        temperature=600.0,
        max_steps=500,
        use_reweighting=True,
    )

    # Load model trained on a smaller lattice
    model_path = "experiments/results/grand_potential_50k/agent_based_final"
    model = PPO.load(model_path)

    print("=" * 80)
    print("DEBUG INFERENCE - Step by step on a larger lattice")
    print("=" * 80)

    obs, info = env.reset(options={"lattice_size": inference_lattice_size})
    print("\nInitial state:")
    print(f"  Lattice size: {env.lattice.size}")
    print(f"  Temperature: {env.temperature:.1f} K")
    print(f"  Num agents: {info.get('n_agents', 0)}")

    # Run 200 steps with detailed logging
    for step in range(200):
        # --- Gumbel-Max Action Selection ---
        # 1. Get observations for all agents
        agent_obs = obs["agent_observations"]
        if not agent_obs:
            print("\nNo agents available to act. Ending episode.")
            break

        # 2. Get logits from the policy network for all agents
        # Note: This requires a model/policy that can handle variable-sized batches.
        # We process them one by one for simplicity here.
        all_logits = []
        for single_agent_obs in agent_obs:
            # The policy expects a batch, so we add a dimension
            obs_tensor, _ = model.policy.obs_to_tensor(np.expand_dims(single_agent_obs, axis=0))
            dist = model.policy.get_distribution(obs_tensor)
            all_logits.append(dist.distribution.logits.flatten())

        # 3. Select action using the Gumbel-Max trick across all agents and actions
        # We need to create a single tensor of all logits
        logits_tensor = torch.cat(all_logits)

        # The action is now an index into the flattened mega-list of all actions
        flat_action_idx = select_action_gumbel_max(logits_tensor)

        # Decode the flat index back to (agent_idx, action_idx)
        num_actions_per_agent = N_ACTIONS
        agent_idx = flat_action_idx // num_actions_per_agent
        action_idx = flat_action_idx % num_actions_per_agent

        action_to_step = (agent_idx, action_idx)
        # --- End Gumbel-Max ---

        print(f"\n--- Step {step + 1} ---")
        print(f"  Action selected: {action_to_step} (Agent {agent_idx}, Action {action_idx})")
        print(f"  Active agents: {len(agent_obs)}")

        obs, reward, terminated, truncated, info = env.step(action_to_step)

        # Get event info
        event_type = "Unknown"
        if "executed_action" in info:
            _, executed_action_idx = info["executed_action"]
            event_type = ActionType(executed_action_idx).name
        print(f"  Event executed: {event_type}")

        # Get lattice stats
        heights = env.lattice.get_height_profile()
        occupied = np.sum(heights > 0)

        # Count atoms
        ti_count = info.get("n_ti", env._count_species(SpeciesType.TI))
        o_count = info.get("n_o", env._count_species(SpeciesType.O))

        print(f"  Reward: {reward:.3f} eV")
        print(f"  Occupied sites: {occupied}")
        print(f"  Ti atoms: {ti_count}, O atoms: {o_count}")
        print(f"  Max height: {heights.max()}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            break

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    debug_inference()
