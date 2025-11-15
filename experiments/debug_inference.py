"""Debug inference to understand why no atoms are deposited."""

import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.agent_env import AgentBasedTiO2Env
from src.rl.action_selection import select_action_gumbel_max


def debug_inference():
    """Run step-by-step inference with debugging."""

    # Create environment matching training config
    env = AgentBasedTiO2Env(
        lattice_size=[5, 5, 5],
        temperature=600.0,
        max_steps=500,
        max_agents=64,
        use_reweighting=True,
    )

    # Load model
    model_path = "experiments/results/grand_potential_50k/agent_based_final"
    model = PPO.load(model_path)

    print("=" * 80)
    print("DEBUG INFERENCE - Step by step")
    print("=" * 80)

    obs, info = env.reset()
    print("\nInitial state:")
    print(f"  Lattice size: {env.lattice.size}")
    print(f"  Temperature: {env.temperature:.1f} K")
    print(f"  Num agents: {len(env.agents)}")
    print(f"  Max agents: {env.max_agents}")

    # Run 200 steps with detailed logging
    for step in range(200):
        # --- Gumbel-Max Action Selection ---
        # 1. Get logits from the policy network for the current observation
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        # The distribution contains the logits
        dist = model.policy.get_distribution(obs_tensor)
        logits = dist.distribution.logits

        # 2. Select action using the Gumbel-Max trick
        # We need to flatten the logits tensor if it's not already 1D
        action = select_action_gumbel_max(logits.flatten())
        # --- End Gumbel-Max ---

        print(f"\n--- Step {step + 1} ---")
        print(f"  Action selected: {action}")
        print(f"  Active agents: {len(env.agents)}")

        obs, reward, terminated, truncated, info = env.step(action)

        # Get event info
        if "event_type" in info:
            event_type = info["event_type"]
            print(f"  Event executed: {event_type}")

        # Get action probabilities for detailed debugging
        # Note: This is computationally expensive and for debugging only
        # We need to convert the single observation to a batch
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()

        # Get the selected agent and its action probabilities
        agent_idx = info.get("agent_idx", -1)
        if agent_idx != -1:
            start_idx = agent_idx * 10  # 10 actions per agent
            end_idx = start_idx + 10
            agent_probs = probs[start_idx:end_idx]

            action_names = [
                "ADS_TI", "ADS_O", "DESORB",
                "D_XP", "D_XN", "D_YP", "D_YN", "D_ZP", "D_ZN",
                "REACT"
            ]

            print("  Probabilidades de acción para el agente seleccionado:")
            for i, p in enumerate(agent_probs):
                print(f"    - {action_names[i]}: {p:.4f}")

        # Get lattice stats
        heights = env.lattice.get_height_profile()
        occupied = np.sum(heights > 0)

        # Count atoms - sites is a list
        ti_count = 0
        o_count = 0
        for site in env.lattice.sites:
            if site.species.name == "TI":
                ti_count += 1
            elif site.species.name == "O":
                o_count += 1

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
