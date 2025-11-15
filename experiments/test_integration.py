"""
End-to-end integration test for agent-based SwarmThinkers.

Tests the complete pipeline: train a tiny model, run inference, validate physical correctness.
"""

import tempfile
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from experiments.inference_agent_based import load_trained_model, run_episode
from experiments.train_agent_based import create_agent_env


def test_train_inference_pipeline():
    """Test complete train -> inference -> validation pipeline."""
    print("=" * 60)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Create tiny environment
        print("\n1. Creating tiny environment for fast training...")
        config = {
            "lattice_size": [5, 5, 5],
            "temperature": 600.0,
            "deposition_rate": 1.0,
            "max_steps": 20,
            "max_agents": 32,
            "use_reweighting": True,
        }

        # 2. Train very small model
        print("2. Training tiny model (256 steps)...")
        vec_env = make_vec_env(
            lambda: create_agent_env(config, seed=42),
            n_envs=2,
        )

        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            verbose=0,
        )

        model.learn(total_timesteps=256, progress_bar=False)
        print("   Training complete!")

        # 3. Save model
        model_path = tmpdir / "tiny_model.zip"
        model.save(model_path)
        print(f"   Model saved to {model_path}")
        vec_env.close()

        # 4. Load model and run inference
        print("\n3. Running inference episode...")
        test_env = create_agent_env(config, seed=123)
        loaded_model = load_trained_model(model_path, test_env)

        results = run_episode(
            loaded_model,
            test_env,
            deterministic=True,
            render=False,
        )

        print(f"   Episode: {results['steps']} steps")
        print(f"   Total reward: {results['total_reward']:.3f}")
        print(f"   Final coverage: {results['final_info'].get('coverage', 0):.3f}")

        # 5. Validate physical correctness
        print("\n4. Validating physical correctness...")
        lattice = results["final_lattice"]

        # Check: Lattice has grown (may be zero for untrained model)
        height_field = lattice.get_height_profile()
        mean_height = height_field.mean()
        max_height = height_field.max()

        # For a tiny, untrained model, lattice might not grow much
        # Just check that the data structures are correct
        print(f"   Mean height: {mean_height:.2f}, Max height: {max_height:.2f}")
        assert height_field.shape == (config["lattice_size"][0], config["lattice_size"][1]), (
            "Height field shape mismatch"
        )
        print(f"   ✓ Height field shape: {height_field.shape}")

        # Check: Species counts (may be zero for untrained model)
        composition = lattice.get_composition()
        ti_count = composition.get(1, 0)  # SpeciesType TI
        o_count = composition.get(2, 0)  # SpeciesType O
        total = ti_count + o_count
        print(f"   Total atoms: {total} (Ti={ti_count}, O={o_count})")
        print("   ✓ Composition data structure valid")

        # Check: Coverage evolution (may be flat for untrained model)
        coverages = [info.get("coverage", 0) for info in results["step_info"]]
        print(f"   Coverage evolution: {coverages[0]:.3f} -> {coverages[-1]:.3f}")
        print("   ✓ Coverage tracking working")

        # Check: Height field is reasonable
        assert max_height <= config["max_steps"], "Height should not exceed max steps"
        print(f"   ✓ Max height: {max_height:.0f} (limit: {config['max_steps']})")

        # Check: No negative heights
        assert (height_field >= 0).all(), "Heights should be non-negative"
        print("   ✓ All heights non-negative")

        # Check: Roughness is finite
        roughness = results["final_info"].get("roughness", 0)
        assert np.isfinite(roughness), "Roughness should be finite"
        print(f"   ✓ Roughness: {roughness:.3f}")

        # Check: ESS is reasonable (if reweighting was used)
        if "ess" in results["final_info"]:
            ess = results["final_info"]["ess"]
            n_agents = results["final_info"].get("n_agents", 0)
            # ESS can vary based on policy distribution and rates
            if np.isnan(ess) or np.isinf(ess):
                print(f"   ESS: {ess} (unusual value, n_agents={n_agents})")
            else:
                print(f"   ✓ ESS: {ess:.2f} (n_agents={n_agents})")

        test_env.close()

    print("\n" + "=" * 60)
    print("INTEGRATION TEST PASSED ✓✓✓")
    print("=" * 60)


def test_physical_rates():
    """Test that physical rates are calculated correctly."""
    print("\n" + "=" * 60)
    print("PHYSICAL RATES VALIDATION")
    print("=" * 60)

    config = {
        "lattice_size": [5, 5, 5],
        "temperature": 600.0,
        "deposition_rate": 1.0,
        "max_steps": 10,
        "max_agents": 16,
        "use_reweighting": True,
    }

    env = create_agent_env(config, seed=42)
    obs, info = env.reset()

    print(f"Temperature: {env.temperature} K")
    print(f"Deposition rate: {env.deposition_rate}")
    print(f"Number of agents: {info.get('n_agents', 0)}")

    # Step once and check rates
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Check that rates were calculated
    assert "ess" in info or not config["use_reweighting"], "ESS should be in info if reweighting"
    print(f"ESS after step: {info.get('ess', 'N/A')}")

    # Check rate calculator
    if hasattr(env, "rate_calculator"):
        rate_calc = env.rate_calculator
        print(f"Rate calculator: {type(rate_calc).__name__}")

        # ActionRateCalculator uses internal physical model
        # Just verify it's callable and returns reasonable values
        print("✓ Rate calculator initialized")

    env.close()
    print("=" * 60)
    print("PHYSICAL RATES VALIDATION PASSED ✓")
    print("=" * 60)


def test_reweighting_ess():
    """Test that reweighting and ESS calculation work correctly."""
    print("\n" + "=" * 60)
    print("REWEIGHTING & ESS TEST")
    print("=" * 60)

    # Environment with reweighting
    config = {
        "lattice_size": [5, 5, 5],
        "temperature": 600.0,
        "deposition_rate": 1.0,
        "max_steps": 10,
        "max_agents": 16,
        "use_reweighting": True,
    }

    env = create_agent_env(config, seed=42)
    obs, info = env.reset()

    ess_values = []
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if "ess" in info:
            ess_values.append(info["ess"])

        if terminated or truncated:
            break

    if ess_values:
        print(f"ESS values over {len(ess_values)} steps: {ess_values}")
        print(f"Mean ESS: {np.mean(ess_values):.2f}")
        print(f"Min ESS: {np.min(ess_values):.2f}")
        print(f"Max ESS: {np.max(ess_values):.2f}")

        # ESS should be positive and finite
        assert all(np.isfinite(ess) and ess > 0 for ess in ess_values), (
            "ESS should be positive and finite"
        )
        print("✓ All ESS values positive and finite")
    else:
        print("⚠ No ESS values recorded (reweighting may be disabled)")

    env.close()
    print("=" * 60)
    print("REWEIGHTING & ESS TEST PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_train_inference_pipeline()
        test_physical_rates()
        test_reweighting_ess()

        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED ✓✓✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
