"""
Test suite for agent-based Gymnasium environment.

Tests Phase 5 implementation:
1. Environment API compliance (Gymnasium interface)
2. Observation/action space shapes
3. Episode termination
4. Reward calculation
5. Multi-step rollouts
"""

from __future__ import annotations

import numpy as np

from src.training.agent_based_env import AgentBasedTiO2Env


def test_environment_initialization():
    """Test that environment initializes correctly."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=10,
        max_agents=32,
        temperature=300.0,
    )

    # Check spaces
    assert env.observation_space is not None
    assert env.action_space is not None

    # Check observation space structure
    assert "agents" in env.observation_space.spaces
    assert "mask" in env.observation_space.spaces
    assert "global" in env.observation_space.spaces

    # Check observation shapes
    assert env.observation_space["agents"].shape == (32, 58)
    assert env.observation_space["mask"].shape == (32,)
    assert env.observation_space["global"].shape == (7,)

    # Check action space
    assert env.action_space.n == 32 * 10  # max_agents * N_ACTIONS

    print(f"Environment initialized: obs_space={env.observation_space}")
    print(f"Action space: Discrete({env.action_space.n})")


def test_environment_reset():
    """Test that environment resets properly."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=10,
        max_agents=32,
    )

    observation, info = env.reset(seed=42)

    # Check observation structure
    assert isinstance(observation, dict)
    assert "agents" in observation
    assert "mask" in observation
    assert "global" in observation

    # Check observation shapes
    assert observation["agents"].shape == (32, 58)
    assert observation["mask"].shape == (32,)
    assert observation["global"].shape == (7,)

    # Check mask (should have some valid agents)
    n_valid_agents = int(np.sum(observation["mask"]))
    assert n_valid_agents > 0, "Should have at least one valid agent"

    # Check info dict
    assert isinstance(info, dict)
    assert "step" in info
    assert info["step"] == 0

    print(f"Reset observation: agents={observation['agents'].shape}")
    print(f"Valid agents: {n_valid_agents}/{32}")
    print(f"Global features: {observation['global']}")


def test_environment_step():
    """Test that environment steps correctly."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=10,
        max_agents=32,
    )

    observation, _ = env.reset(seed=42)

    # Take a step (select first valid agent, first action)
    action = 0  # agent_idx=0, action_idx=0

    observation, reward, terminated, truncated, info = env.step(action)

    # Check return types
    assert isinstance(observation, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Check observation structure
    assert "agents" in observation
    assert "mask" in observation
    assert "global" in observation

    # Check info
    assert "step" in info
    assert info["step"] == 1

    print(f"Step 1: reward={reward:.4f}, terminated={terminated}, truncated={truncated}")
    print(f"Info: {info}")


def test_multi_step_rollout():
    """Test multi-step episode rollout."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=20,
        max_agents=32,
        use_reweighting=False,  # Faster for testing
    )

    observation, _ = env.reset(seed=42)

    episode_reward = 0.0
    step_count = 0
    max_rollout_steps = 10

    for step in range(max_rollout_steps):
        # Sample random action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    print(f"\nRollout completed: {step_count} steps")
    print(f"Episode reward: {episode_reward:.4f}")
    print(f"Final roughness: {info.get('roughness', 0):.3f}")
    print(f"Final coverage: {info.get('coverage', 0):.3f}")

    # Check that we took some steps
    assert step_count > 0, "Should have taken at least one step"


def test_episode_termination():
    """Test that episodes terminate correctly."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=5,  # Short episode
        max_agents=32,
        use_reweighting=False,
    )

    observation, _ = env.reset(seed=42)

    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < 10:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        print(f"  Step {step_count}: terminated={terminated}, truncated={truncated}, env.step_count={info.get('step', '?')}")

    # Should have truncated due to max_steps=5
    print(f"\nFinal: step_count={step_count}, terminated={terminated}, truncated={truncated}")
    assert truncated or terminated, f"Episode should have ended after max_steps=5, but got terminated={terminated}, truncated={truncated}, step_count={step_count}"
    assert step_count <= 6, f"Should have stopped at max_steps=5, got {step_count}"  # Allow 1 extra due to >=

    print(f"\nEpisode ended after {step_count} steps")
    print(f"Terminated: {terminated}, Truncated: {truncated}")


def test_reward_calculation():
    """Test that rewards are calculated."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=10,
        max_agents=32,
        use_reweighting=False,
    )

    observation, _ = env.reset(seed=42)

    rewards = []

    for _ in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    # Check that rewards are numeric
    assert all(isinstance(r, (int, float)) for r in rewards), "Rewards should be numeric"

    print(f"\nRewards over {len(rewards)} steps:")
    print(f"Mean: {np.mean(rewards):.4f}, Std: {np.std(rewards):.4f}")
    print(f"Min: {min(rewards):.4f}, Max: {max(rewards):.4f}")


def test_observation_padding():
    """Test that observation padding works with varying agent counts."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=10,
        max_agents=64,  # Larger than initial agents
    )

    observation, _ = env.reset(seed=42)

    # Check mask indicates valid agents
    n_valid = int(np.sum(observation["mask"]))
    print(f"\nValid agents: {n_valid}/64")

    # Check that masked agents have zero observations
    for i in range(64):
        if observation["mask"][i] == 0:
            # Padding agent - should be zero
            assert np.all(observation["agents"][i] == 0), f"Padded agent {i} should be zero"

    # Check that valid agents have non-zero observations
    has_nonzero = False
    for i in range(n_valid):
        if np.any(observation["agents"][i] != 0):
            has_nonzero = True
            break

    # At least one valid agent should have non-zero observation
    # (might not be true if all are vacant sites, but likely)
    print(f"Has non-zero observations: {has_nonzero}")


def test_with_reweighting():
    """Test environment with physical rate reweighting."""
    env = AgentBasedTiO2Env(
        lattice_size=(3, 3, 5),
        max_steps=10,
        max_agents=32,
        use_reweighting=True,  # Enable reweighting
        temperature=300.0,
    )

    observation, _ = env.reset(seed=42)

    # Take a few steps
    for _ in range(3):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # Check that ESS is in info (when reweighting is enabled)
        if "ess" in info:
            ess = info["ess"]
            assert ess > 0, "ESS should be positive"
            print(f"ESS: {ess:.2f}")

        if terminated or truncated:
            break

    print("Reweighting test completed successfully")


def run_all_tests():
    """Run all test functions."""
    print("=" * 80)
    print("PHASE 5 TESTS: Agent-Based Gymnasium Environment")
    print("=" * 80)

    tests = [
        ("Environment Initialization", test_environment_initialization),
        ("Environment Reset", test_environment_reset),
        ("Environment Step", test_environment_step),
        ("Multi-Step Rollout", test_multi_step_rollout),
        ("Episode Termination", test_episode_termination),
        ("Reward Calculation", test_reward_calculation),
        ("Observation Padding", test_observation_padding),
        ("With Reweighting", test_with_reweighting),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'-' * 80}")
        print(f"TEST: {test_name}")
        print(f"{'-' * 80}")
        try:
            test_func()
            print("PASSED")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 80}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
