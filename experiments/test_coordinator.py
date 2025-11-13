"""
Test suite for Global Softmax Coordinator (Phase 3).

Tests:
1. Coordinator initialization
2. Event selection from multiple agents
3. Global softmax distribution
4. Temperature effects
5. Action masking integration
6. Statistical properties (sampling distribution)
"""

from __future__ import annotations

import numpy as np

from src.kmc.lattice import SpeciesType
from src.kmc.simulator import KMCSimulator
from src.rl.particle_agent import create_agents_from_lattice
from src.rl.shared_policy import SharedPolicyNetwork
from src.rl.swarm_coordinator import SwarmCoordinator


def test_coordinator_initialization():
    """Test 1: Coordinator initialization."""
    print("=" * 60)
    print("Test 1: Coordinator Initialization")
    print("=" * 60)

    # Create policy
    policy = SharedPolicyNetwork()

    # Create coordinator
    coordinator = SwarmCoordinator(policy)

    print(f"\n{coordinator}")
    print(f"Initial stats: {coordinator.get_stats()}")

    assert coordinator.step_count == 0, "Initial step count should be 0"
    assert coordinator.policy is policy, "Policy should be stored"

    print("\n✓ Coordinator initialization test passed")


def test_event_selection():
    """Test 2: Event selection from multiple agents."""
    print("\n" + "=" * 60)
    print("Test 2: Event Selection")
    print("=" * 60)

    # Create lattice with agents
    sim = KMCSimulator(lattice_size=(5, 5, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    # Add some particles
    sim.lattice.deposit_atom(1 + 1 * nx + 1 * nx * ny, SpeciesType.TI)
    sim.lattice.deposit_atom(2 + 2 * nx + 1 * nx * ny, SpeciesType.O)
    sim.lattice.deposit_atom(3 + 3 * nx + 1 * nx * ny, SpeciesType.TI)

    # Create agents
    agents = create_agents_from_lattice(sim.lattice)
    print(f"\nCreated {len(agents)} agents")

    # Create coordinator
    policy = SharedPolicyNetwork()
    coordinator = SwarmCoordinator(policy)

    # Select event
    event = coordinator.select_event(agents)

    print("\nSelected event:")
    print(f"  Agent index: {event.agent_idx}/{len(agents)}")
    print(f"  Action: {event.action.name}")
    print(f"  Probability: {event.probability:.6f}")
    print(f"  Logit: {event.logit:.3f}")
    print(f"  Global rank: {event.global_rank}")

    # Validate
    assert 0 <= event.agent_idx < len(agents), "Agent index out of range"
    assert 0.0 < event.probability <= 1.0, "Probability out of range"
    assert event.global_rank >= 0, "Global rank should be non-negative"

    # Validate action is valid for selected agent
    agent = agents[event.agent_idx]
    valid_actions = agent.get_valid_actions()
    assert event.action in valid_actions, f"Action {event.action} not valid for agent"

    print("\n✓ Event selection test passed")


def test_global_distribution():
    """Test 3: Global softmax distribution."""
    print("\n" + "=" * 60)
    print("Test 3: Global Distribution")
    print("=" * 60)

    # Create small system
    sim = KMCSimulator(lattice_size=(3, 3, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    # Add particles
    sim.lattice.deposit_atom(1 + 1 * nx + 1 * nx * ny, SpeciesType.TI)
    sim.lattice.deposit_atom(2 + 1 * nx + 1 * nx * ny, SpeciesType.O)

    agents = create_agents_from_lattice(sim.lattice)
    print(f"\nAgents: {len(agents)}")

    # Create coordinator
    policy = SharedPolicyNetwork()
    coordinator = SwarmCoordinator(policy)

    # Get distribution
    logits, probs = coordinator.get_global_action_distribution(agents)

    print(f"\nLogits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Total probability: {probs.sum():.6f}")
    print(f"Max probability: {probs.max():.6f}")
    print(f"Min non-zero probability: {probs[probs > 0].min():.6e}")

    # Validate
    assert logits.shape == (len(agents), 10), "Logits shape mismatch"
    assert probs.shape == (len(agents), 10), "Probabilities shape mismatch"
    assert np.isclose(probs.sum(), 1.0), "Probabilities should sum to 1"
    assert np.all(probs >= 0), "Probabilities should be non-negative"

    # Check that invalid actions have zero probability
    from src.rl.action_space import get_batch_action_masks

    masks = get_batch_action_masks(agents)
    invalid_probs = probs[~masks]
    assert np.all(invalid_probs == 0), "Invalid actions should have zero probability"

    print("\n✓ Global distribution test passed")


def test_temperature_effects():
    """Test 4: Temperature effects on selection."""
    print("\n" + "=" * 60)
    print("Test 4: Temperature Effects")
    print("=" * 60)

    # Create system
    sim = KMCSimulator(lattice_size=(4, 4, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    sim.lattice.deposit_atom(1 + 1 * nx + 1 * nx * ny, SpeciesType.TI)
    sim.lattice.deposit_atom(2 + 2 * nx + 1 * nx * ny, SpeciesType.O)

    agents = create_agents_from_lattice(sim.lattice)

    policy = SharedPolicyNetwork()
    coordinator = SwarmCoordinator(policy)

    # Test different temperatures
    temperatures = [0.1, 1.0, 10.0]
    entropies = []

    for temp in temperatures:
        _, probs = coordinator.get_global_action_distribution(agents, temperature=temp)

        # Calculate entropy (measure of randomness)
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
        entropies.append(entropy)

        print(f"\nTemperature: {temp:.1f}")
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Max probability: {probs.max():.6f}")
        print(f"  Effective choices: {np.exp(entropy):.1f}")

    # Validate: higher temperature → higher entropy
    assert entropies[2] > entropies[1] > entropies[0], "Entropy should increase with temperature"

    print("\n✓ Temperature effects test passed")


def test_action_masking_integration():
    """Test 5: Action masking integration."""
    print("\n" + "=" * 60)
    print("Test 5: Action Masking Integration")
    print("=" * 60)

    # Create system with specific agents
    sim = KMCSimulator(lattice_size=(3, 3, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    # Ti agent (7-8 actions depending on neighbors)
    ti_idx = 1 + 1 * nx + 1 * nx * ny
    sim.lattice.deposit_atom(ti_idx, SpeciesType.TI)

    agents = create_agents_from_lattice(sim.lattice)

    policy = SharedPolicyNetwork()
    coordinator = SwarmCoordinator(policy)

    # Get distribution
    logits, probs = coordinator.get_global_action_distribution(agents)

    # Find the Ti agent
    ti_agent = agents[0]  # Should be first non-vacant
    valid_actions = ti_agent.get_valid_actions()

    print(f"\nTi agent valid actions ({len(valid_actions)}):")
    for action in valid_actions:
        print(f"  - {action.name}")

    # Check probabilities
    agent_probs = probs[0]  # First agent
    print("\nAgent probabilities:")
    for i, p in enumerate(agent_probs):
        if p > 0:
            from src.rl.particle_agent import ActionType

            action = ActionType(i)
            print(f"  {action.name}: {p:.6f}")

    # Validate: only valid actions have non-zero probability
    for i in range(10):
        from src.rl.particle_agent import ActionType

        action = ActionType(i)
        if action in valid_actions:
            assert agent_probs[i] > 0, f"Valid action {action.name} should have non-zero prob"
        else:
            assert agent_probs[i] == 0, f"Invalid action {action.name} should have zero prob"

    print("\n✓ Action masking integration test passed")


def test_sampling_distribution():
    """Test 6: Statistical properties of sampling."""
    print("\n" + "=" * 60)
    print("Test 6: Sampling Distribution")
    print("=" * 60)

    # Create simple system
    sim = KMCSimulator(lattice_size=(3, 3, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    sim.lattice.deposit_atom(1 + 1 * nx + 1 * nx * ny, SpeciesType.TI)

    agents = create_agents_from_lattice(sim.lattice)

    policy = SharedPolicyNetwork()
    coordinator = SwarmCoordinator(policy)

    # Sample many times
    n_samples = 1000
    action_counts = {}

    for _ in range(n_samples):
        event = coordinator.select_event(agents)
        key = (event.agent_idx, event.action.value)
        action_counts[key] = action_counts.get(key, 0) + 1

    # Get theoretical distribution
    _, probs = coordinator.get_global_action_distribution(agents)

    print(f"\nSampled {n_samples} events")
    print(f"Unique (agent, action) pairs sampled: {len(action_counts)}")

    # Check that empirical frequencies match theoretical probabilities
    max_diff = 0
    for (agent_idx, action_idx), count in action_counts.items():
        empirical_prob = count / n_samples
        theoretical_prob = probs[agent_idx, action_idx]
        diff = abs(empirical_prob - theoretical_prob)
        max_diff = max(max_diff, diff)

    print(f"Max difference (empirical vs theoretical): {max_diff:.4f}")

    # Should be reasonably close (allow 5% deviation with 1000 samples)
    assert max_diff < 0.05, "Sampling should match theoretical distribution"

    print("\n✓ Sampling distribution test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 3: GLOBAL SOFTMAX COORDINATOR")
    print("Testing SwarmThinkers Event Selection")
    print("=" * 60)

    try:
        test_coordinator_initialization()
        test_event_selection()
        test_global_distribution()
        test_temperature_effects()
        test_action_masking_integration()
        test_sampling_distribution()

        print("\n" + "=" * 60)
        print("✓ ALL PHASE 3 TESTS PASSED")
        print("=" * 60)
        print("\nGlobal softmax coordinator is working correctly:")
        print("  ✓ Coordinator initialization")
        print("  ✓ Event selection (global softmax)")
        print("  ✓ Global probability distribution")
        print("  ✓ Temperature scaling (exploration/exploitation)")
        print("  ✓ Action masking integration")
        print("  ✓ Statistical sampling properties")
        print("\nReady for Phase 4: Physical Rate Calculator + Reweighting")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
