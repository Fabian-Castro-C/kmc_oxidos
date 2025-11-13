"""
Test suite for Shared Policy Network (Phase 2).

Tests:
1. Network initialization
2. Forward pass with batch observations
3. Action masking
4. Probability computation
5. Action sampling
6. Gradient flow (backpropagation)
"""

from __future__ import annotations

import numpy as np
import torch

from src.kmc.lattice import SpeciesType
from src.kmc.simulator import KMCSimulator
from src.rl.action_space import (
    N_ACTIONS,
    get_batch_action_masks,
)
from src.rl.particle_agent import create_agents_from_lattice
from src.rl.shared_policy import SharedPolicyNetwork


def test_network_initialization():
    """Test 1: Network initialization."""
    print("=" * 60)
    print("Test 1: Network Initialization")
    print("=" * 60)

    # Create network
    policy = SharedPolicyNetwork(
        obs_dim=58,
        action_dim=10,
        hidden_dim=256,
        n_hidden_layers=5,
    )

    print(f"\n{policy}")

    # Check parameter count
    # Expected: (58*256) + 256 + 4*(256*256 + 256) + (256*10) + 10
    # = 14848 + 256 + 4*65792 + 2560 + 10 = 280,842
    total_params = sum(p.numel() for p in policy.parameters())
    print("\nExpected params: ~280,842")
    print(f"Actual params: {total_params:,}")

    assert total_params > 200_000, "Network should have >200k parameters"
    print("\n✓ Network initialization test passed")


def test_forward_pass():
    """Test 2: Forward pass with batch observations."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)

    # Create network
    policy = SharedPolicyNetwork()

    # Create dummy observations
    batch_size = 16
    obs_dim = 58
    dummy_obs = np.random.randn(batch_size, obs_dim).astype(np.float32)

    print(f"\nInput shape: {dummy_obs.shape}")

    # Forward pass
    logits = policy.get_action_logits(dummy_obs)

    print(f"Output shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Validate shape
    assert logits.shape == (batch_size, N_ACTIONS), f"Expected shape ({batch_size}, {N_ACTIONS})"
    assert not np.any(np.isnan(logits)), "Logits should not contain NaN"
    assert not np.any(np.isinf(logits)), "Logits should not contain inf (without masking)"

    print("\n✓ Forward pass test passed")


def test_action_masking():
    """Test 3: Action masking."""
    print("\n" + "=" * 60)
    print("Test 3: Action Masking")
    print("=" * 60)

    # Create lattice with agents
    sim = KMCSimulator(lattice_size=(3, 3, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    # Add some particles
    sim.lattice.deposit_atom(1 + 1 * nx + 1 * nx * ny, SpeciesType.TI)
    sim.lattice.deposit_atom(2 + 1 * nx + 1 * nx * ny, SpeciesType.O)

    # Create agents
    agents = create_agents_from_lattice(sim.lattice)
    print(f"\nCreated {len(agents)} agents")

    # Get masks
    masks = get_batch_action_masks(agents)
    print(f"Mask shape: {masks.shape}")

    # Collect observations
    observations = np.array([agent.observe().to_vector() for agent in agents])
    print(f"Observations shape: {observations.shape}")

    # Create policy
    policy = SharedPolicyNetwork()

    # Get logits with masking
    logits_masked = policy.get_action_logits(observations, masks)
    print(f"\nMasked logits shape: {logits_masked.shape}")

    # Check that invalid actions have -inf
    for i, mask in enumerate(masks):
        invalid_actions = ~mask
        if np.any(invalid_actions):
            invalid_logits = logits_masked[i, invalid_actions]
            assert np.all(np.isinf(invalid_logits)), "Invalid actions should have -inf logits"
            print(f"Agent {i}: {np.sum(invalid_actions)} invalid actions correctly masked")

    print("\n✓ Action masking test passed")


def test_probability_computation():
    """Test 4: Probability computation."""
    print("\n" + "=" * 60)
    print("Test 4: Probability Computation")
    print("=" * 60)

    # Create policy
    policy = SharedPolicyNetwork()

    # Create dummy observations
    batch_size = 8
    dummy_obs = np.random.randn(batch_size, 58).astype(np.float32)

    # Create dummy masks (all actions valid)
    masks = np.ones((batch_size, N_ACTIONS), dtype=bool)

    # Get probabilities
    probs = policy.get_action_probabilities(dummy_obs, masks)

    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Example probabilities (agent 0): {probs[0]}")
    print(f"Sum of probabilities (agent 0): {probs[0].sum():.6f}")

    # Validate
    assert probs.shape == (batch_size, N_ACTIONS), "Shape mismatch"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"
    assert np.all(probs >= 0), "Probabilities should be non-negative"
    assert not np.any(np.isnan(probs)), "Probabilities should not contain NaN"

    print("\n✓ Probability computation test passed")


def test_action_sampling():
    """Test 5: Action sampling."""
    print("\n" + "=" * 60)
    print("Test 5: Action Sampling")
    print("=" * 60)

    # Create policy
    policy = SharedPolicyNetwork()

    # Create dummy observations
    batch_size = 10
    dummy_obs = np.random.randn(batch_size, 58).astype(np.float32)

    # Sample actions
    actions, log_probs = policy.sample_actions(dummy_obs)

    print(f"\nSampled actions: {actions}")
    print(f"Log probabilities: {log_probs}")
    print(f"Actions range: [{actions.min()}, {actions.max()}]")
    print(f"Log probs range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")

    # Validate
    assert actions.shape == (batch_size,), "Actions shape mismatch"
    assert log_probs.shape == (batch_size,), "Log probs shape mismatch"
    assert np.all(actions >= 0) and np.all(actions < N_ACTIONS), "Actions out of range"
    assert np.all(log_probs <= 0), "Log probabilities should be non-positive"

    print("\n✓ Action sampling test passed")


def test_gradient_flow():
    """Test 6: Gradient flow (backpropagation)."""
    print("\n" + "=" * 60)
    print("Test 6: Gradient Flow")
    print("=" * 60)

    # Create policy in training mode
    policy = SharedPolicyNetwork()
    policy.train()

    # Create dummy batch
    batch_size = 4
    dummy_obs = torch.randn(batch_size, 58, requires_grad=False)
    dummy_targets = torch.randint(0, N_ACTIONS, (batch_size,))

    print(f"\nInput batch size: {batch_size}")

    # Forward pass
    logits = policy(dummy_obs)

    # Compute loss (cross-entropy)
    loss = torch.nn.functional.cross_entropy(logits, dummy_targets)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_grad = False
    grad_norms = []
    for name, param in policy.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 0:
                print(f"{name}: grad_norm = {grad_norm:.6f}")

    assert has_grad, "Network should have gradients after backward()"
    assert len(grad_norms) > 0, "Should have gradient norms"

    print(f"\nTotal layers with gradients: {len(grad_norms)}")
    print(f"Mean gradient norm: {np.mean(grad_norms):.6f}")

    print("\n✓ Gradient flow test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 2: SHARED POLICY NETWORK")
    print("Testing SwarmThinkers Neural Network")
    print("=" * 60)

    try:
        test_network_initialization()
        test_forward_pass()
        test_action_masking()
        test_probability_computation()
        test_action_sampling()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("✓ ALL PHASE 2 TESTS PASSED")
        print("=" * 60)
        print("\nShared policy network is working correctly:")
        print("  ✓ Network initialization (5×256 MLP)")
        print("  ✓ Batch forward pass (58 → 10 dims)")
        print("  ✓ Action masking (invalid = -inf)")
        print("  ✓ Probability computation (softmax)")
        print("  ✓ Action sampling (categorical)")
        print("  ✓ Gradient backpropagation")
        print("\nReady for Phase 3: Global Softmax Coordinator")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
