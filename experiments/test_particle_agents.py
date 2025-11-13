"""
Test script for Phase 1: Particle Agent Observations.

Validates:
- Agent creation from lattice
- Local observation extraction (1st/2nd neighbors)
- Neighbor encoding
- Species handling (VACANT, Ti, O)
- Valid action determination
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kmc.lattice import Lattice, SpeciesType
from src.kmc.simulator import KMCSimulator
from src.rl.particle_agent import ActionType, ParticleAgent, create_agents_from_lattice


def test_agent_creation():
    """Test 1: Create agents from lattice."""
    print("=" * 60)
    print("Test 1: Agent Creation")
    print("=" * 60)

    # Create small lattice
    sim = KMCSimulator(lattice_size=(5, 5, 3), temperature=600.0, deposition_rate=1.0)

    # Deposit some atoms manually for testing
    # Note: z=0 is substrate, so deposit at z=1
    nx, ny, _ = sim.lattice.size
    # Position (2,2,1) → idx = 2 + 2*5 + 1*5*5 = 37
    sim.lattice.deposit_atom(2 + 2 * nx + 1 * nx * ny, SpeciesType.TI)
    # Position (2,3,1) → idx = 2 + 3*5 + 1*5*5 = 42
    sim.lattice.deposit_atom(2 + 3 * nx + 1 * nx * ny, SpeciesType.O)
    # Position (3,2,1) → idx = 3 + 2*5 + 1*5*5 = 38
    sim.lattice.deposit_atom(3 + 2 * nx + 1 * nx * ny, SpeciesType.TI)
    # Position (2,2,2) → idx = 2 + 2*5 + 2*5*5 = 62
    sim.lattice.deposit_atom(2 + 2 * nx + 2 * nx * ny, SpeciesType.O)

    # Create agents
    agents = create_agents_from_lattice(sim.lattice)

    print(f"Lattice size: {sim.lattice.size}")
    print(f"Total agents created: {len(agents)}")
    print("\nAgent details:")
    for i, agent in enumerate(agents):
        print(f"  {i}: {agent}")

    # Validate
    assert len(agents) > 0, "Should create at least one agent"
    assert all(isinstance(a, ParticleAgent) for a in agents), "All should be ParticleAgent"

    print("\n✓ Agent creation test passed")
    return sim, agents


def test_local_observations(sim, agents):
    """Test 2: Local observation extraction."""
    print("\n" + "=" * 60)
    print("Test 2: Local Observations")
    print("=" * 60)

    if not agents:
        print("No agents to test")
        return

    # Test first agent
    agent = agents[0]
    obs = agent.observe()

    print(f"\nAgent: {agent}")
    print(f"Position: {agent.position}")
    print(f"Species: {agent.species.name}")
    print(f"\n1st neighbors (6): {obs.neighbors_1st}")
    print(f"2nd neighbors (12): {obs.neighbors_2nd}")
    print(f"Own species: {obs.own_species}")
    print(f"Height: {obs.height}")

    # Validate shapes
    assert obs.neighbors_1st.shape == (6,), "1st neighbors should have 6 elements"
    assert obs.neighbors_2nd.shape == (12,), "2nd neighbors should have 12 elements"
    assert isinstance(obs.own_species, (int, np.integer)), "Species should be int"
    assert isinstance(obs.height, (int, np.integer)), "Height should be int"

    # Validate species values (0=VACANT, 1=TI, 2=O, 3=SUBSTRATE)
    assert all(0 <= s <= 3 for s in obs.neighbors_1st), "1st neighbor species out of range"
    assert all(0 <= s <= 3 for s in obs.neighbors_2nd), "2nd neighbor species out of range"
    assert 0 <= obs.own_species <= 3, "Own species out of range"

    print("\n✓ Local observation test passed")


def test_observation_encoding(sim, agents):
    """Test 3: Observation vector encoding."""
    print("\n" + "=" * 60)
    print("Test 3: Observation Encoding")
    print("=" * 60)

    if not agents:
        print("No agents to test")
        return

    agent = agents[0]
    obs = agent.observe()
    vec = obs.to_vector()

    print(f"\nAgent: {agent}")
    print(f"Observation vector shape: {vec.shape}")
    print(f"Observation vector dtype: {vec.dtype}")
    print(f"First 10 values: {vec[:10]}")
    print(f"Last 10 values: {vec[-10:]}")

    # Expected: 6*3 (1st neighbors) + 12*3 (2nd neighbors) + 3 (own) + 1 (height) = 58
    assert vec.shape == (58,), f"Expected shape (58,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected dtype float32, got {vec.dtype}"
    assert np.all(vec >= 0.0), "All values should be non-negative"
    assert np.all(vec <= 1.0), "All values should be <= 1.0 (one-hot + normalized)"

    print("\n✓ Observation encoding test passed")


def test_valid_actions():
    """Test 4: Valid actions per species."""
    print("\n" + "=" * 60)
    print("Test 4: Valid Actions")
    print("=" * 60)

    # Create simple lattice
    sim = KMCSimulator(lattice_size=(3, 3, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    # Test VACANT site (use z=1 surface layer)
    # Position (1,1,1) → idx = 1 + 1*3 + 1*3*3 = 13
    vacant_idx = 1 + 1 * nx + 1 * nx * ny
    vacant_agent = ParticleAgent(vacant_idx, sim.lattice)
    vacant_actions = vacant_agent.get_valid_actions()
    print(f"\nVACANT site actions ({len(vacant_actions)}):")
    for action in vacant_actions:
        print(f"  - {action.name}")
    assert ActionType.ADSORB_TI in vacant_actions, "VACANT should be able to adsorb Ti"
    assert ActionType.ADSORB_O in vacant_actions, "VACANT should be able to adsorb O"
    assert len(vacant_actions) == 2, "VACANT should have exactly 2 actions"

    # Deposit Ti for testing occupied site
    # Position (1,2,1) → idx = 1 + 2*3 + 1*3*3 = 16
    ti_idx = 1 + 2 * nx + 1 * nx * ny
    sim.lattice.deposit_atom(ti_idx, SpeciesType.TI)
    ti_agent = ParticleAgent(ti_idx, sim.lattice)
    ti_actions = ti_agent.get_valid_actions()
    print(f"\nTi particle actions ({len(ti_actions)}):")
    for action in ti_actions:
        print(f"  - {action.name}")
    assert ActionType.DIFFUSE_X_POS in ti_actions, "Ti should be able to diffuse"
    assert ActionType.DESORB in ti_actions, "Ti should be able to desorb"
    assert len(ti_actions) == 7, "Ti should have 7 actions (6 diffuse + 1 desorb)"

    # Deposit O for testing
    # Position (2,1,1) → idx = 2 + 1*3 + 1*3*3 = 14
    o_idx = 2 + 1 * nx + 1 * nx * ny
    sim.lattice.deposit_atom(o_idx, SpeciesType.O)
    o_agent = ParticleAgent(o_idx, sim.lattice)
    o_actions = o_agent.get_valid_actions()
    print(f"\nO particle actions ({len(o_actions)}):")
    for action in o_actions:
        print(f"  - {action.name}")
    assert len(o_actions) == 7, "O should have 7 actions (6 diffuse + 1 desorb)"

    # Test Ti with O neighbors for reaction (use fresh lattice)
    print("\nTesting REACT_TIO2 action availability...")
    sim_react = KMCSimulator(lattice_size=(3, 3, 3), temperature=600.0, deposition_rate=1.0)
    
    # Position (1,1,1) → idx = 1 + 1*3 + 1*3*3 = 13
    ti_center_idx = 1 + 1 * nx + 1 * nx * ny
    sim_react.lattice.deposit_atom(ti_center_idx, SpeciesType.TI)
    
    # Add 2 O neighbors
    # Position (0,1,1) → idx = 0 + 1*3 + 1*3*3 = 12
    sim_react.lattice.deposit_atom(0 + 1 * nx + 1 * nx * ny, SpeciesType.O)
    # Position (2,1,1) → idx = 2 + 1*3 + 1*3*3 = 14
    sim_react.lattice.deposit_atom(2 + 1 * nx + 1 * nx * ny, SpeciesType.O)
    
    ti_with_o_agent = ParticleAgent(ti_center_idx, sim_react.lattice)
    ti_with_o_actions = ti_with_o_agent.get_valid_actions()
    print(f"Ti with 2 O neighbors actions ({len(ti_with_o_actions)}):")
    for action in ti_with_o_actions:
        print(f"  - {action.name}")
    assert ActionType.REACT_TIO2 in ti_with_o_actions, "Ti with 2+ O neighbors should be able to react"
    assert len(ti_with_o_actions) == 8, "Ti with O neighbors should have 8 actions (6 diffuse + 1 desorb + 1 react)"

    print("\n✓ Valid actions test passed")


def test_batch_observations():
    """Test 5: Batch observation collection."""
    print("\n" + "=" * 60)
    print("Test 5: Batch Observations")
    print("=" * 60)

    # Create lattice with multiple particles
    sim = KMCSimulator(lattice_size=(5, 5, 3), temperature=600.0, deposition_rate=1.0)
    nx, ny, _ = sim.lattice.size

    # Add particles at z=1 layer
    # Position (1,1,1) → idx = 1 + 1*5 + 1*5*5 = 31
    sim.lattice.deposit_atom(1 + 1 * nx + 1 * nx * ny, SpeciesType.TI)
    # Position (2,2,1) → idx = 2 + 2*5 + 1*5*5 = 37
    sim.lattice.deposit_atom(2 + 2 * nx + 1 * nx * ny, SpeciesType.O)
    # Position (3,3,1) → idx = 3 + 3*5 + 1*5*5 = 43
    sim.lattice.deposit_atom(3 + 3 * nx + 1 * nx * ny, SpeciesType.TI)
    # Position (1,2,1) → idx = 1 + 2*5 + 1*5*5 = 36
    sim.lattice.deposit_atom(1 + 2 * nx + 1 * nx * ny, SpeciesType.O)

    # Create agents
    agents = create_agents_from_lattice(sim.lattice)

    print(f"\nTotal agents: {len(agents)}")

    # Collect batch of observations
    obs_batch = []
    for agent in agents:
        obs = agent.observe()
        vec = obs.to_vector()
        obs_batch.append(vec)

    obs_batch = np.stack(obs_batch)

    print(f"Observation batch shape: {obs_batch.shape}")
    print(f"Expected: ({len(agents)}, 58)")

    assert obs_batch.shape == (
        len(agents),
        58,
    ), f"Expected shape ({len(agents)}, 58), got {obs_batch.shape}"
    assert obs_batch.dtype == np.float32, "Batch should be float32"

    print("\n✓ Batch observations test passed")


def main():
    """Run all Phase 1 tests."""
    print("\n" + "=" * 60)
    print("PHASE 1: PARTICLE AGENT OBSERVATIONS")
    print("Testing SwarmThinkers Agent Architecture")
    print("=" * 60)

    try:
        # Run tests
        sim, agents = test_agent_creation()
        test_local_observations(sim, agents)
        test_observation_encoding(sim, agents)
        test_valid_actions()
        test_batch_observations()

        # Summary
        print("\n" + "=" * 60)
        print("✓ ALL PHASE 1 TESTS PASSED")
        print("=" * 60)
        print("\nParticle agents are working correctly:")
        print("  ✓ Agent creation from lattice")
        print("  ✓ Local observations (1st/2nd neighbors)")
        print("  ✓ Observation vector encoding (58 dims)")
        print("  ✓ Valid actions per species")
        print("  ✓ Batch observation collection")
        print("\nReady for Phase 2: Shared Policy Network")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
