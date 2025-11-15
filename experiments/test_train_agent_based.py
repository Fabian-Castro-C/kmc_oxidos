"""
Quick test for agent-based training pipeline.
Tests environment creation, vectorization, and policy initialization.
"""

from pathlib import Path
import tempfile
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from experiments.train_agent_based import create_agent_env


def test_env_creation():
    """Test creating agent-based environment."""
    print("Testing environment creation...")
    
    config = {
        "lattice_size": [5, 5, 5],
        "temperature": 600.0,
        "deposition_rate": 1.0,
        "max_steps": 10,
        "max_agents": 32,
        "use_reweighting": True,
    }
    
    env = create_agent_env(config, seed=42)
    print(f"  - Environment created: {env}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"  - Observation keys: {obs.keys()}")
    print(f"  - Agent obs shape: {obs['agents'].shape}")
    print(f"  - Global obs shape: {obs['global'].shape}")
    print(f"  - Mask shape: {obs['mask'].shape}")
    
    env.close()
    print("  ✓ Environment creation PASSED")


def test_vectorized_env():
    """Test vectorized environment creation."""
    print("\nTesting vectorized environments...")
    
    config = {
        "lattice_size": [5, 5, 5],
        "temperature": 600.0,
        "deposition_rate": 1.0,
        "max_steps": 10,
        "max_agents": 32,
        "use_reweighting": True,
    }
    
    n_envs = 2
    vec_env = make_vec_env(
        lambda: create_agent_env(config, seed=0),
        n_envs=n_envs,
    )
    
    print(f"  - Created {n_envs} parallel environments")
    print(f"  - Observation space: {vec_env.observation_space}")
    print(f"  - Action space: {vec_env.action_space}")
    
    # Test reset
    obs = vec_env.reset()
    print(f"  - Batch observation keys: {obs.keys()}")
    print(f"  - Batch agent obs shape: {obs['agents'].shape}")
    print(f"  - Batch global obs shape: {obs['global'].shape}")
    
    # Test step
    actions = [vec_env.action_space.sample() for _ in range(n_envs)]
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"  - Step rewards: {rewards}")
    
    vec_env.close()
    print("  ✓ Vectorized environments PASSED")


def test_policy_initialization():
    """Test PPO policy initialization with Dict observation space."""
    print("\nTesting policy initialization...")
    
    config = {
        "lattice_size": [5, 5, 5],
        "temperature": 600.0,
        "deposition_rate": 1.0,
        "max_steps": 10,
        "max_agents": 32,
        "use_reweighting": True,
    }
    
    vec_env = make_vec_env(
        lambda: create_agent_env(config, seed=0),
        n_envs=2,
    )
    
    # Initialize PPO with MultiInputPolicy for Dict obs
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        verbose=0,
    )
    
    print(f"  - Policy type: {type(model.policy)}")
    print(f"  - Policy architecture: {model.policy}")
    
    # Test predict
    obs = vec_env.reset()
    action, _states = model.predict(obs, deterministic=True)
    print(f"  - Sample action: {action}")
    
    vec_env.close()
    print("  ✓ Policy initialization PASSED")


def test_short_training():
    """Test a very short training run."""
    print("\nTesting short training run...")
    
    config = {
        "lattice_size": [5, 5, 5],
        "temperature": 600.0,
        "deposition_rate": 1.0,
        "max_steps": 10,
        "max_agents": 32,
        "use_reweighting": True,
    }
    
    vec_env = make_vec_env(
        lambda: create_agent_env(config, seed=0),
        n_envs=2,
    )
    
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=64,  # Very small for testing
        batch_size=32,
        verbose=0,
    )
    
    # Train for just 128 steps
    print("  - Starting training (128 steps)...")
    model.learn(total_timesteps=128, progress_bar=False)
    print("  - Training completed")
    
    # Test saving
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.zip"
        model.save(model_path)
        print(f"  - Model saved to {model_path}")
        
        # Test loading
        loaded_model = PPO.load(model_path, env=vec_env)
        print("  - Model loaded successfully")
        
        # Test prediction with loaded model
        obs = vec_env.reset()
        action, _states = loaded_model.predict(obs)
        print(f"  - Loaded model prediction: {action}")
    
    vec_env.close()
    print("  ✓ Short training PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT-BASED TRAINING PIPELINE TESTS")
    print("=" * 60)
    
    try:
        test_env_creation()
        test_vectorized_env()
        test_policy_initialization()
        test_short_training()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓✓✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
