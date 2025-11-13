"""
Quick test to validate the training pipeline works end-to-end.

This runs a minimal training session (500 steps, tiny lattice) to ensure:
- Environment creation works
- Vectorized envs work
- PPO initialization works
- Callbacks work
- Training loop executes
- Model saving works
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_policy import train_stage


def main():
    """Main test function."""
    print("=" * 60)
    print("Training Pipeline Quick Test")
    print("=" * 60)
    print("\nThis will run a tiny training session (500 steps) to validate")
    print("that all components work correctly.\n")

    # Create test output directory
    test_output = Path("experiments/test_training_output")
    test_output.mkdir(exist_ok=True)

    # Run test training
    config_path = Path("experiments/configs/test_tiny.yaml")

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        print("Please ensure test_tiny.yaml exists")
        sys.exit(1)

    print(f"Config: {config_path}")
    print(f"Output: {test_output}\n")

    try:
        final_model = train_stage(
            config_path=config_path,
            output_dir=test_output,
            prev_model_path=None,
            stage_name="test",
        )

        print("\n" + "=" * 60)
        print("✓ Training pipeline test PASSED!")
        print("=" * 60)
        print(f"\nFinal model saved at: {final_model}")
        print("\nYou can now run full training with:")
        print("  uv run python experiments/train_policy.py --stage 1")
        print("\nTo view TensorBoard logs:")
        print(f"  tensorboard --logdir {test_output / 'test' / 'tensorboard'}")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Training pipeline test FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
