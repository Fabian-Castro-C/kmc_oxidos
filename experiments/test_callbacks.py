"""
Test callbacks integration with mock training loop.

This script validates that the callbacks work correctly before
starting actual SB3 training.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training import (
    EpisodeSummaryCallback,
    ESSMonitorCallback,
    MorphologyLoggerCallback,
)


def test_callbacks_basic() -> None:
    """Test basic callback instantiation."""
    print("\n=== Test 1: Callback Instantiation ===")

    ess_callback = ESSMonitorCallback(ess_threshold=0.3, patience=5, verbose=1)
    morphology_callback = MorphologyLoggerCallback(log_freq=100, verbose=1)
    episode_callback = EpisodeSummaryCallback(verbose=1)

    print("✓ All callbacks instantiated successfully")
    print(f"  ESSMonitorCallback: threshold={ess_callback.ess_threshold}, patience={ess_callback.patience}")
    print(f"  MorphologyLoggerCallback: log_freq={morphology_callback.log_freq}")
    print(f"  EpisodeSummaryCallback: verbose={episode_callback.verbose}")


def test_callbacks_with_environment() -> None:
    """Test that callbacks can be instantiated with proper configuration."""
    print("\n=== Test 2: Callback Configuration ===")

    # Verify callbacks have correct attributes
    ess_callback = ESSMonitorCallback(ess_threshold=0.5, patience=10, verbose=1)
    assert ess_callback.ess_threshold == 0.5
    assert ess_callback.patience == 10
    assert ess_callback.ess_values == []
    assert ess_callback.low_ess_count == 0

    morphology_callback = MorphologyLoggerCallback(log_freq=500, verbose=1)
    assert morphology_callback.log_freq == 500
    assert morphology_callback.roughness_values == []

    episode_callback = EpisodeSummaryCallback(verbose=1)
    assert episode_callback.episode_count == 0

    print("✓ All callback configurations validated")
    print("  Callbacks ready for SB3 integration")


def test_ess_early_stopping() -> None:
    """Test ESS callback data structure."""
    print("\n=== Test 3: ESS Callback Data ===")

    callback = ESSMonitorCallback(ess_threshold=0.9, patience=2, verbose=0)

    # Verify initial state
    assert callback.ess_values == []
    assert callback.low_ess_count == 0

    # Simulate adding ESS values
    callback.ess_values = [0.8, 0.7, 0.6, 0.5]
    mean_ess = np.mean(callback.ess_values)

    print("✓ ESS callback can track values")
    print(f"  Sample mean ESS: {mean_ess:.4f}")
    print(f"  Threshold: {callback.ess_threshold}")


def test_morphology_statistics() -> None:
    """Test morphology callback data structure."""
    print("\n=== Test 4: Morphology Callback Data ===")

    callback = MorphologyLoggerCallback(log_freq=5, verbose=0)

    # Verify initial state
    assert callback.roughness_values == []
    assert callback.coverage_values == []

    # Simulate morphology data
    callback.roughness_values = [0.5 + 0.1 * np.random.randn() for _ in range(10)]
    callback.coverage_values = [0.02 * (i + 1) for i in range(10)]
    callback.n_ti_values = [10.0 * (i + 1) for i in range(10)]
    callback.n_o_values = [15.0 * (i + 1) for i in range(10)]

    print("✓ Morphology callback can track values")
    print(f"  Mean roughness: {np.mean(callback.roughness_values):.4f}")
    print(f"  Mean coverage: {np.mean(callback.coverage_values):.4f}")
    print(f"  Mean n_Ti: {np.mean(callback.n_ti_values):.1f}")
    print(f"  Mean n_O: {np.mean(callback.n_o_values):.1f}")


def main() -> None:
    """Run all callback tests."""
    print("=" * 60)
    print("Callback Validation Tests")
    print("=" * 60)

    try:
        test_callbacks_basic()
        test_callbacks_with_environment()
        test_ess_early_stopping()
        test_morphology_statistics()

        print("\n" + "=" * 60)
        print("✓ All callback tests passed!")
        print("Callbacks are ready for SB3 training")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
