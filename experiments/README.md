# Experiments - Validation and Simulation Suite

This directory contains validation experiments and production simulation scripts for the kmc_oxidos project.

## 📁 Structure

```
experiments/
├── validate_kmc_basic.py           # ✅ Validation: Core KMC without RL
├── validate_reaction_formation.py  # 🔜 Validation: Ti+2O→TiO2 reactions
├── validate_scaling_exponents.py   # 🔜 Validation: α, β vs literature
├── run_simulations.py              # Production: Systematic parameter sweeps
├── train_policy.py                 # Production: RL policy training
└── results/                        # All experiment outputs
    ├── validate_kmc_basic/
    │   └── {timestamp}/
    │       ├── experiment_config.json
    │       ├── metrics.json
    │       ├── timeseries.json
    │       ├── plot_*.png
    │       └── logs.txt
    └── ...
```

## 🎯 Validation Experiments (Priority Order)

### 1. `validate_kmc_basic.py` - Core KMC Validation ✅

**Purpose:** Validate fundamental KMC physics without RL complexity

**What it tests:**
- ✅ Surface site identification (vacants with support)
- ✅ Different adsorption rates Ti vs O
- ✅ Diffusion and desorption events
- ✅ Temporal evolution (roughness, coverage)
- ✅ Performance metrics (steps/second)

**Run:**
```bash
uv run python -m experiments.validate_kmc_basic
```

**Expected output:**
- `results/validate_kmc_basic/{timestamp}/`
- Metrics JSON with validation status
- 4 plots: roughness, coverage, composition, height profile
- Exit code 0 if all checks pass

**Critical checks:**
- Roughness increases over time
- Coverage is positive and increasing
- Ti and O have different rates
- Performance > 10 steps/s
- Both species present

---

### 2. `validate_reaction_formation.py` - Reaction Implementation 🔜

**Purpose:** Validate Ti + 2O → TiO2 formation events

**What it tests:**
- Formation of oxide when Ti has 2+ O neighbors
- Stoichiometry tracking (Ti:O:TiO2)
- Impact on morphology vs no-reaction case
- Reaction rate correctness

**Status:** To be implemented after reaction events added to simulator

---

### 3. `validate_scaling_exponents.py` - Literature Comparison 🔜

**Purpose:** Compare simulated α, β with experimental/literature values

**What it tests:**
- Roughness exponent α (expected: 0.5 - 0.9)
- Growth exponent β (expected: 0.24 - 0.75)
- Parameter sensitivity (T, deposition rate)
- Match with Ti/TiO2 thin film literature

**Status:** To be implemented after basic KMC validated

---

## 🚀 Production Experiments

### `run_simulations.py`

Systematic parameter sweeps for production runs. Configure in `src/settings/config.py`.

### `train_policy.py`

RL policy training using PPO with SwarmThinkers approach.

---

## 📊 Results Structure

Each experiment creates a timestamped directory:

```
results/{experiment_name}/{YYYY-MM-DD_HH-MM-SS}/
├── experiment_config.json      # Complete configuration
├── metrics.json                # Final metrics + validation status
├── timeseries.json             # Time series data (for post-processing)
├── plot_01_*.png              # Plot 1
├── plot_02_*.png              # Plot 2
└── ...
```

### `metrics.json` Schema

```json
{
  "experiment_name": "validate_kmc_basic",
  "timestamp": "2025-11-11_14-30-25",
  "config": { ... },
  "results": {
    "final_step": 10000,
    "final_time": 125.34,
    "final_roughness": 2.45,
    "composition": { ... },
    "scaling_exponents": { "alpha": 0.73, "beta": 0.42 }
  },
  "performance": {
    "steps_per_second": 79.7,
    "total_duration_s": 125.5
  },
  "validation_status": {
    "roughness_increased": true,
    "coverage_positive": true,
    "rates_ti_neq_o": true,
    "performance_acceptable": true,
    "_overall_pass": true
  }
}
```

---

## 🔍 Workflow

### Running a Validation Experiment

1. **Execute:**
   ```bash
   uv run python -m experiments.validate_kmc_basic
   ```

2. **Check logs:** Real-time console output with progress

3. **Review results:**
   ```bash
   # Go to latest results
   cd experiments/results/validate_kmc_basic/
   ls -lt | head -n 2  # Latest timestamp folder
   
   # Check if passed
   cat {timestamp}/metrics.json | grep "_overall_pass"
   
   # View plots
   open {timestamp}/plot_*.png
   ```

4. **Verify validation:**
   - Exit code 0 = PASS
   - Exit code 1 = FAIL
   - Check `validation_status` in metrics.json

### Adding a New Validation Experiment

1. Copy `validate_kmc_basic.py` as template
2. Modify `ExperimentConfig` for your test
3. Update `_run_validation_checks()` with your assertions
4. Add specific plots in `generate_plots()`
5. Document in this README

---

## 📈 Comparing Results Across Runs

To compare multiple runs:

```python
import json
from pathlib import Path

results_dir = Path("experiments/results/validate_kmc_basic")

for run_dir in sorted(results_dir.iterdir()):
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"{run_dir.name}: Pass={metrics['validation_status']['_overall_pass']}, "
              f"α={metrics['results']['scaling_exponents']['alpha']:.3f}")
```

---

## 🐛 Debugging Failed Experiments

If validation fails:

1. **Check logs:** Console output shows which check failed
2. **Inspect plots:** Visual inspection often reveals issues
3. **Review metrics.json:** See exact values that failed assertions
4. **Run with debugger:** Add breakpoints in experiment script
5. **Compare with previous passing runs:** Use diff on metrics.json

---

## 📝 Notes

- **Notebooks are NOT for validation** - use only for exploration
- **All validation must be reproducible** - fixed seeds, logged configs
- **Plots are auto-generated** - no manual intervention
- **Exit codes matter** - use in CI/CD pipelines
- **Timestamps prevent overwrites** - safe to run multiple times

---

## 🎯 Next Steps

1. ✅ Run `validate_kmc_basic.py` to establish baseline
2. ⚠️ Implement reaction events in simulator
3. 🔜 Create `validate_reaction_formation.py`
4. 🔜 Create `validate_scaling_exponents.py`
5. 🔜 Compare all validations with literature values in `docs/VALIDATION.md`

---

**Last updated:** 2025-11-11  
**Maintainer:** See `docs/agent/GUIDEMENT.md` for agent instructions
