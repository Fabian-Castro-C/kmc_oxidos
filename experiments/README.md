# Experiments - Validation and Simulation Suite

This directory contains validation experiments and production simulation scripts for the kmc_oxidos project.

## 📁 Structure

```
experiments/
├── validate_kmc_basic.py           # ✅ Validation: Core KMC without RL
├── validate_swarmthinkers.py       # ✅ Validation: SwarmThinkers vs KMC Classic
├── run_simulations.py              # Production: Systematic parameter sweeps
├── train_policy.py                 # 🔜 Production: RL policy training
└── results/                        # All experiment outputs
    ├── validate_kmc_basic/
    ├── validate_swarmthinkers/
    └── ...
```

## 🎯 Validation Experiments

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
uv run python experiments/validate_kmc_basic.py
```

**Expected output:**
- `results/validate_kmc_basic/{timestamp}/`
- Metrics JSON with validation status
- 4 plots: roughness, coverage, composition, height profile
- Exit code 0 if all checks pass

---

### 2. `validate_swarmthinkers.py` - SwarmThinkers Framework Validation ✅

**Purpose:** Validate multi-policy SwarmThinkers implementation against KMC Classic

**What it tests:**
- ✅ Multi-policy framework (diffusion, adsorption, desorption, reaction)
- ✅ Statistical equivalence (KS tests)
- ✅ Effective Sample Size (ESS) > 0.5
- ✅ Importance sampling correctness
- ✅ Performance comparison

**Run:**
```bash
uv run python experiments/validate_swarmthinkers.py --max-steps 1000 --n-trials 20
```

**Arguments:**
- `--lattice-size X Y Z`: Lattice dimensions (default: 20 20 10)
- `--temperature`: Temperature in K (default: 180.0)
- `--max-steps`: Steps per trial (default: 1000)
- `--n-trials`: Number of trials (default: 50)
- `--swarm-size`: Proposals per step (default: 32)
- `--policy-checkpoint`: Path to trained policies (default: None = random)

**Expected output:**
- `results/validate_swarmthinkers/{timestamp}/`
- Metrics JSON with KS test results, ESS metrics
- 4 plots: roughness, coverage, importance weights, trajectories
- Exit code 0 if validation passes

**Critical checks:**
- KS test p-values > 0.05 (distributions statistically identical)
- ESS/N > 0.5 (efficient importance sampling)
- Both methods complete all trials

**Note:** With random policies (no training), distributions will differ. This validates the **technical infrastructure** works correctly. For physics validation, policies must be trained first.

---

## 🚀 Production Experiments

### `run_simulations.py`

Systematic parameter sweeps for production runs. Configure in `src/settings/config.py`.

### `train_policy.py` 🔜

RL policy training using PPO with SwarmThinkers approach. To be implemented in Phase 2.

---

## 📊 Results Structure

Each experiment creates a timestamped directory:

```
results/{experiment_name}/{YYYY-MM-DD_HH-MM-SS}/
├── experiment_config.json      # Complete configuration
├── metrics.json                # Final metrics + validation status
├── plot_*.png                  # Auto-generated plots
```

---

## 🔍 Workflow

### Running a Validation Experiment

1. **Execute:**
   ```bash
   uv run python experiments/validate_swarmthinkers.py --max-steps 100 --n-trials 5
   ```

2. **Check logs:** Real-time console output with progress

3. **Review results:**
   ```bash
   cd experiments/results/validate_swarmthinkers/
   ls -Recurse | Sort-Object LastWriteTime | Select-Object -Last 5
   ```

4. **Verify validation:**
   - Exit code 0 = PASS
   - Exit code 1 = FAIL
   - Check `validation_status` in metrics.json

---

## 📝 Notes

- **Notebooks are NOT for validation** - use only for exploration
- **All validation must be reproducible** - fixed seeds, logged configs
- **Plots are auto-generated** - no manual intervention
- **Exit codes matter** - use in CI/CD pipelines
- **Timestamps prevent overwrites** - safe to run multiple times

---

## 🎯 Implementation Status

- ✅ Phase 1: SwarmThinkers infrastructure (4 policies, multi-event framework)
- 🔜 Phase 2: Policy training (train_policy.py)
- 🔜 Phase 3: Trained policy validation (--policy-checkpoint)

---

**Last updated:** 2025-11-13  
**Maintainer:** See `docs/agent/GUIDEMENT.md` for agent instructions
