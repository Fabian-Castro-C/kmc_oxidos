# Training Configuration Files

This directory contains YAML configuration files for the 3-stage curriculum learning strategy.

## Curriculum Learning Strategy

### Stage 1: Small Lattice (8×8×5)
- **File**: `stage1_small.yaml`
- **Total Timesteps**: 100k
- **Parallel Envs**: 8
- **Focus**: Learn basic event proposal patterns
- **Learning Rate**: 3e-4 (standard PPO)
- **Entropy**: 0.01 (moderate exploration)

### Stage 2: Medium Lattice (20×20×10)
- **File**: `stage2_medium.yaml`
- **Total Timesteps**: 500k
- **Parallel Envs**: 8
- **Focus**: Transfer learning to larger systems
- **Learning Rate**: 1e-4 (fine-tuning)
- **Entropy**: 0.005 (reduced exploration)
- **Transfer**: Loads final model from Stage 1

### Stage 3: Large Lattice (50×50×20)
- **File**: `stage3_large.yaml`
- **Total Timesteps**: 1M
- **Parallel Envs**: 4 (larger system, fewer parallel envs)
- **Focus**: Production-scale refinement
- **Learning Rate**: 3e-5 (minimal updates)
- **Entropy**: 0.001 (pure exploitation)
- **Transfer**: Loads final model from Stage 2

## Configuration Structure

Each YAML file contains:

```yaml
environment:
  lattice_size: [nx, ny, nz]
  temperature: float
  deposition_rate: float
  max_steps: int
  n_swarm: int
  n_proposals: int
  reward_weights:
    roughness_weight: float
    coverage_weight: float
    ratio_weight: float
    ess_weight: float

ppo:
  learning_rate: float
  n_steps: int
  batch_size: int
  n_epochs: int
  gamma: float
  gae_lambda: float
  clip_range: float
  ent_coef: float
  vf_coef: float
  max_grad_norm: float

total_timesteps: int
n_envs: int

ess_threshold: float
ess_patience: int
```

## Usage

### Train all stages:
```bash
uv run python experiments/train_policy.py --stage all
```

### Train specific stage:
```bash
uv run python experiments/train_policy.py --stage 1
uv run python experiments/train_policy.py --stage 2
uv run python experiments/train_policy.py --stage 3
```

### Resume from checkpoint:
```bash
uv run python experiments/train_policy.py --resume-from experiments/training_runs/run_XXX/stage1/stage1_final.zip
```

## Hyperparameter Rationale

### Learning Rate Schedule
- **Stage 1 (3e-4)**: Standard PPO learning rate for initial exploration
- **Stage 2 (1e-4)**: Lower LR for fine-tuning, avoid catastrophic forgetting
- **Stage 3 (3e-5)**: Minimal LR for final refinement

### Entropy Coefficient
- **Stage 1 (0.01)**: Moderate exploration to discover diverse strategies
- **Stage 2 (0.005)**: Reduced exploration, start exploiting learned patterns
- **Stage 3 (0.001)**: Minimal exploration, pure exploitation

### Batch Size & Steps
- **n_steps=2048**: Standard rollout buffer size for PPO
- **batch_size=64**: Must divide (n_steps × n_envs) evenly
- **n_epochs=10**: Standard for PPO

### ESS Monitoring
- **Threshold=0.3**: If mean ESS < 0.3, policies are too concentrated
- **Patience=10**: Allow 10 evaluations (10k steps) before early stopping

## Expected Training Time

Approximate wall-clock time on CPU (8 cores):
- **Stage 1**: ~2-4 hours (100k steps, 8 envs, small lattice)
- **Stage 2**: ~12-24 hours (500k steps, 8 envs, medium lattice)
- **Stage 3**: ~48-72 hours (1M steps, 4 envs, large lattice)

**Total**: ~3-4 days for full curriculum

With GPU acceleration (if available), expect 2-3x speedup.

## Monitoring

### TensorBoard
```bash
tensorboard --logdir experiments/training_runs/run_XXX/
```

### Metrics to Watch
- `train/ess_mean`: Should stay > 0.3
- `morphology/roughness_mean`: Should decrease
- `morphology/coverage_mean`: Should increase
- `morphology/roughness_coverage_ratio`: Should decrease
- `episode/reward`: Should increase (become less negative)
- `eval/mean_reward`: Best model metric

## Troubleshooting

### ESS too low
- Policies are generating very concentrated proposals
- Increase `ent_coef` to encourage exploration
- Check reward function weights

### Roughness not decreasing
- May need more training steps
- Adjust `roughness_weight` to penalize more
- Check that coverage is increasing

### Training too slow
- Reduce `n_envs` (less parallelism but faster per step)
- Reduce `max_steps` in environment
- Use smaller lattice for testing

### Out of memory
- Reduce `n_envs`
- Reduce `n_steps` (rollout buffer)
- Reduce `batch_size`
