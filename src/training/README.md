# Training Module - SwarmThinkers Policy Optimization

This module implements the complete training pipeline for SwarmThinkers policies using **Stable-Baselines3** and **Gymnasium**.

## 🎯 Purpose

Train neural network policies to propose events in KMC simulations that:
1. Match physical rates (π(a) ≈ Γ(a)/Z) for sampling efficiency
2. Optimize morphology (minimize roughness, maximize coverage)
3. Maintain high ESS (>0.5) throughout training

## 📦 Installation

All dependencies (including training tools) are installed by default:

```bash
# Standard installation
uv pip install -e .

# Or sync environment
uv sync
```

Dependencies included:
- `stable-baselines3[extra]`: PPO, A2C, SAC algorithms
- `gymnasium`: Environment API
- `tensorboard`: Training monitoring
- `torch`: Neural networks
- `numpy`, `scipy`, `matplotlib`: Scientific computing

## 🏗️ Module Structure

### `gym_environment.py`
Gymnasium environment wrapper for TiO2 thin film growth.

**Key Features:**
- Wraps `KMCSimulator` + `SwarmEngine` into Gym API
- Observation: Height profile + local features
- Action: Select proposal from SwarmEngine's n_swarm candidates
- Reward: Multi-objective (roughness, coverage, ESS)

**Example:**
```python
from src.training import TiO2GrowthEnv
from stable_baselines3.common.env_checker import check_env

env = TiO2GrowthEnv(lattice_size=(20, 20, 10))
check_env(env)  # Validate Gymnasium API compliance
```

### `reward_functions.py`
Modular reward engineering with tunable weights.

**Components:**
- Roughness penalty: -w_roughness * σ(heights)
- Coverage reward: +w_coverage * mean(heights)
- Stoichiometry penalty: -w_ratio * |O/Ti - 2.0|
- ESS penalty: -w_ess * max(0, 0.5 - ESS_ratio)

### `callbacks.py`
Custom SB3 callbacks for monitoring training.

**Callbacks:**
- `ESSMonitorCallback`: Track ESS during training, early stop if ESS < threshold
- `MorphologyLoggerCallback`: Log roughness, coverage to TensorBoard
- `CheckpointCallback`: Save models periodically (built-in SB3)

### `curriculum.py`
Curriculum learning scheduler for progressive lattice scaling.

**Stages:**
1. Small (8×8×5): Learn basic dynamics
2. Medium (20×20×10): Transfer + refine
3. Large (50×50×20): Production scale

### `hyperparameters.py`
Dataclasses for training configuration.

## 🚀 Usage

### Quick Start

```python
from src.training import TiO2GrowthEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environments (8 parallel)
env = make_vec_env(
    lambda: TiO2GrowthEnv(lattice_size=(8, 8, 5)),
    n_envs=8,
    vec_env_cls=SubprocVecEnv
)

# Initialize PPO
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train
model.learn(total_timesteps=100_000)
model.save("stage1_final")
```

### Full Training Pipeline

```bash
# Stage 1: Small lattice (8×8×5)
python experiments/train_policy.py --config experiments/configs/stage1_small.yaml

# Stage 2: Medium lattice (20×20×10) with transfer learning
python experiments/train_policy.py --config experiments/configs/stage2_medium.yaml \
    --pretrained ./results/stage1_final.zip

# Stage 3: Large lattice (50×50×20)
python experiments/train_policy.py --config experiments/configs/stage3_large.yaml \
    --pretrained ./results/stage2_final.zip
```

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir ./experiments/results/training_runs/tensorboard/
```

**Metrics Logged:**
- `rollout/ep_rew_mean`: Episode reward
- `train/policy_loss`: PPO policy loss
- `train/value_loss`: Value function loss
- `custom/ess_ratio`: Effective sample size
- `custom/roughness`: Surface roughness
- `custom/coverage`: Surface coverage

### Checkpoints
Models are saved every 10k steps to:
```
experiments/results/training_runs/checkpoints/stage{N}/
├── rl_model_10000_steps.zip
├── rl_model_20000_steps.zip
└── ...
```

## 🎛️ Configuration

Example YAML config (`experiments/configs/stage1_small.yaml`):

```yaml
# Environment
lattice_size: [8, 8, 5]
temperature: 180.0
deposition_rate: 0.1
max_steps: 1000

# Training
n_envs: 8
total_timesteps: 100000
checkpoint_freq: 10000

# PPO Hyperparameters
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01

# Reward Weights
w_roughness: 1.0
w_coverage: 0.5
w_ratio: 0.2
w_ess: 0.3
```

## 🧪 Testing

```python
# Validate environment
from stable_baselines3.common.env_checker import check_env
from src.training import TiO2GrowthEnv

env = TiO2GrowthEnv(lattice_size=(8, 8, 5))
check_env(env, warn=True)

# Test reward function
from src.training import compute_reward, RewardConfig

config = RewardConfig(w_roughness=1.0, w_coverage=0.5)
reward, info = compute_reward(simulator, config)
print(f"Reward: {reward}, Roughness: {info['roughness']}")
```

## 📈 Expected Results

After curriculum training (Stages 1-3):

**Stage 1 (100k steps)**:
- ESS ratio: ~0.7-0.8
- Converges to basic event proposal strategy

**Stage 2 (500k steps)**:
- ESS ratio: ~0.8-0.9
- Learns morphology optimization

**Stage 3 (1M steps)**:
- ESS ratio: >0.9
- Production-ready policies

## 🔬 Research Integration

For production simulations (paper results):

```python
# Load trained policy
from stable_baselines3 import PPO
model = PPO.load("./results/stage3_final.zip")

# Run production simulation (200×200×50)
from src.training import TiO2GrowthEnv
env = TiO2GrowthEnv(lattice_size=(200, 200, 50), max_steps=100000)

obs, _ = env.reset()
for _ in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Extract results
final_roughness = info['roughness']
final_coverage = info['coverage']
```

## 📝 Notes

- **Curriculum is essential**: Don't skip stages, policies won't converge on large lattices directly
- **ESS monitoring is critical**: If ESS drops below 0.3, reduce learning rate or increase entropy
- **Vectorized envs required**: Single env is too slow for meaningful training
- **GPU recommended**: Stage 3 benefits from GPU (policy forward pass)
- **Checkpoint often**: Training can take days, save frequently

## 🆘 Troubleshooting

**ESS drops during training:**
- Increase `ent_coef` (0.01 → 0.05) for more exploration
- Reduce `learning_rate` (3e-4 → 1e-4)
- Add `--ess-threshold 0.3` to early stop

**Rewards not improving:**
- Tune reward weights (`w_roughness`, `w_coverage`)
- Check observation normalization
- Verify environment with `check_env()`

**Out of memory:**
- Reduce `n_envs` (8 → 4)
- Reduce `n_steps` (2048 → 1024)
- Use smaller lattice for current stage

---

**Last Updated**: 2025-11-13  
**Status**: Structure created, ready for implementation  
**Next**: Implement `gym_environment.py`
