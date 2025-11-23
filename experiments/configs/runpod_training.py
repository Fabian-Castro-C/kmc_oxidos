"""
RunPod GPU Training Configuration for TiO2 Growth RL

Optimized hyperparameters for long training runs on high-end GPUs (RTX 4090, A100, etc.)
This configuration is designed for production-quality training with proper exploration,
stability, and convergence monitoring.

Usage:
    python experiments/train_scalable_agent.py --config experiments/configs/runpod_training.py
"""

from datetime import datetime
from pathlib import Path

# ============================================================================
# EXPERIMENT IDENTIFICATION
# ============================================================================
RUN_NAME = f"runpod_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
PROJECT_NAME = "TiO2_SwarmThinking_FineTuning"
DESCRIPTION = "Fine-tuning with reduced flux and lower LR for kinetic regime optimization"

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
ENV_CONFIG = {
    # Lattice dimensions - AGGRESSIVE for A100 GPU
    # Thin film geometry: large substrate (x,y), sufficient height (z) for growth
    # A100-40GB: (60, 60, 120) → 432,000 sites
    # A100-80GB: (80, 80, 150) → 960,000 sites (recommended)
    "lattice_size": (20, 20, 50),  # Standard size for GPU training
    # Physical parameters (MUST match between training and inference)
    # LOWERED TEMPERATURE FOR TRAINING VALIDATION:
    # At 600K, diffusion is 10^5x faster than deposition, so 2048 steps = 1 atom moving.
    # At 350K, diffusion is comparable to deposition, allowing film growth in 2048 steps.
    "temperature": 600.0,  # Kelvin - Restored to paper value for correct physics
    "deposition_flux_ti": 0.5,  # ML/s - Low flux for diffusion-dominated regime
    "deposition_flux_o": 1.0,  # ML/s - Low flux for diffusion-dominated regime
    # Episode configuration
    "max_steps_per_episode": 5000,  # Longer episodes for larger system equilibration
    # Random seed for reproducibility
    "seed": 4242,
}

# ============================================================================
# FLUX SCHEDULE (Progressive Curriculum)
# ============================================================================
FLUX_SCHEDULE_CONFIG = {
    "enable_flux_schedule": False,
    # Progressive flux reduction for balanced growth + kinetics
    # Flux values in ML/s (Monolayers per second) - physical units
    # With Poisson: P(deposit) = 1 - exp(-λ), where λ = flux_total * n_sites * 0.01
    # For lattice 10×10 (100 sites): λ = flux_total * 100 * 0.01 = flux_total
    #
    # NEW STRATEGY: Much lower flux to give agent control over surface kinetics
    # Target: ~20-30% deposition probability, ~70-80% agent actions
    #
    # Phase 1 (Updates 0-99): Moderate growth regime
    #   λ = 0.3 → P ≈ 26% → Agent gets 74% control for kinetics
    #   ~1-2 depositions every 5-7 steps
    #
    # Phase 2 (Updates 100-199): Balanced regime
    #   λ = 0.2 → P ≈ 18% → Agent gets 82% control
    #   ~1 deposition every 5-6 steps
    #
    # Phase 3 (Updates 200-399): Low flux - Kinetics-dominated
    #   λ = 0.1 → P ≈ 9.5% → Agent gets 90% control
    #   ~1 deposition every 10 steps
    #
    # Phase 4 (Updates 400+): Ultra-low flux - Pure refinement
    #   λ = 0.05 → P ≈ 4.9% → Agent gets 95% control
    #   ~1 deposition every 20 steps
    #
    "flux_stages": [
        {"at_update": 0, "flux_ti": 0.5, "flux_o": 1.0},  # Low flux for ideal training
    ],
}

# ============================================================================
# REWARD SHAPING (Conservative, Physics-Motivated)
# ============================================================================
REWARD_SHAPING_CONFIG = {
    "enable_reward_shaping": True,
    # A. Exploration Bonus for DIFFUSE/DESORB actions
    # Motivation: Compensate thermodynamic penalty to allow exploration
    # PERMANENTLY DISABLED: Causes catastrophic policy collapse (actor learns to desorb everything)
    "exploration_bonus_enabled": False,
    "exploration_bonus_amount": 0.0,  # KEEP AT 0.0 - Do not enable
    "exploration_bonus_threshold": 0.0,  # Not used when disabled
    # B. Deposition Logit Scaling
    # Motivation: Prevent deposition from dominating action selection at high flux
    # Without scaling: ln(10.0 * 100 sites) ≈ 6.9 >> diffusion logits (~[-5, 5])
    # With scaling (e.g., 2.5): 6.9 / 2.5 ≈ 2.8 (more balanced competition)
    "deposition_logit_scale": 2.5,  # Divide deposition logit by this factor
    # Structural metrics logging (NO reward impact, just monitoring)
    "log_structural_metrics": True,
    "structural_metrics": [
        "roughness",  # Surface roughness (RMS height variation)
        "avg_coordination",  # Average coordination number
        "ti_o_ratio",  # Ti:O stoichiometric ratio
        "bond_density",  # Total bonds per atom
        "ti_o_bonds_fraction",  # Fraction of Ti-O bonds vs all bonds
    ],
}

# ============================================================================
# PPO HYPERPARAMETERS
# ============================================================================
PPO_CONFIG = {
    # Learning rate schedule
    "learning_rate": 3e-4,  # Standard PPO rate for fresh training
    "lr_schedule": "constant",  # Options: "constant", "linear_decay", "cosine"
    "lr_end_factor": 0.1,  # Final LR = initial_lr * lr_end_factor (if using decay)
    # Discount and advantage estimation
    "gamma": 0.999,  # Discount factor - INCREASED to 0.999 to value long-term rewards (finding bonds)
    "gae_lambda": 0.95,  # GAE lambda - balances bias/variance
    # PPO-specific
    "clip_coef": 0.2,  # PPO clipping coefficient - standard value
    "target_kl": 0.015,  # Early stopping if KL divergence exceeds this (None to disable)
    # Loss coefficients
    "vf_coef": 0.5,  # Value function loss coefficient
    "ent_coef": 0.05,  # Entropy bonus - INCREASED to 0.05 to encourage exploration (was 0.01)
    "max_grad_norm": 0.5,  # Gradient clipping for stability
    # Optimization
    "adam_eps": 1e-5,  # Adam epsilon for numerical stability
    "update_epochs": 4,  # Number of epochs per PPO update (optimized for large batch size)
    "minibatch_size": 64,  # Small minibatch to fit Swarm agents in VRAM (64 * 8000 = 512k agents)
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    # Total training budget
    "total_timesteps": 100_000_000,  # 100M steps for deep convergence on RTX 5090
    # Rollout collection
    "num_steps": 256,  # Short rollouts for frequent updates
    "num_envs": 256,  # Reduced to 256 to fit in VRAM (Batch = 256 * 256 = 65k)
    # Note: Swarm architecture processes EVERY atom as an agent.
    # 256 envs * 8000 sites = 2,048,000 agents per forward pass.
    # This consumes ~5-6GB VRAM for activations.
    # --- NEW ---
    # Path to a checkpoint to resume training from. Set to None to train from scratch.
    # Example: "experiments/results/train/runpod_XXXXXXXXXX/models/best_model.pt"
    "resume_from_checkpoint": None,  # Train from scratch
    # --- END NEW ---
    # Checkpointing
    "checkpoint_frequency": 50,  # Save checkpoint every N updates
    "keep_checkpoints": 5,  # Keep only last N checkpoints to save space
    # Evaluation
    "eval_frequency": 25,  # Run evaluation every N updates
    "eval_episodes": 5,  # Number of episodes for evaluation
    # Logging
    "log_frequency": 5,  # Log metrics every N updates
    "tensorboard_enabled": True,
    "save_trajectories": True,  # Save full trajectories for analysis
}

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================
NETWORK_CONFIG = {
    # Actor (decentralized policy)
    # Following SwarmThinkers paper: 5-layer MLP with constant width 256
    "actor_hidden_dims": [256, 256, 256, 256, 256],  # Paper-validated architecture
    "actor_activation": "relu",  # Paper uses ReLU, not tanh
    # Critic (centralized value function)
    # Paper uses same architecture for both actor and critic
    "critic_hidden_dims": [256, 256, 256, 256, 256],  # Matching paper architecture
    "critic_activation": "relu",  # Consistent with paper
    # Initialization
    "orthogonal_init": True,  # Orthogonal weight initialization (helps training)
    "init_scale_actor": 0.01,  # Small initial policy for exploration
    "init_scale_critic": 1.0,
}

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================
COMPUTE_CONFIG = {
    # GPU settings
    "device": "cuda",  # Will auto-detect if CUDA is available
    "cuda_deterministic": False,  # Set True for reproducibility (slower)
    "torch_threads": 4,  # CPU threads for DataLoader
    # Mixed precision training (faster on modern GPUs)
    "use_amp": True,  # Automatic Mixed Precision
    "amp_dtype": "float16",  # Options: "float16", "bfloat16" (if supported)
    # Memory optimization
    "gradient_accumulation_steps": 1,  # Accumulate gradients over N steps
    "pin_memory": True,  # Pin memory for faster GPU transfer
}

# ============================================================================
# MONITORING AND DEBUGGING
# ============================================================================
MONITORING_CONFIG = {
    # Metrics to track
    "track_kl_divergence": True,
    "track_explained_variance": True,
    "track_action_distribution": True,
    "track_episode_length": True,
    # Debugging
    "debug_mode": False,  # Enable verbose logging
    "check_gradients": False,  # Check for NaN/Inf gradients (slower)
    "profile_performance": False,  # Enable PyTorch profiler (slower)
    # Anomaly detection
    "detect_anomalies": False,  # PyTorch anomaly detection (very slow, only for debugging)
}

# ============================================================================
# OUTPUT PATHS
# ============================================================================
PATHS_CONFIG = {
    "results_dir": Path("experiments/results/train"),
    "checkpoints_dir": Path("experiments/results/train") / RUN_NAME / "checkpoints",
    "logs_dir": Path("experiments/results/train") / RUN_NAME / "logs",
    "tensorboard_dir": Path("experiments/results/train") / RUN_NAME / "tensorboard",
    "trajectories_dir": Path("experiments/results/train") / RUN_NAME / "trajectories",
}

# ============================================================================
# EARLY STOPPING AND CONVERGENCE
# ============================================================================
CONVERGENCE_CONFIG = {
    # Early stopping criteria
    "enable_early_stopping": True,
    "patience": 100,  # Fine-tuning: more patience for convergence
    "min_delta": 0.01,  # Minimum improvement to reset patience
    # Success criteria (optional)
    "target_mean_reward": 5.0,  # Stop if mean reward exceeds this
    "target_roughness": 0.3,  # Stop if roughness is below this (nm)
    # Convergence detection
    "check_convergence_window": 100,  # Check last N updates
    "convergence_std_threshold": 0.05,  # Std of mean rewards must be below this
}

# ============================================================================
# CURRICULUM LEARNING (Optional)
# ============================================================================
CURRICULUM_CONFIG = {
    "enable_curriculum": False,  # Enable curriculum learning
    # Start with easier task
    "initial_lattice_size": (10, 10, 15),
    "initial_max_steps": 500,
    # Gradually increase difficulty
    "curriculum_stages": [
        {"at_update": 0, "lattice_size": (10, 10, 15), "max_steps": 500},
        {"at_update": 200, "lattice_size": (15, 15, 20), "max_steps": 1000},
        {"at_update": 500, "lattice_size": (20, 20, 30), "max_steps": 2000},
    ],
}

# ============================================================================
# CONSOLIDATED CONFIG (for easy import)
# ============================================================================
CONFIG = {
    "run_name": RUN_NAME,
    "project_name": PROJECT_NAME,
    "description": DESCRIPTION,
    **ENV_CONFIG,
    **PPO_CONFIG,
    **TRAINING_CONFIG,
    **NETWORK_CONFIG,
    **COMPUTE_CONFIG,
    **MONITORING_CONFIG,
    **CONVERGENCE_CONFIG,
    **CURRICULUM_CONFIG,
    **FLUX_SCHEDULE_CONFIG,
    **REWARD_SHAPING_CONFIG,
}

# Add paths
CONFIG["paths"] = PATHS_CONFIG

# ============================================================================
# ESTIMATED RESOURCE USAGE
# ============================================================================
"""
Estimated resource usage for this configuration:

GPU Memory (VRAM):
- Model parameters: ~5-10 MB (Actor + Critic)
- Batch storage (2048 steps): ~500 MB
- Forward/backward passes: ~2-3 GB
- TOTAL: ~4-5 GB (well within 24GB for RTX 4090 / 40GB for A100)

Training Time Estimates:
- RTX 4090: ~30-40 steps/sec → ~35-45 hours for 5M steps
- A100 (80GB): ~50-60 steps/sec → ~23-28 hours for 5M steps
- RTX 3090: ~20-25 steps/sec → ~55-70 hours for 5M steps

Disk Space:
- Checkpoints: ~100 MB each × 5 kept = ~500 MB
- TensorBoard logs: ~1-2 GB
- Trajectories: ~5-10 GB (if saved)
- TOTAL: ~7-13 GB

Recommended RunPod Instance:
- GPU: RTX 4090 (24GB) or A100 (40/80GB)
- RAM: 32GB+
- Storage: 50GB+ (SSD recommended)
- vCPU: 8-16 cores
"""

if __name__ == "__main__":
    # Print configuration summary
    print("=" * 80)
    print("RunPod Training Configuration Summary")
    print("=" * 80)
    print(f"Run Name: {RUN_NAME}")
    print(f"Project: {PROJECT_NAME}")
    print()
    print("Environment:")
    print(f"  Lattice Size: {ENV_CONFIG['lattice_size']}")
    print(f"  Max Steps/Episode: {ENV_CONFIG['max_steps_per_episode']}")
    print(f"  Temperature: {ENV_CONFIG['temperature']} K")
    print()
    print("Training:")
    print(f"  Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  Rollout Length: {TRAINING_CONFIG['num_steps']}")
    print(f"  Updates: {TRAINING_CONFIG['total_timesteps'] // TRAINING_CONFIG['num_steps']:,}")
    print()
    print("Network:")
    print(f"  Actor: {NETWORK_CONFIG['actor_hidden_dims']}")
    print(f"  Critic: {NETWORK_CONFIG['critic_hidden_dims']}")
    print()
    print("Compute:")
    print(f"  Device: {COMPUTE_CONFIG['device']}")
    print(f"  Mixed Precision: {COMPUTE_CONFIG['use_amp']}")
    print("=" * 80)
