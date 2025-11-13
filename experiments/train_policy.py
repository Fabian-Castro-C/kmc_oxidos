"""
Training script for SwarmThinkers policies with Stable-Baselines3.

This script implements curriculum learning across 3 stages:
- Stage 1: 8×8×5 lattice, 100k steps
- Stage 2: 20×20×10 lattice, 500k steps (transfer learning)
- Stage 3: 50×50×20 lattice, 1M steps (transfer learning)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.settings import settings
from src.training import (
    EpisodeSummaryCallback,
    ESSMonitorCallback,
    MorphologyLoggerCallback,
    TiO2GrowthEnv,
)

# Setup logging using project settings
logger = settings.setup_logging()


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_env(config: dict, seed: int = 0):
    """
    Create a single environment instance.

    Args:
        config: Environment configuration.
        seed: Random seed.

    Returns:
        TiO2GrowthEnv instance.
    """
    return TiO2GrowthEnv(
        lattice_size=tuple(config["lattice_size"]),
        temperature=config["temperature"],
        deposition_rate=config["deposition_rate"],
        max_steps=config["max_steps"],
        n_swarm=config.get("n_swarm", 4),
        n_proposals=config.get("n_proposals", 32),
        reward_weights=config.get("reward_weights"),
        seed=seed,
    )


def train_stage(
    config_path: Path,
    output_dir: Path,
    prev_model_path: Path | None = None,
    stage_name: str = "stage1",
) -> Path:
    """
    Train a single curriculum stage.

    Args:
        config_path: Path to stage configuration YAML.
        output_dir: Output directory for checkpoints and logs.
        prev_model_path: Path to previous stage model for transfer learning.
        stage_name: Name of the current stage.

    Returns:
        Path to the final trained model.
    """
    logger.info("=" * 60)
    logger.info(f"Starting {stage_name}: {config_path}")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration: {config}")

    # Create output directories
    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir = stage_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # Create vectorized environments
    n_envs = config.get("n_envs", 4)
    logger.info(f"Creating {n_envs} parallel environments")

    # Use SubprocVecEnv for true parallelism
    env = make_vec_env(
        lambda: create_env(config["environment"], seed=0),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Create evaluation environment
    eval_env = make_vec_env(
        lambda: create_env(config["environment"], seed=1000),
        n_envs=1,
    )

    # Initialize or load model
    if prev_model_path is not None and prev_model_path.exists():
        logger.info(f"Loading model from {prev_model_path} for transfer learning")
        model = PPO.load(prev_model_path, env=env, tensorboard_log=str(tensorboard_dir))
        logger.info("✓ Model loaded successfully")
    else:
        logger.info("Initializing new PPO model")
        ppo_config = config.get("ppo", {})
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.01),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
        )
        logger.info("✓ Model initialized")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback: save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,  # Adjust for vectorized envs
        save_path=str(checkpoint_dir),
        name_prefix=f"{stage_name}_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback: evaluate every 25k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(stage_dir / "best_model"),
        log_path=str(stage_dir / "eval_logs"),
        eval_freq=25_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # ESS monitor callback
    ess_callback = ESSMonitorCallback(
        ess_threshold=config.get("ess_threshold", 0.3),
        patience=config.get("ess_patience", 10),
        verbose=1,
    )
    callbacks.append(ess_callback)

    # Morphology logger callback
    morphology_callback = MorphologyLoggerCallback(
        log_freq=1000,
        verbose=1,
    )
    callbacks.append(morphology_callback)

    # Episode summary callback
    episode_callback = EpisodeSummaryCallback(verbose=1)
    callbacks.append(episode_callback)

    callback_list = CallbackList(callbacks)

    # Train
    total_timesteps = config.get("total_timesteps", 100_000)
    logger.info(f"Starting training for {total_timesteps:,} timesteps")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
        logger.info("✓ Training completed successfully")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")

    # Save final model
    final_model_path = stage_dir / f"{stage_name}_final.zip"
    model.save(final_model_path)
    logger.info(f"✓ Final model saved to {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()

    logger.info("=" * 60)
    logger.info(f"{stage_name} completed")
    logger.info("=" * 60)

    return final_model_path


def main() -> None:
    """Main training pipeline with curriculum learning."""
    parser = argparse.ArgumentParser(description="Train SwarmThinkers policies with curriculum learning")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/configs"),
        help="Directory containing stage config files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/training_runs"),
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which stage to train (1, 2, 3, or all)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to model checkpoint to resume from",
    )
    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SwarmThinkers Policy Training")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config directory: {args.config_dir}")

    # Define stage configurations
    stages = {
        "1": ("stage1_small.yaml", None),
        "2": ("stage2_medium.yaml", "stage1"),
        "3": ("stage3_large.yaml", "stage2"),
    }

    # Determine which stages to run
    stages_to_run = ["1", "2", "3"] if args.stage == "all" else [args.stage]

    # Track model paths for transfer learning
    model_paths = {}

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        # Determine which stage to start from based on filename
        if "stage1" in str(args.resume_from):
            stages_to_run = ["2", "3"]
            model_paths["stage1"] = args.resume_from
        elif "stage2" in str(args.resume_from):
            stages_to_run = ["3"]
            model_paths["stage2"] = args.resume_from

    # Run curriculum learning
    for stage_num in stages_to_run:
        config_file, prev_stage = stages[stage_num]
        config_path = args.config_dir / config_file

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            logger.error("Please create the configuration file first")
            continue

        # Get previous model path for transfer learning
        prev_model_path = model_paths.get(prev_stage) if prev_stage else None

        # Train stage
        final_model = train_stage(
            config_path=config_path,
            output_dir=output_dir,
            prev_model_path=prev_model_path,
            stage_name=f"stage{stage_num}",
        )

        # Store model path for next stage
        model_paths[f"stage{stage_num}"] = final_model

    logger.info("=" * 60)
    logger.info("✓ All stages completed successfully!")
    logger.info(f"Models saved in: {output_dir}")
    logger.info("=" * 60)

    # Print summary
    logger.info("\nFinal models:")
    for stage_name, model_path in model_paths.items():
        logger.info(f"  {stage_name}: {model_path}")


if __name__ == "__main__":
    main()
