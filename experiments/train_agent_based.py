"""
Training script for agent-based SwarmThinkers with Stable-Baselines3.

Simplified training script specifically for the faithful SwarmThinkers implementation
with per-particle agents and global softmax coordination.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from src.settings import settings
from src.training.agent_based_env import AgentBasedTiO2Env

# Setup logging
logger = settings.setup_logging()


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def create_agent_env(config: dict, seed: int = 0) -> AgentBasedTiO2Env:
    """
    Create agent-based environment instance.

    Args:
        config: Environment configuration dict
        seed: Random seed

    Returns:
        AgentBasedTiO2Env instance
    """
    return AgentBasedTiO2Env(
        lattice_size=tuple(config["lattice_size"]),
        temperature=config["temperature"],
        deposition_rate=config["deposition_rate"],
        max_steps=config["max_steps"],
        max_agents=config["max_agents"],
        use_reweighting=config.get("use_reweighting", True),
        reward_weights=config.get("reward_weights"),
        seed=seed,
    )


def train_agent_based(
    config_path: Path,
    output_dir: Path,
    resume_from: Path | None = None,
) -> Path:
    """
    Train agent-based SwarmThinkers model.

    Args:
        config_path: Path to configuration YAML
        output_dir: Output directory for checkpoints and logs
        resume_from: Optional path to checkpoint to resume from

    Returns:
        Path to final trained model
    """
    logger.info("=" * 80)
    logger.info("AGENT-BASED SWARMTHINKERS TRAINING")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(config_path)
    env_config = config["environment"]
    ppo_config = config.get("ppo", {})
    
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Lattice: {env_config['lattice_size']}")
    logger.info(f"Max agents: {env_config['max_agents']}")
    logger.info(f"Reweighting: {env_config.get('use_reweighting', True)}")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # Create vectorized environments
    n_envs = config.get("n_envs", 4)
    logger.info(f"Creating {n_envs} parallel environments")

    env = make_vec_env(
        lambda: create_agent_env(env_config, seed=0),
        n_envs=n_envs,
    )

    # Initialize or load model
    if resume_from is not None and resume_from.exists():
        logger.info(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=env, tensorboard_log=str(tensorboard_dir))
        logger.info("Model loaded successfully")
    else:
        logger.info("Initializing new PPO model")

        # Note: SB3 doesn't natively support Dict obs with MultiInputPolicy
        # For now, we'll use MlpPolicy and flatten the observation
        # TODO: Implement custom feature extractor for Dict obs

        model = PPO(
            "MultiInputPolicy",  # For Dict observation space
            env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 1024),
            batch_size=ppo_config.get("batch_size", 32),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.02),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
        )
        logger.info("Model initialized")
        logger.info(f"Policy architecture: {model.policy}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_freq = config.get("checkpoint_freq", 10_000)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="agent_based_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

    # Train
    total_timesteps = config.get("total_timesteps", 50_000)
    logger.info(f"Starting training for {total_timesteps:,} timesteps")
    logger.info(f"Checkpoint frequency: {checkpoint_freq:,} steps")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            log_interval=10,
        )
        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save final model
    final_model_path = output_dir / "agent_based_final.zip"
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Cleanup
    env.close()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)

    return final_model_path


def main() -> None:
    """Main entry point for agent-based training."""
    parser = argparse.ArgumentParser(
        description="Train agent-based SwarmThinkers policies"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/agent_based_test.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/agent_based_runs"),
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: timestamp)",
    )

    args = parser.parse_args()

    # Create timestamped output directory
    if args.name:
        run_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    output_dir = args.output_dir / run_name

    logger.info(f"Run name: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {args.config}")

    # Train model
    final_model_path = train_agent_based(
        config_path=args.config,
        output_dir=output_dir,
        resume_from=args.resume_from,
    )

    logger.info(f"Training complete! Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
