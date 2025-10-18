"""
Training script for PPO policy on TiO2 growth environment.

This script trains the SwarmThinkers policy using PPO.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.rl.environment import TiO2GrowthEnv
from src.settings import settings

# Setup logging
logger = settings.setup_logging()


def main() -> None:
    """Main training loop."""
    logger.info("Starting PPO training")

    # Create environment
    env = TiO2GrowthEnv(
        lattice_size=(
            settings.kmc.lattice_size_x,
            settings.kmc.lattice_size_y,
            settings.kmc.lattice_size_z,
        ),
        temperature=settings.kmc.temperature,
        deposition_rate=settings.kmc.deposition_rate,
        max_steps=settings.kmc.max_steps,
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=settings.rl.learning_rate,
        n_steps=settings.rl.n_steps,
        batch_size=settings.rl.batch_size,
        n_epochs=settings.rl.epochs,
        gamma=settings.rl.gamma,
        gae_lambda=settings.rl.gae_lambda,
        clip_range=settings.rl.clip_range,
        ent_coef=settings.rl.ent_coef,
        vf_coef=settings.rl.vf_coef,
        max_grad_norm=settings.rl.max_grad_norm,
        verbose=1,
        device=settings.get_device(),
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(settings.paths.checkpoints_dir),
        name_prefix="ppo_tio2",
    )

    # Train
    logger.info(f"Training for {settings.rl.total_timesteps} timesteps")
    model.learn(
        total_timesteps=settings.rl.total_timesteps,
        callback=checkpoint_callback,
    )

    # Save final model
    model_path = settings.paths.checkpoints_dir / "ppo_tio2_final.zip"
    model.save(model_path)
    logger.info(f"Training completed. Model saved to {model_path}")


if __name__ == "__main__":
    main()
