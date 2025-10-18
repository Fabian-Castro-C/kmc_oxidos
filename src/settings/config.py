"""
Configuration module using Pydantic Settings.

This module handles all project configuration including environment variables,
logging setup, KMC simulation parameters, and RL training parameters.
"""

import logging
import sys
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    file: str = Field(default="logs/kmc_oxidos.log", description="Log file path")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()


class KMCConfig(BaseSettings):
    """KMC simulation configuration."""

    # Lattice dimensions
    lattice_size_x: int = Field(default=50, description="Lattice size in X direction", gt=0)
    lattice_size_y: int = Field(default=50, description="Lattice size in Y direction", gt=0)
    lattice_size_z: int = Field(default=20, description="Lattice size in Z direction", gt=0)

    # Physical parameters
    temperature: float = Field(default=600.0, description="Temperature in Kelvin", gt=0)
    deposition_rate: float = Field(default=1.0, description="Deposition rate in ML/s", ge=0)

    # Simulation parameters
    simulation_time: float = Field(
        default=1000.0, description="Total simulation time in seconds", gt=0
    )
    max_steps: int = Field(default=1000000, description="Maximum number of KMC steps", gt=0)
    snapshot_interval: int = Field(default=1000, description="Steps between snapshots", gt=0)

    # Boltzmann constant (eV/K)
    k_boltzmann: float = Field(default=8.617333e-5, description="Boltzmann constant in eV/K")


class RLConfig(BaseSettings):
    """Reinforcement Learning configuration."""

    # PPO hyperparameters
    learning_rate: float = Field(default=5e-4, description="Learning rate for optimizer", gt=0)
    batch_size: int = Field(default=256, description="Mini-batch size", gt=0)
    epochs: int = Field(default=10, description="Number of PPO epochs", gt=0)
    gamma: float = Field(default=0.99, description="Discount factor", ge=0, le=1)
    clip_range: float = Field(default=0.2, description="PPO clip range", gt=0)
    n_steps: int = Field(default=2048, description="Steps per environment per update", gt=0)
    total_timesteps: int = Field(default=1000000, description="Total training timesteps", gt=0)

    # Neural network architecture
    policy_hidden_layers: int = Field(default=5, description="Number of hidden layers", gt=0)
    policy_hidden_units: int = Field(default=256, description="Units per hidden layer", gt=0)

    # Training settings
    use_gae: bool = Field(default=True, description="Use Generalized Advantage Estimation")
    gae_lambda: float = Field(default=0.95, description="GAE lambda parameter", ge=0, le=1)
    max_grad_norm: float = Field(default=0.5, description="Maximum gradient norm", gt=0)

    # Entropy and value loss coefficients
    ent_coef: float = Field(default=0.0, description="Entropy coefficient", ge=0)
    vf_coef: float = Field(default=0.5, description="Value function coefficient", ge=0)


class PathConfig(BaseSettings):
    """Path configuration."""

    # Base directories
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    results_dir: Path = Field(default=Path("results"), description="Results directory")
    checkpoints_dir: Path = Field(
        default=Path("checkpoints"), description="Model checkpoints directory"
    )
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    @field_validator("data_dir", "results_dir", "checkpoints_dir", "logs_dir")
    @classmethod
    def create_dir_if_not_exists(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class HardwareConfig(BaseSettings):
    """Hardware and performance configuration."""

    device: Literal["auto", "cpu", "cuda"] = Field(default="auto", description="Compute device")
    n_jobs: int = Field(default=4, description="Number of parallel jobs", gt=0)
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class Settings(BaseSettings):
    """Main settings class combining all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    project_name: str = Field(default="kmc_oxidos", description="Project name")
    environment: Literal["development", "production", "testing"] = Field(
        default="development", description="Environment"
    )

    # Configuration sections
    log: LogConfig = Field(default_factory=LogConfig)
    kmc: KMCConfig = Field(default_factory=KMCConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    def setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.

        Returns:
            Configured logger instance.
        """
        # Create logs directory
        log_path = Path(self.log.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log.level),
            format=self.log.format,
            handlers=[
                logging.FileHandler(self.log.file),
                logging.StreamHandler(sys.stdout),
            ],
        )

        logger = logging.getLogger(self.project_name)
        logger.info(f"Logging initialized at level {self.log.level}")
        logger.info(f"Environment: {self.environment}")

        return logger

    def get_device(self) -> str:
        """
        Get the appropriate compute device.

        Returns:
            Device string ('cpu' or '
            cuda').
        """
        if self.hardware.device == "auto":
            try:
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.hardware.device

    def model_dump_summary(self) -> dict[str, dict]:
        """
        Get a summary of all configuration settings.

        Returns:
            Dictionary containing all settings organized by section.
        """
        return {
            "project": {
                "name": self.project_name,
                "environment": self.environment,
            },
            "kmc": self.kmc.model_dump(),
            "rl": self.rl.model_dump(),
            "paths": {k: str(v) for k, v in self.paths.model_dump().items()},
            "hardware": self.hardware.model_dump(),
            "log": self.log.model_dump(),
        }


# Global settings instance
settings = Settings()
