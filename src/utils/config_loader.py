"""
Configuration Loader
===================

Load and manage YAML configuration files with validation and defaults.

Author: Combat Racing RL Team
Date: 2024-2025
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from omegaconf import OmegaConf, DictConfig
from .logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """
    Professional configuration management system.
    
    Features:
        - Load YAML files with validation
        - Merge multiple configs
        - Environment variable interpolation
        - Type checking
        - Default values
    
    Example:
        >>> config = ConfigLoader()
        >>> game_cfg = config.load("config/game_config.yaml")
        >>> print(game_cfg.display.window_width)
        1280
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            base_dir: Base directory for relative paths (default: project root).
        """
        if base_dir is None:
            # Find project root (directory containing config folder)
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / "config").exists():
                    base_dir = current
                    break
                current = current.parent
            
            if base_dir is None:
                base_dir = Path.cwd()
        
        self.base_dir = Path(base_dir)
        logger.info(f"ConfigLoader initialized with base_dir: {self.base_dir}")
    
    def load(
        self,
        config_path: Union[str, Path],
        defaults: Optional[Dict[str, Any]] = None,
    ) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML file (relative to base_dir or absolute).
            defaults: Default values to use if keys are missing.
        
        Returns:
            OmegaConf DictConfig object with loaded configuration.
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file has invalid YAML syntax.
        
        Example:
            >>> config = loader.load("config/game_config.yaml")
        """
        # Resolve path
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.base_dir / config_path
        
        # Check file exists
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading config from: {config_path}")
        
        # Load YAML
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {config_path}: {e}")
            raise
        
        # Convert to OmegaConf
        config = OmegaConf.create(config_dict)
        
        # Merge with defaults if provided
        if defaults:
            defaults_cfg = OmegaConf.create(defaults)
            config = OmegaConf.merge(defaults_cfg, config)
        
        logger.success(f"Config loaded successfully: {config_path.name}")
        return config
    
    def load_all(
        self,
        config_dir: Union[str, Path] = "config",
    ) -> Dict[str, DictConfig]:
        """
        Load all YAML config files from a directory.
        
        Args:
            config_dir: Directory containing config files.
        
        Returns:
            Dictionary mapping filename (without extension) to config.
        
        Example:
            >>> configs = loader.load_all("config")
            >>> game_cfg = configs["game_config"]
            >>> rl_cfg = configs["rl_config"]
        """
        config_dir = Path(config_dir)
        if not config_dir.is_absolute():
            config_dir = self.base_dir / config_dir
        
        if not config_dir.exists():
            logger.warning(f"Config directory not found: {config_dir}")
            return {}
        
        configs = {}
        for yaml_file in config_dir.glob("*.yaml"):
            name = yaml_file.stem
            try:
                configs[name] = self.load(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
        
        logger.info(f"Loaded {len(configs)} config files from {config_dir}")
        return configs
    
    def merge(self, *configs: DictConfig) -> DictConfig:
        """
        Merge multiple configurations (later configs override earlier ones).
        
        Args:
            *configs: Variable number of configs to merge.
        
        Returns:
            Merged configuration.
        
        Example:
            >>> default_cfg = loader.load("default.yaml")
            >>> custom_cfg = loader.load("custom.yaml")
            >>> merged = loader.merge(default_cfg, custom_cfg)
        """
        if not configs:
            return OmegaConf.create({})
        
        merged = configs[0]
        for config in configs[1:]:
            merged = OmegaConf.merge(merged, config)
        
        return merged
    
    def save(
        self,
        config: DictConfig,
        output_path: Union[str, Path],
    ) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save.
            output_path: Output file path.
        
        Example:
            >>> config.game.fps = 120
            >>> loader.save(config, "config/custom_config.yaml")
        """
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        
        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.success(f"Config saved to: {output_path}")
    
    def to_dict(self, config: DictConfig) -> Dict[str, Any]:
        """
        Convert OmegaConf config to plain Python dictionary.
        
        Args:
            config: OmegaConf configuration.
        
        Returns:
            Plain Python dict.
        """
        return OmegaConf.to_container(config, resolve=True)
    
    @staticmethod
    def get_value(
        config: DictConfig,
        key_path: str,
        default: Any = None,
    ) -> Any:
        """
        Get nested config value using dot notation.
        
        Args:
            config: Configuration object.
            key_path: Dot-separated path (e.g., "display.window_width").
            default: Default value if key doesn't exist.
        
        Returns:
            Config value or default.
        
        Example:
            >>> width = ConfigLoader.get_value(config, "display.window_width", 1280)
        """
        try:
            return OmegaConf.select(config, key_path, default=default)
        except Exception:
            return default
