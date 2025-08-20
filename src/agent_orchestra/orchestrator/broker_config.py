
"""
CallBroker Configuration System for Agent Orchestra.

Provides easy configuration of CallBroker with model limits, retry settings,
and integration helpers for common scenarios.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .call_broker import CallBroker, ModelLimits


@dataclass
class BrokerConfig:
    """Configuration for CallBroker with sensible defaults."""
    
    # Default model limits
    default_rpm: int = 30
    default_rpd: int = 500
    default_max_concurrency: int = 5
    
    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    
    # Model-specific overrides
    model_limits: Dict[str, Dict[str, int]] = field(default_factory=dict) # type: ignore
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> BrokerConfig:
        """Create BrokerConfig from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> BrokerConfig:
        """Load BrokerConfig from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Broker config file not found: {config_path}")
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> BrokerConfig:
        """Create BrokerConfig from environment variables."""
        config = cls()
        
        # Override defaults from environment
        if rpm := os.getenv("AGENT_ORCHESTRA_DEFAULT_RPM"):
            config.default_rpm = int(rpm)
        if rpd := os.getenv("AGENT_ORCHESTRA_DEFAULT_RPD"):
            config.default_rpd = int(rpd)
        if concurrency := os.getenv("AGENT_ORCHESTRA_DEFAULT_CONCURRENCY"):
            config.default_max_concurrency = int(concurrency)
        
        # Retry settings
        if retries := os.getenv("AGENT_ORCHESTRA_MAX_RETRIES"):
            config.max_retries = int(retries)
        if delay := os.getenv("AGENT_ORCHESTRA_BASE_DELAY"):
            config.base_delay = float(delay)
        if max_delay := os.getenv("AGENT_ORCHESTRA_MAX_DELAY"):
            config.max_delay = float(max_delay)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert BrokerConfig to dictionary."""
        return asdict(self)
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save BrokerConfig to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def create_broker(self) -> CallBroker:
        """Create CallBroker instance from this configuration."""
        # Convert model limits to ModelLimits objects
        model_limits_objects = {}
        for model, limits in self.model_limits.items():
            model_limits_objects[model] = ModelLimits(
                rpm=limits.get("rpm", self.default_rpm),
                rpd=limits.get("rpd", self.default_rpd),
                max_concurrency=limits.get("max_concurrency", self.default_max_concurrency)
            )
        
        # Default limits for unknown models
        default_limits = ModelLimits(
            rpm=self.default_rpm,
            rpd=self.default_rpd,
            max_concurrency=self.default_max_concurrency
        )
        
        return CallBroker(model_limits_objects, default_limits) # type: ignore


# Predefined configurations for common scenarios
OPENAI_TIER_1_CONFIG = BrokerConfig(
    default_rpm=3,
    default_rpd=200,
    default_max_concurrency=2,
    model_limits={
        "openai:gpt-4o-mini": {"rpm": 3, "rpd": 200, "max_concurrency": 2},
        "openai:gpt-4o": {"rpm": 10, "rpd": 500, "max_concurrency": 5},
        "openai:gpt-3.5-turbo": {"rpm": 60, "rpd": 1000, "max_concurrency": 10},
    }
)

OPENAI_TIER_2_CONFIG = BrokerConfig(
    default_rpm=50,
    default_rpd=5000,
    default_max_concurrency=10,
    model_limits={
        "openai:gpt-4o-mini": {"rpm": 50, "rpd": 5000, "max_concurrency": 10},
        "openai:gpt-4o": {"rpm": 100, "rpd": 10000, "max_concurrency": 20},
        "openai:gpt-3.5-turbo": {"rpm": 500, "rpd": 50000, "max_concurrency": 50},
    }
)

ANTHROPIC_CONFIG = BrokerConfig(
    default_rpm=20,
    default_rpd=1000,
    default_max_concurrency=5,
    model_limits={
        "anthropic:claude-3-haiku": {"rpm": 50, "rpd": 1000, "max_concurrency": 5},
        "anthropic:claude-3-sonnet": {"rpm": 20, "rpd": 500, "max_concurrency": 3},
        "anthropic:claude-3-opus": {"rpm": 5, "rpd": 100, "max_concurrency": 2},
    }
)

MIXED_PROVIDER_CONFIG = BrokerConfig(
    default_rpm=30,
    default_rpd=1000,
    default_max_concurrency=5,
    model_limits={
        # OpenAI
        "openai:gpt-4o-mini": {"rpm": 50, "rpd": 5000, "max_concurrency": 10},
        "openai:gpt-4o": {"rpm": 100, "rpd": 10000, "max_concurrency": 20},
        "openai:gpt-3.5-turbo": {"rpm": 500, "rpd": 50000, "max_concurrency": 50},
        # Anthropic
        "anthropic:claude-3-haiku": {"rpm": 50, "rpd": 1000, "max_concurrency": 5},
        "anthropic:claude-3-sonnet": {"rpm": 20, "rpd": 500, "max_concurrency": 3},
        "anthropic:claude-3-opus": {"rpm": 5, "rpd": 100, "max_concurrency": 2},
    }
)

DEVELOPMENT_CONFIG = BrokerConfig(
    default_rpm=10,
    default_rpd=100,
    default_max_concurrency=2,
    max_retries=2,
    base_delay=0.5,
    max_delay=10.0,
    model_limits={
        "openai:gpt-4o-mini": {"rpm": 10, "rpd": 100, "max_concurrency": 2},
        "openai:gpt-3.5-turbo": {"rpm": 20, "rpd": 200, "max_concurrency": 3},
    }
)


def get_config_by_name(name: str) -> BrokerConfig:
    """Get predefined configuration by name."""
    configs = {
        "openai_tier_1": OPENAI_TIER_1_CONFIG,
        "openai_tier_2": OPENAI_TIER_2_CONFIG,
        "anthropic": ANTHROPIC_CONFIG,
        "mixed_provider": MIXED_PROVIDER_CONFIG,
        "development": DEVELOPMENT_CONFIG,
    }
    
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config name '{name}'. Available: {available}")
    
    return configs[name]


def create_broker_from_config(
    config: Optional[Union[str, Dict[str, Any], BrokerConfig, Path]] = None
) -> CallBroker:
    """
    Create CallBroker from various configuration sources.
    
    Args:
        config: Configuration source. Can be:
            - str: Name of predefined config or path to config file
            - Dict: Configuration dictionary
            - BrokerConfig: Configuration object
            - Path: Path to config file
            - None: Use environment variables or defaults
    
    Returns:
        Configured CallBroker instance
    """
    if config is None:
        # Try environment, fall back to development config
        try:
            broker_config = BrokerConfig.from_env()
        except Exception:
            broker_config = DEVELOPMENT_CONFIG
    
    elif isinstance(config, str):
        # Check if it's a predefined config name
        try:
            broker_config = get_config_by_name(config)
        except ValueError:
            # Assume it's a file path
            broker_config = BrokerConfig.from_file(config)
    
    elif isinstance(config, dict):
        broker_config = BrokerConfig.from_dict(config)
    
    elif isinstance(config, Path):
        broker_config = BrokerConfig.from_file(config)
    
    elif isinstance(config, BrokerConfig): # type: ignore
        broker_config = config
    
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")
    
    return broker_config.create_broker()


# Convenience function for common use cases
def create_development_broker() -> CallBroker:
    """Create broker with development-friendly settings."""
    return DEVELOPMENT_CONFIG.create_broker()


def create_production_broker(provider: str = "mixed") -> CallBroker:
    """Create broker with production settings for specific provider."""
    if provider.lower() in ["openai", "openai_tier_2"]:
        return OPENAI_TIER_2_CONFIG.create_broker()
    elif provider.lower() == "anthropic":
        return ANTHROPIC_CONFIG.create_broker()
    elif provider.lower() in ["mixed", "multi"]:
        return MIXED_PROVIDER_CONFIG.create_broker()
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'mixed'")