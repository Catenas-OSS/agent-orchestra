"""
Tests for CallBroker configuration system.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from agent_orchestra.orchestrator.broker_config import (
    BrokerConfig,
    ModelLimits,
    create_broker_from_config,
    create_development_broker,
    create_production_broker,
    get_config_by_name,
    OPENAI_TIER_1_CONFIG,
    ANTHROPIC_CONFIG,
    DEVELOPMENT_CONFIG,
)
from agent_orchestra.orchestrator.call_broker import CallBroker


class TestBrokerConfig:
    """Test BrokerConfig class functionality."""
    
    def test_default_initialization(self):
        """Test default BrokerConfig initialization."""
        config = BrokerConfig()
        
        assert config.default_rpm == 30
        assert config.default_rpd == 500
        assert config.default_max_concurrency == 5
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.model_limits == {}
    
    def test_custom_initialization(self):
        """Test BrokerConfig with custom values."""
        config = BrokerConfig(
            default_rpm=100,
            default_rpd=2000,
            default_max_concurrency=10,
            max_retries=5,
            model_limits={"test:model": {"rpm": 50, "rpd": 1000, "max_concurrency": 8}}
        )
        
        assert config.default_rpm == 100
        assert config.default_rpd == 2000
        assert config.default_max_concurrency == 10
        assert config.max_retries == 5
        assert "test:model" in config.model_limits
    
    def test_from_dict_creation(self):
        """Test creating BrokerConfig from dictionary."""
        config_dict = {
            "default_rpm": 75,
            "default_rpd": 1500,
            "model_limits": {
                "openai:gpt-4": {"rpm": 20, "rpd": 500, "max_concurrency": 5}
            }
        }
        
        config = BrokerConfig.from_dict(config_dict)
        
        assert config.default_rpm == 75
        assert config.default_rpd == 1500
        assert "openai:gpt-4" in config.model_limits
        assert config.model_limits["openai:gpt-4"]["rpm"] == 20
    
    def test_to_dict_conversion(self):
        """Test converting BrokerConfig to dictionary."""
        config = BrokerConfig(
            default_rpm=60,
            model_limits={"test:model": {"rpm": 30, "rpd": 300, "max_concurrency": 3}}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["default_rpm"] == 60
        assert "model_limits" in config_dict
        assert "test:model" in config_dict["model_limits"]
    
    def test_file_save_and_load(self):
        """Test saving and loading BrokerConfig from file."""
        config = BrokerConfig(
            default_rpm=45,
            default_rpd=900,
            model_limits={"file:test": {"rpm": 15, "rpd": 150, "max_concurrency": 2}}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            # Save config
            config.save_to_file(config_file)
            assert config_file.exists()
            
            # Load config
            loaded_config = BrokerConfig.from_file(config_file)
            
            assert loaded_config.default_rpm == 45
            assert loaded_config.default_rpd == 900
            assert "file:test" in loaded_config.model_limits
            assert loaded_config.model_limits["file:test"]["rpm"] == 15
    
    def test_file_load_nonexistent(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            BrokerConfig.from_file("/nonexistent/path/config.json")
    
    @patch.dict(os.environ, {
        "AGENT_ORCHESTRA_DEFAULT_RPM": "85",
        "AGENT_ORCHESTRA_DEFAULT_RPD": "1700",
        "AGENT_ORCHESTRA_DEFAULT_CONCURRENCY": "12",
        "AGENT_ORCHESTRA_MAX_RETRIES": "4",
        "AGENT_ORCHESTRA_BASE_DELAY": "2.5",
        "AGENT_ORCHESTRA_MAX_DELAY": "120.0"
    })
    def test_from_env_creation(self):
        """Test creating BrokerConfig from environment variables."""
        config = BrokerConfig.from_env()
        
        assert config.default_rpm == 85
        assert config.default_rpd == 1700
        assert config.default_max_concurrency == 12
        assert config.max_retries == 4
        assert config.base_delay == 2.5
        assert config.max_delay == 120.0
    
    def test_from_env_partial_override(self):
        """Test environment variables override only specified values."""
        with patch.dict(os.environ, {"AGENT_ORCHESTRA_DEFAULT_RPM": "200"}, clear=False):
            config = BrokerConfig.from_env()
            
            assert config.default_rpm == 200
            # Others should remain defaults
            assert config.default_rpd == 500  # default
            assert config.default_max_concurrency == 5  # default
    
    def test_create_broker_from_config(self):
        """Test creating CallBroker from BrokerConfig."""
        config = BrokerConfig(
            default_rpm=55,
            default_rpd=1100,
            model_limits={
                "broker:test": {"rpm": 25, "rpd": 250, "max_concurrency": 4}
            }
        )
        
        broker = config.create_broker()
        
        assert isinstance(broker, CallBroker)
        
        # Check that limits were applied correctly
        test_limits = broker._get_model_limits("broker:test")
        assert test_limits.rpm == 25
        assert test_limits.rpd == 250
        assert test_limits.max_concurrency == 4
        
        default_limits = broker._get_model_limits("unknown:model")
        assert default_limits.rpm == 55
        assert default_limits.rpd == 1100


class TestPredefinedConfigurations:
    """Test predefined configuration constants."""
    
    def test_openai_tier_1_config(self):
        """Test OpenAI Tier 1 configuration."""
        config = OPENAI_TIER_1_CONFIG
        
        assert config.default_rpm == 3
        assert config.default_rpd == 200
        assert config.default_max_concurrency == 2
        
        assert "openai:gpt-4o-mini" in config.model_limits
        assert "openai:gpt-4o" in config.model_limits
        assert "openai:gpt-3.5-turbo" in config.model_limits
    
    def test_anthropic_config(self):
        """Test Anthropic configuration."""
        config = ANTHROPIC_CONFIG
        
        assert config.default_rpm == 20
        assert config.default_rpd == 1000
        
        assert "anthropic:claude-3-haiku" in config.model_limits
        assert "anthropic:claude-3-sonnet" in config.model_limits
        assert "anthropic:claude-3-opus" in config.model_limits
    
    def test_development_config(self):
        """Test development configuration."""
        config = DEVELOPMENT_CONFIG
        
        assert config.default_rpm == 10
        assert config.default_rpd == 100
        assert config.max_retries == 2
        assert config.base_delay == 0.5
        assert config.max_delay == 10.0
    
    def test_get_config_by_name_valid(self):
        """Test getting predefined config by name."""
        config = get_config_by_name("development")
        assert config.default_rpm == 10
        
        config = get_config_by_name("anthropic")
        assert config.default_rpm == 20
        
        config = get_config_by_name("openai_tier_1")
        assert config.default_rpm == 3
    
    def test_get_config_by_name_invalid(self):
        """Test getting invalid config name raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_config_by_name("nonexistent_config")
        
        assert "Unknown config name" in str(exc_info.value)
        assert "nonexistent_config" in str(exc_info.value)


class TestConfigurationHelpers:
    """Test configuration helper functions."""
    
    @pytest.mark.asyncio
    async def test_create_broker_from_config_none(self):
        """Test creating broker with None config (uses env/defaults)."""
        broker = create_broker_from_config(None)
        
        assert isinstance(broker, CallBroker)
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_broker_from_config_string_name(self):
        """Test creating broker from predefined config name."""
        broker = create_broker_from_config("development")
        
        assert isinstance(broker, CallBroker)
        
        # Should use development config limits
        assert len(broker.model_limits) >= 1  # Should have models configured
        
        stats = await broker.get_stats()
        assert isinstance(stats, dict)  # Stats should be available
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_broker_from_config_string_file(self):
        """Test creating broker from config file path."""
        config = BrokerConfig(default_rpm=123, model_limits={
            "file:model": {"rpm": 45, "rpd": 450, "max_concurrency": 6}
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "broker_config.json"
            config.save_to_file(config_file)
            
            broker = create_broker_from_config(str(config_file))
            
            assert isinstance(broker, CallBroker)
            
            # Verify limits were loaded correctly
            default_limits = broker._get_model_limits("unknown:model")
            assert default_limits.rpm == 123
            
            file_limits = broker._get_model_limits("file:model")
            assert file_limits.rpm == 45
            
            await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_broker_from_config_dict(self):
        """Test creating broker from config dictionary."""
        config_dict = {
            "default_rpm": 67,
            "default_rpd": 670,
            "model_limits": {
                "dict:model": {"rpm": 33, "rpd": 330, "max_concurrency": 7}
            }
        }
        
        broker = create_broker_from_config(config_dict)
        
        assert isinstance(broker, CallBroker)
        
        # Verify limits
        default_limits = broker._get_model_limits("unknown:model")
        assert default_limits.rpm == 67
        
        dict_limits = broker._get_model_limits("dict:model")
        assert dict_limits.rpm == 33
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_broker_from_config_path_object(self):
        """Test creating broker from Path object."""
        config = BrokerConfig(default_rpm=89)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "path_config.json"
            config.save_to_file(config_file)
            
            broker = create_broker_from_config(config_file)  # Pass Path object
            
            assert isinstance(broker, CallBroker)
            default_limits = broker._get_model_limits("unknown:model")
            assert default_limits.rpm == 89
            
            await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_broker_from_config_object(self):
        """Test creating broker from BrokerConfig object."""
        config = BrokerConfig(default_rpm=77, default_rpd=777)
        
        broker = create_broker_from_config(config)
        
        assert isinstance(broker, CallBroker)
        default_limits = broker._get_model_limits("unknown:model")
        assert default_limits.rpm == 77
        assert default_limits.rpd == 777
        
        await broker.shutdown()
    
    def test_create_broker_from_config_invalid_type(self):
        """Test creating broker with invalid config type."""
        with pytest.raises(TypeError):
            create_broker_from_config(12345)  # Invalid type
    
    @pytest.mark.asyncio
    async def test_create_development_broker(self):
        """Test development broker creation helper."""
        broker = create_development_broker()
        
        assert isinstance(broker, CallBroker)
        
        # Should have development-friendly limits
        default_limits = broker._get_model_limits("unknown:model")
        assert default_limits.rpm == 10  # Development config default
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_production_broker_openai(self):
        """Test production broker creation for OpenAI."""
        broker = create_production_broker("openai")
        
        assert isinstance(broker, CallBroker)
        
        # Should have production limits for OpenAI
        gpt4_limits = broker._get_model_limits("openai:gpt-4o")
        assert gpt4_limits.rpm > 10  # Should be higher than development
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_production_broker_anthropic(self):
        """Test production broker creation for Anthropic."""
        broker = create_production_broker("anthropic")
        
        assert isinstance(broker, CallBroker)
        
        # Should have production limits for Anthropic
        claude_limits = broker._get_model_limits("anthropic:claude-3-haiku")
        assert claude_limits.rpm > 10  # Should be higher than development
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_production_broker_mixed(self):
        """Test production broker creation for mixed providers."""
        broker = create_production_broker("mixed")
        
        assert isinstance(broker, CallBroker)
        
        # Should have limits for both providers
        openai_limits = broker._get_model_limits("openai:gpt-4o")
        anthropic_limits = broker._get_model_limits("anthropic:claude-3-haiku")
        
        assert openai_limits.rpm > 10
        assert anthropic_limits.rpm > 10
        
        await broker.shutdown()
    
    def test_create_production_broker_invalid_provider(self):
        """Test production broker creation with invalid provider."""
        with pytest.raises(ValueError) as exc_info:
            create_production_broker("invalid_provider")
        
        assert "Unknown provider" in str(exc_info.value)


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions in configuration."""
    
    def test_empty_model_limits(self):
        """Test config with empty model limits."""
        config = BrokerConfig(model_limits={})
        broker = config.create_broker()
        
        assert isinstance(broker, CallBroker)
        
        # Should use defaults for any model
        default_limits = broker._get_model_limits("any:model")
        assert default_limits.rpm == config.default_rpm
    
    def test_partial_model_limits(self):
        """Test model limits with missing fields use defaults."""
        config = BrokerConfig(
            default_rpm=100,
            default_rpd=1000,
            default_max_concurrency=5,
            model_limits={
                "partial:model": {"rpm": 50}  # Missing rpd and max_concurrency
            }
        )
        
        broker = config.create_broker()
        partial_limits = broker._get_model_limits("partial:model")
        
        assert partial_limits.rpm == 50  # Specified
        assert partial_limits.rpd == 1000  # Default
        assert partial_limits.max_concurrency == 5  # Default
    
    def test_invalid_json_in_file(self):
        """Test loading config from file with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid.json"
            config_file.write_text("{ invalid json }")
            
            with pytest.raises(json.JSONDecodeError):
                BrokerConfig.from_file(config_file)
    
    @patch.dict(os.environ, {
        "AGENT_ORCHESTRA_DEFAULT_RPM": "not_a_number"
    })
    def test_invalid_env_var_types(self):
        """Test handling invalid environment variable types."""
        with pytest.raises(ValueError):
            BrokerConfig.from_env()
    
    def test_negative_values_in_config(self):
        """Test config with negative values (should be allowed)."""
        config = BrokerConfig(
            default_rpm=-1,  # Technically invalid but config allows it
            default_rpd=0,
            default_max_concurrency=1
        )
        
        # Should create broker without error (validation is broker's responsibility)
        broker = config.create_broker()
        assert isinstance(broker, CallBroker)