#!/usr/bin/env python3
"""
CallBroker Configuration Example

This example demonstrates how to configure the CallBroker system
for production-grade rate limiting and retry handling.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_orchestra.orchestrator.broker_config import (
    BrokerConfig,
    create_broker_from_config,
    create_development_broker,
    create_production_broker,
    get_config_by_name,
    OPENAI_TIER_1_CONFIG,
    MIXED_PROVIDER_CONFIG
)
from agent_orchestra.orchestrator.call_broker import ModelLimits


async def demonstrate_broker_configurations():
    """Show different ways to configure CallBroker."""
    
    print("🏗️  CallBroker Configuration Examples")
    print("=" * 50)
    print()
    
    # 1. Using predefined configurations
    print("📋 1. Predefined Configurations")
    print("-" * 30)
    
    dev_broker = create_development_broker()
    print(f"✅ Development broker created")
    
    prod_broker = create_production_broker("mixed")
    print(f"✅ Production broker created (mixed providers)")
    
    stats = await dev_broker.get_stats()
    print(f"📊 Development broker models: {len(stats)} configured")
    
    print()
    
    # 2. Using configuration by name
    print("🏷️  2. Configuration by Name")
    print("-" * 30)
    
    openai_broker = create_broker_from_config("openai_tier_1")
    print(f"✅ OpenAI Tier 1 broker created")
    
    anthropic_broker = create_broker_from_config("anthropic")
    print(f"✅ Anthropic broker created")
    
    print()
    
    # 3. Custom configuration from dictionary
    print("📝 3. Custom Configuration")
    print("-" * 30)
    
    custom_config = {
        "default_rpm": 25,
        "default_rpd": 1500,
        "default_max_concurrency": 8,
        "max_retries": 2,
        "base_delay": 2.0,
        "model_limits": {
            "openai:gpt-4o": {"rpm": 50, "rpd": 2000, "max_concurrency": 15},
            "custom:my-model": {"rpm": 10, "rpd": 500, "max_concurrency": 3}
        }
    }
    
    custom_broker = create_broker_from_config(custom_config)
    print(f"✅ Custom broker created from dict")
    
    custom_stats = await custom_broker.get_stats()
    print(f"📊 Custom broker configured models: {list(custom_stats.keys())}")
    
    print()
    
    # 4. Configuration object approach
    print("🎛️  4. Configuration Object")
    print("-" * 30)
    
    config = BrokerConfig(
        default_rpm=40,
        default_rpd=2000,
        default_max_concurrency=6,
        model_limits={
            "openai:gpt-4o-mini": {"rpm": 100, "rpd": 5000, "max_concurrency": 20},
            "special:research-model": {"rpm": 5, "rpd": 50, "max_concurrency": 1}
        }
    )
    
    object_broker = config.create_broker()
    print(f"✅ Broker created from BrokerConfig object")
    
    print()
    
    # 5. Save and load configuration
    print("💾 5. Save/Load Configuration")
    print("-" * 30)
    
    # Save config to file
    config_file = Path("/tmp/agent_orchestra_broker.json")
    MIXED_PROVIDER_CONFIG.save_to_file(config_file)
    print(f"✅ Config saved to: {config_file}")
    
    # Load config from file
    loaded_config = BrokerConfig.from_file(config_file)
    loaded_broker = loaded_config.create_broker()
    print(f"✅ Config loaded from file and broker created")
    
    # Clean up
    config_file.unlink(missing_ok=True)
    
    print()
    
    # 6. Environment-based configuration
    print("🌍 6. Environment Configuration")
    print("-" * 30)
    
    # This would read from environment variables like:
    # AGENT_ORCHESTRA_DEFAULT_RPM=50
    # AGENT_ORCHESTRA_DEFAULT_RPD=2000
    # etc.
    env_config = BrokerConfig.from_env()
    env_broker = env_config.create_broker()
    print(f"✅ Broker created from environment variables")
    
    print()
    
    # 7. Integration example with MCPExecutor
    print("🔗 7. Integration with MCPExecutor")
    print("-" * 30)
    
    print("```python")
    print("# Easy integration example:")
    print("from agent_orchestra.orchestrator.executors_mcp import MCPExecutor")
    print("from agent_orchestra.orchestrator.broker_config import create_production_broker")
    print("")
    print("# Create broker with production settings")
    print("broker = create_production_broker('openai')")
    print("")
    print("# Create executor with broker")
    print("executor = MCPExecutor(agent, broker=broker)")
    print("")
    print("# The executor will now automatically use rate limiting!")
    print("```")
    
    print()
    
    # 8. Show actual broker stats
    print("📈 8. Broker Statistics")
    print("-" * 30)
    
    prod_stats = await prod_broker.get_stats()
    for model, stats in prod_stats.items():
        print(f"🤖 {model}:")
        print(f"   RPM: {stats['rpm_used']}/{stats['rpm_limit']}")
        print(f"   RPD: {stats['rpd_used']}/{stats['rpd_limit']}")
        print(f"   Concurrent: {stats['concurrent_used']}/{stats['concurrent_limit']}")
    
    print()
    print("🌟 CallBroker Configuration Complete!")
    print("💡 Use these patterns to configure rate limiting for your agents")
    
    # Cleanup brokers
    await dev_broker.shutdown()
    await prod_broker.shutdown()
    await openai_broker.shutdown()
    await anthropic_broker.shutdown()
    await custom_broker.shutdown()
    await object_broker.shutdown()
    await loaded_broker.shutdown()
    await env_broker.shutdown()


async def main():
    """Main entry point."""
    try:
        await demonstrate_broker_configurations()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🚀 CallBroker configuration demo complete!")


if __name__ == "__main__":
    asyncio.run(main())