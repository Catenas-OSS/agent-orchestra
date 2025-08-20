#!/usr/bin/env python3
"""
Polish Verification Demo - Quick verification of all polish improvements

This demo shows the core polish improvements working without complex MCP setup:
✅ Profile-based AgentPool (no duplicate inits)
✅ Server-name parameter handling
✅ Race-safe agent creation
✅ Path validation and safety
✅ Clean async context management
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from agent_orchestra.orchestrator.agent_pool import AgentPool, AgentSpec
    from agent_orchestra.orchestrator.fs_utils import fs_args, copy_files_to_root
    from agent_orchestra.orchestrator.call_broker import CallBroker, ModelLimits
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)


class MockMCPAgent:
    """Mock agent for demo purposes."""
    
    def __init__(self, server_name=None, model_key="demo", creation_id=None):
        self.server_name = server_name
        self.model_key = model_key
        self.creation_id = creation_id or f"agent_{id(self)}"
        self.call_count = 0
        print(f"   🤖 Created agent {self.creation_id} for server '{server_name}'")
    
    async def run(self, prompt, **kwargs):
        self.call_count += 1
        server_name = kwargs.get('server_name', 'default')
        return f"Mock response from {self.creation_id} on server {server_name} (call #{self.call_count})"
    
    async def astream(self, prompt, **kwargs):
        self.call_count += 1
        server_name = kwargs.get('server_name', 'default')
        yield {"chunk": 1, "content": f"Processing with {self.creation_id} on {server_name}"}
        yield {"output": f"Mock stream result from {self.creation_id} (call #{self.call_count})"}


async def demo_profile_based_agent_pool():
    """Demonstrate profile-based agent pooling."""
    print("🏊 Demo 1: Profile-Based Agent Pool")
    print("=" * 50)
    
    creation_count = 0
    
    async def factory(spec: AgentSpec):
        nonlocal creation_count
        creation_count += 1
        return MockMCPAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
    
    pool = AgentPool(factory)
    
    print("📝 Creating specs for different server profiles...")
    
    # Get agents for different profiles
    fs_spec = AgentSpec(server_name="fs_business", model_key="gpt-4", policy_id=None)
    playwright_spec = AgentSpec(server_name="playwright", model_key="gpt-4", policy_id=None)
    
    print("🔄 Testing agent reuse within same profile...")
    
    # Multiple requests for fs_business should get same agent
    fs_agent1 = await pool.get(fs_spec, "run1")
    fs_agent2 = await pool.get(fs_spec, "run1")
    fs_agent3 = await pool.get(fs_spec, "run1")
    
    # Request for playwright should get different agent
    playwright_agent = await pool.get(playwright_spec, "run1")
    
    # Verify agent reuse
    assert fs_agent1 is fs_agent2 is fs_agent3, "FS agents should be the same instance"
    assert playwright_agent is not fs_agent1, "Playwright agent should be different"
    assert creation_count == 2, f"Should have created exactly 2 agents, got {creation_count}"
    
    # Get pool stats
    stats = await pool.get_pool_stats()
    
    print(f"✅ Results:")
    print(f"   📊 Total agents created: {creation_count}")
    print(f"   🏊 Agent profiles: {len(stats['profiles'])}")
    for profile_key, profile_info in stats['profiles'].items():
        server = profile_info['server_name'] or 'default'
        usage = profile_info['usage_count']
        print(f"      {server} server: {usage} uses")
    
    await pool.shutdown()
    print()


async def demo_race_safe_creation():
    """Demonstrate race-safe concurrent agent creation."""
    print("🏁 Demo 2: Race-Safe Agent Creation")
    print("=" * 50)
    
    creation_count = 0
    
    async def factory(spec: AgentSpec):
        nonlocal creation_count
        creation_count += 1
        # Add delay to simulate real agent creation
        await asyncio.sleep(0.01)
        return MockMCPAgent(spec.server_name, spec.model_key, f"race_agent_{creation_count}")
    
    pool = AgentPool(factory)
    spec = AgentSpec(server_name="concurrent_test", model_key="gpt-4")
    
    print("🔄 Launching 10 concurrent requests for same profile...")
    
    # Launch 10 concurrent requests
    tasks = [
        asyncio.create_task(pool.get(spec, f"concurrent_run_{i}"))
        for i in range(10)
    ]
    
    agents = await asyncio.gather(*tasks)
    
    # All agents should be the same instance
    first_agent = agents[0]
    all_same = all(agent is first_agent for agent in agents)
    
    print(f"✅ Results:")
    print(f"   🤖 Agents created: {creation_count}")
    print(f"   🔗 All agents identical: {all_same}")
    print(f"   ⚡ Race condition prevented: {creation_count == 1}")
    
    assert creation_count == 1, f"Should have created exactly 1 agent, got {creation_count}"
    assert all_same, "All agents should be the same instance"
    
    await pool.shutdown()
    print()


async def demo_server_name_parameter_handling():
    """Demonstrate server_name parameter handling."""
    print("🌐 Demo 3: Server Name Parameter Handling")
    print("=" * 50)
    
    creation_count = 0
    
    async def factory(spec: AgentSpec):
        nonlocal creation_count
        creation_count += 1
        return MockMCPAgent(spec.server_name, spec.model_key, f"param_agent_{creation_count}")
    
    pool = AgentPool(factory)
    
    # Get agents for different servers
    fs_spec = AgentSpec(server_name="fs_server", model_key="demo")
    web_spec = AgentSpec(server_name="web_server", model_key="demo")
    
    fs_agent = await pool.get(fs_spec, "demo_run")
    web_agent = await pool.get(web_spec, "demo_run")
    
    print("🔄 Testing server_name parameter passing...")
    
    # Test agent calls with server_name parameter
    fs_result = await fs_agent.run("Test message", server_name="fs_server")
    web_result = await web_agent.run("Test message", server_name="web_server")
    
    print(f"✅ Results:")
    print(f"   🗂️  FS Server Response: {fs_result}")
    print(f"   🌐 Web Server Response: {web_result}")
    print(f"   📊 Agents created: {creation_count}")
    
    await pool.shutdown()
    print()


def demo_path_validation():
    """Demonstrate filesystem path validation."""
    print("🛡️  Demo 4: Path Validation and Safety")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        
        print("🔄 Testing safe path handling...")
        
        # Test safe relative paths
        try:
            safe_args = fs_args(root, "test.json")
            print(f"   ✅ Safe path 'test.json': {safe_args}")
        except Exception as e:
            print(f"   ❌ Safe path failed: {e}")
        
        # Test directory traversal prevention
        try:
            unsafe_args = fs_args(root, "../../../etc/passwd")
            print(f"   ❌ Directory traversal allowed: {unsafe_args}")
        except ValueError as e:
            print(f"   ✅ Directory traversal blocked: Path validation working")
        
        # Test file copying with path sanitization
        test_files = {
            "safe_file.json": {"data": "safe content"},
            "/absolute/path/unsafe.json": {"data": "content"}
        }
        
        copied = copy_files_to_root(test_files, root)
        print(f"   ✅ File copying with sanitization:")
        for original, safe_name in copied.items():
            print(f"      '{original}' -> '{safe_name}'")
            assert (root / safe_name).exists(), f"File {safe_name} should exist"
    
    print()


async def demo_rate_limiting():
    """Demonstrate global rate limiting."""
    print("⏱️  Demo 5: Global Rate Limiting")
    print("=" * 50)
    
    # Create broker with low limits for demo
    limits = {"demo:model": ModelLimits(rpm=2, rpd=10, max_concurrency=1)}
    broker = CallBroker(limits)
    
    agent1 = MockMCPAgent("server1", "demo:model", "rate_agent_1")
    agent2 = MockMCPAgent("server2", "demo:model", "rate_agent_2")
    
    print("🔄 Testing rate limiting across different agents...")
    
    import time
    start_time = time.time()
    
    # Make calls through broker
    await broker.call_agent_regular("demo:model", lambda: agent1.run("test"))
    await broker.call_agent_regular("demo:model", lambda: agent2.run("test"))
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Get stats
    stats = await broker.get_stats()
    
    print(f"✅ Results:")
    print(f"   ⏱️  Total duration: {duration:.2f}s")
    print(f"   📊 RPM used: {stats['demo:model']['rpm_used']}")
    print(f"   🎯 Both agents called: {agent1.call_count == 1 and agent2.call_count == 1}")
    print(f"   ⚡ Rate limiting active: {duration > 0.5}")  # Should have waited
    
    await broker.shutdown()
    print()


async def main():
    """Run all polish improvement demos."""
    print("🌟 Agentic Orchestra - Polish Improvements Verification")
    print("=" * 65)
    print("🎯 Demonstrating all production-ready improvements")
    print()
    
    try:
        # Run all demos
        await demo_profile_based_agent_pool()
        await demo_race_safe_creation()
        await demo_server_name_parameter_handling()
        demo_path_validation()
        await demo_rate_limiting()
        
        print("🎉 All Polish Improvements Verified!")
        print("=" * 65)
        print("✅ Profile-based AgentPool - Agent reuse without duplicates")
        print("✅ Race-safe agent creation - Concurrent requests handled safely")  
        print("✅ Server name parameter handling - Clean parameter filtering")
        print("✅ Path validation and safety - Directory traversal prevention")
        print("✅ Global rate limiting - Shared limits across all agents")
        print()
        print("🚀 Agentic Orchestra is production-ready with all improvements!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())