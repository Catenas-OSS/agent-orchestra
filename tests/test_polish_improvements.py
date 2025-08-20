"""
Sanity tests for Part 4 polish improvements.

Tests the key improvements like per-server singletons, race safety,
global rate limiting, and path validation.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from agent_orchestra.orchestrator.agent_pool import AgentPool, AgentSpec
from agent_orchestra.orchestrator.fs_utils import fs_args, copy_files_to_root, create_multi_server_config
from agent_orchestra.orchestrator.call_broker import CallBroker, ModelLimits
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.orchestrator.types import NodeSpec


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, server_name=None, model_key="test", creation_id=None):
        self.server_name = server_name
        self.model_key = model_key
        self.policy_id = None
        self.creation_id = creation_id or asyncio.current_task().get_name()
        self.call_count = 0
    
    async def run(self, prompt, **kwargs):
        self.call_count += 1
        return f"Result from {self.creation_id} (call #{self.call_count})"
    
    async def astream(self, prompt, **kwargs):
        self.call_count += 1
        yield {"chunk": 1, "content": f"Processing with {self.creation_id}"}
        yield {"output": f"Stream result from {self.creation_id} (call #{self.call_count})"}


class TestPerServerSingletons:
    """Test that agents are correctly shared per server profile."""
    
    @pytest.mark.asyncio
    async def test_same_server_same_agent(self):
        """3 nodes on same server should share 1 agent."""
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
        
        pool = AgentPool(factory)
        
        # Create specs for same server
        spec_fs = AgentSpec(server_name="fs", model_key="gpt-4", policy_id=None)
        
        # Get agents for 3 nodes on same server
        agent1 = await pool.get(spec_fs, "run1")
        agent2 = await pool.get(spec_fs, "run1") 
        agent3 = await pool.get(spec_fs, "run1")
        
        # Should be the same agent
        assert agent1 is agent2
        assert agent2 is agent3
        assert creation_count == 1
        assert agent1.creation_id == "agent_1"
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_different_servers_different_agents(self):
        """Different servers should get different agents."""
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
        
        pool = AgentPool(factory)
        
        # Different server specs
        spec_fs = AgentSpec(server_name="fs", model_key="gpt-4")
        spec_playwright = AgentSpec(server_name="playwright", model_key="gpt-4")
        
        # Get agents
        agent_fs = await pool.get(spec_fs, "run1")
        agent_playwright = await pool.get(spec_playwright, "run1")
        
        # Should be different agents
        assert agent_fs is not agent_playwright
        assert creation_count == 2
        assert agent_fs.server_name == "fs"
        assert agent_playwright.server_name == "playwright"
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_exact_count_3_fs_1_playwright(self):
        """3 nodes on 'fs' + 1 on 'playwright' → exactly 2 agents created."""
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
        
        pool = AgentPool(factory)
        
        # 3 nodes on fs server
        spec_fs = AgentSpec(server_name="fs", model_key="gpt-4")
        fs_agent1 = await pool.get(spec_fs, "run1")
        fs_agent2 = await pool.get(spec_fs, "run1") 
        fs_agent3 = await pool.get(spec_fs, "run1")
        
        # 1 node on playwright server
        spec_playwright = AgentSpec(server_name="playwright", model_key="gpt-4")
        playwright_agent = await pool.get(spec_playwright, "run1")
        
        # Should have exactly 2 agents created
        assert creation_count == 2
        
        # FS agents should be the same
        assert fs_agent1 is fs_agent2 is fs_agent3
        
        # Playwright agent should be different
        assert playwright_agent is not fs_agent1
        
        # Verify server assignments
        assert fs_agent1.server_name == "fs"
        assert playwright_agent.server_name == "playwright"
        
        await pool.shutdown()


class TestRaceSafety:
    """Test race safety in concurrent agent creation."""
    
    @pytest.mark.asyncio
    async def test_concurrent_same_spec_creates_one_agent(self):
        """10 concurrent calls for same spec should create exactly 1 agent."""
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            # Add small delay to simulate real agent creation
            await asyncio.sleep(0.01)
            return MockAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
        
        pool = AgentPool(factory)
        spec = AgentSpec(server_name="test", model_key="gpt-4")
        
        # Launch 10 concurrent requests for the same spec
        tasks = [
            asyncio.create_task(pool.get(spec, f"concurrent_run_{i}"))
            for i in range(10)
        ]
        
        agents = await asyncio.gather(*tasks)
        
        # All agents should be the same instance
        first_agent = agents[0]
        assert all(agent is first_agent for agent in agents)
        
        # Should have created exactly 1 agent
        assert creation_count == 1
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_different_specs_create_different_agents(self):
        """Concurrent calls for different specs should create different agents."""
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            await asyncio.sleep(0.01)
            return MockAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
        
        pool = AgentPool(factory)
        
        # Create 5 different specs
        specs = [
            AgentSpec(server_name=f"server_{i}", model_key="gpt-4")
            for i in range(5)
        ]
        
        # Launch concurrent requests for different specs
        tasks = [
            asyncio.create_task(pool.get(spec, f"run_{i}"))
            for i, spec in enumerate(specs)
        ]
        
        agents = await asyncio.gather(*tasks)
        
        # All agents should be different
        for i, agent in enumerate(agents):
            for j, other_agent in enumerate(agents):
                if i != j:
                    assert agent is not other_agent
        
        # Should have created 5 agents
        assert creation_count == 5
        
        await pool.shutdown()


class TestGlobalRateLimiting:
    """Test that rate limiting works across different agents."""
    
    @pytest.mark.asyncio 
    async def test_shared_broker_across_agents(self):
        """Two different agents with same model_key share RPM limits."""
        # Create broker with very low limits for testing
        limits = {"openai:gpt-4": ModelLimits(rpm=2, rpd=10, max_concurrency=1)}
        broker = CallBroker(limits)
        
        # Create two different mock agents
        agent1 = MockAgent("fs", "openai:gpt-4", "agent1")
        agent2 = MockAgent("playwright", "openai:gpt-4", "agent2")
        
        # Make calls through the broker
        await broker.call_agent_regular("openai:gpt-4", lambda: agent1.run("test"))
        await broker.call_agent_regular("openai:gpt-4", lambda: agent2.run("test"))
        
        # Check stats - should show 2 requests for the shared model
        stats = await broker.get_stats()
        assert "openai:gpt-4" in stats
        assert stats["openai:gpt-4"]["rpm_used"] == 2
        
        # Both agents should have been called
        assert agent1.call_count == 1
        assert agent2.call_count == 1
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement_across_agents(self):
        """Rate limits should be enforced across different agent instances."""
        import time
        
        # Very restrictive limits
        limits = {"test:model": ModelLimits(rpm=1, rpd=5, max_concurrency=1)}
        broker = CallBroker(limits)
        
        agent1 = MockAgent("fs", "test:model", "agent1")  
        agent2 = MockAgent("playwright", "test:model", "agent2")
        
        # First call should succeed quickly
        start_time = time.time()
        await broker.call_agent_regular("test:model", lambda: agent1.run("test1"))
        first_call_time = time.time() - start_time
        
        # Second call should be delayed by rate limiting
        start_time = time.time()
        await broker.call_agent_regular("test:model", lambda: agent2.run("test2"))
        second_call_time = time.time() - start_time
        
        # Second call should take longer due to rate limiting
        assert second_call_time > first_call_time
        assert second_call_time > 0.5  # Should have waited
        
        # Both agents called
        assert agent1.call_count == 1
        assert agent2.call_count == 1
        
        await broker.shutdown()


class TestPathValidation:
    """Test filesystem path validation and safety."""
    
    def test_fs_args_relative_path(self):
        """Valid relative paths should work."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            
            # Valid relative path
            args = fs_args(root, "test.json")
            assert args == {"path": "test.json"}
            
            # Valid nested relative path
            args = fs_args(root, "data/test.json")
            assert args == {"path": "data/test.json"}
    
    def test_fs_args_prevents_directory_traversal(self):
        """Directory traversal attempts should be blocked."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            
            # Attempt directory traversal
            with pytest.raises(ValueError) as exc_info:
                fs_args(root, "../../../etc/passwd")
            
            assert "outside root" in str(exc_info.value)
    
    def test_fs_args_prevents_absolute_path_escape(self):
        """Absolute paths that escape root should be blocked.""" 
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            
            # Attempt absolute path escape
            with pytest.raises(ValueError) as exc_info:
                fs_args(root, "/etc/passwd")
            
            assert "outside root" in str(exc_info.value)
    
    def test_copy_files_to_root(self):
        """File copying should create safe relative paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            
            files = {
                "test1.json": {"data": "value1"},
                "test2.json": {"data": "value2"},
                "/absolute/path/test3.json": {"data": "value3"}  # Should be sanitized
            }
            
            result = copy_files_to_root(files, root)
            
            # Should return safe relative paths
            assert result["test1.json"] == "test1.json"
            assert result["test2.json"] == "test2.json" 
            assert result["/absolute/path/test3.json"] == "test3.json"  # Sanitized to basename
            
            # Files should actually exist
            assert (root / "test1.json").exists()
            assert (root / "test2.json").exists()
            assert (root / "test3.json").exists()
    
    def test_create_multi_server_config(self):
        """Multi-server config creation should work."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            configs = {
                "fs_sales": {"root": f"{tmp_dir}/sales"},
                "fs_reports": {"root": f"{tmp_dir}/reports"}, 
                "playwright": {"type": "playwright"}
            }
            
            result = create_multi_server_config(configs)
            
            # Should have mcpServers section
            assert "mcpServers" in result
            assert len(result["mcpServers"]) == 3
            
            # Filesystem servers should have proper config
            fs_sales = result["mcpServers"]["fs_sales"]
            assert fs_sales["command"] == "npx"
            assert "--root" in fs_sales["args"]
            
            # Playwright server should have proper config  
            playwright = result["mcpServers"]["playwright"]
            assert playwright["command"] == "npx"
            assert "@modelcontextprotocol/server-playwright" in playwright["args"]


class TestMCPExecutorImprovements:
    """Test MCPExecutor improvements."""
    
    @pytest.mark.asyncio
    async def test_server_name_routing(self):
        """Server name should be passed through to agent calls."""
        call_log = []
        
        class MockAgent:
            async def run(self, prompt, **kwargs):
                call_log.append(kwargs)
                return "test_result"
        
        # Create executor with agent pool
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent()
        
        pool = AgentPool(factory)
        executor = MCPExecutor(
            agent=None,
            default_server="default_server",
            agent_pool=pool,
            model_key="test"
        )
        executor.set_run_context("test_run")
        
        # Test with specific server_name
        node = NodeSpec(id="test", type="task", server_name="specific_server")
        ctx = {"blackboard": {}}
        
        await executor.execute(node, ctx)
        
        # Should have passed server_name through
        assert len(call_log) == 1
        assert call_log[0]["server_name"] == "specific_server"
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_fallback_to_default_server(self):
        """Should use default server when node has no server_name."""
        call_log = []
        
        class MockAgent:
            async def run(self, prompt, **kwargs):
                call_log.append(kwargs)
                return "test_result"
        
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent()
        
        pool = AgentPool(factory)
        executor = MCPExecutor(
            agent=None,
            default_server="fallback_server",
            agent_pool=pool,
            model_key="test"
        )
        executor.set_run_context("test_run")
        
        # Test without server_name
        node = NodeSpec(id="test", type="task")  # No server_name
        ctx = {"blackboard": {}}
        
        await executor.execute(node, ctx)
        
        # Should have used default server
        assert len(call_log) == 1
        assert call_log[0]["server_name"] == "fallback_server"
        
        await pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_output_capture_fallback(self):
        """Should capture text when no output key is present."""
        
        class MockAgent:
            async def astream(self, prompt, **kwargs):
                yield {"chunk": 1, "content": "processing"}
                yield {"text": "final text result"}  # No "output" key
        
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent()
        
        pool = AgentPool(factory)
        executor = MCPExecutor(
            agent=None,
            agent_pool=pool,
            model_key="test"
        )
        executor.set_run_context("test_run")
        
        node = NodeSpec(id="test", type="task")
        ctx = {"blackboard": {}}
        
        chunks = []
        
        async def collect_chunk(chunk):
            chunks.append(chunk)
        
        result = await executor.execute_with_stream(node, ctx, collect_chunk)
        
        # Should have captured text as fallback
        assert result["output"] == "final text result"
        assert len(chunks) == 2
        
        await pool.shutdown()


class TestIntegrationScenarios:
    """Integration tests combining multiple improvements."""
    
    @pytest.mark.asyncio
    async def test_complete_integration(self):
        """Test all improvements working together."""
        # This would be a complex integration test
        # For now, just verify the components can be created together
        
        creation_count = 0
        
        async def factory(spec: AgentSpec):
            nonlocal creation_count
            creation_count += 1
            return MockAgent(spec.server_name, spec.model_key, f"agent_{creation_count}")
        
        # Create all components
        pool = AgentPool(factory)
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        executor = MCPExecutor(
            agent=None,
            default_server="fs",
            broker=broker,
            agent_pool=pool,
            model_key="openai:gpt-4"
        )
        
        # Verify they can work together
        executor.set_run_context("integration_test")
        
        # Clean up
        await broker.shutdown()
        await pool.shutdown()
        
        # Test passed if no exceptions raised
        assert True