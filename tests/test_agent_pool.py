"""
Tests for AgentPool system - agent reuse and lifecycle management.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from agent_orchestra.orchestrator.agent_pool import AgentPool, get_global_agent_pool, set_global_agent_pool


class MockSidecarMCPAgent:
    """Mock SidecarMCPAgent for testing."""
    
    def __init__(self, llm=None, client=None, max_steps=10):
        self.llm = llm or Mock()
        self.client = client or Mock()
        self.max_steps = max_steps
        self._pool_agent_id = None
    
    async def run(self, prompt: str):
        return f"mock_result_for_{prompt}"
    
    async def astream(self, prompt: str):
        yield {"chunk": 1, "content": f"processing_{prompt}"}
        yield {"output": f"mock_result_for_{prompt}"}


class TestAgentPool:
    """Test AgentPool functionality."""
    
    @pytest.fixture
    def agent_pool(self):
        """Provide a fresh AgentPool instance."""
        return AgentPool(max_agents_per_run=3)
    
    @pytest.fixture
    def template_agent(self):
        """Provide a template agent for testing."""
        return MockSidecarMCPAgent()
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, agent_pool):
        """Test AgentPool initializes correctly."""
        assert agent_pool.max_agents_per_run == 3
        assert len(agent_pool._run_pools) == 0
        assert len(agent_pool._active_runs) == 0
        
        stats = await agent_pool.get_pool_stats()
        assert stats["active_runs"] == 0
        assert stats["total_agents"] == 0
    
    @pytest.mark.asyncio
    async def test_get_agent_for_run_creates_new(self, agent_pool, template_agent):
        """Test getting agent for run creates new agent first time."""
        run_id = "test_run_001"
        
        agent = await agent_pool.get_agent_for_run(run_id, template_agent)
        
        assert agent is not None
        assert agent != template_agent  # Should be new instance
        assert hasattr(agent, '_pool_agent_id')
        assert agent._pool_agent_id == f"{run_id}:agent:0"
        
        # Should track the run
        assert run_id in agent_pool._active_runs
        assert run_id in agent_pool._run_pools
        assert len(agent_pool._run_pools[run_id]) == 1
    
    @pytest.mark.asyncio
    async def test_get_agent_for_run_reuses_existing(self, agent_pool, template_agent):
        """Test getting agent for same run reuses existing agent."""
        run_id = "test_run_002"
        
        # Get agent first time
        agent1 = await agent_pool.get_agent_for_run(run_id, template_agent)
        
        # Get agent second time
        agent2 = await agent_pool.get_agent_for_run(run_id, template_agent)
        
        # Should be the same agent instance
        assert agent1 is agent2
        assert len(agent_pool._run_pools[run_id]) == 1
        
        # Usage count should increase
        agent_id = agent1._pool_agent_id
        assert agent_pool._run_agent_usage[run_id][agent_id] == 2
    
    @pytest.mark.asyncio
    async def test_get_agent_for_foreach_item_reuses_run_agent(self, agent_pool, template_agent):
        """Test getting agent for foreach item reuses run's primary agent."""
        run_id = "test_run_003"
        
        # Get run agent
        run_agent = await agent_pool.get_agent_for_run(run_id, template_agent)
        
        # Get foreach item agents
        foreach_agent1 = await agent_pool.get_agent_for_foreach_item(run_id, 0, template_agent)
        foreach_agent2 = await agent_pool.get_agent_for_foreach_item(run_id, 1, template_agent)
        
        # All should be the same agent
        assert run_agent is foreach_agent1
        assert run_agent is foreach_agent2
        
        # Usage count should reflect all accesses
        agent_id = run_agent._pool_agent_id
        assert agent_pool._run_agent_usage[run_id][agent_id] == 3  # 1 run + 2 foreach
    
    @pytest.mark.asyncio
    async def test_multiple_runs_get_separate_agents(self, agent_pool, template_agent):
        """Test multiple runs get separate agent instances."""
        run_id1 = "test_run_004"
        run_id2 = "test_run_005"
        
        agent1 = await agent_pool.get_agent_for_run(run_id1, template_agent)
        agent2 = await agent_pool.get_agent_for_run(run_id2, template_agent)
        
        # Should be different agents
        assert agent1 is not agent2
        assert agent1._pool_agent_id != agent2._pool_agent_id
        
        # Should track both runs
        assert run_id1 in agent_pool._active_runs
        assert run_id2 in agent_pool._active_runs
        assert len(agent_pool._active_runs) == 2
    
    @pytest.mark.asyncio
    async def test_agent_cloning_success(self, agent_pool):
        """Test successful agent cloning."""
        # Create template with specific attributes
        llm_mock = Mock()
        client_mock = Mock()
        template = MockSidecarMCPAgent(llm=llm_mock, client=client_mock, max_steps=5)
        
        run_id = "test_run_006"
        
        agent = await agent_pool.get_agent_for_run(run_id, template)
        
        # New agent should have same LLM and client
        assert agent.llm is llm_mock
        assert agent.client is client_mock
        assert agent.max_steps == 5
        assert agent._pool_agent_id == f"{run_id}:agent:0"
    
    @pytest.mark.asyncio
    async def test_agent_cloning_failure_fallback(self, agent_pool):
        """Test agent cloning failure falls back to template."""
        # Create template that will cause cloning to fail
        template = Mock()
        del template.llm  # Missing required attribute
        
        run_id = "test_run_007"
        
        agent = await agent_pool.get_agent_for_run(run_id, template)
        
        # Should fall back to template agent
        assert agent is template
    
    @pytest.mark.asyncio
    async def test_finish_run_cleanup(self, agent_pool, template_agent):
        """Test finishing a run cleans up properly."""
        run_id = "test_run_008"
        
        # Create and use some agents
        agent = await agent_pool.get_agent_for_run(run_id, template_agent)
        await agent_pool.get_agent_for_foreach_item(run_id, 0, template_agent)
        
        # Verify setup
        assert run_id in agent_pool._active_runs
        assert run_id in agent_pool._run_pools
        assert run_id in agent_pool._run_agent_usage
        
        # Finish the run
        await agent_pool.finish_run(run_id)
        
        # Verify cleanup
        assert run_id not in agent_pool._active_runs
        assert run_id not in agent_pool._run_pools
        assert run_id not in agent_pool._run_agent_usage
        
        agent_id = agent._pool_agent_id
        assert agent_id not in agent_pool._agent_refs
        assert agent_id not in agent_pool._agent_run_mapping
    
    @pytest.mark.asyncio
    async def test_finish_run_nonexistent(self, agent_pool):
        """Test finishing nonexistent run is safe."""
        # Should not raise error
        await agent_pool.finish_run("nonexistent_run")
    
    @pytest.mark.asyncio
    async def test_get_pool_stats_detailed(self, agent_pool, template_agent):
        """Test detailed pool statistics."""
        run_id1 = "stats_run_001"
        run_id2 = "stats_run_002"
        
        # Create agents and usage
        await agent_pool.get_agent_for_run(run_id1, template_agent)
        await agent_pool.get_agent_for_foreach_item(run_id1, 0, template_agent)
        await agent_pool.get_agent_for_foreach_item(run_id1, 1, template_agent)
        
        await agent_pool.get_agent_for_run(run_id2, template_agent)
        
        stats = await agent_pool.get_pool_stats()
        
        assert stats["active_runs"] == 2
        assert stats["total_agents"] == 2
        
        assert run_id1 in stats["runs"]
        assert run_id2 in stats["runs"]
        
        run1_stats = stats["runs"][run_id1]
        assert run1_stats["agents"] == 1
        assert run1_stats["total_usage"] == 3  # 1 run + 2 foreach
        
        run2_stats = stats["runs"][run_id2]
        assert run2_stats["agents"] == 1
        assert run2_stats["total_usage"] == 1  # 1 run
    
    @pytest.mark.asyncio
    async def test_agent_pool_shutdown(self, agent_pool, template_agent):
        """Test agent pool shutdown cleans up everything."""
        run_id1 = "shutdown_run_001"
        run_id2 = "shutdown_run_002"
        
        # Create some agents
        await agent_pool.get_agent_for_run(run_id1, template_agent)
        await agent_pool.get_agent_for_run(run_id2, template_agent)
        
        # Verify setup
        stats = await agent_pool.get_pool_stats()
        assert stats["active_runs"] == 2
        assert stats["total_agents"] == 2
        
        # Shutdown
        await agent_pool.shutdown()
        
        # Verify complete cleanup
        final_stats = await agent_pool.get_pool_stats()
        assert final_stats["active_runs"] == 0
        assert final_stats["total_agents"] == 0
        assert len(agent_pool._run_pools) == 0
        assert len(agent_pool._active_runs) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_requests(self, agent_pool, template_agent):
        """Test concurrent requests for agents work correctly."""
        run_id = "concurrent_run_001"
        
        # Request multiple agents concurrently
        tasks = [
            asyncio.create_task(agent_pool.get_agent_for_run(run_id, template_agent))
            for _ in range(5)
        ]
        
        agents = await asyncio.gather(*tasks)
        
        # All should be the same agent instance
        first_agent = agents[0]
        assert all(agent is first_agent for agent in agents)
        
        # Usage count should reflect all requests
        agent_id = first_agent._pool_agent_id
        assert agent_pool._run_agent_usage[run_id][agent_id] == 5
    
    @pytest.mark.asyncio
    async def test_agent_id_generation(self, agent_pool):
        """Test agent ID generation is deterministic."""
        run_id = "id_test_run"
        
        agent_id1 = agent_pool._generate_agent_id(run_id, 0)
        agent_id2 = agent_pool._generate_agent_id(run_id, 1)
        agent_id3 = agent_pool._generate_agent_id(run_id, 0)  # Same as first
        
        assert agent_id1 == f"{run_id}:agent:0"
        assert agent_id2 == f"{run_id}:agent:1"
        assert agent_id3 == agent_id1  # Should be deterministic
    
    @pytest.mark.asyncio
    async def test_agent_without_pool_support(self, agent_pool):
        """Test handling agents without pool support (no agent pool)."""
        run_id = "no_pool_run"
        template_agent = MockSidecarMCPAgent()
        
        # Simulate no agent pool support
        agent_pool._agent_pool = None
        
        agent = await agent_pool.get_agent_for_run(run_id, template_agent)
        
        # Should work and create new agent
        assert agent is not template_agent
        assert hasattr(agent, '_pool_agent_id')


class TestGlobalAgentPool:
    """Test global agent pool functionality."""
    
    def test_get_global_agent_pool_creates_instance(self):
        """Test getting global agent pool creates instance."""
        # Reset global instance
        set_global_agent_pool(None)
        
        pool = get_global_agent_pool()
        
        assert isinstance(pool, AgentPool)
        assert pool.max_agents_per_run == 3  # Default
    
    def test_get_global_agent_pool_returns_same(self):
        """Test getting global agent pool returns same instance."""
        # Reset global instance
        set_global_agent_pool(None)
        
        pool1 = get_global_agent_pool()
        pool2 = get_global_agent_pool()
        
        assert pool1 is pool2
    
    def test_set_global_agent_pool_custom(self):
        """Test setting custom global agent pool."""
        custom_pool = AgentPool(max_agents_per_run=5)
        
        set_global_agent_pool(custom_pool)
        
        retrieved_pool = get_global_agent_pool()
        assert retrieved_pool is custom_pool
        assert retrieved_pool.max_agents_per_run == 5


class TestAgentPoolPerformance:
    """Test agent pool performance under load."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_agent_requests(self):
        """Test agent pool under high concurrent load."""
        agent_pool = AgentPool(max_agents_per_run=5)
        template_agent = MockSidecarMCPAgent()
        
        # Create many concurrent runs with agent requests
        async def create_run_with_agents(run_id: str):
            # Each run gets an agent and does some foreach work
            await agent_pool.get_agent_for_run(run_id, template_agent)
            for i in range(3):
                await agent_pool.get_agent_for_foreach_item(run_id, i, template_agent)
        
        # Execute many runs concurrently
        tasks = [
            asyncio.create_task(create_run_with_agents(f"perf_run_{i}"))
            for i in range(50)
        ]
        
        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        elapsed = end_time - start_time
        
        # Verify all runs completed
        stats = await agent_pool.get_pool_stats()
        assert stats["active_runs"] == 50
        assert stats["total_agents"] == 50  # One agent per run
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # Should be fast
        
        # Cleanup
        await agent_pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test agent pool doesn't leak memory with many runs."""
        agent_pool = AgentPool(max_agents_per_run=2)
        template_agent = MockSidecarMCPAgent()
        
        # Create and finish many runs
        for i in range(100):
            run_id = f"memory_run_{i}"
            
            # Create agent
            await agent_pool.get_agent_for_run(run_id, template_agent)
            
            # Finish immediately
            await agent_pool.finish_run(run_id)
        
        # Pool should be clean
        stats = await agent_pool.get_pool_stats()
        assert stats["active_runs"] == 0
        assert stats["total_agents"] == 0
        assert len(agent_pool._run_pools) == 0
        assert len(agent_pool._agent_refs) == 0
    
    @pytest.mark.asyncio
    async def test_agent_reuse_efficiency(self):
        """Test that agent reuse actually reduces agent creation."""
        agent_pool = AgentPool(max_agents_per_run=1)
        
        creation_count = 0
        
        class CountingMockAgent(MockSidecarMCPAgent):
            def __init__(self, *args, **kwargs):
                nonlocal creation_count
                creation_count += 1
                super().__init__(*args, **kwargs)
        
        template_agent = CountingMockAgent()
        run_id = "reuse_run"
        
        # Get agent multiple times
        agents = []
        for _ in range(10):
            agent = await agent_pool.get_agent_for_run(run_id, template_agent)
            agents.append(agent)
        
        # Should have created only limited new agents (template + pool agents)
        assert creation_count <= 3  # Template + up to 2 pool agents
        
        # All agents should be the same instance
        assert all(agent is agents[0] for agent in agents[1:])
        
        await agent_pool.finish_run(run_id)