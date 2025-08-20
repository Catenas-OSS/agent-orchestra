"""
Tests for Part 4 backward compatibility - ensuring all new features are optional
and existing code continues to work unchanged.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import Executor
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.orchestrator.call_broker import CallBroker, ModelLimits
from agent_orchestra.orchestrator.agent_pool import AgentPool


class MockSidecarMCPAgent:
    """Mock agent for backward compatibility testing."""
    
    def __init__(self, llm=None, client=None, max_steps=10):
        self.llm = llm or Mock()
        self.client = client or Mock()
        self.max_steps = max_steps
        self._call_count = 0
    
    async def run(self, prompt: str):
        self._call_count += 1
        return f"mock_result_{self._call_count}_{prompt[:10]}"
    
    async def astream(self, prompt: str):
        self._call_count += 1
        yield {"chunk": 1, "content": f"processing_{prompt[:10]}"}
        yield {"output": f"mock_stream_result_{self._call_count}_{prompt[:10]}"}


class TestPart4BackwardCompatibility:
    """Test that Part 4 features don't break existing functionality."""
    
    @pytest.mark.asyncio
    async def test_mcp_executor_without_broker_or_pool(self):
        """Test MCPExecutor works exactly as before without broker/pool."""
        # Create agent and executor WITHOUT Part 4 features
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)  # No broker, no agent_pool
        
        # Verify no Part 4 components
        assert executor._broker is None
        assert executor._agent_pool is None
        assert executor._current_run_id is None
        
        # Create simple node
        node = NodeSpec(
            id="test_node",
            type="task",
            inputs={"instruction": "test task"}
        )
        ctx = {"blackboard": {}}
        
        # Execute should work exactly as before
        result = await executor.execute(node, ctx)
        
        assert "output" in result
        assert "mock_result_1" in result["output"]
        assert agent._call_count == 1
    
    @pytest.mark.asyncio
    async def test_mcp_executor_streaming_without_broker_or_pool(self):
        """Test MCPExecutor streaming works without Part 4 features."""
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)  # No Part 4 features
        
        node = NodeSpec(
            id="stream_node",
            type="task",
            inputs={"instruction": "streaming test"}
        )
        ctx = {"blackboard": {}}
        
        chunks = []
        
        async def collect_chunk(chunk):
            chunks.append(chunk)
        
        result = await executor.execute_with_stream(node, ctx, collect_chunk)
        
        # Should work exactly as before
        assert len(chunks) == 2
        assert chunks[0]["chunk"] == 1
        assert "output" in chunks[1]
        assert "output" in result
        assert agent._call_count == 1
    
    @pytest.mark.asyncio
    async def test_orchestrator_without_part4_features(self):
        """Test full orchestrator workflow without any Part 4 features."""
        # Create executor without Part 4 features
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)
        orchestrator = Orchestrator(executor)
        
        # Define workflow using only Part 1-3 features
        workflow = GraphSpec(
            nodes=[
                NodeSpec(
                    id="task1",
                    type="task",
                    inputs={"instruction": "first task"}
                ),
                NodeSpec(
                    id="task2", 
                    type="task",
                    inputs={"instruction": "second task", "from": "task1"}
                )
            ],
            edges=[("task1", "task2")]
        )
        
        run_spec = RunSpec(
            run_id="compat_run_001",
            goal="Test backward compatibility"
        )
        
        # Execute workflow
        events = []
        async for event in orchestrator.run(workflow, run_spec):
            events.append(event)
        
        # Should work exactly as Parts 1-3
        event_types = [e.type for e in events]
        assert "RUN_START" in event_types
        assert "NODE_START" in event_types
        assert "NODE_COMPLETE" in event_types
        assert "RUN_COMPLETE" in event_types
        
        # Both tasks should have executed
        assert agent._call_count == 2
    
    @pytest.mark.asyncio
    async def test_orchestrator_streaming_without_part4(self):
        """Test orchestrator streaming without Part 4 features."""
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)
        orchestrator = Orchestrator(executor)
        
        workflow = GraphSpec(
            nodes=[
                NodeSpec(
                    id="stream_task",
                    type="task",
                    inputs={"instruction": "streaming task"}
                )
            ],
            edges=[]
        )
        
        run_spec = RunSpec(
            run_id="stream_compat_001",
            goal="Test streaming compatibility"
        )
        
        events = []
        async for event in orchestrator.run_streaming(workflow, run_spec):
            events.append(event)
        
        # Should have AGENT_CHUNK events from streaming
        event_types = [e.type for e in events]
        assert "AGENT_CHUNK" in event_types
        assert "RUN_COMPLETE" in event_types
        
        agent_chunks = [e for e in events if e.type == "AGENT_CHUNK"]
        assert len(agent_chunks) >= 1
        assert agent._call_count == 1
    
    @pytest.mark.asyncio
    async def test_foreach_without_part4_features(self):
        """Test foreach nodes work without agent pool."""
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)  # No agent pool
        orchestrator = Orchestrator(executor)
        
        workflow = GraphSpec(
            nodes=[
                NodeSpec(
                    id="foreach_task",
                    type="foreach",
                    inputs={
                        "items": ["item1", "item2", "item3"],
                        "instruction": "process each item"
                    },
                    concurrency=2
                )
            ],
            edges=[]
        )
        
        run_spec = RunSpec(
            run_id="foreach_compat_001", 
            goal="Test foreach without agent pool"
        )
        
        events = []
        async for event in orchestrator.run(workflow, run_spec):
            events.append(event)
        
        # Should work with regular task execution fallback
        event_types = [e.type for e in events]
        assert "NODE_COMPLETE" in event_types
        
        # Should have processed all items
        assert agent._call_count == 3
    
    @pytest.mark.asyncio
    async def test_mcp_executor_with_only_broker(self):
        """Test MCPExecutor with broker but no agent pool."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent, broker=broker)  # Only broker, no pool
        
        assert executor._broker is broker
        assert executor._agent_pool is None
        
        node = NodeSpec(
            id="broker_only_node",
            type="task", 
            inputs={"instruction": "test with broker"}
        )
        ctx = {"blackboard": {}}
        
        result = await executor.execute(node, ctx)
        
        # Should work with broker rate limiting
        assert "output" in result
        assert agent._call_count == 1
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_mcp_executor_with_only_agent_pool(self):
        """Test MCPExecutor with agent pool but no broker."""
        agent_pool = AgentPool(max_agents_per_run=2)
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent, agent_pool=agent_pool)  # Only pool, no broker
        
        assert executor._broker is None
        assert executor._agent_pool is agent_pool
        
        # Set run context for pool to work
        executor.set_run_context("test_run_001")
        
        node = NodeSpec(
            id="pool_only_node",
            type="task",
            inputs={"instruction": "test with pool"}
        )
        ctx = {"blackboard": {}}
        
        result = await executor.execute(node, ctx)
        
        # Should work with agent pooling
        assert "output" in result
        
        await agent_pool.shutdown()
    
    @pytest.mark.asyncio
    async def test_orchestrator_partial_part4_features(self):
        """Test orchestrator with some but not all Part 4 features."""
        # Only use broker, not agent pool
        broker = CallBroker({}, ModelLimits(rpm=50, rpd=500, max_concurrency=3))
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent, broker=broker)
        orchestrator = Orchestrator(executor)
        
        workflow = GraphSpec(
            nodes=[
                NodeSpec(
                    id="partial_task1",
                    type="task",
                    inputs={"instruction": "first partial task"}
                ),
                NodeSpec(
                    id="partial_foreach",
                    type="foreach",
                    inputs={
                        "items": ["a", "b"], 
                        "instruction": "process items with broker only"
                    },
                    concurrency=1
                )
            ],
            edges=[("partial_task1", "partial_foreach")]
        )
        
        run_spec = RunSpec(
            run_id="partial_part4_001",
            goal="Test partial Part 4 features"
        )
        
        events = []
        async for event in orchestrator.run_streaming(workflow, run_spec):
            events.append(event)
        
        # Should work fine with broker rate limiting
        event_types = [e.type for e in events]
        assert "RUN_COMPLETE" in event_types
        
        # First task + 2 foreach items
        assert agent._call_count == 3
        
        await broker.shutdown()
    
    @pytest.mark.asyncio 
    async def test_non_mcp_executor_unaffected(self):
        """Test that non-MCP executors are completely unaffected."""
        # Create a basic executor (no MCP, no Part 4 features)
        class SimpleExecutor(Executor):
            def __init__(self):
                self.call_count = 0
            
            async def execute(self, node, ctx):
                self.call_count += 1
                return {"output": f"simple_result_{self.call_count}"}
        
        executor = SimpleExecutor()
        orchestrator = Orchestrator(executor)
        
        # Should not have any Part 4 methods or attributes
        assert not hasattr(executor, '_broker')
        assert not hasattr(executor, '_agent_pool')
        assert not hasattr(executor, 'set_run_context')
        assert not hasattr(executor, 'execute_foreach_item')
        
        workflow = GraphSpec(
            nodes=[
                NodeSpec(
                    id="simple_task",
                    type="task",
                    inputs={"instruction": "simple test"}
                )
            ],
            edges=[]
        )
        
        run_spec = RunSpec(
            run_id="simple_executor_001",
            goal="Test non-MCP executor"
        )
        
        events = []
        async for event in orchestrator.run(workflow, run_spec):
            events.append(event)
        
        # Should work exactly as before
        assert executor.call_count == 1
        event_types = [e.type for e in events]
        assert "RUN_COMPLETE" in event_types
    
    def test_import_compatibility(self):
        """Test that all existing imports still work."""
        # These imports should work exactly as before Part 4
        from agent_orchestra.orchestrator import (
            Orchestrator,
            Event, 
            EventType,
            NodeSpec,
            GraphSpec,
            RunSpec,
            Executor,
            CallableExecutor,
            MCPExecutor,
            topo_sort
        )
        
        # All should be available
        assert Orchestrator is not None
        assert Event is not None
        assert EventType is not None
        assert NodeSpec is not None
        assert GraphSpec is not None
        assert RunSpec is not None
        assert Executor is not None
        assert CallableExecutor is not None
        assert MCPExecutor is not None
        assert topo_sort is not None
    
    def test_new_part4_imports_available(self):
        """Test that new Part 4 imports are available but optional."""
        # These should be available for users who want Part 4 features
        from agent_orchestra.orchestrator import (
            CallBroker,
            ModelLimits,
            AgentPool,
            BrokerConfig,
            create_broker_from_config,
            create_development_broker,
            create_production_broker,
        )
        
        # All new features should be available
        assert CallBroker is not None
        assert ModelLimits is not None
        assert AgentPool is not None
        assert BrokerConfig is not None
        assert create_broker_from_config is not None
        assert create_development_broker is not None
        assert create_production_broker is not None
    
    @pytest.mark.asyncio
    async def test_part4_components_optional_cleanup(self):
        """Test that Part 4 components clean up gracefully when not used."""
        # Create executor with Part 4 components
        broker = CallBroker({}, ModelLimits(rpm=10, rpd=100, max_concurrency=2))
        agent_pool = AgentPool(max_agents_per_run=1)
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent, broker=broker, agent_pool=agent_pool)
        
        # Use the executor
        executor.set_run_context("cleanup_test_001")
        
        node = NodeSpec(id="cleanup_node", type="task", inputs={"instruction": "test"})
        ctx = {"blackboard": {}}
        
        await executor.execute(node, ctx)
        
        # Cleanup should work without errors
        await broker.shutdown()
        await agent_pool.shutdown()
        
        # Should not affect the core functionality
        assert agent._call_count == 1


class TestMCPExecutorEnhancements:
    """Test specific MCPExecutor enhancements are backward compatible."""
    
    @pytest.mark.asyncio
    async def test_set_run_context_optional(self):
        """Test that set_run_context is optional and safe to ignore."""
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)
        
        # Should work fine without ever calling set_run_context
        node = NodeSpec(id="no_context", type="task", inputs={"instruction": "test"})
        ctx = {"blackboard": {}}
        
        result = await executor.execute(node, ctx)
        assert "output" in result
        
        # Calling set_run_context should be safe even without agent pool
        executor.set_run_context("safe_context_001")
        
        result2 = await executor.execute(node, ctx)
        assert "output" in result2
    
    @pytest.mark.asyncio
    async def test_foreach_fallback_behavior(self):
        """Test that foreach execution falls back gracefully."""
        agent = MockSidecarMCPAgent()
        executor = MCPExecutor(agent)  # No agent pool
        
        # Direct foreach item execution should use regular execution path
        node = NodeSpec(
            id="fallback_foreach_item", 
            type="task",
            inputs={"item": "test_item", "instruction": "process item"}
        )
        ctx = {"blackboard": {}}
        
        # This should work even though execute_foreach_item method exists
        result = await executor.execute(node, ctx)
        assert "output" in result
        assert agent._call_count == 1
    
    def test_executor_initialization_signatures(self):
        """Test that MCPExecutor initialization is backward compatible."""
        agent = MockSidecarMCPAgent()
        
        # Original signature should still work
        executor1 = MCPExecutor(agent)
        assert executor1._template_agent is agent
        assert executor1._broker is None
        assert executor1._agent_pool is None
        
        # With default_server (existing parameter)
        executor2 = MCPExecutor(agent, default_server="test_server")
        assert executor2._default_server == "test_server"
        assert executor2._broker is None
        assert executor2._agent_pool is None
        
        # All parameters should work
        broker = CallBroker({}, ModelLimits())
        pool = AgentPool()
        executor3 = MCPExecutor(agent, "test_server", broker, pool)
        assert executor3._default_server == "test_server"
        assert executor3._broker is broker
        assert executor3._agent_pool is pool
    
    @pytest.mark.asyncio
    async def test_model_detection_graceful_fallback(self):
        """Test that model detection fails gracefully."""
        # Agent without model attributes
        agent = Mock()
        executor = MCPExecutor(agent)
        
        # Should default to unknown model
        assert executor._model_name == "unknown:default"
        
        # Should still work for execution (though agent is mock)
        assert executor._broker is None  # No broker to fail on model detection