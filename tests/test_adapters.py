"""Tests for adapter system."""

import pytest
from unittest.mock import Mock, AsyncMock

from agent_orchestra.adapters.base import (
    BaseAdapter, AdapterResult, UsageMetrics, Artifact, AdapterRegistry
)


def test_usage_metrics():
    """Test UsageMetrics creation and functionality."""
    metrics = UsageMetrics(tokens=100, cost=2.5, input_tokens=80, output_tokens=20)
    
    assert metrics.tokens == 100
    assert metrics.cost == 2.5
    assert metrics.input_tokens == 80
    assert metrics.output_tokens == 20
    assert metrics.latency_ms > 0  # Should be auto-set


def test_artifact():
    """Test Artifact creation."""
    artifact = Artifact(
        uri="file://test.txt",
        mime_type="text/plain",
        size_bytes=1024,
        sha256="abc123",
        metadata={"source": "test"}
    )
    
    assert artifact.uri == "file://test.txt"
    assert artifact.mime_type == "text/plain"
    assert artifact.size_bytes == 1024
    assert artifact.metadata["source"] == "test"


def test_adapter_result():
    """Test AdapterResult functionality."""
    # Successful result
    success_result = AdapterResult(
        output="Success!",
        usage=UsageMetrics(tokens=50, cost=1.0),
        artifacts=[],
        errors=[]
    )
    
    assert success_result.success is True
    assert success_result.output == "Success!"
    
    # Failed result
    error_result = AdapterResult(
        output=None,
        usage=UsageMetrics(),
        errors=["Something went wrong"]
    )
    
    assert error_result.success is False
    assert len(error_result.errors) == 1


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""
    
    def __init__(self):
        self.calls = []
    
    async def call(self, agent, tool, *, inputs, meta):
        self.calls.append((agent, tool, inputs, meta))
        return AdapterResult(
            output=f"Mock result for {tool}",
            usage=UsageMetrics(tokens=10, cost=0.1)
        )
    
    async def get_available_tools(self, agent):
        return ["mock_tool", "another_tool"]
    
    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_mock_adapter():
    """Test the mock adapter."""
    adapter = MockAdapter()
    
    # Test call
    result = await adapter.call(
        agent="test-agent",
        tool="mock_tool",
        inputs={"param": "value"},
        meta={"trace_id": "123"}
    )
    
    assert result.success is True
    assert "Mock result" in result.output
    assert result.usage.tokens == 10
    
    # Verify call was recorded
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call[0] == "test-agent"
    assert call[1] == "mock_tool"
    assert call[2]["param"] == "value"
    
    # Test tools
    tools = await adapter.get_available_tools("test-agent")
    assert "mock_tool" in tools
    
    # Test health check
    healthy = await adapter.health_check()
    assert healthy is True


def test_adapter_registry():
    """Test AdapterRegistry functionality."""
    registry = AdapterRegistry()
    
    # Register adapter
    mock_adapter = MockAdapter()
    registry.register("mock", mock_adapter)
    
    # Test retrieval
    adapter = registry.get("mock")
    assert adapter is mock_adapter
    
    # Test non-existent adapter
    assert registry.get("nonexistent") is None
    
    # Test listing
    adapters = registry.list_adapters()
    assert "mock" in adapters


def test_adapter_registry_entry_points():
    """Test adapter registry entry point loading."""
    registry = AdapterRegistry()
    
    # This should not crash even if no entry points exist
    registry.load_from_entry_points()
    
    # Should have at least loaded any registered adapters
    # In a real environment, this would load mcp_use adapter
    adapters = registry.list_adapters()
    assert isinstance(adapters, list)


class FailingAdapter(BaseAdapter):
    """Adapter that always fails for testing error handling."""
    
    async def call(self, agent, tool, *, inputs, meta):
        raise Exception("Adapter failure")
    
    async def get_available_tools(self, agent):
        return []
    
    async def health_check(self):
        return False


@pytest.mark.asyncio
async def test_failing_adapter():
    """Test adapter error handling."""
    adapter = FailingAdapter()
    
    # Health check should return False
    healthy = await adapter.health_check()
    assert healthy is False
    
    # Call should raise exception
    with pytest.raises(Exception, match="Adapter failure"):
        await adapter.call("agent", "tool", inputs={}, meta={})


def test_adapter_protocol():
    """Test that adapters implement the required protocol."""
    adapter = MockAdapter()
    
    # Check that all required methods exist
    assert hasattr(adapter, 'call')
    assert hasattr(adapter, 'get_available_tools')
    assert hasattr(adapter, 'health_check')
    
    # Check that they're async
    import inspect
    assert inspect.iscoroutinefunction(adapter.call)
    assert inspect.iscoroutinefunction(adapter.get_available_tools)
    assert inspect.iscoroutinefunction(adapter.health_check)