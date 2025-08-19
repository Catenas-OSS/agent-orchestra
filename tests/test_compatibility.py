"""Golden tests for ensuring API compatibility with mcp-use."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from agent_orchestra import SidecarMCPClient, SidecarMCPAgent, SidecarLangChainAdapter


class MockMCPClient:
    """Mock MCPClient for testing compatibility."""
    
    def __init__(self, *args, **kwargs):
        self.sessions = {}
        self.debug = kwargs.get('debug', False)
    
    def create_session(self, server_name, auto_initialize=True):
        session = MockSession(server_name)
        self.sessions[server_name] = session
        return session
    
    def create_all_sessions(self, auto_initialize=True):
        # Mock implementation
        return self.sessions
    
    def get_session(self, server_name):
        return self.sessions.get(server_name)
    
    def close_all_sessions(self):
        self.sessions.clear()
    
    @classmethod
    def from_dict(cls, config_dict, debug=False):
        return cls(debug=debug)
    
    @classmethod
    def from_config_file(cls, path, debug=False):
        return cls(debug=debug)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockSession:
    """Mock session for testing."""
    
    def __init__(self, server_name):
        self.server_name = server_name
    
    def list_tools(self):
        return [
            {"name": "test_tool", "description": "A test tool"},
            {"name": "another_tool", "description": "Another test tool"}
        ]
    
    def call_tool(self, name, arguments=None):
        return f"Mock result from {name}"


class MockMCPAgent:
    """Mock MCPAgent for testing compatibility."""
    
    def __init__(self, *args, **kwargs):
        self.client = kwargs.get('client')
        self.disallowed_tools = kwargs.get('disallowed_tools', [])
        self.use_server_manager = kwargs.get('use_server_manager', False)
    
    def run(self, message, *args, **kwargs):
        return f"Mock result for: {message}"
    
    def stream(self, message, *args, **kwargs):
        async def mock_stream():
            yield f"Stream chunk 1 for: {message}"
            yield f"Stream chunk 2 for: {message}"
        return mock_stream()
    
    async def astream(self, message, *args, **kwargs):
        yield f"Async stream chunk 1 for: {message}"
        yield f"Async stream chunk 2 for: {message}"


class MockLangChainAdapter:
    """Mock LangChain adapter for testing."""
    
    def create_tools(self, client, allowed_tools=None, disallowed_tools=None):
        tools = []
        for session in client.sessions.values():
            for tool_info in session.list_tools():
                tool = Mock()
                tool.name = tool_info["name"]
                tool.description = tool_info["description"]
                tools.append(tool)
        return tools
    
    def _convert_tool(self, tool_info, server_name):
        tool = Mock()
        tool.name = tool_info["name"]
        tool.description = tool_info["description"]
        return tool


# Monkey patch for tests
import agent_orchestra.sidecar_client
import agent_orchestra.sidecar_agent  
import agent_orchestra.sidecar_adapter

agent_orchestra.sidecar_client.MCPClient = MockMCPClient
agent_orchestra.sidecar_agent.MCPAgent = MockMCPAgent
agent_orchestra.sidecar_adapter.LangChainAdapter = MockLangChainAdapter


class TestSidecarMCPClientCompatibility:
    """Test SidecarMCPClient maintains MCPClient compatibility."""
    
    def test_inheritance(self):
        """Test that SidecarMCPClient is an instance of MCPClient."""
        client = SidecarMCPClient()
        assert isinstance(client, MockMCPClient)
    
    def test_factory_methods(self):
        """Test factory methods maintain same signatures."""
        config = {"mcpServers": {"test": {"command": "echo"}}}
        
        # Test from_dict
        client1 = SidecarMCPClient.from_dict(config, debug=True)
        assert client1.debug is True
        
        # Test from_dict with sidecar kwargs
        client2 = SidecarMCPClient.from_dict(
            config, 
            debug=False,
            policy={"allowed_tools": ["test"]},
            telemetry=Mock()
        )
        assert client2._sidecar_policy == {"allowed_tools": ["test"]}
        assert client2._sidecar_telemetry is not None
    
    def test_session_lifecycle(self):
        """Test session lifecycle methods work identically."""
        client = SidecarMCPClient()
        
        # Test create_session with default args
        session = client.create_session("test")
        assert session._inner_session.server_name == "test"
        assert "test" in client.sessions
        
        # Test create_session with explicit args
        session2 = client.create_session("test2", auto_initialize=False)
        assert session2._inner_session.server_name == "test2"
        
        # Test get_session
        retrieved = client.get_session("test")
        assert retrieved is not None
        
        # Test sessions dict interface compatibility
        assert isinstance(client.sessions, dict)
        assert len(client.sessions) == 2
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager behavior is preserved."""
        client = SidecarMCPClient()
        
        async with client as c:
            assert c is client
            session = c.create_session("test")
            assert session is not None
    
    def test_golden_behavior_no_sidecar(self):
        """Test that without sidecar options, behavior is identical."""
        config = {"mcpServers": {"test": {"command": "echo"}}}
        
        # Create without sidecar options
        client = SidecarMCPClient.from_dict(config)
        session = client.create_session("test")
        
        # Should behave identically to base MCPClient
        tools = session.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "test_tool"
        
        result = session.call_tool("test_tool")
        assert result == "Mock result from test_tool"


class TestSidecarMCPAgentCompatibility:
    """Test SidecarMCPAgent maintains MCPAgent compatibility."""
    
    def test_inheritance(self):
        """Test that SidecarMCPAgent is an instance of MCPAgent."""
        client = SidecarMCPClient()
        agent = SidecarMCPAgent(client=client)
        assert isinstance(agent, MockMCPAgent)
    
    def test_run_method_signature(self):
        """Test run method maintains same signature and behavior."""
        client = SidecarMCPClient()
        agent = SidecarMCPAgent(client=client)
        
        result = agent.run("test message")
        assert result == "Mock result for: test message"
        
        # Test with additional args/kwargs
        result2 = agent.run("test", server_name="specific")
        assert result2 == "Mock result for: test"
    
    @pytest.mark.asyncio 
    async def test_streaming_methods(self):
        """Test streaming methods maintain compatibility."""
        client = SidecarMCPClient()
        agent = SidecarMCPAgent(client=client)
        
        # Test astream
        chunks = []
        async for chunk in agent.astream("test message"):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert "test message" in chunks[0]
    
    def test_server_manager_compatibility(self):
        """Test use_server_manager and server_name override work."""
        client = SidecarMCPClient()
        
        # Test with use_server_manager
        agent = SidecarMCPAgent(client=client, use_server_manager=True)
        assert agent.use_server_manager is True
        
        # Test server_name override in run
        result = agent.run("test", server_name="override")
        assert result == "Mock result for: test"
    
    def test_additive_disallowed_tools(self):
        """Test that agent-level disallowed_tools are preserved."""
        client = SidecarMCPClient()
        agent = SidecarMCPAgent(
            client=client,
            disallowed_tools=["dangerous_tool"]
        )
        
        assert agent.disallowed_tools == ["dangerous_tool"]
        
        # Session-level policy should be additive, not overriding
        session = client.create_session("test")
        session._disallowed_tools = ["session_blocked_tool"]
        
        # Both should be effective (this would be tested in integration tests)
        state = agent.get_sidecar_state()
        assert state["existing_disallowed_tools"] == ["dangerous_tool"]


class TestSidecarLangChainAdapterCompatibility:
    """Test SidecarLangChainAdapter maintains LangChainAdapter compatibility."""
    
    def test_inheritance(self):
        """Test adapter inheritance."""
        adapter = SidecarLangChainAdapter()
        assert isinstance(adapter, MockLangChainAdapter)
    
    def test_create_tools_signature(self):
        """Test create_tools maintains exact signature."""
        client = SidecarMCPClient()
        client.create_session("test")
        
        adapter = SidecarLangChainAdapter()
        
        # Test without filters
        tools = adapter.create_tools(client)
        assert len(tools) == 2
        
        # Test with allowed_tools
        tools_filtered = adapter.create_tools(
            client, 
            allowed_tools=["test_tool"]
        )
        assert len(tools_filtered) <= len(tools)
        
        # Test with disallowed_tools
        tools_filtered2 = adapter.create_tools(
            client,
            disallowed_tools=["another_tool"]
        )
        assert len(tools_filtered2) <= len(tools)
    
    def test_metadata_attachment(self):
        """Test that Sidecar metadata is attached without breaking compatibility."""
        client = SidecarMCPClient()
        client.create_session("test")
        
        adapter = SidecarLangChainAdapter()
        tools = adapter.create_tools(client)
        
        # Tools should have Sidecar metadata
        for tool in tools:
            metadata = adapter.get_tool_metadata(tool)
            assert "_sidecar_schema_hash" in metadata
            assert "_sidecar_server_origin" in metadata
            assert "_sidecar_safety_flags" in metadata


class TestGoldenBehaviorTests:
    """Tests that ensure identical behavior when sidecar features are disabled."""
    
    def test_identical_output_no_telemetry(self):
        """Test that without telemetry, output is identical to base classes."""
        config = {"mcpServers": {"test": {"command": "echo"}}}
        
        # Sidecar client without any sidecar features
        sidecar_client = SidecarMCPClient.from_dict(config)
        sidecar_session = sidecar_client.create_session("test")
        
        # Should produce identical results
        tools = sidecar_session.list_tools()
        result = sidecar_session.call_tool("test_tool")
        
        assert len(tools) == 2
        assert result == "Mock result from test_tool"
    
    @pytest.mark.asyncio
    async def test_streaming_passthrough_identical(self):
        """Test that streaming passes through identically without modification."""
        client = SidecarMCPClient()
        agent = SidecarMCPAgent(client=client)
        
        # Collect chunks
        chunks = []
        async for chunk in agent.astream("test"):
            chunks.append(chunk)
        
        # Should be identical to what MockMCPAgent.astream produces
        expected_chunks = [
            "Async stream chunk 1 for: test",
            "Async stream chunk 2 for: test"
        ]
        assert chunks == expected_chunks
    
    def test_create_all_sessions_default_behavior(self):
        """Test that create_all_sessions maintains default behavior."""
        client = SidecarMCPClient()
        
        # Should work with default parameters
        sessions = client.create_all_sessions(auto_initialize=True)
        assert isinstance(sessions, dict)
        
        # Should work without parameters
        sessions2 = client.create_all_sessions()
        assert isinstance(sessions2, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])