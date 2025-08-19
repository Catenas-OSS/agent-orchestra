"""Tests for policy enforcement functionality."""

import pytest
from unittest.mock import Mock
from agent_orchestra import SidecarMCPClient, SidecarSession


class MockSession:
    """Mock session for policy testing."""
    
    def __init__(self, server_name="test"):
        self.server_name = server_name
    
    def list_tools(self):
        return [
            {"name": "safe_tool", "description": "A safe tool"},
            {"name": "dangerous_tool", "description": "A dangerous tool"},
            {"name": "file_read", "description": "Read files"},
            {"name": "file_delete", "description": "Delete files"},
            {"name": "web_fetch", "description": "Fetch web content"}
        ]
    
    def call_tool(self, name, arguments=None):
        return f"Result from {name} with {arguments}"


class TestPolicyEnforcement:
    """Test policy enforcement at session level."""
    
    def test_allowlist_policy(self):
        """Test allowlist enforcement."""
        inner_session = MockSession()
        policy = {"allowed_tools": ["safe_tool", "file_read"]}
        
        session = SidecarSession(inner_session, policy=policy)
        
        # Allowed tool should work
        result = session.call_tool("safe_tool", {"param": "value"})
        assert result == "Result from safe_tool with {'param': 'value'}"
        
        # Disallowed tool should raise PermissionError
        with pytest.raises(PermissionError, match="not in allowed tools list"):
            session.call_tool("dangerous_tool")
    
    def test_denylist_policy(self):
        """Test denylist enforcement."""
        inner_session = MockSession()
        policy = {"disallowed_tools": ["dangerous_tool", "file_delete"]}
        
        session = SidecarSession(inner_session, policy=policy)
        
        # Non-blocked tool should work
        result = session.call_tool("safe_tool")
        assert result == "Result from safe_tool with None"
        
        # Blocked tool should raise PermissionError
        with pytest.raises(PermissionError, match="is disallowed by policy"):
            session.call_tool("dangerous_tool")
    
    def test_combined_allowlist_denylist(self):
        """Test combined allowlist and denylist."""
        inner_session = MockSession()
        policy = {
            "allowed_tools": ["safe_tool", "file_read", "dangerous_tool"],
            "disallowed_tools": ["dangerous_tool"]
        }
        
        session = SidecarSession(inner_session, policy=policy)
        
        # Tool in allowlist but not in denylist should work
        result = session.call_tool("safe_tool")
        assert result == "Result from safe_tool with None"
        
        # Tool in allowlist but also in denylist should be blocked
        with pytest.raises(PermissionError, match="is disallowed by policy"):
            session.call_tool("dangerous_tool")
        
        # Tool not in allowlist should be blocked
        with pytest.raises(PermissionError, match="not in allowed tools list"):
            session.call_tool("web_fetch")
    
    def test_list_tools_filtering(self):
        """Test that list_tools applies policy filtering."""
        inner_session = MockSession()
        policy = {"allowed_tools": ["safe_tool", "file_read"]}
        
        session = SidecarSession(inner_session, policy=policy)
        
        # Should only return allowed tools
        tools = session.list_tools()
        tool_names = [tool["name"] for tool in tools]
        
        assert "safe_tool" in tool_names
        assert "file_read" in tool_names
        assert "dangerous_tool" not in tool_names
        assert "file_delete" not in tool_names
        assert "web_fetch" not in tool_names
        assert len(tools) == 2
    
    def test_no_policy_no_filtering(self):
        """Test that without policy, no filtering occurs."""
        inner_session = MockSession()
        session = SidecarSession(inner_session)
        
        # All tools should be available
        tools = session.list_tools()
        assert len(tools) == 5
        
        # All tools should be callable
        result = session.call_tool("dangerous_tool")
        assert result == "Result from dangerous_tool with None"


class TestTelemetryIntegration:
    """Test telemetry integration with policy enforcement."""
    
    def test_telemetry_emitted_on_tool_call(self):
        """Test that telemetry is emitted for tool calls."""
        inner_session = MockSession()
        telemetry = Mock()
        
        session = SidecarSession(
            inner_session, 
            telemetry=telemetry,
            run_context={"trace_id": "test-123"}
        )
        
        # Call a tool
        session.call_tool("safe_tool", {"arg": "value"})
        
        # Check telemetry was called
        assert telemetry.emit.call_count >= 2  # start and success events
        
        # Check event structure
        calls = telemetry.emit.call_args_list
        start_event = calls[0][0][0]
        success_event = calls[1][0][0]
        
        assert start_event["event_type"] == "tool_call_start"
        assert start_event["data"]["tool_name"] == "safe_tool"
        assert start_event["trace_id"] == "test-123"
        
        assert success_event["event_type"] == "tool_call_success"
        assert success_event["data"]["tool_name"] == "safe_tool"
    
    def test_telemetry_emitted_on_policy_error(self):
        """Test that telemetry is emitted when policy blocks a tool."""
        inner_session = MockSession()
        telemetry = Mock()
        policy = {"disallowed_tools": ["dangerous_tool"]}
        
        session = SidecarSession(
            inner_session,
            policy=policy,
            telemetry=telemetry
        )
        
        # Try to call blocked tool
        with pytest.raises(PermissionError):
            session.call_tool("dangerous_tool")
        
        # Should not emit start event since policy check happens first
        assert telemetry.emit.call_count == 0
    
    def test_telemetry_emitted_on_tool_error(self):
        """Test telemetry when tool execution fails."""
        inner_session = Mock()
        inner_session.call_tool.side_effect = ValueError("Tool failed")
        
        telemetry = Mock()
        session = SidecarSession(inner_session, telemetry=telemetry)
        
        # Call tool that will fail
        with pytest.raises(ValueError, match="Tool failed"):
            session.call_tool("failing_tool")
        
        # Check error telemetry was emitted
        assert telemetry.emit.call_count >= 2  # start and error events
        calls = telemetry.emit.call_args_list
        error_event = calls[1][0][0]
        
        assert error_event["event_type"] == "tool_call_error"
        assert error_event["data"]["error"] == "Tool failed"
        assert error_event["data"]["error_type"] == "ValueError"


class TestSessionProxyBehavior:
    """Test that SidecarSession properly proxies all session methods."""
    
    def test_getattr_passthrough(self):
        """Test that unknown attributes are passed through to inner session."""
        inner_session = Mock()
        inner_session.custom_method.return_value = "custom_result"
        inner_session.custom_attribute = "custom_value"
        
        session = SidecarSession(inner_session)
        
        # Method call should be passed through
        result = session.custom_method("arg")
        assert result == "custom_result"
        inner_session.custom_method.assert_called_once_with("arg")
        
        # Attribute access should be passed through
        assert session.custom_attribute == "custom_value"
    
    def test_string_representation(self):
        """Test string representations of SidecarSession."""
        inner_session = MockSession("test_server")
        session = SidecarSession(inner_session)
        
        # Test __str__
        str_repr = str(session)
        assert "SidecarSession" in str_repr
        assert "inner=" in str_repr
        
        # Test __repr__
        repr_str = repr(session)
        assert "SidecarSession" in repr_str
        assert "inner=" in repr_str
        assert "policy=" in repr_str
        assert "telemetry=" in repr_str


class TestClientIntegration:
    """Test policy enforcement through SidecarMCPClient."""
    
    def test_client_passes_policy_to_session(self):
        """Test that client passes policy configuration to sessions."""
        
        class MockMCPClient:
            def __init__(self, *args, **kwargs):
                self.sessions = {}
            
            def create_session(self, server_name, auto_initialize=True):
                return MockSession(server_name)
        
        # Monkey patch
        import agent_orchestra.sidecar_client
        agent_orchestra.sidecar_client.MCPClient = MockMCPClient
        
        policy = {"disallowed_tools": ["dangerous_tool"]}
        client = SidecarMCPClient(policy=policy)
        
        session = client.create_session("test")
        
        # Session should have policy
        assert session._policy == policy
        
        # Policy should be enforced
        with pytest.raises(PermissionError):
            session.call_tool("dangerous_tool")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])