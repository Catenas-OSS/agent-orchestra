"""SidecarSession - Proxy wrapper for MCP sessions with policy enforcement and telemetry.

This module provides a transparent proxy wrapper around MCPSession objects that adds
policy enforcement and telemetry features while maintaining complete interface
compatibility with the original MCPSession interface.

Typical usage:
    session = SidecarSession(
        inner_session=original_mcp_session,
        policy={"disallowed_tools": ["system_exec"]},
        telemetry=telemetry_collector
    )
    
    # Use exactly like a regular MCPSession
    tools = session.list_tools()
    result = session.call_tool("tool_name", {"arg": "value"})
"""

from typing import Any, Dict, List, Optional, cast
import logging

logger = logging.getLogger(__name__)


class SidecarSession:
    """
    Proxy wrapper around MCPSession that adds policy enforcement and telemetry.
    
    This class wraps an MCPSession object and adds policy enforcement for tool access
    and telemetry collection while maintaining complete interface compatibility with
    the original MCPSession class. It uses the proxy pattern to transparently intercept
    method calls and add the additional functionality.
    
    Attributes:
        _inner_session: The original MCPSession being wrapped
        _policy: Dictionary with policy configuration (allowed/disallowed tools)
        _telemetry: Optional telemetry collector object with emit() method
        _run_context: Dictionary with context for telemetry events
        _allowed_tools: List of explicitly allowed tool names or None
        _disallowed_tools: List of explicitly disallowed tool names
    """
    
    def __init__(
        self, 
        inner_session: Any,
        policy: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Any] = None,
        run_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize SidecarSession proxy.
        
        Args:
            inner_session: The original MCPSession to wrap. Must implement the standard
                MCPSession interface including list_tools() and call_tool() methods.
            policy: Optional policy configuration with allowlist/denylist. Dictionary that
                can contain 'allowed_tools' and/or 'disallowed_tools' keys with lists of
                tool names as values.
            telemetry: Optional telemetry collector that implements an emit(event) method.
            run_context: Optional context for telemetry (trace_id, etc.) that will be
                included in all telemetry events.  
                
        Raises:
            TypeError: If inner_session does not implement the required methods
            ValueError: If policy contains invalid configuration
        """
        self._inner_session = inner_session
        self._policy = policy or {}
        self._telemetry = telemetry
        self._run_context = run_context or {}
        
        # Extract policy rules
        self._allowed_tools = self._policy.get("allowed_tools")
        self._disallowed_tools = self._policy.get("disallowed_tools", [])
        
    def _check_tool_policy(self, tool_name: str) -> None:
        """
        Check if tool usage is allowed by current policy configuration.
        
        This method implements the policy enforcement logic, checking if the
        requested tool is allowed according to the configured policy rules.
        
        Args:
            tool_name: Name of the tool to check against the policy
            
        Raises:
            PermissionError: If tool is not allowed by the current policy configuration,
                either because it's not in the allowlist (if one exists) or because
                it's explicitly in the denylist.
        """
        # If allowlist exists, tool must be in it
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            raise PermissionError(f"Tool '{tool_name}' not in allowed tools list")
            
        # Tool must not be in denylist
        if tool_name in self._disallowed_tools:
            raise PermissionError(f"Tool '{tool_name}' is disallowed by policy")
    
    def _emit_telemetry(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit telemetry event if telemetry collector is configured.
        
        This method sends an event to the telemetry collector if one is configured.
        It enriches the event with the run context and handles errors during emission.
        
        Args:
            event_type: String identifier for the event type
            data: Dictionary containing event-specific data
            
        Note:
            If telemetry emission fails, the error is logged but not propagated.
        """
        if self._telemetry:
            event = { # pyright: ignore[reportUnknownVariableType]
                "event_type": event_type,
                "data": data,
                **self._run_context
            }
            try:
                self._telemetry.emit(event)
            except Exception as e:
                logger.warning(f"Failed to emit telemetry: {e}")
    
    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        Call a tool with policy enforcement and telemetry tracking.
        
        This method intercepts tool calls, applies policy enforcement, emits telemetry
        events, and then delegates to the inner session's call_tool method.
        
        Args:
            name: Tool name to call
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            Any: Tool execution result, with type depending on the specific tool
            
        Raises:
            PermissionError: If tool is not allowed by the current policy
            Exception: Any exception that the underlying tool might raise
        """
        # Policy enforcement (additive to agent-level disallowed_tools)
        self._check_tool_policy(name)
        
        # Emit telemetry before tool call
        self._emit_telemetry("tool_call_start", {
            "tool_name": name,
            "arguments": arguments,
            "server": getattr(self._inner_session, 'server_name', 'unknown')
        })
        
        try:
            # Forward to inner session
            result = self._inner_session.call_tool(name, arguments)
            
            # Emit success telemetry
            self._emit_telemetry("tool_call_success", {
                "tool_name": name,
                "result_type": type(result).__name__
            })
            
            return result
            
        except Exception as e:
            # Emit error telemetry
            self._emit_telemetry("tool_call_error", {
                "tool_name": name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools, applying policy filters.
        
        This method intercepts list_tools calls, applies the configured policy filters
        to the tool list, emits telemetry, and returns the filtered list of tools.
        
        Returns:
            List of tool information dictionaries, filtered according to policy.
            Each dictionary typically contains 'name', 'description', and 'parameters' keys.
        """
        tools = self._inner_session.list_tools()
        
        # Apply policy filtering to tool list
        if self._allowed_tools is not None or self._disallowed_tools:
            filtered_tools: List[Dict[str, Any]] = []
            for tool in tools:
                tool_name = tool.get('name', '')
                
                # Check allowlist
                if self._allowed_tools is not None and tool_name not in self._allowed_tools:
                    continue
                    
                # Check denylist  
                if tool_name in self._disallowed_tools:
                    continue
                    
                filtered_tools.append(tool)
            
            tools = filtered_tools
        
        # Emit telemetry
        self._emit_telemetry("tools_listed", {
            "tool_count": len(tools),
            "filtered": len(tools) != len(self._inner_session.list_tools())
        })
        
        return cast(List[Dict[str, Any]], tools)
    
    def __getattr__(self, name: str) -> Any:
        """
        Forward all other attribute access to the inner session.
        
        This method implements the proxy pattern, ensuring complete interface
        compatibility with the inner session. Any methods or attributes not
        explicitly overridden in this class will be delegated to the inner session.
        
        Args:
            name: The attribute name being accessed
            
        Returns:
            The attribute value from the inner session
            
        Raises:
            AttributeError: If the inner session doesn't have the requested attribute
        """
        return getattr(self._inner_session, name)
    
    def __str__(self) -> str:
        """Return string representation of this SidecarSession.
        
        Returns:
            A string representation that includes the inner session
        """
        return f"SidecarSession(inner={self._inner_session})"
    
    def __repr__(self) -> str:
        """Return detailed representation of this SidecarSession.
        
        Returns:
            A detailed string representation that includes the inner session,
            policy configuration, and telemetry status
        """
        return (f"SidecarSession(inner={self._inner_session!r}, "
                f"policy={self._policy}, telemetry={self._telemetry is not None})")