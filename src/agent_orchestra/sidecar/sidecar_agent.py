"""SidecarMCPAgent - Extended MCPAgent with telemetry and additive policy support.

This module provides an enhanced version of MCPAgent that adds telemetry,
additional policy support, and error handling while maintaining complete
API compatibility with the base MCPAgent.

Typical usage:
    # Create agent with telemetry
    agent = SidecarMCPAgent(
        llm=llm,
        client=client,
        sidecar_telemetry=telemetry_collector,
        sidecar_run_context={"user_id": "user123", "task": "summarize_file"}
    )
    
    # Use like a normal MCPAgent
    result = await agent.run("Analyze this data and provide insights")
    
    # Use streaming with telemetry tracking
    async for chunk in agent.stream_events("Create a summary of the file"):
        process_chunk(chunk)
"""

from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, TypeVar, Type, cast
import logging

from mcp_use import MCPAgent

try:
    from pydantic import BaseModel # pyright: ignore[reportAssignmentType]
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:  # type: ignore
        pass

# Type variable for structured output
T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class SidecarMCPAgent(MCPAgent):  # pyright: ignore[reportInheritanceFromUntyped]
    """
    Extended MCPAgent with telemetry hooks and enhanced policy support.
    
    This class extends the MCPAgent from mcp-use to add telemetry collection,
    enhanced error handling, and additional policy support while maintaining
    complete API compatibility with the base MCPAgent.
    
    Attributes:
        _sidecar_telemetry: Telemetry collector instance
        _sidecar_run_context: Dictionary with context for telemetry and tracing
    """
    """
    Extended MCPAgent that adds telemetry hooks while maintaining complete
    API compatibility and respecting existing agent-level disallowed_tools.
    """
    
    def __init__(
        self,
        *args: Any,
        sidecar_telemetry: Optional[Any] = None,
        sidecar_run_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize SidecarMCPAgent.
        
        Args:
            *args: Arguments passed to MCPAgent
            sidecar_telemetry: Optional telemetry collector  
            sidecar_run_context: Optional context for telemetry (trace_id, etc.)
            **kwargs: Additional arguments passed to MCPAgent
        """
        super().__init__(*args, **kwargs) # pyright: ignore[reportUnknownMemberType]
        
        self._sidecar_telemetry = sidecar_telemetry
        self._sidecar_run_context = sidecar_run_context or {}
        
        # Emit agent initialization telemetry
        self._emit_telemetry("agent_init", {
            "model": getattr(self, 'model', 'unknown'),
            "agent_id": id(self),
            "use_server_manager": getattr(self, 'use_server_manager', False),
            "existing_disallowed_tools": getattr(self, 'disallowed_tools', [])
        })
    
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
        if self._sidecar_telemetry:
            event = { # pyright: ignore[reportUnknownVariableType]
                "event_type": event_type,
                "data": data,
                **self._sidecar_run_context
            }
            try:
                self._sidecar_telemetry.emit(event)
            except Exception as e:
                logger.warning(f"Failed to emit telemetry: {e}")
    
    async def run( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        message: str,
        **kwargs: Any
    ) -> str:
        """
        Run agent with telemetry tracking.
        
        This method extends the parent run method to add telemetry tracking
        for the start, success, and error events of agent execution.
        
        Args:
            message: The message to process
            **kwargs: Additional keyword arguments to pass to the parent run method,
                which may include server_name, history, etc.
            
        Returns:
            str: Agent execution result text
            
        Raises:
            Exception: Any exception that the underlying agent execution might raise
        """
        """
        Run agent with telemetry tracking.
        
        Args:
            message: The message to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Agent execution result
        """
        # Extract parameters for telemetry
        server_name = kwargs.get('server_name')
        
        # Emit run start telemetry
        self._emit_telemetry("agent_run_start", {
            "message_length": len(message),
            "server_name": server_name,
            "use_server_manager": getattr(self, 'use_server_manager', False)
        })
        
        try:
            # Filter out server_name from kwargs before passing to parent
            # since the base MCPAgent doesn't support it
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'server_name'}
            
            # Execute parent run method with filtered kwargs
            result = await super().run(message, **filtered_kwargs)
            
            # Emit success telemetry
            self._emit_telemetry("agent_run_success", {
                "result_type": type(result).__name__,
                "server_name": server_name
            })
            
            return cast(str, result)
            
        except Exception as e:
            # Emit error telemetry
            self._emit_telemetry("agent_run_error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "server_name": server_name
            })
            raise
    
    def stream( # type: ignore
        self,
        query: str,
        max_steps: Optional[int] = None,
        manage_connector: bool = True,
        external_history: Optional[List[Any]] = None,
        track_execution: bool = True,
        output_schema: Optional[Type[T]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """
        Stream agent execution with telemetry mirroring.
        
        This method wraps the parent stream method to add telemetry tracking
        for streaming execution, while maintaining the same interface.
        
        Args:
            query: The query to run
            max_steps: Optional maximum number of steps to take
            manage_connector: Whether to handle the connector lifecycle internally
            external_history: Optional external history to use instead of internal history
            track_execution: Whether to track execution for telemetry
            output_schema: Optional Pydantic BaseModel class for structured output
            **kwargs: Additional keyword arguments to pass to the parent method
            
        Yields:
            Stream chunks from parent method (unchanged)
            
        Raises:
            Exception: Any exception that the underlying streaming might raise
        """
        """
        Stream agent execution with telemetry mirroring.
        
        Args:
            query: The query to run
            max_steps: Optional maximum number of steps to take
            manage_connector: Whether to handle the connector lifecycle internally
            external_history: Optional external history to use instead of internal history
            track_execution: Whether to track execution for telemetry
            output_schema: Optional Pydantic BaseModel class for structured output
            **kwargs: Additional keyword arguments
            
        Yields:
            Stream chunks from parent method (unchanged)
        """
        # Extract server_name for telemetry but don't pass to parent
        server_name = kwargs.get('server_name')
        # Filter out server_name from kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'server_name'}
        
        return self._stream_with_telemetry(
            super().stream(
                query, 
                max_steps=max_steps,
                manage_connector=manage_connector,
                external_history=external_history,
                track_execution=track_execution,
                output_schema=output_schema, # pyright: ignore[reportArgumentType]
                **filtered_kwargs
            ), # pyright: ignore[reportArgumentType]
            "stream",
            query,
            server_name
        )
    
    async def astream(
        self,
        query: str,
        max_steps: Optional[int] = None,
        manage_connector: bool = True,
        external_history: Optional[List[Any]] = None,
        track_execution: bool = True,
        output_schema: Optional[Type[T]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """
        Async stream agent execution with telemetry mirroring.
        
        This method provides an asynchronous interface to stream execution
        with telemetry tracking. It delegates to the _stream_with_telemetry helper.
        
        Args:
            query: The query to run
            max_steps: Optional maximum number of steps to take
            manage_connector: Whether to handle the connector lifecycle internally
            external_history: Optional external history to use instead of internal history
            track_execution: Whether to track execution for telemetry
            output_schema: Optional Pydantic BaseModel class for structured output
            **kwargs: Additional keyword arguments to pass to the parent method
            
        Yields:
            Stream chunks from parent method (unchanged)
            
        Raises:
            Exception: Any exception that the underlying streaming might raise
        """
        """
        Async stream agent execution with telemetry mirroring.
        
        Args:
            query: The query to run
            max_steps: Optional maximum number of steps to take
            manage_connector: Whether to handle the connector lifecycle internally
            external_history: Optional external history to use instead of internal history
            track_execution: Whether to track execution for telemetry
            output_schema: Optional Pydantic BaseModel class for structured output
            **kwargs: Additional keyword arguments
            
        Yields:
            Stream chunks from parent method (unchanged)
        """
        # Extract server_name for telemetry but don't pass to parent
        server_name = kwargs.get('server_name')
        # Filter out server_name from kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'server_name'}
        
        # Use parent's stream method (assuming it's async)
        stream_gen = super().stream(
            query,
            max_steps=max_steps,
            manage_connector=manage_connector,
            external_history=external_history,
            track_execution=track_execution,
            output_schema=output_schema, # pyright: ignore[reportArgumentType]
            **filtered_kwargs
        )
        
        async for chunk in self._stream_with_telemetry(
            stream_gen, # pyright: ignore[reportArgumentType]
            "astream", 
            query,
            server_name
        ):
            yield chunk
    
    async def stream_events( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        max_steps: Optional[int] = None,
        manage_connector: bool = True,
        external_history: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Asynchronous streaming interface that yields string chunks with telemetry.
        
        This method extends the parent stream_events method to add telemetry
        tracking for streaming events.
        
        Args:
            query: The query to run
            max_steps: Optional maximum number of steps to take
            manage_connector: Whether to handle the connector lifecycle internally
            external_history: Optional external history to use instead of internal history
            **kwargs: Additional keyword arguments to pass to the parent method
            
        Yields:
            String chunks from the agent execution
            
        Raises:
            Exception: Any exception that the underlying streaming might raise
        """
        """
        Asynchronous streaming interface that yields string chunks.
        
        Args:
            query: The query to run
            max_steps: Optional maximum number of steps to take
            manage_connector: Whether to handle the connector lifecycle internally
            external_history: Optional external history to use instead of internal history
            **kwargs: Additional keyword arguments
            
        Yields:
            String chunks from the agent execution
        """
        server_name = kwargs.get('server_name')
        # Filter out server_name from kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'server_name'}
        
        # Emit stream events start telemetry
        self._emit_telemetry("agent_stream_events_start", {
            "query_length": len(query),
            "server_name": server_name,
            "max_steps": max_steps,
            "use_server_manager": getattr(self, 'use_server_manager', False)
        })
        
        chunk_count = 0
        try:
            # Use parent's stream_events method
            async for chunk in super().stream_events( # type: ignore
                query=query,
                max_steps=max_steps,
                manage_connector=manage_connector,
                external_history=external_history,
                **filtered_kwargs
            ): # pyright: ignore[reportGeneralTypeIssues]
                chunk_count += 1
                
                # Mirror chunk to telemetry without modification
                self._emit_telemetry("agent_stream_events_chunk", {
                    "chunk_number": chunk_count,
                    "chunk_type": type(chunk).__name__, # type: ignore
                    "server_name": server_name
                })
                
                # Pass through chunk unchanged
                yield chunk
                
            # Emit completion telemetry
            self._emit_telemetry("agent_stream_events_complete", {
                "total_chunks": chunk_count,
                "server_name": server_name
            })
            
        except Exception as e:
            # Emit error telemetry
            self._emit_telemetry("agent_stream_events_error", {
                "chunks_processed": chunk_count,
                "error": str(e),
                "error_type": type(e).__name__,
                "server_name": server_name
            })
            raise
    
    async def _stream_with_telemetry(
        self,
        stream_generator: AsyncGenerator[Any, None],
        method_name: str,
        message: str,
        server_name: Optional[str]
    ) -> AsyncGenerator[Any, None]:
        """
        Wrap streaming generator with telemetry mirroring.
        
        This helper method wraps a streaming generator with telemetry tracking
        for the start, each chunk, completion, and error events.
        
        Args:
            stream_generator: The original stream generator
            method_name: Name of the streaming method being used
            message: Input message that triggered the streaming
            server_name: Optional server name for context
            
        Yields:
            Stream chunks unchanged (pure pass-through)
            
        Raises:
            Exception: Any exception that the underlying streaming might raise
        """
        """
        Wrap streaming generator with telemetry mirroring.
        
        Args:
            stream_generator: The original stream generator
            method_name: Name of the streaming method
            message: Input message
            server_name: Optional server name
            
        Yields:
            Stream chunks unchanged (pure pass-through)
        """
        chunk_count = 0
        
        # Emit stream start telemetry
        self._emit_telemetry("agent_stream_start", {
            "method": method_name,
            "message_length": len(message),
            "server_name": server_name,
            "use_server_manager": getattr(self, 'use_server_manager', False)
        })
        
        try:
            async for chunk in stream_generator:
                chunk_count += 1
                
                # Mirror chunk to telemetry without modification
                self._emit_telemetry("agent_stream_chunk", {
                    "method": method_name,
                    "chunk_number": chunk_count,
                    "chunk_type": type(chunk).__name__,
                    "server_name": server_name
                })
                
                # Pass through chunk unchanged
                yield chunk
                
            # Emit stream completion telemetry
            self._emit_telemetry("agent_stream_complete", {
                "method": method_name,
                "total_chunks": chunk_count,
                "server_name": server_name
            })
            
        except Exception as e:
            # Emit stream error telemetry
            self._emit_telemetry("agent_stream_error", {
                "method": method_name,
                "chunks_processed": chunk_count,
                "error": str(e),
                "error_type": type(e).__name__,
                "server_name": server_name
            })
            raise
    
    def get_sidecar_state(self) -> Dict[str, Any]:
        """
        Get current sidecar configuration state.
        
        This method returns the current sidecar configuration state as a dictionary,
        which can be useful for inspection, logging, or troubleshooting.
        
        Returns:
            Dict[str, Any]: Dictionary containing the current sidecar state including
                telemetry status, run context, agent ID, and other configuration details
        """
        return {
            "telemetry_enabled": self._sidecar_telemetry is not None,
            "run_context": self._sidecar_run_context,
            "agent_id": id(self),
            "use_server_manager": getattr(self, 'use_server_manager', False),
            "existing_disallowed_tools": getattr(self, 'disallowed_tools', [])
        }
    
    async def initialize(self) -> None:
        """Initialize the MCP client and agent with telemetry tracking.
        
        This method extends the parent initialize method to add telemetry
        tracking for initialization start, success, and error events.
        
        Raises:
            Exception: Any exception that might occur during initialization
        """
        # Emit initialization start telemetry
        self._emit_telemetry("agent_initialize_start", {
            "agent_id": id(self)
        })
        
        try:
            await super().initialize()
            
            # Emit initialization success telemetry
            self._emit_telemetry("agent_initialize_success", {
                "agent_id": id(self)
            })
            
        except Exception as e:
            # Emit initialization error telemetry
            self._emit_telemetry("agent_initialize_error", {
                "agent_id": id(self),
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    async def close(self) -> None:
        """Close the MCP connection with telemetry tracking and improved error handling.
        
        This method extends the parent close method to add telemetry tracking
        for close start, success, and error events.
        
        Raises:
            Exception: Any exception that might occur during connection closure
        """
        # Emit close start telemetry
        self._emit_telemetry("agent_close_start", {
            "agent_id": id(self)
        })
        
        try:
            await super().close()
            
            # Emit close success telemetry
            self._emit_telemetry("agent_close_success", {
                "agent_id": id(self)
            })
            
        except Exception as e:
            # Emit close error telemetry
            self._emit_telemetry("agent_close_error", {
                "agent_id": id(self),
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    def get_conversation_history(self) -> List[Any]:
        """Get the current conversation history.
        
        Returns:
            List[Any]: The current conversation history items
        """
        return cast(List[Any], super().get_conversation_history())  # pyright: ignore[reportUnknownReturnType]
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history with telemetry tracking.
        
        This method extends the parent clear_conversation_history method to add
        telemetry tracking when the history is cleared.
        """
        self._emit_telemetry("conversation_history_cleared", {
            "agent_id": id(self)
        })
        super().clear_conversation_history()
    
    def add_to_history(self, message: Any) -> None:
        """Add a message to the conversation history with telemetry tracking.
        
        This method extends the parent add_to_history method to add telemetry
        tracking when a message is added to the history.
        
        Args:
            message: The message to add to the conversation history
        """
        super().add_to_history(message)
        
        self._emit_telemetry("message_added_to_history", {
            "agent_id": id(self),
            "message_type": type(message).__name__
        })
    
    def get_system_message(self) -> Optional[Any]:
        """Get the current system message.
        
        Returns:
            Optional[Any]: The current system message or None if not set
        """
        return cast(Optional[Any], super().get_system_message())  # pyright: ignore[reportUnknownReturnType]
    
    def set_system_message(self, message: str) -> None:
        """Set a new system message with telemetry tracking.
        
        This method extends the parent set_system_message method to add telemetry
        tracking when the system message is updated.
        
        Args:
            message: The new system message to set
        """
        self._emit_telemetry("system_message_updated", {
            "agent_id": id(self),
            "message_length": len(message)
        })
        
        super().set_system_message(message)
    
    def set_disallowed_tools(self, disallowed_tools: List[str]) -> None:
        """Set the list of tools that should not be available to the agent.
        
        This method extends the parent set_disallowed_tools method to add telemetry
        tracking when the disallowed tools list is updated.
        
        Args:
            disallowed_tools: List of tool names that should not be available
        """
        self._emit_telemetry("disallowed_tools_updated", {
            "agent_id": id(self),
            "disallowed_tools": disallowed_tools,
            "count": len(disallowed_tools)
        })
        
        super().set_disallowed_tools(disallowed_tools)
    
    def get_disallowed_tools(self) -> List[str]:
        """Get the list of tools that are not available to the agent.
        
        Returns:
            List[str]: List of tool names that are not available to the agent
        """
        return cast(List[str], super().get_disallowed_tools())  # pyright: ignore[reportUnnecessaryCast, reportUnknownReturnType]