"""SidecarMCPClient - Extended MCPClient with state management and sidecar configuration.

This module provides an enhanced version of MCPClient that adds telemetry,
policy enforcement, and additional configuration options while maintaining
complete API compatibility with the base MCPClient.

Typical usage:
    # Create with configuration file
    client = SidecarMCPClient.from_config_file(
        "config.json",
        telemetry=telemetry_collector,
        policy={"disallowed_tools": ["file_delete"]}
    )
    
    # Create from configuration dictionary
    client = SidecarMCPClient.from_dict(
        config_dict,
        run_context={"session_id": "abc123", "user": "user123"}
    )
    
    # Use like a normal MCPClient
    session = await client.create_session("server_name")
"""

from typing import Any, Dict, List, Optional, Union,TypeVar

# Type variable for covariant session type
T_Session = TypeVar('T_Session', covariant=True)

# Forward reference for return type annotations
SidecarSession_T = TypeVar('SidecarSession_T', bound='SidecarSession')
import json
import logging
from pathlib import Path

from mcp_use import MCPClient

from .sidecar_session import SidecarSession
from .types import SandboxOptions

logger = logging.getLogger(__name__)


class SidecarMCPClient(MCPClient):  # pyright: ignore[reportInheritanceFromUntyped]
    """
    Extended MCPClient that adds sidecar functionality with complete API compatibility.
    
    This class extends the MCPClient from mcp-use to add sidecar functionality including
    telemetry collection, policy enforcement, and additional configuration options.
    It maintains complete API compatibility with the base MCPClient while adding
    these enhanced features.
    
    Attributes:
        sessions: Dictionary mapping server names to session instances
        _sidecar_policy: Policy configuration dictionary for tool access control
        _sidecar_needs: List of requirement tags for agent capability specification
        _sidecar_hints: Dictionary of preference hints for agent behavior
        _sidecar_telemetry: Telemetry collector instance
        _sidecar_cache_ttl_s: Cache time-to-live in seconds
        _sidecar_run_context: Dictionary with context for telemetry and tracing
    """
    
    def __init__(
        self,
        *args: Any,
        policy: Optional[Dict[str, Any]] = None,
        needs: Optional[List[str]] = None, 
        hints: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Any] = None,
        cache_ttl_s: Optional[int] = None,
        run_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize SidecarMCPClient.
        
        Args:
            *args: Arguments passed to MCPClient
            policy: Optional policy configuration (allow/deny tools)
            needs: Optional list of requirement tags
            hints: Optional preference hints
            telemetry: Optional telemetry collector
            cache_ttl_s: Optional cache TTL in seconds
            run_context: Optional context for telemetry (trace_id, etc.)
            **kwargs: Additional arguments passed to MCPClient
        """
        super().__init__(*args, **kwargs)
        
        # Sidecar state
        self._sidecar_policy = policy
        self._sidecar_needs = needs
        self._sidecar_hints = hints
        self._sidecar_telemetry = telemetry
        self._sidecar_cache_ttl_s = cache_ttl_s
        self._sidecar_run_context = run_context or {}
        
        # Emit initialization telemetry
        self._emit_telemetry("client_init", {
            "needs": self._sidecar_needs,
            "hints": self._sidecar_hints,
            "policy_enabled": self._sidecar_policy is not None,
            "cache_ttl_s": self._sidecar_cache_ttl_s
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
            event = { # type: ignore
                "event_type": event_type,
                "data": data,
                **self._sidecar_run_context
            }
            try:
                self._sidecar_telemetry.emit(event)
            except Exception as e:
                logger.warning(f"Failed to emit telemetry: {e}")
    
    async def create_session(self, server_name: str, auto_initialize: bool = True) -> Any:  # type: ignore
        """
        Create a session wrapped in SidecarSession proxy.
        
        This method extends the parent create_session method to wrap the original
        MCPSession in a SidecarSession proxy that adds policy enforcement and
        telemetry collection while maintaining the same interface.
        
        Args:
            server_name: Name of the MCP server to create a session for
            auto_initialize: Whether to automatically initialize the session
            
        Returns:
            SidecarSession: A proxy wrapping the original session with added functionality
            
        Raises:
            ValueError: If the server name is not found in the configuration
            ConnectionError: If the connection to the server fails
        """
        # Create original session via parent
        original_session = await super().create_session(server_name, auto_initialize)
        
        # Wrap in SidecarSession proxy
        sidecar_session = SidecarSession(
            inner_session=original_session,
            policy=self._sidecar_policy,
            telemetry=self._sidecar_telemetry,
            run_context=self._sidecar_run_context
        )
        
        # Update sessions dict to maintain interface compatibility
        # Note: self.sessions must remain Dict[str, MCPSession] interface
        self.sessions[server_name] = sidecar_session # type: ignore
        
        self._emit_telemetry("session_created", {
            "server_name": server_name,
            "auto_initialize": auto_initialize
        })
        
        return sidecar_session
    
    async def create_all_sessions(self, auto_initialize: bool = True) -> Dict[str, Any]:  # type: ignore
        """
        Create sessions for all configured servers.
        
        This method creates a session for each configured server, wrapping each one
        in a SidecarSession proxy. It reuses existing sessions if they already exist.
        
        Args:
            auto_initialize: Whether to automatically initialize the sessions
            
        Returns:
            Dict[str, SidecarSession]: Dictionary mapping server names to SidecarSession instances
            
        Raises:
            ConnectionError: If connection to any server fails
        """
        sessions = {}
        for server_name in self.get_server_names():
            if server_name not in self.sessions: # type: ignore
                sessions[server_name] = await self.create_session(server_name, auto_initialize)
            else:
                sessions[server_name] = self.sessions[server_name] # type: ignore
        
        self._emit_telemetry("all_sessions_created", {
            "session_count": len(sessions), # type: ignore
            "auto_initialize": auto_initialize
        })
        
        return sessions # type: ignore
    
    @classmethod
    def from_dict( # pyright: ignore[reportIncompatibleMethodOverride]
        cls, 
        config_dict: Dict[str, Any], 
        sandbox: bool = False,
        sandbox_options: Optional[SandboxOptions] = None,
        sampling_callback: Optional[Any] = None,
        elicitation_callback: Optional[Any] = None,
        message_handler: Optional[Any] = None,
        logging_callback: Optional[Any] = None,
        **kwargs: Any
    ) -> "SidecarMCPClient":
        """
        Create SidecarMCPClient from configuration dictionary.
        
        This factory method creates a SidecarMCPClient instance from a configuration
        dictionary, extracting sidecar-specific configuration and passing the rest
        to the parent class constructor.
        
        Args:
            config_dict: Configuration dictionary with server configurations and
                optional 'sidecar' section
            sandbox: Whether to use sandboxed execution mode for running MCP servers
            sandbox_options: Optional sandbox configuration options
            sampling_callback: Optional sampling callback function
            elicitation_callback: Optional elicitation callback function
            message_handler: Optional message handler function
            logging_callback: Optional logging callback function
            **kwargs: Additional sidecar options (policy, needs, hints, etc.)
                which take precedence over those in the config_dict
            
        Returns:
            SidecarMCPClient: Instance configured according to the provided options
            
        Raises:
            ValueError: If the configuration is invalid
        """
        """
        Create SidecarMCPClient from configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            sandbox: Whether to use sandboxed execution mode for running MCP servers
            sandbox_options: Optional sandbox configuration options
            sampling_callback: Optional sampling callback function
            elicitation_callback: Optional elicitation callback function
            message_handler: Optional message handler function
            logging_callback: Optional logging callback function
            **kwargs: Additional sidecar options (policy, needs, hints, etc.)
            
        Returns:
            SidecarMCPClient instance
        """
        # Extract sidecar config if present
        sidecar_config = config_dict.get("sidecar", {})
        
        # Merge sidecar config with kwargs (kwargs take precedence)
        sidecar_kwargs = {
            "policy": sidecar_config.get("policy"),
            "needs": sidecar_config.get("needs"),
            "hints": sidecar_config.get("hints"), 
            "telemetry": sidecar_config.get("telemetry"),
            "cache_ttl_s": sidecar_config.get("cache_ttl_s"),
            "run_context": sidecar_config.get("run_context"),
            **kwargs
        }
        
        # Remove None values
        sidecar_kwargs = {k: v for k, v in sidecar_kwargs.items() if v is not None}
        
        # Create a clean config dict without sidecar section for parent class
        clean_config = {k: v for k, v in config_dict.items() if k != "sidecar"}
        
        # Create SidecarMCPClient with same parameters as parent
        return cls(
            config=clean_config,
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
            message_handler=message_handler,
            logging_callback=logging_callback,
            **sidecar_kwargs
        )
    
    @classmethod  
    def from_config_file( # type: ignore
        cls,
        path: Union[str, Path],
        sandbox: bool = False,
        sandbox_options: Optional[SandboxOptions] = None,
        sampling_callback: Optional[Any] = None,
        elicitation_callback: Optional[Any] = None,
        message_handler: Optional[Any] = None,
        logging_callback: Optional[Any] = None,
        **kwargs: Any
    ) -> "SidecarMCPClient":
        """
        Create SidecarMCPClient from configuration file.
        
        This factory method loads a configuration file and delegates to from_dict
        to create a SidecarMCPClient instance.
        
        Args:
            path: Path to the configuration file (JSON format)
            sandbox: Whether to use sandboxed execution mode for running MCP servers
            sandbox_options: Optional sandbox configuration options
            sampling_callback: Optional sampling callback function
            elicitation_callback: Optional elicitation callback function
            message_handler: Optional message handler function
            logging_callback: Optional logging callback function
            **kwargs: Additional sidecar options (policy, needs, hints, etc.)
                which take precedence over those in the config file
            
        Returns:
            SidecarMCPClient: Instance configured according to the provided file
            
        Raises:
            FileNotFoundError: If the configuration file does not exist
            ValueError: If the configuration file contains invalid JSON
            ValueError: If the configuration is invalid
        """
        """
        Create SidecarMCPClient from configuration file.
        
        Args:
            path: Path to configuration file
            sandbox: Whether to use sandboxed execution mode for running MCP servers
            sandbox_options: Optional sandbox configuration options
            sampling_callback: Optional sampling callback function
            elicitation_callback: Optional elicitation callback function
            message_handler: Optional message handler function
            logging_callback: Optional logging callback function
            **kwargs: Additional sidecar options (policy, needs, hints, etc.)
            
        Returns:
            SidecarMCPClient instance
        """
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(
            config_dict, 
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
            message_handler=message_handler,
            logging_callback=logging_callback,
            **kwargs
        )
    
    def get_sidecar_state(self) -> Dict[str, Any]:
        """
        Get current sidecar configuration state.
        
        This method returns the current sidecar configuration state as a dictionary,
        which can be useful for inspection, logging, or troubleshooting.
        
        Returns:
            Dict[str, Any]: Dictionary containing the current sidecar state including
                policy, needs, hints, cache configuration, run context, and telemetry status
        """
        return {
            "policy": self._sidecar_policy,
            "needs": self._sidecar_needs,
            "hints": self._sidecar_hints,
            "cache_ttl_s": self._sidecar_cache_ttl_s,
            "run_context": self._sidecar_run_context,
            "telemetry_enabled": self._sidecar_telemetry is not None
        }
    
    def add_server(self, name: str, server_config: Dict[str, Any]) -> None:
        """Add a server configuration.
        
        This method extends the parent add_server method to add telemetry
        tracking for server additions.
        
        Args:
            name: The name to identify this server
            server_config: The server configuration dictionary with command, args, and
                optional environment variables
            
        Raises:
            ValueError: If the server name already exists or the configuration is invalid
        """
        super().add_server(name, server_config)
        
        # Emit telemetry for server addition
        self._emit_telemetry("server_added", {
            "server_name": name,
            "config_keys": list(server_config.keys())
        })
    
    def remove_server(self, name: str) -> None:
        """Remove a server configuration.
        
        This method extends the parent remove_server method to add telemetry
        tracking for server removals.
        
        Args:
            name: The name of the server to remove
            
        Raises:
            ValueError: If the server name does not exist
            RuntimeError: If the server has active sessions
        """
        super().remove_server(name)
        
        # Emit telemetry for server removal
        self._emit_telemetry("server_removed", {
            "server_name": name
        })
    
    def get_server_names(self) -> List[str]:
        """Get the list of configured server names.
        
        This method returns a list of all currently configured server names.
        
        Returns:
            List[str]: List of server names that have been configured
        """
        server_names: List[str] = super().get_server_names()
        return server_names  # pyright: ignore[reportUnknownReturnType]
    
    def save_config(self, filepath: str) -> None:
        """Save the current configuration to a file.
        
        This method extends the parent save_config method to add telemetry
        tracking for configuration saves.
        
        Args:
            filepath: The path to save the configuration to
            
        Raises:
            IOError: If the file cannot be written
        """
        super().save_config(filepath)
        
        # Emit telemetry for config save
        self._emit_telemetry("config_saved", {
            "filepath": filepath,
            "server_count": len(self.get_server_names())
        })
    
    async def close_session(self, server_name: str) -> None:
        """Close a session.
        
        This method extends the parent close_session method to add telemetry
        tracking for session closures.
        
        Args:
            server_name: The name of the server to close the session for
            
        Raises:
            ValueError: If the server name does not exist
            ValueError: If no session exists for the server
        """
        # Emit telemetry before closing
        self._emit_telemetry("session_closing", {
            "server_name": server_name
        })
        
        await super().close_session(server_name)
        
        # Emit telemetry after closing
        self._emit_telemetry("session_closed", {
            "server_name": server_name
        })
    
    def get_all_active_sessions(self) -> Dict[str, Any]:
        """Get all active sessions.
        
        This method returns a dictionary of all currently active sessions.
        
        Returns:
            Dict[str, Any]: Dictionary mapping server names to their SidecarSession instances
        """
        sessions: Dict[str, Any] = super().get_all_active_sessions()
        return sessions  # pyright: ignore[reportUnknownReturnType]