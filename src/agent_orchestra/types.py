"""Type definitions for Agent Orchestra.

This module contains TypedDict definitions and other type annotations used throughout
the Agent Orchestra library. These types define the structure for configuration options,
policies, and other shared data structures.
"""

from typing import TypedDict, Any, Dict, List, Optional
from typing_extensions import NotRequired


class SandboxOptions(TypedDict):
    """Configuration options for sandbox execution.

    This type defines the configuration options available when running
    MCP servers in a sandboxed environment (e.g., using E2B).
    
    Attributes:
        api_key: Direct API key for sandbox provider (e.g., E2B API key).
            If not provided, will use E2B_API_KEY environment variable.
        sandbox_template_id: Template ID for the sandbox environment.
            Default: 'base'
        supergateway_command: Command to run supergateway.
            Default: 'npx -y supergateway'
    """

    api_key: str
    """Direct API key for sandbox provider (e.g., E2B API key).
    If not provided, will use E2B_API_KEY environment variable."""

    sandbox_template_id: NotRequired[str]
    """Template ID for the sandbox environment.
    Default: 'base'"""

    supergateway_command: NotRequired[str]
    """Command to run supergateway.
    Default: 'npx -y supergateway'"""


class SidecarPolicy(TypedDict, total=False):
    """Sidecar policy configuration for tool access control.
    
    This TypedDict defines the structure for policy configuration which controls
    which tools agents can access. It supports both allowlist and denylist
    approaches to access control.
    
    Attributes:
        allowed_tools: List of tools that are explicitly allowed. If provided, only these
            tools will be accessible, overriding any default tool access.
        disallowed_tools: List of tools that are explicitly disallowed. These tools will
            not be accessible, even if they would be allowed by default.
    """
    
    allowed_tools: Optional[List[str]]
    """List of tools that are explicitly allowed."""
    
    disallowed_tools: Optional[List[str]]
    """List of tools that are explicitly disallowed."""


class SidecarConfig(TypedDict, total=False):
    """Sidecar configuration block for Agent Orchestra extensions.
    
    This TypedDict defines the complete configuration structure for the sidecar
    functionality, including policy enforcement, requirements specification,
    behavior hints, caching, and run context metadata.
    
    Attributes:
        policy: Policy configuration for tool access control.
        needs: List of requirement tags that specify capabilities the agent needs.
        hints: Preference hints that guide agent behavior without strictly enforcing it.
        cache_ttl_s: Cache time-to-live in seconds for caching agent responses.
        run_context: Context dictionary for telemetry and tracing, used to associate
            agent actions with specific runs, trace IDs, or other metadata.
    """
    
    policy: Optional[SidecarPolicy]
    """Policy configuration for tool access control."""
    
    needs: Optional[List[str]]
    """List of requirement tags."""
    
    hints: Optional[Dict[str, Any]]
    """Preference hints for agent behavior."""
    
    cache_ttl_s: Optional[int]
    """Cache TTL in seconds."""
    
    run_context: Optional[Dict[str, Any]]
    """Context for telemetry and tracing."""