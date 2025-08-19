"""Sidecar module for Agent Orchestra - Enhanced mcp-use with telemetry and policy enforcement."""

from .sidecar_client import SidecarMCPClient
from .sidecar_agent import SidecarMCPAgent
from .sidecar_session import SidecarSession

# Adapter is optional - only available if mcp-use is installed
try:
    from .sidecar_adapter import SidecarLangChainAdapter
except ImportError:
    SidecarLangChainAdapter = None  # type: ignore

__all__ = [
    "SidecarMCPClient",
    "SidecarMCPAgent", 
    "SidecarSession",
    "SidecarLangChainAdapter",
]