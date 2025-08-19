"""Agent Orchestra - Sidecar extension for mcp-use with state management, policy enforcement, and telemetry."""

from .sidecar_client import SidecarMCPClient
from .sidecar_agent import SidecarMCPAgent
from .sidecar_session import SidecarSession

# Adapter is optional - only available if mcp-use is installed
try:
    from .sidecar_adapter import SidecarLangChainAdapter
except ImportError:
    SidecarLangChainAdapter = None  # type: ignore

__version__ = "0.1.0"

__all__ = [
    "SidecarMCPClient",
    "SidecarMCPAgent", 
    "SidecarSession",
    "SidecarLangChainAdapter",
]