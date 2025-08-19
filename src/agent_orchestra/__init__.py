"""Agent Orchestra - Sidecar extension for mcp-use with state management, policy enforcement, and telemetry."""

from .sidecar.sidecar_client import SidecarMCPClient
from .sidecar.sidecar_agent import SidecarMCPAgent
from .sidecar.sidecar_session import SidecarSession

# Adapter is optional - only available if mcp-use is installed
try:
    from .sidecar.sidecar_adapter import SidecarLangChainAdapter
except ImportError:
    SidecarLangChainAdapter = None  # type: ignore

# Orchestrator functionality
from .orchestrator.core import Orchestrator
from .orchestrator.types import Event, NodeSpec, GraphSpec, RunSpec
from .orchestrator.executors_mcp import MCPExecutor

__version__ = "0.1.0"

__all__ = [
    "SidecarMCPClient",
    "SidecarMCPAgent", 
    "SidecarSession",
    "SidecarLangChainAdapter",
    "Orchestrator",
    "Event", 
    "NodeSpec", 
    "GraphSpec", 
    "RunSpec",
    "MCPExecutor",
]