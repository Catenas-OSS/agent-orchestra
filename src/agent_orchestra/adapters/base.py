"""Base adapter interface and data structures.

This module defines the core contracts that all adapters must implement.
These interfaces are frozen and should not change without major version bumps.
"""

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class UsageMetrics:
    """Standardized usage metrics across all adapters."""

    tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        """Initialize latency if not provided.
        
        Sets latency_ms to current timestamp if it's 0.
        """
        if self.latency_ms == 0:
            self.latency_ms = int(time.time() * 1000)


@dataclass
class Artifact:
    """Represents files, images, or other data produced by agents."""

    uri: str
    mime_type: str
    size_bytes: int
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterResult:
    """Standardized result format from adapter calls."""

    output: Any
    usage: UsageMetrics
    artifacts: list[Artifact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the adapter call was successful.
        
        Returns:
            True if no errors occurred during execution.
        """
        return len(self.errors) == 0


class BaseAdapter(Protocol):
    """Protocol that all adapters must implement.
    
    This is the frozen contract for adapter implementations.
    Breaking changes to this interface require a major version bump.
    """

    @abstractmethod
    async def call(
        self,
        agent: str,
        tool: str,
        *,
        inputs: dict[str, Any],
        meta: dict[str, Any]
    ) -> AdapterResult:
        """Execute a tool call through the adapter.
        
        Args:
            agent: Agent identifier/configuration
            tool: Tool name to execute
            inputs: Input parameters for the tool
            meta: Metadata including trace_id, span_id, etc.
            
        Returns:
            AdapterResult with output, usage metrics, and any artifacts
        """
        ...

    @abstractmethod
    async def get_available_tools(self, agent: str) -> list[str]:
        """Get list of available tools for an agent.
        
        Args:
            agent: Agent identifier/configuration.
            
        Returns:
            List of tool names available for this agent.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the adapter is healthy and ready to serve requests."""
        ...


class AdapterRegistry:
    """Registry for discovering and loading adapters via entry points."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    def register(self, name: str, adapter: BaseAdapter) -> None:
        """Register an adapter instance.
        
        Args:
            name: Name to register the adapter under.
            adapter: Adapter instance to register.
        """
        self._adapters[name] = adapter

    def get(self, name: str) -> BaseAdapter | None:
        """Get an adapter by name."""
        return self._adapters.get(name)

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    def load_from_entry_points(self) -> None:
        """Load adapters from setuptools entry points.
        
        Attempts to load adapters using importlib.metadata first,
        then falls back to pkg_resources for older Python versions.
        """
        try:
            from importlib.metadata import entry_points

            for ep in entry_points(group="agent_orchestra.adapters"):
                try:
                    adapter_class = ep.load()
                    self.register(ep.name, adapter_class())
                except Exception as e:
                    # Log error but don't fail startup
                    print(f"Failed to load adapter {ep.name}: {e}")
        except ImportError:
            # Python < 3.10 fallback
            try:
                import pkg_resources
                for ep in pkg_resources.iter_entry_points("agent_orchestra.adapters"):
                    try:
                        adapter_class = ep.load()
                        self.register(ep.name, adapter_class())
                    except Exception as e:
                        print(f"Failed to load adapter {ep.name}: {e}")
            except ImportError:
                pass


# Global adapter registry instance
registry = AdapterRegistry()

