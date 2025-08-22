"""
Agent Pool Management for Agent Orchestra.

Provides agent reuse across foreach operations and runs to minimize
initialization overhead and improve performance. Supports profile-based
agent creation to avoid duplicate initialization.
"""

from __future__ import annotations
import asyncio
from typing import Dict, Optional, Any, Set, Tuple, Callable, Awaitable
from dataclasses import dataclass
from ..sidecar import sidecar_agent
import logging

logger = logging.getLogger(__name__)

# Type alias for profile keys
ProfileKey = Tuple[Optional[str], str, Optional[str]]


@dataclass
class AgentSpec:
    """Specification for creating an agent with specific profile."""
    server_name: Optional[str] = None
    model_key: str = "default"
    policy_id: Optional[str] = None
    use_server_manager: bool = True


class AgentPool:
    """
    Manages a pool of sidecar_agent.SidecarMCPAgent instances for reuse across operations.
    
    Features:
    - Profile-based agent creation (server_name, model_key, policy_id)
    - Agent reuse across nodes with same profile
    - No duplicate agent initialization
    - Race-safe agent creation
    - Automatic cleanup
    """
    
    def __init__(self, factory: Callable[[AgentSpec], Awaitable[sidecar_agent.SidecarMCPAgent]], max_agents_per_run: int = 10):
        self._factory = factory
        self.max_agents_per_run = max_agents_per_run
        
        # Profile-based agent pools
        self._agents: Dict[ProfileKey, sidecar_agent.SidecarMCPAgent] = {}  # profile_key -> agent
        self._agent_usage: Dict[ProfileKey, int] = {}  # profile_key -> usage_count
        
        # Per-profile locks for race-safe creation
        self._locks: Dict[ProfileKey, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        
        # Run tracking for cleanup
        self._active_runs: Set[str] = set()
        self._run_profiles: Dict[str, Set[ProfileKey]] = {}  # run_id -> set of profiles used
        
        logger.info(f"AgentPool initialized with profile-based agent creation")
    
    def _profile_key(self, spec: AgentSpec) -> ProfileKey:
        """Generate a profile key from an agent specification."""
        return (spec.server_name, spec.model_key, spec.policy_id)
    
    async def get(self, spec: AgentSpec, run_id: Optional[str] = None) -> sidecar_agent.SidecarMCPAgent:
        """
        Get an agent for the specified profile.
        
        Creates agent if it doesn't exist, otherwise reuses existing agent.
        This method is race-safe - multiple concurrent calls with the same
        spec will result in only one agent being created.
        
        Args:
            spec: Agent specification defining the profile
            run_id: Optional run ID for tracking purposes
            
        Returns:
            sidecar_agent.SidecarMCPAgent instance for the profile
        """
        key = self._profile_key(spec)
        
        # Check if agent already exists
        agent = self._agents.get(key)
        if agent:
            self._agent_usage[key] = self._agent_usage.get(key, 0) + 1
            if run_id:
                self._track_run_usage(run_id, key)
            logger.debug(f"Reusing agent for profile {key} (usage: {self._agent_usage[key]})")
            return agent
        
        # Get or create per-profile lock
        async with self._global_lock:
            lock = self._locks.setdefault(key, asyncio.Lock())
        
        # Create agent in a race-safe manner
        async with lock:
            # Double-check after acquiring lock
            agent = self._agents.get(key)
            if agent:
                self._agent_usage[key] = self._agent_usage.get(key, 0) + 1
                if run_id:
                    self._track_run_usage(run_id, key)
                logger.debug(f"Reusing agent for profile {key} (usage: {self._agent_usage[key]})")
                return agent
            
            # Create new agent
            logger.info(f"Creating new agent for profile {key}")
            try:
                agent = await self._factory(spec)
                self._agents[key] = agent
                self._agent_usage[key] = 1
                
                if run_id:
                    self._track_run_usage(run_id, key)
                
                logger.info(f"Created agent for profile {key}")
                return agent
                
            except Exception as e:
                logger.error(f"Failed to create agent for profile {key}: {e}")
                raise
    
    def _track_run_usage(self, run_id: str, profile_key: ProfileKey) -> None:
        """Track which profiles are used by which runs."""
        self._active_runs.add(run_id)
        if run_id not in self._run_profiles:
            self._run_profiles[run_id] = set()
        self._run_profiles[run_id].add(profile_key)
    
    async def get_agent_for_run(self, run_id: str, template_agent: sidecar_agent.SidecarMCPAgent) -> sidecar_agent.SidecarMCPAgent:
        """
        Get or create an agent for a specific run (legacy method).
        
        This method is kept for backward compatibility but now uses the
        profile-based system under the hood.
        """
        # Create a default spec from template agent
        spec = AgentSpec(
            server_name=None,
            model_key=getattr(template_agent, 'model_key', 'default'),
            policy_id=None,
            use_server_manager=True
        )
        
        return await self.get(spec, run_id)
    
    async def get_agent_for_foreach_item(self, run_id: str, item_index: int, template_agent: sidecar_agent.SidecarMCPAgent) -> sidecar_agent.SidecarMCPAgent:
        """
        Get an agent for a specific foreach item (legacy method).
        
        This reuses the run's primary agent to avoid initialization overhead
        while still allowing concurrent execution.
        """
        # For foreach items, we reuse the run's primary agent
        # This gives us the benefits of agent reuse while still allowing
        # TaskGroup concurrency (the agent calls are async)
        return await self.get_agent_for_run(run_id, template_agent)
    
    
    async def finish_run(self, run_id: str) -> None:
        """
        Mark a run as finished and clean up its tracking.
        
        Note: This doesn't destroy agents as they may be reused across runs
        with the same profile. Agents are only cleaned up during shutdown.
        """
        
        if run_id not in self._active_runs:
            return
        
        self._active_runs.discard(run_id)
        
        # Log usage statistics for this run
        profiles = self._run_profiles.get(run_id, set())
        for profile in profiles:
            usage_count = self._agent_usage.get(profile, 0)
            logger.info(f"Profile {profile} completed run {run_id} with {usage_count} total uses")
        
        # Clean up run tracking (but keep agents for reuse)
        self._run_profiles.pop(run_id, None)
        
        logger.debug(f"Finished run {run_id}, agents remain available for reuse")
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        stats: Dict[str, Any] = {
            "active_runs": len(self._active_runs),
            "total_agents": len(self._agents),
            "profiles": {},
            "runs": {}
        }
        
        # Profile statistics
        for profile_key, usage in self._agent_usage.items():
            server_name, model_key, policy_id = profile_key
            stats["profiles"][str(profile_key)] = {
                "server_name": server_name,
                "model_key": model_key,
                "policy_id": policy_id,
                "usage_count": usage
            }
        
        # Run statistics
        for run_id in self._active_runs:
            profiles = self._run_profiles.get(run_id, set())
            total_usage = sum(self._agent_usage.get(profile, 0) for profile in profiles)
            
            stats["runs"][run_id] = {
                "profiles_used": len(profiles),
                "total_usage": total_usage,
                "profiles": [str(profile) for profile in profiles]
            }
        
        return stats
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent pool."""
        logger.info("Shutting down AgentPool")
        
        # Finish all active runs
        active_runs = list(self._active_runs)
        for run_id in active_runs:
            await self.finish_run(run_id)
        
        # TODO: Consider calling cleanup methods on agents if available
        # For now, rely on Python GC and MCP client cleanup
        
        # Clear all tracking
        self._agents.clear()
        self._agent_usage.clear()
        self._locks.clear()
        self._active_runs.clear()
        self._run_profiles.clear()
        
        logger.info("AgentPool shutdown complete")


# Factory function type
AgentFactory = Callable[[AgentSpec], Awaitable[sidecar_agent.SidecarMCPAgent]]


def create_default_agent_factory(client: Any, llm: Optional[Any] = None) -> AgentFactory:
    """
    Create a default agent factory that can create agents for different profiles.
            @property        @property        @property
        def policy_id(self):
            raise NotImplementedError

        @policy_id.setter
        def policy_id(self, value):
            raise NotImplementedError


        def server_name(self):
            raise NotImplementedError

        @server_name.setter
        def server_name(self, value):
            raise NotImplementedError


        def model_key(self):
            raise NotImplementedError

        @model_key.setter
        def model_key(self, value):
            raise NotImplementedError


    Args:
        client: SidecarMCPClient instance to reuse across agents
        llm: Default LLM to use (optional)
        
    Returns:
        Factory function for creating agents
    """
    async def factory(spec: AgentSpec) -> sidecar_agent.SidecarMCPAgent:
        """Create an agent for the given specification."""
        from ..sidecar import sidecar_agent
        
        # Create agent with the shared client
        agent = sidecar_agent.SidecarMCPAgent(
            llm=llm,
            client=client,
            use_server_manager=spec.use_server_manager
        )
        
        # Store profile information on the agent
        agent.model_key = spec.model_key # type: ignore
        agent.server_name = spec.server_name # type: ignore
        agent.policy_id = spec.policy_id # type: ignore
        
        return agent
    
    return factory


# Global agent pool instance (optional convenience)
_global_agent_pool: Optional[AgentPool] = None


def get_global_agent_pool(factory: Optional[AgentFactory] = None) -> AgentPool:
    """Get or create the global agent pool instance."""
    global _global_agent_pool
    if _global_agent_pool is None:
        if factory is None:
            raise ValueError("Must provide factory for first call to get_global_agent_pool")
        _global_agent_pool = AgentPool(factory)
    return _global_agent_pool


def set_global_agent_pool(pool: AgentPool) -> None:
    """Set the global agent pool instance."""
    global _global_agent_pool
    _global_agent_pool = pool