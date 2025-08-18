"""MCP-USE adapter implementation.

This is the ONLY module that should import from mcp-use.
All other modules should use the adapter interface.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from agent_orchestra.adapters.base import AdapterResult, BaseAdapter, UsageMetrics
from agent_orchestra.events import Event, EventEmitter, EventType
from agent_orchestra.tools_loader import ToolsLoader

# Only mcp-use imports allowed in this file
try:
    from mcp_use import MCPAgent, MCPClient
    MCP_USE_AVAILABLE = True
except ImportError:
    MCP_USE_AVAILABLE = False
    # Create type-safe stubs for when mcp-use is not available
    MCPAgent = Any  # type: ignore
    MCPClient = Any  # type: ignore


class MCPUseAdapter(BaseAdapter):
    """Adapter for MCP-USE agents.
    
    Provides integration with the MCP-USE library for executing
    Model Context Protocol (MCP) agents and tools. Handles client
    creation, agent management, and execution lifecycle.
    """

    def __init__(self, event_emitter: EventEmitter | None = None) -> None:
        """Initialize the MCP-USE adapter.
        
        Args:
            event_emitter: Optional event emitter for lifecycle events
            
        Raises:
            ImportError: If mcp-use is not installed
        """
        if not MCP_USE_AVAILABLE:
            raise ImportError(
                "mcp-use is not installed. Install with: pip install mcp-use"
            )

        self._clients: dict[str, Any] = {}
        self._agents: dict[str, Any] = {}
        self._event_emitter = event_emitter
        self._tools_loader = ToolsLoader()

    async def call(
        self,
        agent: str,
        tool: str,
        *,
        inputs: dict[str, Any],
        meta: dict[str, Any]
    ) -> AdapterResult:
        """Execute a tool call through MCP-USE.
        
        Args:
            agent: Agent configuration string or identifier
            tool: Name of the tool to execute
            inputs: Input parameters for the tool
            meta: Metadata including trace/span IDs and execution context
            
        Returns:
            AdapterResult containing output, usage metrics, and metadata
        """
        start_time = time.time()

        try:
            # Get or create agent
            mcp_agent = await self._get_agent(agent, meta)

            # Format the input for the agent
            prompt = self._format_tool_call(tool, inputs)

            # Execute the agent
            result = await mcp_agent.run(prompt)

            # Calculate usage metrics
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            usage = UsageMetrics(
                tokens=self._estimate_tokens(prompt, str(result)),
                cost=0.0,  # TODO: Calculate actual cost based on provider
                latency_ms=latency_ms
            )

            return AdapterResult(
                output=result,
                usage=usage,
                metadata={
                    "agent_config": agent,
                    "tool": tool,
                    "trace_id": meta.get("trace_id"),
                    "span_id": meta.get("span_id")
                }
            )

        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)

            return AdapterResult(
                output=None,
                usage=UsageMetrics(latency_ms=latency_ms),
                errors=[str(e)],
                metadata={
                    "agent_config": agent,
                    "tool": tool,
                    "trace_id": meta.get("trace_id"),
                    "span_id": meta.get("span_id")
                }
            )

    async def get_available_tools(self, agent: str) -> list[str]:
        """Get available tools for an MCP agent.
        
        Args:
            agent: Agent configuration string or identifier
            
        Returns:
            List of available tool names for the agent
        """
        try:
            client = await self._get_client(agent)
            # TODO: Query MCP client for available tools
            # This is a placeholder - actual implementation depends on mcp-use API
            return ["web_search", "browser", "analysis", "file_operations"]
        except Exception:
            return []

    async def health_check(self) -> bool:
        """Check if MCP-USE is available and working.
        
        Returns:
            True if MCP-USE is functional, False otherwise
        """
        try:
            # Simple test to ensure mcp-use is working
            test_config = {
                "mcpServers": {
                    "test": {
                        "command": "echo",
                        "args": ["test"]
                    }
                }
            }
            client = MCPClient.from_dict(test_config)  # type: ignore
            return True
        except Exception:
            return False

    async def _get_client(self, agent_config: str | dict[str, Any]) -> Any:
        """Get or create MCP client for agent configuration.
        
        Args:
            agent_config: Agent configuration as string (JSON/file path) or dict
            
        Returns:
            MCPClient instance for the configuration
        """
        config_key = self._get_config_key(agent_config)

        if config_key not in self._clients:
            if isinstance(agent_config, str):
                # Try to parse as JSON
                try:
                    config = json.loads(agent_config)
                except json.JSONDecodeError:
                    # Assume it's a config file path
                    self._clients[config_key] = MCPClient.from_config_file(agent_config)  # type: ignore
                else:
                    self._clients[config_key] = MCPClient.from_dict(config)  # type: ignore
            else:
                self._clients[config_key] = MCPClient.from_dict(agent_config)  # type: ignore

        return self._clients[config_key]

    async def _get_agent(self, agent_config: str | dict[str, Any], meta: dict[str, Any]) -> Any:
        """Get or create MCP agent.
        
        Args:
            agent_config: Agent configuration as string or dict
            meta: Metadata containing LLM and execution parameters
            
        Returns:
            MCPAgent instance ready for execution
        """
        config_key = self._get_config_key(agent_config)

        if config_key not in self._agents:
            client = await self._get_client(agent_config)

            # Get LLM from meta or use default
            llm = meta.get("llm")
            if llm is None:
                # Default LLM - this should be configurable
                try:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model="gpt-4")
                except ImportError:
                    raise ImportError(
                        "No LLM provided and langchain-openai not available. "
                        "Install with: pip install langchain-openai"
                    )

            max_steps = meta.get("max_steps", 30)
            self._agents[config_key] = MCPAgent(  # type: ignore
                llm=llm,
                client=client,
                max_steps=max_steps
            )

        return self._agents[config_key]

    def _get_config_key(self, agent_config: str | dict[str, Any]) -> str:
        """Generate a key for caching clients/agents.
        
        Args:
            agent_config: Agent configuration as string or dict
            
        Returns:
            String key suitable for caching
        """
        if isinstance(agent_config, str):
            return agent_config
        return str(hash(json.dumps(agent_config, sort_keys=True)))

    def _format_tool_call(self, tool: str, inputs: dict[str, Any]) -> str:
        """Format tool call as a prompt for the agent.
        
        Args:
            tool: Name of the tool to call
            inputs: Input parameters for the tool
            
        Returns:
            Formatted prompt string for the agent
        """
        if not inputs:
            return f"Use the {tool} tool"

        # Convert inputs to a natural language prompt
        input_parts = []
        for key, value in inputs.items():
            input_parts.append(f"{key}: {value}")

        return f"Use the {tool} tool with the following inputs: {', '.join(input_parts)}"

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Rough token estimation (replace with actual tokenizer if needed).
        
        Args:
            prompt: Input prompt text
            response: Response text
            
        Returns:
            Estimated token count for prompt and response
        """
        # Very rough estimate: ~4 characters per token
        total_chars = len(prompt) + len(response)
        return total_chars // 4

    async def build_client(
        self,
        tools_config_dict_or_path: dict[str, Any] | str | Path,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None
    ) -> Any:
        """Build MCPClient from config dict or path.
        
        Args:
            tools_config_dict_or_path: Either a config dictionary or path to config file
            run_id: Run ID for event emission
            trace_id: Trace ID for event emission
            span_id: Span ID for event emission
            
        Returns:
            MCPClient instance
            
        Raises:
            ToolsLoaderError: If loading/validation fails
        """
        # Load and validate config
        if isinstance(tools_config_dict_or_path, (str, Path)):
            config = self._tools_loader.load(tools_config_dict_or_path)
        else:
            # Validate dict config
            self._tools_loader._validate(tools_config_dict_or_path)
            self._tools_loader._validate_edge_cases(tools_config_dict_or_path)
            config = tools_config_dict_or_path

        # Emit client.start event
        if self._event_emitter and run_id and trace_id and span_id:
            server_count = len(config.get("mcpServers", {}))
            await self._event_emitter.emit(Event(
                type=EventType.CLIENT_START,
                run_id=run_id,
                trace_id=trace_id,
                span_id=span_id,
                payload={"server_count": server_count}
            ))

            # Emit server.start events
            for server_name, server_config in config.get("mcpServers", {}).items():
                transport = "http-sse" if "url" in server_config else "stdio"
                await self._event_emitter.emit(Event(
                    type=EventType.SERVER_START,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={"name": server_name, "transport": transport}
                ))

        # Create client
        client = MCPClient.from_dict(config)  # type: ignore

        # Optional readiness check
        if self._event_emitter and run_id and trace_id and span_id:
            await self._check_server_readiness(client, config, run_id, trace_id, span_id)

        return client

    def build_agent(
        self,
        llm: Any,
        client: Any,
        max_steps: int,
        *,
        disallowed_tools: list[str] | None = None,
        use_server_manager: bool = False
    ) -> Any:
        """Build MCPAgent with specified configuration.
        
        Args:
            llm: Language model instance
            client: MCPClient instance
            max_steps: Maximum number of steps for agent execution
            disallowed_tools: Optional list of tools to disallow
            use_server_manager: Whether to use server manager
            
        Returns:
            MCPAgent instance
        """
        return MCPAgent(  # type: ignore
            llm=llm,
            client=client,
            max_steps=max_steps,
            disallowed_tools=disallowed_tools,
            use_server_manager=use_server_manager
        )

    async def run(
        self,
        agent: Any,
        prompt: str,
        *,
        server_name: str | None = None
    ) -> Any:
        """Run agent with prompt.
        
        Args:
            agent: MCPAgent instance
            prompt: Input prompt for the agent
            server_name: Optional server name to pass through
            
        Returns:
            Agent execution result
        """
        if server_name is not None:
            return await agent.run(prompt, server_name=server_name)  # type: ignore
        else:
            return await agent.run(prompt)  # type: ignore

    async def close_all(
        self,
        client: Any,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None
    ) -> None:
        """Close all client sessions.
        
        Args:
            client: MCPClient instance to close
            run_id: Run ID for event emission
            trace_id: Trace ID for event emission
            span_id: Span ID for event emission
        """
        try:
            await client.close_all_sessions()  # type: ignore
        finally:
            # Always emit client.stop event
            if self._event_emitter and run_id and trace_id and span_id:
                await self._event_emitter.emit(Event(
                    type=EventType.CLIENT_STOP,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={}
                ))

    async def _check_server_readiness(
        self,
        client: Any,
        config: dict[str, Any],
        run_id: str,
        trace_id: str,
        span_id: str
    ) -> None:
        """Check server readiness with optional timeout.
        
        Args:
            client: MCPClient instance
            config: Server configuration
            run_id: Run ID for events
            trace_id: Trace ID for events
            span_id: Span ID for events
        """
        for server_name, server_config in config.get("mcpServers", {}).items():
            timeout_s = server_config.get("timeout_s")
            if timeout_s is None:
                continue  # Skip readiness check if no timeout specified

            start_time = time.time()
            try:
                # Simple readiness check - try to list tools with timeout
                await asyncio.wait_for(
                    self._ping_server_readiness(client, server_name),
                    timeout=timeout_s
                )

                # Emit server.ready
                elapsed_ms = int((time.time() - start_time) * 1000)
                if self._event_emitter:
                    await self._event_emitter.emit(Event(
                        type=EventType.SERVER_READY,
                        run_id=run_id,
                        trace_id=trace_id,
                        span_id=span_id,
                        payload={"name": server_name, "ms": elapsed_ms}
                    ))

            except TimeoutError:
                if self._event_emitter:
                    await self._event_emitter.emit(Event(
                        type=EventType.SERVER_ERROR,
                        run_id=run_id,
                        trace_id=trace_id,
                        span_id=span_id,
                        payload={"name": server_name, "reason": f"Timeout after {timeout_s}s"}
                    ))
            except Exception as e:
                if self._event_emitter:
                    await self._event_emitter.emit(Event(
                        type=EventType.SERVER_ERROR,
                        run_id=run_id,
                        trace_id=trace_id,
                        span_id=span_id,
                        payload={"name": server_name, "reason": str(e)}
                    ))

    async def _ping_server_readiness(self, client: Any, server_name: str) -> None:
        """Ping server for readiness (light check).
        
        Args:
            client: MCPClient instance
            server_name: Name of server to ping
        """
        try:
            # Attempt a light operation like listing tools
            # This is a placeholder - actual implementation depends on mcp-use API
            # For now, just do a minimal check
            pass
        except Exception:
            # If this method is not supported, silently skip
            pass

