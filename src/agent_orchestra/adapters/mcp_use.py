"""MCP-USE adapter implementation.

This is the ONLY module that should import from mcp-use.
All other modules should use the adapter interface.
"""

import json
import time
from typing import Any

from agent_orchestra.adapters.base import AdapterResult, BaseAdapter, UsageMetrics

# Only mcp-use imports allowed in this file
try:
    from mcp_use import MCPAgent, MCPClient
    MCP_USE_AVAILABLE = True
except ImportError:
    MCP_USE_AVAILABLE = False
    MCPAgent = None
    MCPClient = None


class MCPUseAdapter(BaseAdapter):
    """Adapter for MCP-USE agents."""

    def __init__(self) -> None:
        if not MCP_USE_AVAILABLE:
            raise ImportError(
                "mcp-use is not installed. Install with: pip install mcp-use"
            )

        self._clients: dict[str, MCPClient] = {}
        self._agents: dict[str, MCPAgent] = {}

    async def call(
        self,
        agent: str,
        tool: str,
        *,
        inputs: dict[str, Any],
        meta: dict[str, Any]
    ) -> AdapterResult:
        """Execute a tool call through MCP-USE."""
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
        """Get available tools for an MCP agent."""
        try:
            client = await self._get_client(agent)
            # TODO: Query MCP client for available tools
            # This is a placeholder - actual implementation depends on mcp-use API
            return ["web_search", "browser", "analysis", "file_operations"]
        except Exception:
            return []

    async def health_check(self) -> bool:
        """Check if MCP-USE is available and working."""
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
            client = MCPClient.from_dict(test_config)
            return True
        except Exception:
            return False

    async def _get_client(self, agent_config: str | dict[str, Any]) -> MCPClient:
        """Get or create MCP client for agent configuration."""
        config_key = self._get_config_key(agent_config)

        if config_key not in self._clients:
            if isinstance(agent_config, str):
                # Try to parse as JSON
                try:
                    config = json.loads(agent_config)
                except json.JSONDecodeError:
                    # Assume it's a config file path
                    self._clients[config_key] = MCPClient.from_config_file(agent_config)
                else:
                    self._clients[config_key] = MCPClient.from_dict(config)
            else:
                self._clients[config_key] = MCPClient.from_dict(agent_config)

        return self._clients[config_key]

    async def _get_agent(self, agent_config: str | dict[str, Any], meta: dict[str, Any]) -> MCPAgent:
        """Get or create MCP agent."""
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
            self._agents[config_key] = MCPAgent(
                llm=llm,
                client=client,
                max_steps=max_steps
            )

        return self._agents[config_key]

    def _get_config_key(self, agent_config: str | dict[str, Any]) -> str:
        """Generate a key for caching clients/agents."""
        if isinstance(agent_config, str):
            return agent_config
        return str(hash(json.dumps(agent_config, sort_keys=True)))

    def _format_tool_call(self, tool: str, inputs: dict[str, Any]) -> str:
        """Format tool call as a prompt for the agent."""
        if not inputs:
            return f"Use the {tool} tool"

        # Convert inputs to a natural language prompt
        input_parts = []
        for key, value in inputs.items():
            input_parts.append(f"{key}: {value}")

        return f"Use the {tool} tool with the following inputs: {', '.join(input_parts)}"

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Rough token estimation (replace with actual tokenizer if needed)."""
        # Very rough estimate: ~4 characters per token
        total_chars = len(prompt) + len(response)
        return total_chars // 4

