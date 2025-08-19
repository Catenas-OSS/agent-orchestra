from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Awaitable
from ..sidecar.sidecar_agent import SidecarMCPAgent
from .types import NodeSpec


class MCPExecutor:
    """MCP-backed executor that uses SidecarMCPAgent for task execution."""
    
    def __init__(self, agent: SidecarMCPAgent, default_server: Optional[str] = None):
        self._agent = agent
        self._default_server = default_server
    
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node using MCP agent, returning final result."""
        # Prepare context for agent execution
        prompt = self._build_prompt(node, ctx)
        
        # Use agent's run method for non-streaming execution
        # Note: Server selection is handled by the agent/client automatically
        result = await self._agent.run(prompt)
        
        return {"output": result}
    
    async def execute_with_stream(
        self, 
        node: NodeSpec, 
        ctx: Dict[str, Any], 
        on_chunk: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> Dict[str, Any]:
        """Execute node with streaming, calling on_chunk for each chunk."""
        # Prepare context for agent execution
        prompt = self._build_prompt(node, ctx)
        
        # Use agent's astream method for streaming execution
        # Note: Server selection is handled by the agent/client automatically
        final_result = None
        async for chunk in self._agent.astream(prompt):
            await on_chunk(chunk)
            # Capture the final result from the last chunk with output
            if "output" in chunk:
                final_result = chunk["output"]
        
        return {"output": final_result}
    
    def _build_prompt(self, node: NodeSpec, ctx: Dict[str, Any]) -> str:
        """Build prompt for agent execution from node spec and context."""
        prompt_parts = []
        
        if node.name:
            prompt_parts.append(f"Task: {node.name}") # type: ignore
        
        # Add inputs to prompt
        if node.inputs:
            prompt_parts.append("Inputs:") # type: ignore
            for key, value in node.inputs.items():
                # Handle references to other nodes
                if isinstance(value, str) and value in ctx["blackboard"]:
                    actual_value = ctx["blackboard"][value]
                    prompt_parts.append(f"- {key}: {actual_value}") # type: ignore
                else:
                    prompt_parts.append(f"- {key}: {value}") # type: ignore
        
        return "\n".join(prompt_parts) # type: ignore