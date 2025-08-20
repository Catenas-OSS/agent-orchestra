from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Awaitable
from ..sidecar.sidecar_agent import SidecarMCPAgent
from .types import NodeSpec
from .call_broker import CallBroker
from .agent_pool import AgentPool, AgentSpec
import logging

logger = logging.getLogger(__name__)


class MCPExecutor:
    """MCP-backed executor that uses SidecarMCPAgent for task execution."""
    
    def __init__(self, agent: Optional[SidecarMCPAgent] = None, default_server: Optional[str] = None, broker: Optional[CallBroker] = None, agent_pool: Optional[AgentPool] = None, model_key: str = "default"):
        self._template_agent = agent  # Keep as template for backward compatibility
        self._default_server = default_server
        self._broker = broker
        self._agent_pool = agent_pool
        self._model_key = model_key
        
        # Try to detect model from agent for broker routing
        self._model_name = self._detect_model_name(agent) if agent else f"unknown:{model_key}"
        
        # Current run context (set by orchestrator)
        self._current_run_id: Optional[str] = None
        
        if self._broker:
            logger.info(f"MCPExecutor initialized with CallBroker for model: {self._model_name}")
        if self._agent_pool:
            logger.info("MCPExecutor initialized with AgentPool for profile-based agent reuse")
        if not self._broker and not self._agent_pool:
            logger.debug("MCPExecutor initialized with direct agent calls (no broker/pool)")
    
    def _detect_model_name(self, agent: SidecarMCPAgent) -> str:
        """Attempt to detect the model name from the agent for broker routing."""
        try:
            # Try to get model from LLM if available
            if hasattr(agent, 'llm') and hasattr(agent.llm, 'model_name'):
                model = agent.llm.model_name
                if model.startswith('gpt'):
                    return f"openai:{model}"
                elif 'claude' in model:
                    return f"anthropic:{model}"
                return model
            elif hasattr(agent, 'llm') and hasattr(agent.llm, 'model'):
                model = agent.llm.model
                if model.startswith('gpt'):
                    return f"openai:{model}"
                elif 'claude' in model:
                    return f"anthropic:{model}"
                return model
        except Exception:
            pass
        
        # Default fallback
        return "unknown:default"
    
    def set_run_context(self, run_id: str) -> None:
        """Set the current run context for agent pool management."""
        self._current_run_id = run_id
    
    async def _get_agent(self, node: Optional[NodeSpec] = None, for_foreach_item: bool = False, item_index: int = 0) -> SidecarMCPAgent:
        """Get an agent for execution, either from pool or direct."""
        if self._agent_pool:
            # Use profile-based agent pool
            server_name = None
            if node:
                server_name = node.server_name or self._default_server
            else:
                server_name = self._default_server
                
            spec = AgentSpec(
                server_name=server_name,
                model_key=self._model_key,
                policy_id=None,  # TODO: Add policy support
                use_server_manager=True
            )
            
            return await self._agent_pool.get(spec, self._current_run_id)
        elif self._template_agent:
            # Direct agent usage (backward compatibility)
            return self._template_agent
        else:
            raise ValueError("No agent or agent pool provided to MCPExecutor")
    
    async def execute_foreach_item(self, node: NodeSpec, ctx: Dict[str, Any], item_index: int) -> Dict[str, Any]:
        """Execute a foreach item using a profile-managed agent."""
        # Get agent for this profile (server + model combination)
        agent = await self._get_agent(node, for_foreach_item=True, item_index=item_index)
        
        # Prepare context for agent execution
        prompt = self._build_prompt(node, ctx)
        
        # Route through broker if available, otherwise direct agent call
        if self._broker:
            # Use broker for rate limiting and retries
            async def agent_call() -> Any:
                # Pass server_name to agent for routing
                server_name = node.server_name or self._default_server
                kwargs = {}
                if server_name:
                    kwargs["server_name"] = server_name
                return await agent.run(prompt, **kwargs)
            
            result = await self._broker.call_agent_regular(self._model_name, agent_call)
        else:
            # Direct agent call (backward compatibility)
            server_name = node.server_name or self._default_server
            kwargs = {}
            if server_name:
                kwargs["server_name"] = server_name
            result = await agent.run(prompt, **kwargs)
        
        return {"output": result}
    
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node using MCP agent, returning final result."""
        # Get agent for execution (from pool or direct)
        agent = await self._get_agent(node)
        
        # Prepare context for agent execution
        prompt = self._build_prompt(node, ctx)
        
        # Route through broker if available, otherwise direct agent call
        if self._broker:
            # Use broker for rate limiting and retries
            async def agent_call() -> Any:
                # Pass server_name to agent for routing
                server_name = node.server_name or self._default_server
                kwargs = {}
                if server_name:
                    kwargs["server_name"] = server_name
                return await agent.run(prompt, **kwargs)
            
            result = await self._broker.call_agent_regular(self._model_name, agent_call)
        else:
            # Direct agent call (backward compatibility)
            server_name = node.server_name or self._default_server
            kwargs = {}
            if server_name:
                kwargs["server_name"] = server_name
            result = await agent.run(prompt, **kwargs)
        
        return {"output": result}
    
    async def execute_with_stream(
        self, 
        node: NodeSpec, 
        ctx: Dict[str, Any], 
        on_chunk: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> Dict[str, Any]:
        """Execute node with streaming, calling on_chunk for each chunk."""
        # Get agent for execution (from pool or direct)
        agent = await self._get_agent(node)
        
        # Prepare context for agent execution
        prompt = self._build_prompt(node, ctx)
        
        # Route through broker if available, otherwise direct agent call
        final_result = None
        last_text = None
        
        if self._broker:
            # Use broker for rate limiting, retries, and chunk passthrough
            async def agent_stream() -> Any:
                server_name = node.server_name or self._default_server
                kwargs = {}
                if server_name:
                    kwargs["server_name"] = server_name
                async for chunk in agent.astream(prompt, **kwargs):  # type: ignore
                    yield chunk
            
            async for chunk in self._broker.call_agent_streaming(self._model_name, agent_stream):
                # Broker passes chunks through unchanged - perfect for AGENT_CHUNK events
                await on_chunk(chunk)
                # Capture the final result from the last chunk with output
                if isinstance(chunk, dict):
                    if "output" in chunk:
                        final_result = chunk["output"]
                    elif "text" in chunk:
                        last_text = chunk["text"]
                    elif "message" in chunk and isinstance(chunk["message"], str):
                        last_text = chunk["message"]
        else:
            # Direct agent streaming (backward compatibility)
            server_name = node.server_name or self._default_server
            kwargs = {}
            if server_name:
                kwargs["server_name"] = server_name
            async for chunk in agent.astream(prompt, **kwargs):  # type: ignore
                await on_chunk(chunk)
                # Capture the final result from the last chunk with output
                if isinstance(chunk, dict):
                    if "output" in chunk:
                        final_result = chunk["output"]
                    elif "text" in chunk:
                        last_text = chunk["text"]
                    elif "message" in chunk and isinstance(chunk["message"], str):
                        last_text = chunk["message"]
        
        # Return output or fall back to last text if no output was captured
        return {"output": final_result if final_result is not None else last_text}
    
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
                    blackboard_entry = ctx["blackboard"][value]
                    # Extract actual result from blackboard entry structure
                    if isinstance(blackboard_entry, dict) and "result" in blackboard_entry:
                        actual_value = blackboard_entry["result"]
                        # Further extract if nested
                        if isinstance(actual_value, dict) and "output" in actual_value:
                            actual_value = actual_value["output"]
                    else:
                        actual_value = blackboard_entry
                    prompt_parts.append(f"- {key}: {actual_value}") # type: ignore
                else:
                    prompt_parts.append(f"- {key}: {value}") # type: ignore
        
        return "\n".join(prompt_parts) # type: ignore