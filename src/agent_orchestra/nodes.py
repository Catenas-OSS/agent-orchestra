"""Node abstractions and MCP wrappers.

Defines base node types and execution logic.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_orchestra.adapters.base import BaseAdapter
from agent_orchestra.events import Event, EventEmitter, EventType


@dataclass
class NodeResult:
    """Result from node execution."""

    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Any] = field(default_factory=list)
    success: bool = True
    error: str | None = None


class Node(ABC):
    """Base class for all node types."""

    def __init__(
        self,
        node_id: str,
        config: dict[str, Any],
        adapter: BaseAdapter,
        event_emitter: EventEmitter
    ) -> None:
        """Initialize the node.
        
        Args:
            node_id: Unique identifier for this node.
            config: Node-specific configuration dictionary.
            adapter: Adapter instance for backend communication.
            event_emitter: Event emitter for logging node events.
        """
        self.node_id = node_id
        self.config = config
        self.adapter = adapter
        self.event_emitter = event_emitter

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        context: dict[str, Any],
        trace_id: str,
        span_id: str
    ) -> NodeResult:
        """Execute the node with given inputs and context.
        
        Args:
            inputs: Input data for the node execution.
            context: Execution context including run metadata.
            trace_id: Unique identifier for distributed tracing.
            span_id: Unique identifier for this execution span.
            
        Returns:
            NodeResult containing output, metadata, and success status.
        """
        ...


class MCPAgentNode(Node):
    """Node that wraps an MCP agent."""

    async def execute(
        self,
        inputs: dict[str, Any],
        context: dict[str, Any],
        trace_id: str,
        span_id: str
    ) -> NodeResult:
        """Execute MCP agent node.
        
        Args:
            inputs: Input data for the MCP agent.
            context: Execution context including run metadata.
            trace_id: Unique identifier for distributed tracing.
            span_id: Unique identifier for this execution span.
            
        Returns:
            NodeResult containing agent output and execution metadata.
        """
        try:
            # Emit start event
            await self.event_emitter.emit(Event(
                type=EventType.NODE_START,
                run_id=context.get("run_id", ""),
                trace_id=trace_id,
                span_id=span_id,
                node_id=self.node_id,
                payload={"inputs": inputs}
            ))

            # Get tool from config
            tool = self.config.get("tool", "general")

            # Call adapter
            result = await self.adapter.call(
                agent=self.config.get("agent_config", "{}"),
                tool=tool,
                inputs=inputs,
                meta={
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "run_id": context.get("run_id"),
                    "max_steps": self.config.get("max_steps", 30)
                }
            )

            # Create node result
            node_result = NodeResult(
                output=result.output,
                metadata=result.metadata,
                artifacts=result.artifacts,
                success=result.success,
                error=result.errors[0] if result.errors else None
            )

            # Emit completion event
            await self.event_emitter.emit(Event(
                type=EventType.NODE_COMPLETE if node_result.success else EventType.NODE_ERROR,
                run_id=context.get("run_id", ""),
                trace_id=trace_id,
                span_id=span_id,
                node_id=self.node_id,
                payload={
                    "output": result.output,
                    "usage": result.usage.__dict__ if hasattr(result.usage, '__dict__') else {},
                    "success": node_result.success,
                    "error": node_result.error
                }
            ))

            return node_result

        except Exception as e:
            error_result = NodeResult(
                output=None,
                success=False,
                error=str(e)
            )

            await self.event_emitter.emit(Event(
                type=EventType.NODE_ERROR,
                run_id=context.get("run_id", ""),
                trace_id=trace_id,
                span_id=span_id,
                node_id=self.node_id,
                payload={"error": str(e)}
            ))

            return error_result


class CompositeNode(Node):
    """Node that can shard work across multiple sub-agents."""

    def __init__(
        self,
        node_id: str,
        config: dict[str, Any],
        adapter: BaseAdapter,
        event_emitter: EventEmitter
    ) -> None:
        """Initialize the composite node.
        
        Args:
            node_id: Unique identifier for this node.
            config: Node-specific configuration dictionary.
            adapter: Adapter instance for backend communication.
            event_emitter: Event emitter for logging node events.
        """
        super().__init__(node_id, config, adapter, event_emitter)
        self.max_concurrency = config.get("max_concurrency", 5)

    async def execute(
        self,
        inputs: dict[str, Any],
        context: dict[str, Any],
        trace_id: str,
        span_id: str
    ) -> NodeResult:
        """Execute composite node with sharding.
        
        Args:
            inputs: Input data containing shards or single input.
            context: Execution context including run metadata.
            trace_id: Unique identifier for distributed tracing.
            span_id: Unique identifier for this execution span.
            
        Returns:
            NodeResult containing aggregated results from all shards.
        """
        try:
            # Get sharding configuration
            shards = inputs.get("shards", [inputs])

            # Limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def execute_shard(shard_input: dict[str, Any], shard_id: int) -> Any:
                async with semaphore:
                    # Create sub-node for this shard
                    sub_node = MCPAgentNode(
                        node_id=f"{self.node_id}_shard_{shard_id}",
                        config=self.config,
                        adapter=self.adapter,
                        event_emitter=self.event_emitter
                    )

                    result = await sub_node.execute(
                        inputs=shard_input,
                        context=context,
                        trace_id=trace_id,
                        span_id=f"{span_id}_shard_{shard_id}"
                    )

                    return result.output

            # Execute all shards
            tasks = [
                execute_shard(shard, i)
                for i, shard in enumerate(shards)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and collect successful results
            successful_results = [
                result for result in results
                if not isinstance(result, Exception)
            ]

            errors = [
                str(result) for result in results
                if isinstance(result, Exception)
            ]

            return NodeResult(
                output=successful_results,
                success=len(errors) == 0,
                error="; ".join(errors) if errors else None,
                metadata={"total_shards": len(shards), "successful_shards": len(successful_results)}
            )

        except Exception as e:
            return NodeResult(
                output=None,
                success=False,
                error=str(e)
            )


class FunctionNode(Node):
    """Node that executes a Python function."""

    async def execute(
        self,
        inputs: dict[str, Any],
        context: dict[str, Any],
        trace_id: str,
        span_id: str
    ) -> NodeResult:
        """Execute function node.
        
        Args:
            inputs: Input data for the function.
            context: Execution context including run metadata.
            trace_id: Unique identifier for distributed tracing.
            span_id: Unique identifier for this execution span.
            
        Returns:
            NodeResult containing function output.
            
        Raises:
            ValueError: If function name is not specified in config.
        """
        try:
            function_name = self.config.get("function")
            if not function_name:
                raise ValueError("Function node requires 'function' in config")

            # Import and execute function
            # This is a placeholder - actual implementation would be more sophisticated
            result = f"Function {function_name} executed with inputs: {inputs}"

            return NodeResult(
                output=result,
                success=True
            )

        except Exception as e:
            return NodeResult(
                output=None,
                success=False,
                error=str(e)
            )


def create_node(
    node_id: str,
    node_type: str,
    config: dict[str, Any],
    adapter: BaseAdapter,
    event_emitter: EventEmitter
) -> Node:
    """Factory function to create nodes of different types.
    
    Args:
        node_id: Unique identifier for the node.
        node_type: Type of node to create (mcp_agent, composite, function).
        config: Node-specific configuration dictionary.
        adapter: Adapter instance for backend communication.
        event_emitter: Event emitter for logging node events.
        
    Returns:
        Node instance of the specified type.
        
    Raises:
        ValueError: If node_type is not recognized.
    """

    if node_type == "mcp_agent":
        return MCPAgentNode(node_id, config, adapter, event_emitter)
    elif node_type == "composite":
        return CompositeNode(node_id, config, adapter, event_emitter)
    elif node_type == "function":
        return FunctionNode(node_id, config, adapter, event_emitter)
    else:
        raise ValueError(f"Unknown node type: {node_type}")

