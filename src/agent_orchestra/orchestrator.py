"""Main orchestration engine.

Provides the core Orchestrator class with run/astream/resume capabilities.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_orchestra.adapters.base import registry as adapter_registry
from agent_orchestra.checkpointer import Checkpointer, CheckpointState, create_deterministic_context
from agent_orchestra.events import Event, EventBuffer, EventEmitter, EventType, JSONLSink
from agent_orchestra.graph import Graph, GraphSpec
from agent_orchestra.nodes import create_node
from agent_orchestra.policy import (
    Budget,
    HITLManager,
    Policy,
    create_default_budget,
    create_default_policy,
)
from agent_orchestra.agents_loader import load_agents_config, AgentsLoaderError
from agent_orchestra.tools_loader import load_tools_config, ToolsLoaderError
from agent_orchestra.llm_factory import LLMFactory


@dataclass
class RunResult:
    """Result from a complete orchestration run.
    
    Contains execution results, metrics, and any errors that occurred
    during the orchestration process.
    """

    run_id: str
    trace_id: str
    success: bool
    outputs: dict[str, Any]
    total_tokens: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    error: str | None = None
    checkpoints: list[str] = field(default_factory=list)


class Orchestrator:
    """Main orchestration engine for multi-agent workflows."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "./checkpoints",
        event_dir: str | Path = "./events",
        policy: Policy | None = None,
        budget: Budget | None = None,
        max_concurrency: int = 10
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            checkpoint_dir: Directory to store checkpoints for resumption.
            event_dir: Directory to store JSONL event logs.
            policy: Security policy for tool and domain restrictions.
            budget: Resource budget with token, cost, and time limits.
            max_concurrency: Maximum number of concurrent node executions.
        """
        self.checkpointer = Checkpointer(checkpoint_dir)
        self.event_dir = Path(event_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)

        self.policy = policy or create_default_policy()
        self.budget = budget or create_default_budget()
        self.hitl_manager = HITLManager()

        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Load adapters
        adapter_registry.load_from_entry_points()
        
        # Bootstrap state (set during run)
        self.agents_by_id: dict[str, dict[str, Any]] = {}
        self.llm_cache: dict[str, Any] = {}
        self.llm_factory = LLMFactory()
        self.mcp_client = None

    async def _bootstrap(
        self, 
        tools_file: str | Path | None, 
        agents_path: str | Path | None,
        run_id: str,
        trace_id: str,
        event_emitter: EventEmitter
    ) -> None:
        """Bootstrap the orchestrator with tools and agents."""
        span_id = str(uuid.uuid4())
        
        # Load tools → MCPClient
        if tools_file:
            try:
                tools_config = load_tools_config(tools_file)
                
                # Get adapter to create MCP client with event emission
                mcp_adapter = adapter_registry.get("mcp_use")
                if not mcp_adapter:
                    raise ValueError("mcp_use adapter not found in registry")
                
                # Set event emitter on adapter
                if hasattr(mcp_adapter, '_event_emitter'):
                    mcp_adapter._event_emitter = event_emitter
                
                # Create MCP client using adapter's build_client method
                self.mcp_client = await mcp_adapter.build_client(
                    tools_config,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id
                )
            except ToolsLoaderError as e:
                raise ValueError(f"Failed to load tools config: {e}")
            except ImportError:
                raise ImportError("mcp-use is required for tools loading")

        # Load agents registry → agents_by_id, write canonical agents.json, cache agents_digest
        if agents_path:
            try:
                agents_result = load_agents_config(agents_path)
                self.agents_by_id = agents_result.by_id
                
                # Write canonical agents.json 
                canonical_path = Path("agents.json")
                import json
                with open(canonical_path, "w") as f:
                    json.dump(agents_result.canonical_json, f, indent=2)
                    
                # Cache agents digest (for future use)
                self.agents_digest = hash(json.dumps(agents_result.canonical_json, sort_keys=True))
                
            except AgentsLoaderError as e:
                raise ValueError(f"Failed to load agents config: {e}")
                
        # Calculate tools digest for agent caching
        if tools_file:
            try:
                # Create tools digest from config for agent cache invalidation
                tools_config_str = json.dumps(tools_config, sort_keys=True)
                self.tools_digest = str(hash(tools_config_str))
            except Exception:
                self.tools_digest = "unknown_tools"

        # Initialize empty LLM cache and agent cache
        self.llm_cache = {}
        self.agent_cache = {}
        self.tools_digest = None

    async def _resolve_agent_spec(self, node_config: dict[str, Any]) -> dict[str, Any]:
        """Resolve agent specification from node config and registry."""
        # Check if agent_id is provided (registry lookup)
        agent_id = node_config.get("agent_id")
        if agent_id:
            # Load from registry
            if not self.agents_by_id:
                raise ValueError(f"Agent ID '{agent_id}' specified but no --agents provided")
            if agent_id not in self.agents_by_id:
                available_agents = list(self.agents_by_id.keys())
                raise ValueError(
                    f"Agent ID '{agent_id}' not found in registry. "
                    f"Available agents: {available_agents}"
                )
            
            spec = self.agents_by_id[agent_id].copy()
        else:
            # Inline fallback: synthesize spec from config.model
            model = node_config.get("model")
            if not model:
                if self.agents_by_id:
                    raise ValueError(
                        f"Either agent_id or model must be specified. "
                        f"Available agents: {list(self.agents_by_id.keys())}"
                    )
                else:
                    raise ValueError("Either agent_id or model must be specified in config")
            
            print(f"⚠️  WARNING: Using inline fallback for model '{model}'. "
                  f"Consider using --agents with agent_id instead.")
            
            # Synthesize basic spec
            spec = {
                "id": f"__inline@v0",
                "model": model,
                "max_steps": 30,
                "use_server_manager": True
            }
            
            # Pass through API key if provided
            if "api_key" in node_config:
                spec["api_key"] = node_config["api_key"]
        
        # Apply node-level overrides (Node > agent spec for behavioral knobs)
        if "max_steps" in node_config:
            # Use minimum for safety: min(spec.max_steps, node.max_steps)
            spec["max_steps"] = min(spec.get("max_steps", 30), node_config["max_steps"])
            
        # Apply server pinning precedence: Node server > agent default_server > auto-route
        if "server" in node_config:
            spec["default_server"] = node_config["server"]
            
        return spec

    async def run(
        self,
        graph: Graph | dict[str, Any] | str | Path,
        ctx: dict[str, Any] | None = None,
        run_id: str | None = None,
        tools_file: str | Path | None = None,
        agents_path: str | Path | None = None
    ) -> RunResult:
        """Execute a graph and return the final result.
        
        Args:
            graph: Graph specification as Graph object, dict, or file path.
            ctx: Additional context to merge with graph context.
            run_id: Custom run ID, generates UUID if not provided.
            
        Returns:
            RunResult with execution metrics and outputs.
            
        Raises:
            ValueError: If graph specification is invalid.
            RuntimeError: If execution fails unexpectedly.
        """

        # Prepare execution
        run_id = run_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        # Parse graph
        if isinstance(graph, (str, Path)):
            graph = Graph.from_file(graph, run_id=run_id, trace_id=trace_id)
        elif isinstance(graph, dict):
            spec = GraphSpec(**graph)
            graph = Graph(spec=spec, run_id=run_id, trace_id=trace_id)

        # Merge context
        if ctx:
            graph.spec.ctx.update(ctx)

        # Set up event logging
        event_emitter = EventEmitter([
            JSONLSink(self.event_dir / f"{run_id}.jsonl"),
            EventBuffer()  # For inspection
        ])

        # Bootstrap (once per run)
        await self._bootstrap(tools_file, agents_path, run_id, trace_id, event_emitter)

        try:
            # Execute the full workflow
            async for _ in self.astream(graph, event_emitter):
                pass  # Events are handled by the emitter

            # Get final results
            buffer = event_emitter.sinks[1]  # EventBuffer
            if isinstance(buffer, EventBuffer):
                end_events = buffer.get_events(EventType.RUN_END)
            else:
                end_events = []

            if end_events:
                final_event = end_events[-1]
                return RunResult(
                    run_id=run_id,
                    trace_id=trace_id,
                    success=True,
                    outputs=final_event.payload.get("outputs", {}),
                    total_tokens=final_event.payload.get("total_tokens", 0),
                    total_cost=final_event.payload.get("total_cost", 0.0),
                    total_time=final_event.payload.get("total_time", 0.0)
                )
            else:
                return RunResult(
                    run_id=run_id,
                    trace_id=trace_id,
                    success=False,
                    outputs={},
                    error="Execution did not complete properly"
                )

        except Exception as e:
            await event_emitter.emit(Event(
                type=EventType.RUN_ERROR,
                run_id=run_id,
                trace_id=trace_id,
                span_id=str(uuid.uuid4()),
                payload={"error": str(e)}
            ))

            return RunResult(
                run_id=run_id,
                trace_id=trace_id,
                success=False,
                outputs={},
                error=str(e)
            )

        finally:
            # Clean up MCP client
            if self.mcp_client:
                try:
                    mcp_adapter = adapter_registry.get("mcp_use")
                    if mcp_adapter and hasattr(mcp_adapter, 'close_all'):
                        await mcp_adapter.close_all(
                            self.mcp_client,
                            run_id=run_id,
                            trace_id=trace_id,
                            span_id=str(uuid.uuid4())
                        )
                    self.mcp_client = None
                except Exception as e:
                    print(f"Error cleaning up MCP client: {e}")
            
            await event_emitter.close()

    async def astream(
        self,
        graph: Graph,
        event_emitter: EventEmitter | None = None
    ) -> AsyncIterator[Event]:
        """Stream execution events as the graph executes.
        
        Args:
            graph: Graph object to execute.
            event_emitter: Optional event emitter, creates default if None.
            
        Yields:
            Event: Execution events including start, node events, and end.
            
        Raises:
            Exception: Any exception during graph execution is re-raised.
        """

        if event_emitter is None:
            event_emitter = EventEmitter([EventBuffer()])

        run_id = graph.run_id
        trace_id = graph.trace_id
        start_time = time.time()

        # Create deterministic context
        det_context = create_deterministic_context()

        # Emit start event
        start_event = Event(
            type=EventType.RUN_START,
            run_id=run_id,
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            payload={
                "graph_spec": graph.spec.model_dump(),
                "context": graph.spec.ctx
            }
        )
        await event_emitter.emit(start_event)
        yield start_event

        try:
            # Execute DAG
            execution_state = {
                "completed_nodes": [],
                "failed_nodes": [],
                "node_results": {},
                "current_context": graph.spec.ctx.copy()
            }
            
            # Pre-resolve all agent specs for mcp_agent nodes
            resolved_agent_specs = {}
            for node_id, node_config in graph.spec.nodes.items():
                if node_config.type == "mcp_agent":
                    try:
                        resolved_spec = await self._resolve_agent_spec(node_config.config)
                        resolved_agent_specs[node_id] = resolved_spec
                    except Exception as e:
                        # Emit error and fail early
                        error_event = Event(
                            type=EventType.NODE_ERROR,
                            run_id=graph.run_id,
                            trace_id=graph.trace_id,
                            span_id=str(uuid.uuid4()),
                            node_id=node_id,
                            payload={"error": f"Agent spec resolution failed: {str(e)}"}
                        )
                        await event_emitter.emit(error_event)
                        yield error_event
                        raise ValueError(f"Failed to resolve agent spec for node {node_id}: {str(e)}")

            # Add bootstrap state to context
            execution_state["current_context"].update({
                "agents_by_id": self.agents_by_id,
                "llm_cache": self.llm_cache,
                "llm_factory": self.llm_factory,
                "mcp_client": self.mcp_client,
                "resolved_agent_specs": resolved_agent_specs,
                "agent_cache": self.agent_cache,
                "tools_digest": self.tools_digest
            })

            async for event in self._execute_dag(
                graph, execution_state, det_context, event_emitter
            ):
                yield event

            # Calculate final metrics
            end_time = time.time()
            total_time = end_time - start_time

            # Emit end event
            end_event = Event(
                type=EventType.RUN_END,
                run_id=run_id,
                trace_id=trace_id,
                span_id=str(uuid.uuid4()),
                payload={
                    "outputs": execution_state["node_results"],
                    "total_time": total_time,
                    "total_tokens": self.budget.used_tokens,
                    "total_cost": self.budget.used_cost,
                    "completed_nodes": execution_state["completed_nodes"],
                    "failed_nodes": execution_state["failed_nodes"]
                }
            )
            await event_emitter.emit(end_event)
            yield end_event

        except Exception as e:
            error_event = Event(
                type=EventType.RUN_ERROR,
                run_id=run_id,
                trace_id=trace_id,
                span_id=str(uuid.uuid4()),
                payload={"error": str(e)}
            )
            await event_emitter.emit(error_event)
            yield error_event
            raise

    async def resume(self, checkpoint_id: str) -> RunResult:
        """Resume execution from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to resume from.
            
        Returns:
            RunResult with execution metrics and outputs.
            
        Raises:
            ValueError: If checkpoint is not found.
        """

        # Load checkpoint
        checkpoint = self.checkpointer.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Recreate graph
        graph = Graph(
            spec=GraphSpec(**checkpoint.graph_spec),
            run_id=checkpoint.run_id,
            trace_id=checkpoint.trace_id
        )

        # Set up event logging
        event_emitter = EventEmitter([
            JSONLSink(self.event_dir / f"{checkpoint.run_id}_resume.jsonl"),
            EventBuffer()
        ])

        # Resume execution
        try:
            async for _ in self._resume_from_checkpoint(checkpoint, graph, event_emitter):
                pass

            # Get final results
            buffer = event_emitter.sinks[1]
            if isinstance(buffer, EventBuffer):
                end_events = buffer.get_events(EventType.RUN_END)
            else:
                end_events = []

            if end_events:
                final_event = end_events[-1]
                return RunResult(
                    run_id=checkpoint.run_id,
                    trace_id=checkpoint.trace_id,
                    success=True,
                    outputs=final_event.payload.get("outputs", {}),
                    total_tokens=final_event.payload.get("total_tokens", 0),
                    total_cost=final_event.payload.get("total_cost", 0.0),
                    total_time=final_event.payload.get("total_time", 0.0)
                )
            else:
                return RunResult(
                    run_id=checkpoint.run_id,
                    trace_id=checkpoint.trace_id,
                    success=False,
                    outputs={},
                    error="Resume did not complete properly"
                )

        finally:
            # Clean up MCP client  
            if self.mcp_client:
                try:
                    mcp_adapter = adapter_registry.get("mcp_use")
                    if mcp_adapter and hasattr(mcp_adapter, 'close_all'):
                        await mcp_adapter.close_all(
                            self.mcp_client,
                            run_id=checkpoint.run_id,
                            trace_id=checkpoint.trace_id,
                            span_id=str(uuid.uuid4())
                        )
                    self.mcp_client = None
                except Exception as e:
                    print(f"Error cleaning up MCP client: {e}")
                    
            await event_emitter.close()

    async def _execute_dag(
        self,
        graph: Graph,
        execution_state: dict[str, Any],
        det_context: Any,
        event_emitter: EventEmitter
    ) -> AsyncIterator[Event]:
        """Execute the DAG with topological ordering.
        
        Args:
            graph: Graph object containing nodes and dependencies.
            execution_state: Current execution state with completed/failed nodes.
            det_context: Deterministic context for reproducible execution.
            event_emitter: Event emitter for logging execution events.
            
        Yields:
            Event: Various execution events including node and budget events.
        """

        execution_order = graph.get_execution_order()

        for node_id in execution_order:
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(node_id, graph, execution_state):
                continue

            # Check budget
            if self.budget.is_exceeded():
                budget_event = Event(
                    type=EventType.BUDGET_EXCEEDED,
                    run_id=graph.run_id,
                    trace_id=graph.trace_id,
                    span_id=str(uuid.uuid4()),
                    node_id=node_id,
                    payload={"budget": self.budget.__dict__}
                )
                await event_emitter.emit(budget_event)
                yield budget_event
                break

            # Execute node
            async for event in self._execute_node(
                node_id, graph, execution_state, det_context, event_emitter
            ):
                yield event

            # Create checkpoint after each node (exclude unpickleable objects)
            checkpoint_context = execution_state["current_context"].copy()
            # Remove unpickleable objects for checkpointing
            checkpoint_context.pop("llm_cache", None)
            checkpoint_context.pop("llm_factory", None) 
            checkpoint_context.pop("mcp_client", None)
            checkpoint_context.pop("agent_cache", None)
            
            checkpoint = self.checkpointer.create_checkpoint(
                run_id=graph.run_id,
                trace_id=graph.trace_id,
                graph_spec=graph.spec.model_dump(),
                node_states={},  # TODO: Collect node states
                completed_nodes=execution_state["completed_nodes"],
                failed_nodes=execution_state["failed_nodes"],
                context=checkpoint_context,
                usage={
                    "total_tokens": self.budget.used_tokens,
                    "total_cost": self.budget.used_cost,
                    "total_children": self.budget.used_children
                }
            )

            self.checkpointer.save_checkpoint(checkpoint)

            checkpoint_event = Event(
                type=EventType.CHECKPOINT_SAVED,
                run_id=graph.run_id,
                trace_id=graph.trace_id,
                span_id=str(uuid.uuid4()),
                payload={"checkpoint_id": checkpoint.checkpoint_id}
            )
            await event_emitter.emit(checkpoint_event)
            yield checkpoint_event

    async def _execute_node(
        self,
        node_id: str,
        graph: Graph,
        execution_state: dict[str, Any],
        _det_context: Any,
        event_emitter: EventEmitter
    ) -> AsyncIterator[Event]:
        """Execute a single node.
        
        Args:
            node_id: ID of the node to execute.
            graph: Graph object containing node configuration.
            execution_state: Current execution state to update.
            _det_context: Deterministic context (unused in current implementation).
            event_emitter: Event emitter for logging node events.
            
        Yields:
            Event: Node execution events including errors.
        """

        async with self._semaphore:
            # Get node configuration
            node_config = graph.get_node_config(node_id)

            # Get adapter
            adapter_name = node_config.adapter
            adapter = adapter_registry.get(adapter_name)
            if not adapter:
                raise ValueError(f"Adapter not found: {adapter_name}")

            # Create node
            node = create_node(
                node_id=node_id,
                node_type=node_config.type,
                config=node_config.config,
                adapter=adapter,
                event_emitter=event_emitter
            )

            # Prepare inputs
            inputs = self._prepare_node_inputs(node_id, graph, execution_state)

            # Execute node
            span_id = str(uuid.uuid4())

            try:
                result = await node.execute(
                    inputs=inputs,
                    context=execution_state["current_context"],
                    trace_id=graph.trace_id,
                    span_id=span_id
                )

                if result.success:
                    execution_state["completed_nodes"].append(node_id)
                    execution_state["node_results"][node_id] = result.output
                else:
                    execution_state["failed_nodes"].append(node_id)

                # Update budget (placeholder - actual usage should come from adapter)
                self.budget.consume_tokens(100)  # Estimate
                self.budget.consume_cost(0.01)   # Estimate

            except Exception as e:
                execution_state["failed_nodes"].append(node_id)
                error_event = Event(
                    type=EventType.NODE_ERROR,
                    run_id=graph.run_id,
                    trace_id=graph.trace_id,
                    span_id=span_id,
                    node_id=node_id,
                    payload={"error": str(e)}
                )
                await event_emitter.emit(error_event)
                yield error_event

    def _dependencies_satisfied(
        self,
        node_id: str,
        graph: Graph,
        execution_state: dict[str, Any]
    ) -> bool:
        """Check if all dependencies for a node are satisfied.
        
        Args:
            node_id: ID of the node to check dependencies for.
            graph: Graph object containing dependency information.
            execution_state: Current execution state with completed nodes.
            
        Returns:
            True if all dependencies are satisfied, False otherwise.
        """
        dependencies = graph.spec.get_dependencies(node_id)
        completed = set(execution_state["completed_nodes"])
        return dependencies.issubset(completed)

    def _prepare_node_inputs(
        self,
        node_id: str,
        graph: Graph,
        execution_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare inputs for a node based on its dependencies.
        
        Args:
            node_id: ID of the node to prepare inputs for.
            graph: Graph object containing dependency information.
            execution_state: Current execution state with node results.
            
        Returns:
            Dictionary of inputs combining dependency outputs and context.
        """
        inputs = {}

        # Get outputs from dependency nodes
        dependencies = graph.spec.get_dependencies(node_id)
        for dep_node in dependencies:
            if dep_node in execution_state["node_results"]:
                inputs[dep_node] = execution_state["node_results"][dep_node]

        # Add initial context
        inputs.update(graph.spec.ctx)

        return inputs

    async def _resume_from_checkpoint(
        self,
        checkpoint: CheckpointState,
        graph: Graph,
        event_emitter: EventEmitter
    ) -> AsyncIterator[Event]:
        """Resume execution from a checkpoint.
        
        Args:
            checkpoint: Checkpoint state to resume from.
            graph: Graph object to continue executing.
            event_emitter: Event emitter for logging execution events.
            
        Yields:
            Event: Execution events from the resumed run.
        """

        # Restore execution state
        execution_state = {
            "completed_nodes": checkpoint.completed_nodes.copy(),
            "failed_nodes": checkpoint.failed_nodes.copy(),
            "node_results": checkpoint.node_states.copy(),
            "current_context": checkpoint.context.copy()
        }

        # Restore deterministic context
        det_context = create_deterministic_context(checkpoint)

        # Restore budget
        self.budget.used_tokens = checkpoint.total_tokens
        self.budget.used_cost = checkpoint.total_cost
        self.budget.used_children = checkpoint.total_children

        # Continue execution
        async for event in self._execute_dag(
            graph, execution_state, det_context, event_emitter
        ):
            yield event

