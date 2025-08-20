from __future__ import annotations
import asyncio
from typing import Any, AsyncGenerator, Dict, Optional, Set
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from .types import Event, GraphSpec, NodeSpec, RunSpec
from .executors import Executor
from .utils import topo_sort

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, executor: Executor):
        self._executor = executor
        self._event_seq = 0

    def _next_event_seq(self) -> int:
        """Generate monotonic event sequence for deterministic logging."""
        self._event_seq += 1
        return self._event_seq

    def _emit_event(self, event_type: str, run_id: str, node_id: Optional[str] = None, 
                   data: Optional[Dict[str, Any]] = None) -> Event:
        """Create event with sequence number."""
        return Event(
            type=event_type,  # type: ignore
            run_id=run_id,
            node_id=node_id,
            data=data or {},
            event_seq=self._next_event_seq()
        )

    async def _run_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                       stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a single node with retry logic, timeout, and streaming support."""
        
        # Emit NODE_START event
        yield self._emit_event("NODE_START", run.run_id, node.id, {
            "type": node.type, 
            "phase": f"{node.type}:start"
        })
        
        # Handle different node types
        if node.type == "task":
            async for event in self._run_task_node(node, ctx, run, stream):
                yield event
        elif node.type == "foreach":
            async for event in self._run_foreach_node(node, ctx, run, stream):
                yield event
        elif node.type == "reduce":
            async for event in self._run_reduce_node(node, ctx, run, stream):
                yield event
        elif node.type == "gate":
            async for event in self._run_gate_node(node, ctx, run, stream):
                yield event
        else:
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": f"Unknown node type: {node.type}"
            })

    async def _timeout_iterator(self, async_generator: AsyncGenerator, timeout: float) -> AsyncGenerator: # type: ignore
        """A helper to iterate over an async generator with a timeout."""
        try:
            while True:
                yield await asyncio.wait_for(async_generator.__anext__(), timeout) # type: ignore
        except StopAsyncIteration:
            pass

    async def _run_task_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                            stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a task node with retries and timeout."""
        
        async def execute_with_retry_streaming() -> AsyncGenerator[Dict[str, Any], None]:
            # Set up retry logic if retries > 0
            if node.retries > 0:
                @retry(
                    stop=stop_after_attempt(node.retries + 1),
                    wait=wait_exponential(multiplier=node.retry_backoff_s),
                    reraise=True
                )
                async def retry_execute() -> AsyncGenerator[Dict[str, Any], None]:
                    async for chunk in self._execute_single_attempt_streaming(node, ctx):
                        yield chunk
                
                async for chunk in retry_execute():
                    yield chunk
            else:
                async for chunk in self._execute_single_attempt_streaming(node, ctx):
                    yield chunk
        
        async def execute_with_retry_regular() -> Any:
            # Set up retry logic if retries > 0
            if node.retries > 0:
                @retry(
                    stop=stop_after_attempt(node.retries + 1),
                    wait=wait_exponential(multiplier=node.retry_backoff_s),
                    reraise=True
                )
                async def retry_execute() -> Any:
                    return await self._execute_single_attempt_regular(node, ctx)
                
                return await retry_execute()
            else:
                return await self._execute_single_attempt_regular(node, ctx)
        
        try:
            if stream:
                stream_generator = execute_with_retry_streaming()
                if node.timeout_s:
                    try:
                        async for chunk in self._timeout_iterator(stream_generator, node.timeout_s): # type: ignore
                            yield self._emit_event("AGENT_CHUNK", run.run_id, node.id, chunk) # type: ignore
                    except asyncio.TimeoutError:
                        yield self._emit_event("ERROR", run.run_id, node.id, {"error": f"Node execution timed out after {node.timeout_s} seconds"})
                        return
                else:
                    async for chunk in stream_generator:
                        yield self._emit_event("AGENT_CHUNK", run.run_id, node.id, chunk)
                
                yield self._emit_event("NODE_COMPLETE", run.run_id, node.id, {
                    "output_meta": list(ctx["blackboard"].get(node.id, {}).keys()),
                    "phase": "task:complete"
                })
            else:
                if node.timeout_s:
                    result = await asyncio.wait_for(execute_with_retry_regular(), timeout=node.timeout_s)
                else:
                    result = await execute_with_retry_regular()
                
                ctx["blackboard"][node.id] = {"result": result, "source": node.id}
                yield self._emit_event("NODE_COMPLETE", run.run_id, node.id, {
                    "output_meta": list(ctx["blackboard"][node.id].keys()),
                    "phase": "task:complete"
                })

        except Exception as e:
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": repr(e), 
                "phase": "task:error"
            })

    async def _execute_single_attempt_streaming(self, node: NodeSpec, ctx: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single streaming attempt of a node."""
        if hasattr(self._executor, 'execute_with_stream'):
            chunk_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
            
            async def on_chunk(chunk: Dict[str, Any]) -> None:
                await chunk_queue.put(chunk)
            
            async def stream_execution() -> Any:
                try:
                    result = await self._executor.execute_with_stream(node, ctx, on_chunk)  # type: ignore
                    await chunk_queue.put(None)  # Sentinel
                    return result
                except Exception as e:
                    await chunk_queue.put(None)
                    raise
            
            execute_task = asyncio.create_task(stream_execution())
            final_result = None
            
            try:
                while True:
                    chunk = await chunk_queue.get()
                    if chunk is None:
                        final_result = await execute_task
                        break
                    else:
                        yield chunk
                
                # Store result in blackboard
                ctx["blackboard"][node.id] = {"result": final_result, "source": node.id}
                
            except Exception:
                execute_task.cancel()
                try:
                    await execute_task
                except asyncio.CancelledError:
                    pass
                raise
        else:
            # No streaming support, do regular execution
            result = await self._executor.execute(node, ctx)
            ctx["blackboard"][node.id] = {"result": result, "source": node.id}
    
    async def _execute_single_attempt_regular(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single non-streaming attempt of a node."""
        result = await self._executor.execute(node, ctx)
        return result

    async def _run_foreach_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                               stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a foreach node with concurrency control."""
        items = node.inputs.get("items", [])
        if not isinstance(items, list):
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": "foreach node requires 'items' list in inputs"
            })
            return
        
        yield self._emit_event("NODE_START", run.run_id, node.id, {
            "phase": "foreach:start", 
            "item_count": len(items)
        })
        
        results = []
        semaphore = asyncio.Semaphore(node.concurrency or len(items))
        
        # Use agent pool for foreach items if available, otherwise regular task execution
        async def process_item(item: Any, index: int) -> Any:
            async with semaphore:
                # Create virtual sub-node
                sub_node = NodeSpec(
                    id=f"{node.id}:{index}",
                    type="task",
                    inputs={"item": item, **node.inputs},
                    timeout_s=node.timeout_s,
                    retries=node.retries,
                    retry_backoff_s=node.retry_backoff_s,
                    retry_on_timeout=node.retry_on_timeout
                )
                
                try:
                    # Use agent pool execution if available
                    if hasattr(self._executor, 'execute_foreach_item'):
                        result = await self._executor.execute_foreach_item(sub_node, ctx, index)
                        ctx["blackboard"][sub_node.id] = {"result": result, "source": sub_node.id}
                        return result
                    else:
                        # Fallback to regular task execution
                        item_result = None
                        async for event in self._run_task_node(sub_node, ctx, run, stream=False):
                            if event.type == "NODE_COMPLETE":
                                item_result = ctx["blackboard"].get(sub_node.id, {}).get("result")
                        return item_result
                except Exception as e:
                    if node.foreach_fail_policy == "fail_fast":
                        raise
                    else:  # skip
                        return None
        
        try:
            # Execute all items with asyncio.gather
            tasks = [process_item(item, i) for i, item in enumerate(items)]
            results = await asyncio.gather(*tasks)
            
            ctx["blackboard"][node.id] = {"items": results, "source": node.id}
            yield self._emit_event("NODE_COMPLETE", run.run_id, node.id, {
                "output_meta": ["items", "source"],
                "phase": "foreach:complete",
                "item_count": len(results)
            })
            
        except Exception as e:
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": repr(e),
                "phase": "foreach:error"
            })

    async def _run_reduce_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                              stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a reduce node gathering from specified sources."""
        from_ids = node.inputs.get("from_ids", [])
        
        # Default to immediate parents if from_ids not specified
        if not from_ids:
            # Find immediate parents in graph
            graph_edges = getattr(ctx, "_graph_edges", [])
            from_ids = [edge[0] for edge in graph_edges if edge[1] == node.id]
        
        # Validate all required inputs exist and normalize shapes
        values = []
        for from_id in from_ids:
            if from_id not in ctx["blackboard"]:
                yield self._emit_event("ERROR", run.run_id, node.id, {
                    "error": f"Required input '{from_id}' not found in blackboard",
                    "phase": "reduce:validation_error"
                })
                return
            
            # Get the blackboard entry
            entry = ctx["blackboard"][from_id]
            
            # Normalize shape - ensure we have a proper dict structure
            if isinstance(entry, dict):
                values.append(entry)
            elif isinstance(entry, str):
                # Wrap raw strings in a dict structure
                values.append({"result": entry, "source": from_id})
            else:
                # Wrap other raw values
                values.append({"result": entry, "source": from_id})
        
        # Create reduce node with values
        reduce_node = NodeSpec(
            id=node.id,
            type="task",
            inputs={"values": values, **node.inputs},
            timeout_s=node.timeout_s,
            retries=node.retries,
            retry_backoff_s=node.retry_backoff_s,
            retry_on_timeout=node.retry_on_timeout
        )
        
        # Execute reduction
        try:
            result = None
            async for event in self._run_task_node(reduce_node, ctx, run, stream):
                if event.type == "AGENT_CHUNK":
                    # Create new data dict with phase info (avoid modifying existing data)
                    if isinstance(event.data, dict):
                        new_data = dict(event.data)  # Create mutable copy
                        new_data["phase"] = "reduce:process"
                    else:
                        # Wrap non-dict data in a proper structure
                        new_data = {"content": event.data, "phase": "reduce:process"}
                    
                    # Create new event with normalized data
                    event = Event(
                        type=event.type,
                        run_id=event.run_id,
                        node_id=event.node_id,
                        data=new_data,
                        event_seq=event.event_seq
                    )
                    yield event
                elif event.type == "NODE_COMPLETE":
                    result = ctx["blackboard"].get(node.id, {}).get("result")
            
            # Store with proper format
            ctx["blackboard"][node.id] = {"reduced": result, "source": node.id}
            yield self._emit_event("NODE_COMPLETE", run.run_id, node.id, {
                "output_meta": ["reduced", "source"],
                "phase": "reduce:complete"
            })
            
        except Exception as e:
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": repr(e),
                "phase": "reduce:error"
            })

    async def _run_gate_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                            stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a gate node for conditional flow control."""
        predicate = node.inputs.get("predicate")
        if predicate is None:
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": "gate node requires 'predicate' in inputs"
            })
            return
        
        try:
            # Directly evaluate the predicate (for simple boolean gates)
            # In a more complex implementation, this could evaluate expressions from blackboard
            passed = bool(predicate)
            ctx["blackboard"][node.id] = {"result": passed, "source": node.id}
            
            if passed:
                yield self._emit_event("NODE_COMPLETE", run.run_id, node.id, {
                    "output_meta": ["result", "source"],
                    "phase": "gate:passed"
                })
            else:
                yield self._emit_event("NODE_COMPLETE", run.run_id, node.id, {
                    "output_meta": ["result", "source"],
                    "phase": "gate:blocked",
                    "skipped": True
                })
                
        except Exception as e:
            yield self._emit_event("ERROR", run.run_id, node.id, {
                "error": repr(e),
                "phase": "gate:error"
            })

    async def run(self, graph: GraphSpec, run: RunSpec) -> AsyncGenerator[Event, None]:
        """Run orchestration with sequential execution (backward compatible)."""
        async for event in self._run_orchestration(graph, run, stream=False):
            yield event

    async def run_streaming(self, graph: GraphSpec, run: RunSpec) -> AsyncGenerator[Event, None]:
        """Run orchestration with streaming support."""
        async for event in self._run_orchestration(graph, run, stream=True):
            yield event

    async def _run_orchestration(self, graph: GraphSpec, run: RunSpec, stream: bool = False) -> AsyncGenerator[Event, None]:
        """Core orchestration logic with concurrency support."""
        ctx: Dict[str, Any] = {"blackboard": {}, "_graph_edges": graph.edges}
        skipped_nodes: Set[str] = set()
        
        # Set run context on executor if it supports it (for agent pool management)
        if hasattr(self._executor, 'set_run_context'):
            self._executor.set_run_context(run.run_id)
        
        yield self._emit_event("RUN_START", run.run_id, data={"goal": run.goal})
        
        # Compute topological order
        order = topo_sort(graph)
        node_map: Dict[str, NodeSpec] = {n.id: n for n in graph.nodes}
        
        # Execute nodes in topological order
        for node_id in order:
            if node_id in skipped_nodes:
                continue
                
            node = node_map[node_id]
            
            try:
                # Execute single node
                gate_skipped = False
                async for event in self._run_node(node, ctx, run, stream):
                    yield event
                    
                    # Check for gate that blocked flow
                    if (event.type == "NODE_COMPLETE" and 
                        event.data.get("phase") == "gate:blocked"):
                        gate_skipped = True
                
                # If gate blocked, skip all transitive successors
                if gate_skipped:
                    successors = self._get_transitive_successors(graph, node_id)
                    skipped_nodes.update(successors)
                    
            except Exception as e:
                yield self._emit_event("ERROR", run.run_id, node_id, {"error": repr(e)})
                return
        
        yield self._emit_event("RUN_COMPLETE", run.run_id, data={"result": ctx["blackboard"]})
        
        # Clean up agent pool if available
        if hasattr(self._executor, '_agent_pool') and self._executor._agent_pool:
            await self._executor._agent_pool.finish_run(run.run_id)

    def _get_transitive_successors(self, graph: GraphSpec, node_id: str) -> Set[str]:
        """Get all transitive successors of a node."""
        successors = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            for edge in graph.edges:
                if edge[0] == current and edge[1] not in successors:
                    successors.add(edge[1])
                    queue.append(edge[1])
        
        return successors