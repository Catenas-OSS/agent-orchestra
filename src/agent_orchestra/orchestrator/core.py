"""
Core orchestration engine for Agent Orchestra.

This module contains the main Orchestrator class that executes workflow graphs
with support for conditional flows, parallel processing, and error handling.
"""
from __future__ import annotations
import asyncio
from typing import Any, AsyncGenerator, Dict, Optional, Set
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from .types import Event, GraphSpec, NodeSpec, RunSpec
from .executors import Executor
from .utils import topo_sort
from .signature import node_signature, foreach_item_signature
from .store import RunStore, SavedNode, SavedForeachItem
from .store_factory import create_store
from ..logging import get_system_logger

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, executor: Executor, store: Optional[RunStore] = None):
        self._executor = executor
        # Auto-create store if none provided - defaults to SQLite for best performance
        self._store = store if store is not None else create_store("sqlite")
        self._event_seq = 0
        self._gate_pruned_nodes: Set[str] = set()
        self._server_configs: Optional[Dict[str, Any]] = None
        self._system_logger = get_system_logger()
        self._system_logger.info("orchestrator", "Orchestrator initialized")

    def _next_event_seq(self) -> int:
        """Generate monotonic event sequence for deterministic logging."""
        self._event_seq += 1
        return self._event_seq

    async def _emit_event(self, event_type: str, run_spec: RunSpec, node: Optional[NodeSpec] = None, 
                         data: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Event, None]:
        """Create and emit event with persistence support."""
        event = Event(
            type=event_type,  # type: ignore
            run_id=run_spec.run_id,
            node_id=node.id if node else None,
            data=data or {},
            event_seq=self._next_event_seq()
        )
        
        # Persist event if store available
        if self._store:
            await self._store.append_event(run_spec.run_id, event)
        
        yield event

    def _emit_event_sync(self, event_type: str, run_id: str, node_id: Optional[str] = None, 
                        data: Optional[Dict[str, Any]] = None) -> Event:
        """Create event with sequence number (backward compatibility helper)."""
        return Event(
            type=event_type,  # type: ignore
            run_id=run_id,
            node_id=node_id,
            data=data or {},
            event_seq=self._next_event_seq()
        )

    async def _run_node_enhanced(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                                stream: bool = False) -> AsyncGenerator[Event, None]:
        """Enhanced node execution with persistence support, wrapping existing logic."""
        
        # Create a wrapper that intercepts events and persists them
        async def event_wrapper(events_generator):
            async for event in events_generator:
                # Persist event if store available
                if self._store:
                    await self._store.append_event(run.run_id, event)
                yield event
        
        # Use existing node execution logic with event wrapping
        async for event in event_wrapper(self._run_node(node, ctx, run, stream)):
            yield event

    async def _run_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                       stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a single node with retry logic, timeout, and streaming support."""
        
        # Emit enhanced NODE_START event with attempt tracking
        node_start_data = {
            "node_type": node.type,
            "type": node.type, 
            "phase": f"{node.type}:start",
            "attempt": 1,
            "max_attempts": node.retries + 1,
            "started_at": asyncio.get_event_loop().time(),
            "server_name": getattr(node, 'server_name', None)
        }
        
        # Add supervisor-specific fields
        if node.type == "supervisor":
            node_start_data.update({
                "available_agents": node.available_agents or {},
                "max_agent_calls": node.max_agent_calls
            })
        
        # Try to get model info from executor
        if hasattr(self._executor, '_model_name'):
            node_start_data["model"] = self._executor._model_name
        elif hasattr(self._executor, '_model_key'):
            node_start_data["model"] = self._executor._model_key
        
        yield self._emit_event_sync("NODE_START", run.run_id, node.id, node_start_data)
        
        # Emit AGENT_INSTRUCTIONS event with task details
        if node.type == "task":
            # Extract task information
            task_info = {
                "task": {
                    "title": node.name,
                    "body": self._build_task_prompt(node, ctx)
                },
                "system": "You are a helpful AI assistant executing a workflow task.",
                "tools": self._get_available_tools(node),
                "policy": {"allow": True, "redact": False},
                "context": {
                    "run_id": run.run_id,
                    "node_id": node.id,
                    "server_name": getattr(node, 'server_name', None)
                }
            }
            yield self._emit_event_sync("AGENT_INSTRUCTIONS", run.run_id, node.id, task_info)
        
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
        elif node.type == "supervisor":
            async for event in self._run_supervisor_node(node, ctx, run, stream):
                yield event
        else:
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
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
                            yield self._emit_event_sync("AGENT_CHUNK", run.run_id, node.id, chunk) # type: ignore
                    except asyncio.TimeoutError:
                        yield self._emit_event_sync("ERROR", run.run_id, node.id, {"error": f"Node execution timed out after {node.timeout_s} seconds"})
                        return
                else:
                    async for chunk in stream_generator:
                        yield self._emit_event_sync("AGENT_CHUNK", run.run_id, node.id, chunk)
                
                # Enhanced NODE_COMPLETE event
                node_result = ctx["blackboard"].get(node.id, {})
                result_data = node_result.get("result", {})
                
                complete_data = {
                    "output_meta": list(node_result.keys()),
                    "phase": "task:complete",
                    "ended_at": asyncio.get_event_loop().time(),
                    "resumed": False  # TODO: Track actual resume state
                }
                
                # Extract output summary (normalize result|output|text)
                output_summary = self._extract_output_summary(result_data)
                if output_summary:
                    complete_data["output_summary"] = output_summary
                
                # Add token/cost info if available
                if isinstance(result_data, dict):
                    if "usage" in result_data:
                        usage = result_data["usage"]
                        complete_data["tokens"] = {
                            "prompt": usage.get("prompt_tokens", 0),
                            "completion": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0)
                        }
                    if "cost" in result_data:
                        complete_data["cost"] = result_data["cost"]
                
                # TODO: Add artifacts detection
                complete_data["artifacts"] = []
                
                yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, complete_data)
            else:
                if node.timeout_s:
                    result = await asyncio.wait_for(execute_with_retry_regular(), timeout=node.timeout_s)
                else:
                    result = await execute_with_retry_regular()
                
                ctx["blackboard"][node.id] = {"result": result, "source": node.id}
                
                # Debug logging for blackboard storage
                logger.info(f"BLACKBOARD STORE: Node {node.id} storing result type: {type(result)}")
                logger.info(f"BLACKBOARD STORE: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                if isinstance(result, dict) and "output" in result:
                    logger.info(f"BLACKBOARD STORE: Output preview: {str(result['output'])[:200]}...")
                else:
                    logger.info(f"BLACKBOARD STORE: Raw result preview: {str(result)[:200]}...")
                
                # Enhanced NODE_COMPLETE event (non-streaming)
                complete_data = {
                    "output_meta": list(ctx["blackboard"][node.id].keys()),
                    "phase": "task:complete",
                    "ended_at": asyncio.get_event_loop().time(),
                    "resumed": False
                }
                
                # Extract output summary
                output_summary = self._extract_output_summary(result)
                if output_summary:
                    complete_data["output_summary"] = output_summary
                
                # Add token/cost info if available
                if isinstance(result, dict):
                    if "usage" in result:
                        usage = result["usage"]
                        complete_data["tokens"] = {
                            "prompt": usage.get("prompt_tokens", 0),
                            "completion": usage.get("completion_tokens", 0),
                            "total": usage.get("total_tokens", 0)
                        }
                    if "cost" in result:
                        complete_data["cost"] = result["cost"]
                
                complete_data["artifacts"] = []
                
                yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, complete_data)

        except Exception as e:
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
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
                
                # Debug logging for streaming blackboard storage
                logger.info(f"BLACKBOARD STORE (STREAM): Node {node.id} storing result type: {type(final_result)}")
                logger.info(f"BLACKBOARD STORE (STREAM): Result keys: {list(final_result.keys()) if isinstance(final_result, dict) else 'Not dict'}")
                if isinstance(final_result, dict) and "output" in final_result:
                    logger.info(f"BLACKBOARD STORE (STREAM): Output preview: {str(final_result['output'])[:200]}...")
                else:
                    logger.info(f"BLACKBOARD STORE (STREAM): Raw result preview: {str(final_result)[:200]}...")
                
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
        """Execute a foreach node with concurrency control and granular resume support."""
        items = node.inputs.get("items", [])
        if not isinstance(items, list):
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
                "error": "foreach node requires 'items' list in inputs"
            })
            return
        
        # Load existing foreach item results for granular resume
        saved_items = {}
        if self._store:
            try:
                saved_items = await self._store.load_foreach_items(run.run_id, node.id)
            except Exception as e:
                logger.warning(f"Failed to load foreach items for {node.id}: {e}")
        
        yield self._emit_event_sync("NODE_START", run.run_id, node.id, {
            "phase": "foreach:start", 
            "item_count": len(items),
            "cached_items": len(saved_items)
        })
        
        results = [None] * len(items)  # Pre-allocate results array
        semaphore = asyncio.Semaphore(node.concurrency or len(items))
        
        # Use agent pool for foreach items if available, otherwise regular task execution
        async def process_item(item: Any, index: int) -> Any:
            async with semaphore:
                # Check if we have a cached result first (granular resume)
                if index in saved_items:
                    # Validate cached result by checking signature
                    saved_item = saved_items[index]
                    current_sig = foreach_item_signature(node, run, index, item, self._server_configs)
                    
                    if saved_item.signature == current_sig:
                        # Cache hit - use saved result
                        results[index] = saved_item.result
                        return saved_item.result
                
                # Cache miss or signature mismatch - execute the item
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
                    else:
                        # Fallback to regular task execution
                        item_result = None
                        async for event in self._run_task_node(sub_node, ctx, run, stream=False):
                            if event.type == "NODE_COMPLETE":
                                item_result = ctx["blackboard"].get(sub_node.id, {}).get("result")
                        result = item_result
                    
                    # Save individual item result for future granular resume
                    if self._store and result is not None:
                        item_sig = foreach_item_signature(node, run, index, item, self._server_configs)
                        saved_foreach_item = SavedForeachItem(node.id, index, item_sig, result)
                        await self._store.save_foreach_item(run.run_id, saved_foreach_item)
                    
                    results[index] = result
                    return result
                    
                except Exception as e:
                    if node.foreach_fail_policy == "fail_fast":
                        raise
                    else:  # skip
                        results[index] = None
                        return None
        
        try:
            # Execute all items with asyncio.gather
            tasks = [process_item(item, i) for i, item in enumerate(items)]
            await asyncio.gather(*tasks)
            
            # Filter out None results if skip policy is used
            final_results = [r for r in results if r is not None] if node.foreach_fail_policy == "skip" else results
            
            ctx["blackboard"][node.id] = {"items": final_results, "source": node.id}
            
            # Count cached vs executed items for reporting
            cached_count = len([i for i in range(len(items)) if i in saved_items and 
                              saved_items[i].signature == foreach_item_signature(node, run, i, items[i], self._server_configs)])
            executed_count = len(items) - cached_count
            
            yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, {
                "output_meta": ["items", "source"],
                "phase": "foreach:complete",
                "item_count": len(final_results),
                "cached_items": cached_count,
                "executed_items": executed_count
            })
            
        except Exception as e:
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
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
                yield self._emit_event_sync("ERROR", run.run_id, node.id, {
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
            yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, {
                "output_meta": ["reduced", "source"],
                "phase": "reduce:complete"
            })
            
        except Exception as e:
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
                "error": repr(e),
                "phase": "reduce:error"
            })

    async def _run_gate_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                            stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a gate node for conditional flow control."""
        predicate = node.inputs.get("predicate")
        if predicate is None:
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
                "error": "gate node requires 'predicate' in inputs"
            })
            return
        
        try:
            # Directly evaluate the predicate (for simple boolean gates)
            # In a more complex implementation, this could evaluate expressions from blackboard
            passed = bool(predicate)
            ctx["blackboard"][node.id] = {"result": passed, "source": node.id}
            
            if passed:
                yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, {
                    "output_meta": ["result", "source"],
                    "phase": "gate:passed"
                })
            else:
                yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, {
                    "output_meta": ["result", "source"],
                    "phase": "gate:blocked",
                    "skipped": True
                })
                
        except Exception as e:
            yield self._emit_event_sync("ERROR", run.run_id, node.id, {
                "error": repr(e),
                "phase": "gate:error"
            })

    async def _run_supervisor_node(self, node: NodeSpec, ctx: Dict[str, Any], run: RunSpec, 
                                 stream: bool = False) -> AsyncGenerator[Event, None]:
        """Execute a supervisor node that dynamically chooses and calls other agents."""
        logger.info(f"SUPERVISOR: Starting node {node.id} with {len(node.available_agents or {})} available agents")
        
        # Build supervisor context with available agents and current task
        supervisor_prompt = self._build_supervisor_prompt(node, ctx)
        
        # Create a dynamic task spec for supervisor reasoning
        supervisor_task = NodeSpec(
            id=f"{node.id}_supervisor_decision",
            type="task",
            name="Agent Selection and Orchestration",
            inputs={"task_prompt": supervisor_prompt},
            server_name=node.server_name
        )
        
        # Execute supervisor decision-making
        supervisor_decision_completed = False
        supervisor_output = ""
        
        async for event in self._run_task_node(supervisor_task, ctx, run, stream=stream):
            # Pass through supervisor decision events
            if event.type == "AGENT_CHUNK" and stream:
                yield event
            elif event.type == "NODE_COMPLETE":
                supervisor_decision_completed = True
                # Extract supervisor decision and execute chosen agents
                decision_result = ctx["blackboard"].get(supervisor_task.id, {}).get("result", {})
                if isinstance(decision_result, dict) and "output" in decision_result:
                    supervisor_output = decision_result["output"]
        
        # After supervisor decision is complete, execute chosen agents
        if supervisor_decision_completed and supervisor_output:
            final_result = await self._execute_supervisor_decisions(
                node, supervisor_output, ctx, run, stream=stream
            )
            
            # Store final result in blackboard
            ctx["blackboard"][node.id] = {"result": final_result, "source": node.id}
            
            # Emit final supervisor completion
            yield self._emit_event_sync("NODE_COMPLETE", run.run_id, node.id, {
                "output_meta": ["result", "source"],
                "phase": "supervisor:complete",
                "agents_called": len(self._parse_agent_calls(supervisor_output))
            })

    def _build_supervisor_prompt(self, node: NodeSpec, ctx: Dict[str, Any]) -> str:
        """Build prompt for supervisor agent to choose and orchestrate other agents."""
        prompt_parts = []
        
        prompt_parts.append("You are a supervisor agent responsible for choosing and orchestrating other specialized agents to complete a task.")
        prompt_parts.append(f"Task: {node.name or 'Complete the assigned task'}")
        
        # Add context from inputs
        if node.inputs:
            prompt_parts.append("\\nContext and Inputs:")
            for key, value in node.inputs.items():
                # Resolve references to other nodes
                if isinstance(value, str) and value in ctx["blackboard"]:
                    blackboard_entry = ctx["blackboard"][value]
                    if isinstance(blackboard_entry, dict) and "result" in blackboard_entry:
                        actual_value = blackboard_entry["result"]
                        if isinstance(actual_value, dict) and "output" in actual_value:
                            actual_value = actual_value["output"]
                    else:
                        actual_value = blackboard_entry
                    prompt_parts.append(f"- {key}: {actual_value}")
                else:
                    prompt_parts.append(f"- {key}: {value}")
        
        # Add available agents
        if node.available_agents:
            prompt_parts.append("\\nAvailable Specialized Agents:")
            for agent_id, agent_info in node.available_agents.items():
                description = agent_info.get("description", "No description")
                capabilities = agent_info.get("capabilities", [])
                server = agent_info.get("server", "default")
                prompt_parts.append(f"- {agent_id}: {description}")
                if capabilities:
                    prompt_parts.append(f"  Capabilities: {', '.join(capabilities)}")
                prompt_parts.append(f"  Server: {server}")
        
        prompt_parts.append(f"\\nYou can call up to {node.max_agent_calls} agents. For each agent you want to call, respond with:")
        prompt_parts.append("CALL_AGENT: <agent_id>")
        prompt_parts.append("TASK: <specific task for this agent>")
        prompt_parts.append("INPUT: <input data for this agent>")
        prompt_parts.append("---")
        prompt_parts.append("\\nEnd your response with:")
        prompt_parts.append("FINAL_RESULT: <your final combined result>")
        
        return "\n".join(prompt_parts)

    def _parse_agent_calls(self, supervisor_output: str) -> list:
        """Parse agent calls from supervisor output."""
        calls = []
        lines = supervisor_output.split("\n")  # Fixed: was using \\n instead of \n
        current_call = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("CALL_AGENT:"):
                if current_call:  # Save previous call
                    calls.append(current_call)
                current_call = {"agent_id": line.replace("CALL_AGENT:", "").strip()}
            elif line.startswith("TASK:") and current_call:
                current_call["task"] = line.replace("TASK:", "").strip()
            elif line.startswith("INPUT:") and current_call:
                current_call["input"] = line.replace("INPUT:", "").strip()
            elif line == "---" and current_call:
                calls.append(current_call)
                current_call = {}
        
        if current_call:  # Don't forget the last call
            calls.append(current_call)
            
        return calls

    async def _execute_supervisor_decisions(self, node: NodeSpec, supervisor_output: str, 
                                          ctx: Dict[str, Any], run: RunSpec, stream: bool = False) -> Dict[str, Any]:
        """Execute the agents chosen by the supervisor."""
        agent_calls = self._parse_agent_calls(supervisor_output)
        results = {}
        
        logger.info(f"SUPERVISOR: Executing {len(agent_calls)} agent calls")
        
        for i, call in enumerate(agent_calls[:node.max_agent_calls]):
            agent_id = call.get("agent_id")
            task = call.get("task", "Complete the assigned task")
            agent_input = call.get("input", "")
            
            if not agent_id or agent_id not in (node.available_agents or {}):
                logger.warning(f"SUPERVISOR: Unknown agent {agent_id}, skipping")
                continue
                
            agent_info = node.available_agents[agent_id]  # type: ignore
            server_name = agent_info.get("server")
            
            # Create dynamic task node for this agent call with enhanced prompting for file creation
            enhanced_task = f"""
{task}

You are a {agent_id} specialist. Use MCP tools to create actual files:

For ui_designer: Create wireframes, design specifications, and asset lists
For frontend_developer: Create HTML, CSS, and JavaScript files using filesystem tools
For content_writer: Create content files, copy, and text assets  
For backend_developer: Create server code, API files, and configuration
For quality_assurance: Create test files and validation reports

Input: {agent_input}

IMPORTANT: Use filesystem and file creation MCP tools to generate real files, not just descriptions.
"""

            agent_task = NodeSpec(
                id=f"{node.id}_agent_{i}_{agent_id}",
                type="task", 
                name=enhanced_task,
                inputs={"task_description": enhanced_task, "input_data": agent_input},
                server_name=server_name
            )
            
            logger.info(f"SUPERVISOR: Calling agent {agent_id} with task: {task[:100]}...")
            
            # Execute the agent task
            async for event in self._run_task_node(agent_task, ctx, run, stream=stream):
                # Let agent chunks pass through if needed
                pass
            
            # Extract result
            agent_result = ctx["blackboard"].get(agent_task.id, {}).get("result", {})
            if isinstance(agent_result, dict) and "output" in agent_result:
                results[agent_id] = agent_result["output"]
            else:
                results[agent_id] = agent_result
        
        # Extract final result from supervisor output
        final_result_text = ""
        lines = supervisor_output.split("\\n")
        capture = False
        for line in lines:
            if line.strip().startswith("FINAL_RESULT:"):
                final_result_text = line.replace("FINAL_RESULT:", "").strip()
                capture = True
            elif capture and line.strip():
                final_result_text += "\\n" + line.strip()
        
        return {
            "output": final_result_text or "Task completed by supervisor",
            "agent_results": results,
            "agents_called": list(results.keys())
        }

    async def run(self, graph: GraphSpec, run: RunSpec, *, resume: bool = False) -> AsyncGenerator[Event, None]:
        """Run orchestration with sequential execution (backward compatible)."""
        async for event in self._run_orchestration(graph, run, stream=False, resume=resume):
            yield event

    async def run_streaming(self, graph: GraphSpec, run: RunSpec, *, resume: bool = False) -> AsyncGenerator[Event, None]:
        """Run orchestration with streaming support."""
        self._system_logger.info("orchestrator", f"Starting streaming run: {run.run_id}")
        self._system_logger.info("orchestrator", f"Graph: {len(graph.nodes)} nodes, Resume: {resume}")
        async for event in self._run_orchestration(graph, run, stream=True, resume=resume):
            yield event

    async def _run_orchestration(self, graph: GraphSpec, run: RunSpec, stream: bool = False, resume: bool = False) -> AsyncGenerator[Event, None]:
        """Enhanced orchestration with checkpoint persistence and resume support."""
        self._system_logger.info("orchestrator", f"Starting orchestration: {run.run_id}")
        ctx: Dict[str, Any] = {"blackboard": {}, "_graph_edges": graph.edges}
        
        # Initialize persistence and load checkpoints
        checkpoint = {}
        try:
            if self._store:
                self._system_logger.info("orchestrator", "Initializing run store")
                await self._store.start_run(run.run_id, run)
                if resume:
                    self._system_logger.info("orchestrator", "Loading checkpoint for resume")
                    checkpoint = await self._store.load_checkpoint(run.run_id)
                    self._gate_pruned_nodes = await self._store.load_gate_pruning(run.run_id)
                    # Resume event sequence continuity
                    last_seq = await self._store.get_last_event_seq(run.run_id)
                    if last_seq > self._event_seq:
                        self._event_seq = last_seq
                        
        except Exception as e:
            self._system_logger.error("orchestrator", f"Failed to initialize persistence: {str(e)}")
            logger.error(f"Failed to initialize persistence for run {run.run_id}: {e}")
            # Continue without persistence
        
        # Set run context on executor if it supports it (for agent pool management)
        if hasattr(self._executor, 'set_run_context'):
            self._executor.set_run_context(run.run_id)
        
        # Emit RUN_START
        async for event in self._emit_event("RUN_START", run, data={"goal": run.goal, "resumed": resume}):
            yield event
        
        try:
            # Compute topological order and create node map
            order = topo_sort(graph)
            node_map: Dict[str, NodeSpec] = {n.id: n for n in graph.nodes}
            
            # Execute nodes in topological order with resume support
            for node_id in order:
                if node_id in self._gate_pruned_nodes:
                    continue  # Skip previously pruned nodes
                    
                node = node_map[node_id]
                
                # Compute node signature for caching
                sig = node_signature(node, run, self._server_configs)
                
                # Check if we can resume from checkpoint
                if node_id in checkpoint and checkpoint[node_id].signature == sig:
                    # Resume: restore cached result
                    ctx["blackboard"][node_id] = checkpoint[node_id].result
                    async for event in self._emit_event("NODE_COMPLETE", run, node, 
                                                       {"resumed": True, "signature": sig}):
                        yield event
                    continue
                
                try:
                    # Execute node normally
                    gate_blocked = False
                    async for event in self._run_node_enhanced(node, ctx, run, stream):
                        yield event
                        
                        # Check for gate that blocked flow
                        if (event.type == "NODE_COMPLETE" and 
                            event.data.get("phase") == "gate:blocked"):
                            gate_blocked = True
                    
                    # Save successful result to checkpoint
                    if self._store and node_id in ctx["blackboard"]:
                        saved_node = SavedNode(node_id, sig, ctx["blackboard"][node_id])
                        await self._store.save_node_result(run.run_id, saved_node)
                    
                    # Handle gate pruning
                    if gate_blocked:
                        successors = self._get_transitive_successors(graph, node_id)
                        self._gate_pruned_nodes.update(successors)
                        
                        # Persist gate pruning state
                        if self._store:
                            await self._store.save_gate_pruning(run.run_id, self._gate_pruned_nodes)
                        
                except Exception as e:
                    # Emit error and mark run as failed
                    async for event in self._emit_event("ERROR", run, node, {"error": repr(e)}):
                        yield event
                    
                    if self._store:
                        await self._store.mark_run_error(run.run_id, repr(e))
                    return
            
            # Emit successful completion
            async for event in self._emit_event("RUN_COMPLETE", run, data={"result": ctx["blackboard"]}):
                yield event
            
            # Mark run as complete
            if self._store:
                await self._store.mark_run_complete(run.run_id)
                
        except asyncio.CancelledError:
            # Handle cancellation
            if self._store:
                await self._store.mark_run_canceled(run.run_id)
            raise
        except Exception as e:
            # Handle unexpected errors
            async for event in self._emit_event("ERROR", run, data={"error": repr(e)}):
                yield event
            
            if self._store:
                await self._store.mark_run_error(run.run_id, repr(e))
            return
        finally:
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
    
    def _extract_output_summary(self, result: Any) -> Optional[str]:
        """
        Extract normalized output summary from result data.
        Handles result|output|text normalization and provides fallback text.
        """
        if result is None:
            return None
        
        # Handle dict results - look for common output keys
        if isinstance(result, dict):
            for key in ["result", "output", "text", "content", "message"]:
                if key in result:
                    value = result[key]
                    if isinstance(value, str) and value.strip():
                        return value.strip()  # No truncation
                    elif value is not None:
                        return str(value)  # No truncation
            
            # Fallback to first non-empty string value
            for value in result.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()  # No truncation
        
        # Handle direct string results
        elif isinstance(result, str) and result.strip():
            return result.strip()  # No truncation
        
        # Fallback to string representation
        elif result is not None:
            return str(result)  # No truncation
        
        return None
    
    def _build_task_prompt(self, node: NodeSpec, ctx: Dict[str, Any]) -> str:
        """Build the task prompt for AGENT_INSTRUCTIONS event."""
        prompt_parts = []
        
        if node.name:
            prompt_parts.append(f"Task: {node.name}")
        
        # Add inputs to prompt  
        if node.inputs:
            prompt_parts.append("Inputs:")
            for key, value in node.inputs.items():
                # Handle references to other nodes
                if isinstance(value, str) and value in ctx["blackboard"]:
                    blackboard_entry = ctx["blackboard"][value]
                    
                    # Debug logging for blackboard retrieval
                    logger.info(f"BLACKBOARD RETRIEVE: Node {node.id} requesting data from {value}")
                    logger.info(f"BLACKBOARD RETRIEVE: Entry type: {type(blackboard_entry)}")
                    logger.info(f"BLACKBOARD RETRIEVE: Entry keys: {list(blackboard_entry.keys()) if isinstance(blackboard_entry, dict) else 'Not dict'}")
                    
                    # Extract actual result from blackboard entry structure
                    if isinstance(blackboard_entry, dict) and "result" in blackboard_entry:
                        actual_value = blackboard_entry["result"]
                        logger.info(f"BLACKBOARD RETRIEVE: Result type: {type(actual_value)}")
                        logger.info(f"BLACKBOARD RETRIEVE: Result keys: {list(actual_value.keys()) if isinstance(actual_value, dict) else 'Not dict'}")
                        # Further extract if nested
                        if isinstance(actual_value, dict) and "output" in actual_value:
                            actual_value = actual_value["output"]
                            logger.info(f"BLACKBOARD RETRIEVE: Final output type: {type(actual_value)}")
                            logger.info(f"BLACKBOARD RETRIEVE: Final output preview: {str(actual_value)[:200]}...")
                    else:
                        actual_value = blackboard_entry
                        logger.info(f"BLACKBOARD RETRIEVE: Direct value type: {type(actual_value)}")
                    prompt_parts.append(f"- {key}: {actual_value}")
                else:
                    prompt_parts.append(f"- {key}: {value}")
        
        return "\n".join(prompt_parts)
    
    def _get_available_tools(self, node: NodeSpec) -> list:
        """Get list of available tools for the agent."""
        # This is a simplified version - in reality we'd query the MCP server
        tools = [
            {
                "name": "analyze_data",
                "purpose": "Analyze and process data inputs",
                "schema_summary": "Takes data input and returns analysis",
                "safety": "safe"
            },
            {
                "name": "generate_content", 
                "purpose": "Generate text content based on inputs",
                "schema_summary": "Takes prompt and returns generated text",
                "safety": "safe"
            }
        ]
        
        # Add server-specific tools if available
        if hasattr(node, 'server_name') and node.server_name:
            tools.append({
                "name": f"{node.server_name}_tools",
                "purpose": f"Server-specific tools for {node.server_name}",
                "schema_summary": "Various server tools",
                "safety": "safe"
            })
        
        return tools