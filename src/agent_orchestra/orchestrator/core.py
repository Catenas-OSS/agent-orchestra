from __future__ import annotations
import asyncio
from typing import Any, AsyncGenerator, Dict, Optional
from .types import Event, GraphSpec, NodeSpec, RunSpec
from .executors import Executor
from .utils import topo_sort

class Orchestrator:
    def __init__(self, executor: Executor):
        self._executor = executor

    async def run(self, graph: GraphSpec, run: RunSpec) -> AsyncGenerator[Event, None]:
        ctx: Dict[str, Any] = {"blackboard": {}}
        yield Event("RUN_START", run_id=run.run_id, data={"goal": run.goal})

        order = topo_sort(graph)
        node_map: Dict[str, NodeSpec] = {n.id: n for n in graph.nodes}

        for node_id in order:
            node = node_map[node_id]
            yield Event("NODE_START", run_id=run.run_id, node_id=node.id, data={"type": node.type})
            try:
                coro = self._executor.execute(node, ctx)
                out = await asyncio.wait_for(coro, timeout=node.timeout_s) if node.timeout_s else await coro
                ctx["blackboard"][node.id] = out
                yield Event("NODE_COMPLETE", run_id=run.run_id, node_id=node.id, data={"output_meta": list(out.keys())})
            except Exception as e:  # stop early in Part 1
                yield Event("ERROR", run_id=run.run_id, node_id=node.id, data={"error": repr(e)})
                return

        yield Event("RUN_COMPLETE", run_id=run.run_id, data={"result": ctx["blackboard"]})

    async def run_streaming(self, graph: GraphSpec, run: RunSpec) -> AsyncGenerator[Event, None]:
        """Run orchestration with streaming support for MCP executors."""
        ctx: Dict[str, Any] = {"blackboard": {}}
        yield Event("RUN_START", run_id=run.run_id, data={"goal": run.goal})

        order = topo_sort(graph)
        node_map: Dict[str, NodeSpec] = {n.id: n for n in graph.nodes}

        for node_id in order:
            node = node_map[node_id]
            yield Event("NODE_START", run_id=run.run_id, node_id=node.id, data={"type": node.type})
            
            try:
                # Check if executor supports streaming
                if hasattr(self._executor, 'execute_with_stream'):
                    # Use streaming execution with queue for chunk handling
                    chunk_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
                    
                    async def on_chunk(chunk: Dict[str, Any]) -> None:
                        await chunk_queue.put(chunk)
                    
                    # Create streaming execution coroutine
                    async def stream_execution() -> Dict[str, Any]:
                        try:
                            result = await self._executor.execute_with_stream(node, ctx, on_chunk)  # type: ignore
                            await chunk_queue.put(None)  # Sentinel value to signal completion
                            return result # type: ignore
                        except Exception as e: # type: ignore
                            await chunk_queue.put(None)  # Ensure sentinel is sent even on error
                            raise
                    
                    # Start the execution task
                    execute_task = asyncio.create_task(stream_execution())
                    result = None
                    
                    # Process chunks until completion
                    try:
                        while True:
                            # Get next chunk with timeout
                            chunk = await asyncio.wait_for(
                                chunk_queue.get(), 
                                timeout=node.timeout_s
                            )
                            
                            if chunk is None:  # Execution completed
                                result = await execute_task
                                break
                            else:
                                # Yield chunk event
                                yield Event("AGENT_CHUNK", run_id=run.run_id, node_id=node.id, data=chunk)
                    
                    except asyncio.TimeoutError:
                        execute_task.cancel()
                        try:
                            await execute_task
                        except asyncio.CancelledError:
                            pass
                        raise
                    
                else:
                    # Fallback to regular execution
                    coro = self._executor.execute(node, ctx)
                    result = await asyncio.wait_for(coro, timeout=node.timeout_s) if node.timeout_s else await coro
                
                ctx["blackboard"][node.id] = result
                yield Event("NODE_COMPLETE", run_id=run.run_id, node_id=node.id, data={"output_meta": list(result.keys())})
                
            except Exception as e:
                yield Event("ERROR", run_id=run.run_id, node_id=node.id, data={"error": repr(e)})
                return

        yield Event("RUN_COMPLETE", run_id=run.run_id, data={"result": ctx["blackboard"]})