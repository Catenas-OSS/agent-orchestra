from __future__ import annotations
import asyncio
from typing import Any, Callable, Dict, Protocol, Awaitable
from .types import NodeSpec
from ..logging import get_system_logger

class Executor(Protocol):  # structural typing per PEP 544
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]: # type: ignore
        """Return a mapping stored under ctx['blackboard'][node.id]."""

class CallableExecutor:
    """Wrap an async callable for tests/dev."""
    def __init__(self, fn: Callable[[NodeSpec, Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        self._fn = fn
        self._system_logger = get_system_logger()
        self._system_logger.info("callable_executor", "CallableExecutor initialized")
        
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._system_logger.info("callable_executor", f"Starting node execution: {node.id}", node_id=node.id)
        try:
            result = await self._fn(node, ctx)
            self._system_logger.info("callable_executor", f"Node execution completed successfully", node_id=node.id)
            return result
        except Exception as e:
            self._system_logger.error("callable_executor", f"Node execution failed: {str(e)}", node_id=node.id)
            raise

class MockExecutor:
    """Mock executor for testing with comprehensive logging."""
    
    def __init__(self):
        self._system_logger = get_system_logger()
        self._system_logger.info("mock_executor", "MockExecutor initialized for testing")
    
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mock node with detailed logging."""
        self._system_logger.info("mock_executor", f"Mock execution starting: {node.id}", node_id=node.id)
        self._system_logger.debug("mock_executor", f"Node task: {getattr(node, 'task', 'No task specified')}", node_id=node.id)
        
        try:
            # Simulate some work
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Log some activity
            self._system_logger.debug("mock_executor", "Simulating task processing", node_id=node.id)
            
            # Simulate different outcomes based on node ID
            if "error" in node.id.lower():
                self._system_logger.warning("mock_executor", "Simulating error condition", node_id=node.id)
                raise Exception(f"Mock error for node {node.id}")
            
            # More simulation
            await asyncio.sleep(0.2)
            
            result = {
                "status": "success",
                "message": f"Mock execution completed for {node.id}",
                "simulated_output": f"Mock result from {node.id}"
            }
            
            self._system_logger.info("mock_executor", f"Mock execution completed successfully", node_id=node.id)
            self._system_logger.debug("mock_executor", f"Result: {result}", node_id=node.id)
            
            return result
            
        except Exception as e:
            self._system_logger.error("mock_executor", f"Mock execution failed: {str(e)}", node_id=node.id)
            raise