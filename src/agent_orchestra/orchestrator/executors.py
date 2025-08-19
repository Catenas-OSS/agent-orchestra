from __future__ import annotations
from typing import Any, Callable, Dict, Protocol, Awaitable
from .types import NodeSpec

class Executor(Protocol):  # structural typing per PEP 544
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]: # type: ignore
        """Return a mapping stored under ctx['blackboard'][node.id]."""

class CallableExecutor:
    """Wrap an async callable for tests/dev."""
    def __init__(self, fn: Callable[[NodeSpec, Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        self._fn = fn
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return await self._fn(node, ctx)