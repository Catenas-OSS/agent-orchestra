"""Orchestrator module for Agent Orchestra.

This module provides graph-based task orchestration with event streaming,
pluggable executors, and robust error handling.
"""

from .core import Orchestrator
from .types import Event, EventType, NodeSpec, GraphSpec, RunSpec
from .executors import Executor, CallableExecutor
from .executors_mcp import MCPExecutor
from .utils import topo_sort

__all__ = [
    "Orchestrator",
    "Event", 
    "EventType",
    "NodeSpec",
    "GraphSpec", 
    "RunSpec",
    "Executor",
    "CallableExecutor",
    "MCPExecutor",
    "topo_sort",
]