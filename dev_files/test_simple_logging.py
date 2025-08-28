#!/usr/bin/env python3
"""
Simple test workflow to verify logging system works correctly.
Uses MockExecutor to avoid complex MCP library interactions.
"""

import asyncio
from typing import Dict, Any
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import MockExecutor

# Create a simple test workflow
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="simple_test",
            name="Simple Logging Test",
            task="Basic test of logging infrastructure"
        )
    ],
    edges=[]
)

# Use MockExecutor to avoid MCP complexity
executor = MockExecutor()

# Simple run specification  
run_spec = RunSpec(
    run_id=f"simple_test_{int(asyncio.get_event_loop().time())}",
    goal="Test basic logging system without external dependencies"
)