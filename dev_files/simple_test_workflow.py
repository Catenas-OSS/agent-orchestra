#!/usr/bin/env python3
"""
Simple test workflow to verify TUI integration
"""

import sys
from pathlib import Path
import uuid

# Add the src directory to the path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec

# Simple workflow with one task
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="test_task",
            type="task", 
            name="Simple Test Task",
            inputs={
                "instruction": "This is a simple test task to verify the TUI is working."
            }
        )
    ],
    edges=[]
)

run_spec = RunSpec(
    run_id=f"test_{uuid.uuid4().hex[:6]}",
    goal="Simple TUI test"
)

# Export for CLI
__all__ = ["graph_spec", "run_spec"]