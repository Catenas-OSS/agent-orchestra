#!/usr/bin/env python3
"""
Minimal supervisor demo that works without external dependencies
"""

import sys
from pathlib import Path
import uuid

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec

# Create minimal workflow with supervisor orchestration
graph_spec = GraphSpec(
    nodes=[
        # Simple supervisor node with mock agents
        NodeSpec(
            id="demo_supervisor",
            type="supervisor",
            name="Demo Supervisor Agent", 
            inputs={
                "task": "Demonstrate supervisor agent orchestration in the TUI interface"
            },
            available_agents={
                "content_writer": {
                    "description": "Creates written content and documentation",
                    "capabilities": ["writing", "editing", "documentation"],
                    "server": "mock"
                },
                "code_reviewer": {
                    "description": "Reviews code for quality and best practices", 
                    "capabilities": ["code review", "testing", "quality assurance"],
                    "server": "mock"
                },
                "data_analyst": {
                    "description": "Analyzes data and provides insights",
                    "capabilities": ["data analysis", "reporting", "visualization"],
                    "server": "mock"
                }
            },
            max_agent_calls=2
        )
    ],
    edges=[]
)

run_spec = RunSpec(
    run_id=f"supervisor_demo_{uuid.uuid4().hex[:6]}", 
    goal="Minimal supervisor TUI demonstration"
)

# Export for CLI
__all__ = ["graph_spec", "run_spec"]