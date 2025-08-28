#!/usr/bin/env python3
"""
Simple Agent Orchestra Workflow Example

A basic workflow that doesn't require external dependencies.
Perfect for testing the CLI and understanding the core concepts.
"""

import uuid
from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor

# Simple task executor that doesn't need external dependencies
async def simple_task(node, ctx):
    """Execute a simple task."""
    task_name = node.inputs.get("task", "unknown task")
    message = node.inputs.get("message", "Hello from Agent Orchestra!")
    
    print(f"🤖 Executing: {task_name}")
    print(f"💬 Message: {message}")
    
    return {
        "output": f"Completed {task_name}: {message}",
        "status": "success"
    }

# Create the workflow
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="welcome",
            type="task",
            name="Welcome Task",
            inputs={
                "task": "greeting",
                "message": "Welcome to Agent Orchestra!"
            }
        ),
        NodeSpec(
            id="demo",
            type="task", 
            name="Demo Task",
            inputs={
                "task": "demonstration",
                "message": "This is a simple workflow demonstration"
            }
        ),
        NodeSpec(
            id="completion",
            type="task",
            name="Completion Task", 
            inputs={
                "task": "completion",
                "message": "Workflow completed successfully!"
            }
        )
    ],
    edges=[
        ("welcome", "demo"),
        ("demo", "completion")
    ]
)

# Create the run specification
run_spec = RunSpec(
    run_id=f"simple_demo_{uuid.uuid4().hex[:6]}",
    goal="Demonstrate basic Agent Orchestra workflow execution"
)

# Create the executor
executor = CallableExecutor(simple_task)

# Export for CLI usage
__all__ = ["graph_spec", "run_spec", "executor"]

if __name__ == "__main__":
    print("🎭 Simple Agent Orchestra Demo")
    print("=" * 40)
    print("This is a basic workflow with 3 sequential tasks.")
    print()
    print("🚀 To run with the CLI:")
    print("   python -m agent_orchestra.cli_py run examples/quickstart/simple_workflow.py --watch")
    print()
    print("📖 To run programmatically:")
    print("   python examples/quickstart/simple_workflow.py")