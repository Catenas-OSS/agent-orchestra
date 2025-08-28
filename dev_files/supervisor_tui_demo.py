#!/usr/bin/env python3
"""
SUPERVISOR TUI DEMO - Integration Test for Smart Agent Orchestration
This workflow demonstrates supervisor agent orchestration integrated with the TUI
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
import uuid

def create_supervisor_tui_demo() -> GraphSpec:
    """Create a workflow showcasing supervisor orchestration for the TUI."""
    
    # Define specialized agents for the supervisor
    demo_agents = {
        "content_creator": {
            "description": "Creates engaging content and marketing copy",
            "capabilities": ["copywriting", "marketing", "creative content"],
            "server": "filesystem"  # Example - could be any MCP server
        },
        "data_analyst": {
            "description": "Analyzes data and provides insights",
            "capabilities": ["data analysis", "statistics", "reporting"],
            "server": "filesystem"
        },
        "code_reviewer": {
            "description": "Reviews code for quality and best practices",
            "capabilities": ["code review", "quality assurance", "testing"],
            "server": "filesystem"
        }
    }
    
    # Create workflow with supervisor orchestration
    workflow = GraphSpec(
        nodes=[
            # Simple task to establish context
            NodeSpec(
                id="setup_context",
                type="task",
                name="Setup Demo Context",
                inputs={
                    "instruction": "Set up context for a product launch project. Define the key requirements: compelling marketing content, data analysis for market research, and code quality review for the product."
                }
            ),
            
            # Supervisor node that dynamically chooses agents
            NodeSpec(
                id="launch_supervisor",
                type="supervisor",
                name="Product Launch Supervisor", 
                inputs={
                    "context": "setup_context",
                    "project_brief": "Coordinate a comprehensive product launch by intelligently selecting and orchestrating specialist agents. Choose the most appropriate agents based on the project requirements."
                },
                available_agents=demo_agents,
                max_agent_calls=2,  # Demonstrate selecting 2 of the 3 available agents
                server_name="filesystem"
            ),
            
            # Final validation task
            NodeSpec(
                id="validate_results",
                type="task",
                name="Validate Supervisor Output",
                inputs={
                    "supervisor_output": "launch_supervisor",
                    "instruction": "Review and validate the work completed by the supervisor and its selected agents. Provide a summary of what was accomplished."
                }
            )
        ],
        edges=[
            ("setup_context", "launch_supervisor"),
            ("launch_supervisor", "validate_results")
        ]
    )
    
    return workflow

# Create and return the workflow
graph_spec = create_supervisor_tui_demo()
run_spec = RunSpec(
    run_id=f"supervisor_tui_demo_{uuid.uuid4().hex[:6]}", 
    goal="Demonstrate supervisor agent orchestration in the TUI interface"
)

# Export for CLI
__all__ = ["graph_spec", "run_spec"]