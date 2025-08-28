#!/usr/bin/env python3
"""
Working Supervisor Demo - Compatible with CLI workflow loader
"""

# Note: Imports are done at module level to be available during exec()
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

# Add src to path before importing
_src_path = str(Path(__file__).parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec

# Define specialist agents
specialist_agents = {
    "ui_designer": {
        "description": "Creates modern, responsive UI designs and layouts",
        "capabilities": ["responsive design", "CSS Grid", "modern aesthetics"],
        "server": "mock"  # Use mock server for demo
    },
    "frontend_developer": {
        "description": "Implements complete HTML, CSS, and JavaScript",
        "capabilities": ["HTML5", "CSS3", "ES6+", "responsive implementation"],
        "server": "mock"
    },
    "content_creator": {
        "description": "Creates engaging website content and copy",
        "capabilities": ["copywriting", "content strategy", "SEO optimization"],
        "server": "mock"
    }
}

# Create the workflow directly at module level
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="setup_context",
            type="task",
            name="Setup Project Context",
            inputs={
                "instruction": "Initialize context for a modern website project requiring design, development, and content creation."
            }
        ),
        
        NodeSpec(
            id="website_supervisor",
            type="supervisor",
            name="Website Building Supervisor", 
            inputs={
                "context": "setup_context",
                "project_brief": "Create a complete website by intelligently selecting and coordinating specialist agents based on project requirements."
            },
            available_agents=specialist_agents,
            max_agent_calls=2,
            server_name="mock"
        ),
        
        NodeSpec(
            id="finalize_project", 
            type="task",
            name="Finalize Website Project",
            inputs={
                "supervisor_output": "website_supervisor",
                "instruction": "Review and finalize the website project completed by the supervisor and its selected agents."
            }
        )
    ],
    edges=[
        ("setup_context", "website_supervisor"),
        ("website_supervisor", "finalize_project")
    ]
)

run_spec = RunSpec(
    run_id=f"supervisor_demo_{uuid.uuid4().hex[:6]}",
    goal="Demonstrate supervisor agent orchestration with TUI interface"
)

# Required exports for CLI
__all__ = ["graph_spec", "run_spec"]