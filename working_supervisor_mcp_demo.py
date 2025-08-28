#!/usr/bin/env python3
"""
Working MCP Supervisor Demo - Real MCP with debugging
"""
import os
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Set API key
os.environ.setdefault('OPENAI_API_KEY', "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")

# Agent Orchestra imports
from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.sidecar.sidecar_client import SidecarMCPClient
from agent_orchestra.sidecar.sidecar_agent import SidecarMCPAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mcp_environment():
    """Setup MCP environment with debugging."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("langchain_openai required. Install with: pip install langchain-openai")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable required")
    
    # Setup output directory
    output_dir = Path("./website_output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Website output directory: {output_dir.absolute()}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=4000
    )
    
    # Configure MCP filesystem server
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(output_dir.absolute())]
            }
        }
    }
    
    # Create client and agent
    logger.info("Creating MCP client...")
    client = SidecarMCPClient.from_dict(config)
    logger.info("MCP client created")
    
    agent = SidecarMCPAgent(
        llm=llm,
        client=client,
        sidecar_run_context={
            "task": "website_building",
            "project": "demo_website",
            "quality_level": "professional"
        }
    )
    logger.info("MCP agent created")
    
    # Create executor
    executor = MCPExecutor(
        agent=agent,
        default_server="filesystem",
        broker=None,
        agent_pool=None,
        model_key="openai:gpt-4o-mini"
    )
    logger.info("MCPExecutor created")
    
    return agent, executor

# Define specialist agents
specialist_agents = {
    "ui_designer": {
        "description": "Creates modern, visually stunning UI designs with advanced CSS and animations",
        "capabilities": ["responsive design", "CSS Grid", "animations", "modern aesthetics"],
        "server": "filesystem"
    },
    "frontend_developer": {
        "description": "Implements HTML, CSS, and JavaScript for interactive web experiences", 
        "capabilities": ["HTML5", "CSS3", "JavaScript", "responsive implementation"],
        "server": "filesystem"
    },
    "content_creator": {
        "description": "Creates compelling website content and copy with SEO optimization",
        "capabilities": ["copywriting", "content strategy", "SEO", "brand messaging"],
        "server": "filesystem"
    }
}

# Create workflow
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="website_supervisor",
            type="supervisor",
            name="AI Website Building Supervisor",
            inputs={
                "brief": "Build a modern portfolio website for a creative agency. Create HTML, CSS, and JavaScript files with professional design, responsive layout, and engaging content. Use the filesystem tools to create actual files."
            },
            available_agents=specialist_agents,
            max_agent_calls=3,
            server_name="filesystem",
            timeout_s=300  # 5 minute timeout
        )
    ],
    edges=[]
)

# Create run specification
timestamp = int(time.time())
run_spec = RunSpec(
    run_id=f"mcp_supervisor_{timestamp}_{uuid.uuid4().hex[:6]}",
    goal="Build professional website using real MCP supervisor with file creation"
)

# Setup environment
try:
    agent, executor = setup_mcp_environment()
    logger.info("Environment setup complete")
except Exception as e:
    logger.error(f"Environment setup failed: {e}")
    # Fallback to a simple mock for CLI compatibility
    from agent_orchestra.orchestrator.executors import CallableExecutor
    
    async def fallback_executor(node, ctx):
        return {"output": f"Fallback execution - MCP setup failed: {e}", "error": True}
    
    agent = None
    executor = CallableExecutor(fallback_executor)
    # Empty graph for fallback
    graph_spec = GraphSpec(nodes=[], edges=[])

# Required exports for CLI
__all__ = ["graph_spec", "run_spec", "executor"]

if __name__ == "__main__":
    print("🎭 Working MCP Supervisor Demo")
    print("=" * 50)
    print(f"Run ID: {run_spec.run_id}")
    print(f"Nodes: {len(graph_spec.nodes)}")
    if agent:
        print("✅ Real MCP executor ready")
        print("🚀 To run: python3 -m agent_orchestra.cli_py run working_supervisor_mcp_demo.py --watch")
    else:
        print("❌ MCP setup failed, using fallback")