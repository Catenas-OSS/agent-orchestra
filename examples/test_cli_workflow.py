#!/usr/bin/env python3
"""
Simple CLI-compatible workflow for testing the restored Python-first catenas CLI.

This file defines all required components at module level:
- workflow (GraphSpec)
- run_spec (RunSpec) 
- executor (MCPExecutor)
- store (SQLiteRunStore)

Usage:
    catenas run examples/test_cli_workflow.py
    catenas run examples/test_cli_workflow.py --watch
"""

import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    from agent_orchestra.orchestrator.broker_config import create_development_broker
    from agent_orchestra.orchestrator.agent_pool import AgentPool, create_default_agent_factory
    from agent_orchestra.orchestrator.store_factory import create_store
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e .")
    sys.exit(1)


# ============================================================================
# Module-level workflow definition for CLI compatibility
# ============================================================================

# 1. Define the workflow graph
workflow = GraphSpec(
    nodes=[
        # Task 1: Generate creative writing prompts
        NodeSpec(
            id="generate_prompts",
            type="task",
            name="Generate Writing Prompts",
            inputs={
                "instruction": (
                    "Generate 3 creative writing prompts for different genres: "
                    "1. A sci-fi story about AI discovering emotions "
                    "2. A mystery involving a missing cookbook "
                    "3. A fantasy tale about a librarian with magic powers. "
                    "Keep each prompt to 2-3 sentences."
                )
            }
        ),
        
        # Task 2: Analyze the prompts for themes
        NodeSpec(
            id="analyze_themes",
            type="task", 
            name="Analyze Themes",
            inputs={
                "from": "generate_prompts",
                "instruction": (
                    "Analyze the writing prompts and identify common themes, "
                    "literary devices, and storytelling techniques present. "
                    "Provide a brief analysis of what makes each prompt engaging."
                )
            }
        ),
        
        # Task 3: Create a summary report
        NodeSpec(
            id="create_summary",
            type="task",
            name="Create Summary Report", 
            inputs={
                "from": "analyze_themes",
                "instruction": (
                    "Create a brief summary report that combines the writing prompts "
                    "with the thematic analysis. Format as a creative writing guide "
                    "with actionable insights for aspiring writers."
                )
            }
        )
    ],
    edges=[
        ("generate_prompts", "analyze_themes"),
        ("analyze_themes", "create_summary")
    ]
)

# 2. Define the run specification
import uuid
run_spec = RunSpec(
    run_id=f"cli_test_{uuid.uuid4().hex[:8]}",
    goal="Test CLI workflow with creative writing prompt generation and analysis"
)

# 3. Create the MCP client (simple config for testing)
client = SidecarMCPClient.from_dict({
    "mcpServers": {}  # No external servers needed for this text-only example
})

# 4. Create LLM (only if API key is available)
llm = None
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)
else:
    print("⚠️  No OPENAI_API_KEY found - using mock mode")

# 5. Create agent factory and pool
agent_factory = create_default_agent_factory(client, llm)
agent_pool = AgentPool(agent_factory, max_agents_per_run=3)

# 6. Create broker for rate limiting
broker = create_development_broker()

# 7. Create the executor
executor = MCPExecutor(
    agent=None,  # Use agent pool instead
    default_server="default",
    broker=broker,
    agent_pool=agent_pool,
    model_key="openai:gpt-4o-mini"
)

# 8. Create the store
store = create_store("sqlite", ".ao_runs/cli_test.sqlite3")

# ============================================================================
# Optional: Demo function for standalone execution
# ============================================================================

async def demo():
    """Optional demo function - not used by CLI."""
    print("🎭 CLI Test Workflow Demo")
    print("=" * 40)
    print(f"📊 Nodes: {len(workflow.nodes)}")
    print(f"🔗 Edges: {len(workflow.edges)}")
    print()
    
    orchestrator = Orchestrator(executor, store=store)
    
    print("🚀 Running workflow...")
    async for event in orchestrator.run_streaming(workflow, run_spec):
        if event.type == "NODE_START":
            print(f"▶️  Starting: {event.node_id}")
        elif event.type == "NODE_COMPLETE":
            print(f"✅ Completed: {event.node_id}")
        elif event.type == "RUN_COMPLETE":
            print("🎉 Workflow completed!")
            break
        elif event.type == "ERROR":
            print(f"❌ Error: {event.data}")
    
    await executor.broker.shutdown()
    await executor.agent_pool.shutdown()


# Only run demo if executed directly, not when loaded by CLI
if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())