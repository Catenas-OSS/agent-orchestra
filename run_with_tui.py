#!/usr/bin/env python3
"""
Run supervisor website builder with TUI - bypasses broken CLI
"""
import os
import asyncio
from pathlib import Path

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "sk-proj-bWT0-kVWxXoyscxZOpLuFwHMPtD8fwM8k1KbwWyA0M_ONoBCAfsvTmMQ9XfKFHxz7M3b0FhoRyT3BlbkFJejby-tp2xOHtzR__bFMEJjBTVEDnKWOT9tsQto7kIJejby-tp2xOHtzR__bFMEJjBTVEDnKWOT9tsQto7koCz17tASVt4racCT5oT0Kza0OpSS3w8WIA"

# Add paths
import sys
sys.path.insert(0, 'examples/quickstart')
sys.path.insert(0, 'src')

# Import the working supervisor demo
from supervisor_website_builder import graph_spec, run_spec, executor
from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.store_sqlite import SQLiteRunStore
from agent_orchestra.tui.main import ProfessionalUrwidTUI
from agent_orchestra.tui.model import RunTUIModel, NodeState
from agent_orchestra.logging import init_logging

async def run_with_tui():
    """Run the supervisor demo with TUI."""
    print("🎭 Agent Orchestra - TUI Demo")
    print("=" * 40)
    print("Loading workflow...")
    
    # Initialize logging for TUI mode
    system_logger = init_logging(tui_mode=True)
    system_logger.enable_tui_mode()
    
    # Create store
    store_path = Path(".ao_runs/tui_demo.sqlite3")
    store_path.parent.mkdir(exist_ok=True)
    store = SQLiteRunStore(str(store_path))
    
    # Create orchestrator
    orchestrator = Orchestrator(executor, store=store)
    
    # Initialize TUI model
    model = RunTUIModel(run_spec.run_id, goal=run_spec.goal)
    
    # Add nodes from graph spec
    for node in graph_spec.nodes:
        node_state = NodeState(
            id=node.id,
            name=node.name,
            node_type=node.type,
            server=getattr(node, 'server_name', None),
            inputs=getattr(node, 'inputs', {})
        )
        
        # Add supervisor-specific fields
        if node.type == "supervisor":
            node_state.available_agents = getattr(node, 'available_agents', {})
            node_state.max_agent_calls = getattr(node, 'max_agent_calls', 5)
        
        model.nodes[node.id] = node_state
    
    # Store DAG edges
    model.dag_edges = [(edge[0], edge[1]) for edge in graph_spec.edges]
    
    # Create and run TUI
    print("Starting TUI...")
    tui = ProfessionalUrwidTUI(orchestrator, model)
    
    # Run the TUI with streaming events
    run_stream = orchestrator.run_streaming(graph_spec, run_spec)
    await tui.run(run_stream)

if __name__ == "__main__":
    asyncio.run(run_with_tui())