#!/usr/bin/env python3
"""
Quick working demo for YC presentation - bypasses broken CLI
"""
import os
import asyncio
from pathlib import Path

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Import the working supervisor demo
from examples.quickstart.supervisor_website_builder import graph_spec, run_spec, executor
from src.agent_orchestra.orchestrator.core import Orchestrator
from src.agent_orchestra.orchestrator.store_sqlite import SQLiteStore

async def run_demo():
    """Run the supervisor demo directly."""
    print("🎭 Agent Orchestra - Working Demo for YC")
    print("=" * 50)
    print(f"Run ID: {run_spec.run_id}")
    print(f"Goal: {run_spec.goal}")
    print(f"Nodes: {len(graph_spec.nodes)}")
    print("Starting execution...\n")
    
    # Create store
    store_path = Path(".ao_runs/demo.sqlite3")
    store_path.parent.mkdir(exist_ok=True)
    store = SQLiteStore(str(store_path))
    
    # Create orchestrator
    orchestrator = Orchestrator(executor, store=store)
    
    # Run with progress output
    try:
        async for event in orchestrator.run_streaming(graph_spec, run_spec):
            event_type = event.type
            node_id = getattr(event, 'node_id', None)
            
            if event_type == "NODE_START":
                print(f"🚀 Starting: {node_id}")
            elif event_type == "AGENT_CHUNK":
                data = getattr(event, 'data', {})
                if isinstance(data, dict) and data.get("text"):
                    text = data["text"][:100]
                    print(f"💭 [{node_id}] {text}")
            elif event_type == "NODE_COMPLETE":
                print(f"✅ Completed: {node_id}")
            elif event_type == "RUN_COMPLETE":
                print("\n🎉 Demo completed successfully!")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_demo())