#!/usr/bin/env python3
"""
Direct execution demo - bypasses broken CLI entirely
"""
import os
import asyncio
from pathlib import Path

# Set API key

# Import what we need
import sys
sys.path.insert(0, 'src')

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.store_sqlite import SQLiteRunStore
from openai_supervisor_demo import graph_spec, run_spec, executor

async def run_direct():
    """Run orchestrator directly with real-time output."""
    print("🎭 Direct Agent Orchestra Demo")
    print("=" * 50)
    print("Bypassing broken CLI, running orchestrator directly")
    print(f"Run ID: {run_spec.run_id}")
    print(f"Nodes: {len(graph_spec.nodes)}")
    print()
    
    # Create store
    store_path = Path(".ao_runs/direct_demo.sqlite3")
    store_path.parent.mkdir(exist_ok=True)
    store = SQLiteRunStore(str(store_path))
    
    # Create orchestrator
    orchestrator = Orchestrator(executor, store=store)
    
    print("Starting execution...")
    print("-" * 30)
    
    try:
        node_count = 0
        async for event in orchestrator.run_streaming(graph_spec, run_spec):
            event_type = event.type
            node_id = getattr(event, 'node_id', None)
            data = getattr(event, 'data', {})
            
            if event_type == "RUN_START":
                print("🚀 Run started")
                
            elif event_type == "NODE_START":
                node_count += 1
                node_name = next((n.name for n in graph_spec.nodes if n.id == node_id), node_id)
                print(f"\n[{node_count}/{len(graph_spec.nodes)}] 🤖 Starting: {node_name}")
                
            elif event_type == "AGENT_CHUNK":
                if isinstance(data, dict) and data.get("text"):
                    text = data["text"][:100] + ("..." if len(data["text"]) > 100 else "")
                    print(f"   💭 {text}")
                    
            elif event_type == "NODE_COMPLETE":
                output = data.get("output_summary", "")
                tokens = data.get("tokens", {})
                cost = data.get("cost", 0)
                
                print(f"   ✅ Completed!")
                if tokens:
                    print(f"   📊 Tokens: {tokens.get('total', 0):,} | Cost: ${cost:.4f}")
                if output:
                    preview = output[:200] + ("..." if len(output) > 200 else "")
                    print(f"   🎯 Output: {preview}")
                    
            elif event_type == "RUN_COMPLETE":
                print(f"\n🎉 Demo completed successfully!")
                print("-" * 30)
                break
                
            elif event_type == "ERROR":
                error = data.get("error", str(data))
                print(f"   ❌ Error: {error}")
                
    except Exception as e:
        print(f"💥 Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_direct())