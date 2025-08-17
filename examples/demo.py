#!/usr/bin/env python3
"""
Hello World Demo for Agent Orchestra

This demo shows how to use the orchestrator programmatically
to run a 5-node workflow with MCP agents.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_orchestra import Orchestrator


async def main():
    """Run the hello world demo."""
    
    print("🎭 Agent Orchestra Hello World Demo")
    print("=" * 50)
    
    # Create orchestrator
    orchestrator = Orchestrator(
        checkpoint_dir="./demo_checkpoints",
        event_dir="./demo_events",
        max_concurrency=3
    )
    
    # Load the demo graph
    graph_file = Path(__file__).parent / "hello_world.json"
    
    print(f"📋 Loading graph from: {graph_file}")
    
    try:
        # Run the orchestration
        result = await orchestrator.run(
            graph=graph_file,
            ctx={
                "demo_mode": True,
                "user_input": "Hello from the demo!"
            }
        )
        
        print("\n🎯 Execution Results:")
        print(f"  Success: {result.success}")
        print(f"  Run ID: {result.run_id}")
        print(f"  Total tokens: {result.total_tokens}")
        print(f"  Total cost: ${result.total_cost:.4f}")
        print(f"  Total time: {result.total_time:.2f}s")
        
        if result.outputs:
            print("\n📊 Node Outputs:")
            for node_id, output in result.outputs.items():
                print(f"  {node_id}: {output}")
        
        if result.error:
            print(f"\n❌ Error: {result.error}")
        
        print(f"\n📁 Events logged to: ./demo_events/{result.run_id}.jsonl")
        print(f"📁 Checkpoints saved to: ./demo_checkpoints/")
        
        print("\n✨ Demo completed! Try these commands:")
        print(f"  catenas tail {result.run_id} --event-dir ./demo_events")
        print(f"  catenas inspect {result.run_id} --event-dir ./demo_events")
        print(f"  catenas list-checkpoints --checkpoint-dir ./demo_checkpoints")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)