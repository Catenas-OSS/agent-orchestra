#!/usr/bin/env python3
"""
Demo script showing Agentic Orchestra streaming orchestrator functionality.

This example demonstrates how to use the new run_streaming() method with 
AGENT_CHUNK events for real-time monitoring of agent execution.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor


class MockStreamingExecutor:
    """Demo executor that simulates MCP streaming with realistic chunks."""
    
    async def execute_with_stream(self, node, ctx, on_chunk):
        """Simulate agent streaming with realistic message flow."""
        # Simulate different types of chunks that mcp-use might send
        chunks = [
            {"message": f"Starting task: {node.name or node.id}"},
            {"action": "thinking", "status": "analyzing_inputs"},
            {"step": 1, "description": "Processing input data"},
            {"action": "tool_call", "tool": "web_search", "status": "executing"},
            {"step": 2, "description": "Analyzing search results"},
            {"message": "Generating final response"},
        ]
        
        for i, chunk in enumerate(chunks):
            await on_chunk(chunk)
            await asyncio.sleep(0.1)  # Simulate processing time
            print(f"  📦 Chunk {i+1}: {chunk}")
        
        # Final output chunk
        result = f"Completed {node.name or node.id}"
        final_chunk = {"output": result}
        await on_chunk(final_chunk)
        print(f"  ✅ Final: {final_chunk}")
        
        return {"output": result}
    
    async def execute(self, node, ctx):
        """Fallback for non-streaming execution."""
        await asyncio.sleep(0.2)
        return {"output": f"Non-streaming result for {node.name or node.id}"}


async def demo_streaming_orchestrator():
    """Demonstrate streaming orchestrator with real-time events."""
    print("🎭 Agentic Orchestra Streaming Orchestrator Demo")
    print("=" * 50)
    
    # Create a streaming executor
    executor = MockStreamingExecutor()
    orchestrator = Orchestrator(executor)
    
    # Create a simple task graph
    task_a = NodeSpec(
        id="research",
        type="task",
        name="Market Research",
        inputs={"topic": "AI automation tools", "depth": "comprehensive"}
    )
    
    task_b = NodeSpec(
        id="analysis", 
        type="task",
        name="Competitive Analysis",
        inputs={"from": "research", "focus": "pricing_models"}
    )
    
    graph = GraphSpec(
        nodes=[task_a, task_b],
        edges=[("research", "analysis")]
    )
    
    run_spec = RunSpec(
        run_id="demo_001",
        goal="Analyze AI automation market and competitors"
    )
    
    print(f"🚀 Starting orchestration: {run_spec.goal}")
    print(f"📊 Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print()
    
    # Run with streaming
    event_count = 0
    chunk_count = 0
    
    async for event in orchestrator.run_streaming(graph, run_spec):
        event_count += 1
        
        if event.type == "RUN_START":
            print(f"🎬 RUN_START: {event.data['goal']}")
        
        elif event.type == "NODE_START":
            print(f"\n🔧 NODE_START: {event.node_id} ({event.data['type']})")
        
        elif event.type == "AGENT_CHUNK":
            chunk_count += 1
            chunk_data = event.data
            if "message" in chunk_data:
                print(f"  💬 Message: {chunk_data['message']}")
            elif "action" in chunk_data:
                print(f"  ⚡ Action: {chunk_data['action']} - {chunk_data.get('status', 'N/A')}")
            elif "step" in chunk_data:
                print(f"  📝 Step {chunk_data['step']}: {chunk_data.get('description', 'Processing...')}")
            elif "output" in chunk_data:
                print(f"  🎯 Output: {chunk_data['output']}")
            else:
                print(f"  📦 Chunk: {chunk_data}")
        
        elif event.type == "NODE_COMPLETE":
            print(f"✅ NODE_COMPLETE: {event.node_id}")
            print(f"   Output metadata: {event.data['output_meta']}")
        
        elif event.type == "RUN_COMPLETE":
            print(f"\n🎉 RUN_COMPLETE!")
            print(f"📋 Final results:")
            for node_id, result in event.data["result"].items():
                print(f"   {node_id}: {result}")
        
        elif event.type == "ERROR":
            print(f"❌ ERROR in {event.node_id}: {event.data['error']}")
    
    print(f"\n📈 Summary:")
    print(f"   Total events: {event_count}")
    print(f"   Streaming chunks: {chunk_count}")
    print(f"   Status: ✅ Success")


async def demo_fallback_behavior():
    """Demonstrate fallback to non-streaming execution."""
    print("\n" + "=" * 50)
    print("🔄 Fallback Behavior Demo (Non-streaming executor)")
    print("=" * 50)
    
    # Create a regular (non-streaming) executor
    async def simple_task(node, ctx):
        await asyncio.sleep(0.1)
        return {"output": f"Simple result for {node.name or node.id}"}
    
    executor = CallableExecutor(simple_task)
    orchestrator = Orchestrator(executor)
    
    # Simple single-node graph
    task = NodeSpec(id="simple", type="task", name="Simple Task")
    graph = GraphSpec(nodes=[task], edges=[])
    run_spec = RunSpec(run_id="fallback_001", goal="Test fallback behavior")
    
    print(f"🚀 Starting non-streaming orchestration")
    
    chunk_count = 0
    async for event in orchestrator.run_streaming(graph, run_spec):
        if event.type == "AGENT_CHUNK":
            chunk_count += 1
        elif event.type == "RUN_START":
            print(f"🎬 RUN_START: {event.data['goal']}")
        elif event.type == "NODE_COMPLETE":
            print(f"✅ NODE_COMPLETE: {event.node_id}")
        elif event.type == "RUN_COMPLETE":
            print(f"🎉 RUN_COMPLETE: {event.data['result']}")
    
    print(f"📈 No streaming chunks (fallback worked): {chunk_count == 0}")


async def main():
    """Run both demos."""
    await demo_streaming_orchestrator()
    await demo_fallback_behavior()
    
    print(f"\n🎭 Agentic Orchestra streaming demo complete!")
    print(f"💡 Key features demonstrated:")
    print(f"   ✅ Real-time AGENT_CHUNK events")
    print(f"   ✅ Clean separation of concerns (executor vs orchestrator)")
    print(f"   ✅ Graceful fallback for non-streaming executors")
    print(f"   ✅ Proper timeout and error handling")


if __name__ == "__main__":
    asyncio.run(main())