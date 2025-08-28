#!/usr/bin/env python3
"""
Agent Orchestra Streaming Orchestrator Example

This example demonstrates the streaming orchestrator functionality with real MCP servers.
Shows how to use run_streaming() to get real-time AGENT_CHUNK events during execution.

Usage:
    OPENAI_API_KEY=<your-key> python orchestrator_streaming_example.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path


try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("❌ Install required dependency: pip install langchain-openai")
    sys.exit(1)

from agent_orchestra import SidecarMCPAgent, SidecarMCPClient, MCPExecutor, Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec

async def main():
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable")
        return
    
    print("🎭 Agent Orchestra Streaming Orchestrator")
    print("=" * 50)
    
    # Create temp directory with test files (like the working example)
    temp_dir = tempfile.mkdtemp()
    test_dir = Path(temp_dir)
    (test_dir / "hello.txt").write_text("Hello from Agent Orchestra!")
    (test_dir / "data.json").write_text('{"project": "agent-orchestra", "status": "testing"}')
    
    print(f"📁 Test files created in: {temp_dir}")
    
    # Configure MCP client (exactly like working example)
    CONFIG = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(temp_dir)]
            }
        }
    }
    
    # Create client and agent (exactly like working example)
    client = SidecarMCPClient.from_dict(CONFIG)
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    agent = SidecarMCPAgent(llm=llm, client=client, max_steps=5)
    
    try:
        print("🚀 Initializing streaming orchestrator...")
        
        print("🎬 Starting streaming orchestration demo...")
        
        # Create orchestrator components
        executor = MCPExecutor(agent)
        orchestrator = Orchestrator(executor)
        
        # Simple single task
        task = NodeSpec(
            id="test_task",
            type="task", 
            name="File Analysis",
            inputs={"instruction": "Read hello.txt and tell me what it says"}
        )
        
        graph = GraphSpec(nodes=[task], edges=[])
        run_spec = RunSpec(run_id="streaming_demo", goal="Demonstrate real-time streaming orchestration")
        
        # Run streaming orchestrator
        events = []
        chunk_count = 0
        
        print("📡 Starting streaming execution...")
        async for event in orchestrator.run_streaming(graph, run_spec):
            events.append(event)
            
            if event.type == "RUN_START":
                print(f"▶️  {event.data['goal']}")
            elif event.type == "NODE_START":
                print(f"🔧 {event.node_id}: {task.name}")
            elif event.type == "AGENT_CHUNK":
                chunk_count += 1
                chunk = event.data
                print(f"  📦 Chunk {chunk_count}: {str(chunk)[:80]}...")
            elif event.type == "NODE_COMPLETE":
                print(f"✅ {event.node_id} completed")
            elif event.type == "RUN_COMPLETE":
                print(f"🎉 Orchestration complete!")
                result = event.data["result"][task.id]
                print(f"📋 Final result: {result}")
            elif event.type == "ERROR":
                print(f"❌ Error: {event.data['error']}")
        
        print(f"\n📊 Streaming Test Results:")
        print(f"   Events: {len(events)}")
        print(f"   Chunks: {chunk_count}")
        print(f"   Success: {'✅' if any(e.type == 'RUN_COMPLETE' for e in events) else '❌'}")
        
        # Verify we got the expected events
        event_types = [e.type for e in events]
        expected = ["RUN_START", "NODE_START", "NODE_COMPLETE", "RUN_COMPLETE"]
        
        print(f"\n🔍 Event Flow Analysis:")
        for expected_event in expected:
            if expected_event in event_types:
                print(f"   ✅ {expected_event}")
            else:
                print(f"   ❌ Missing {expected_event}")
        
        if chunk_count > 0:
            print(f"   ✅ AGENT_CHUNK events: {chunk_count}")
        else:
            print(f"   ⚠️  No AGENT_CHUNK events (may be normal)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup like working example
        await client.close_all_sessions()
        print(f"🧹 Cleaned up: {temp_dir}")

if __name__ == "__main__":
    asyncio.run(main())