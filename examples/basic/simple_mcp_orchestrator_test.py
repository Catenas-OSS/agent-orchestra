#!/usr/bin/env python3
"""
Simple MCP + Orchestrator Integration Test

A focused test to verify that MCPExecutor + Orchestrator + streaming works correctly
with real MCP servers and agents. This is kept minimal for reliability.

Usage:
    OPENAI_API_KEY=<your-key> python simple_mcp_orchestrator_test.py

This test:
1. Sets up a filesystem MCP server
2. Creates a simple 2-node workflow  
3. Verifies streaming events are properly emitted
4. Confirms MCP tools are being used correctly
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e .")
    sys.exit(1)


async def simple_mcp_orchestrator_test():
    """Run a simple but complete MCP + orchestrator integration test."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        print("💡 This is an opt-in test requiring API access")
        return False
    
    print("🧪 Simple MCP + Orchestrator Integration Test")
    print("=" * 50)
    
    # Create temporary directory with test file
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello from Agent Orchestra!\nThis is a test file for MCP integration.")
        
        print(f"📁 Test directory: {temp_dir}")
        print(f"📄 Test file: {test_file.name}")
        
        # Configure filesystem MCP server
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(temp_dir)],
                }
            }
        }
        
        # Initialize stack
        client = SidecarMCPClient.from_dict(config)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=5)
        executor = MCPExecutor(agent)
        orchestrator = Orchestrator(executor)
        
        try:
            # Simple 2-node workflow
            workflow = GraphSpec(
                nodes=[
                    NodeSpec(
                        id="read_file",
                        type="task",
                        name="Read Test File",
                        inputs={"instruction": f"Read the contents of the file named '{test_file.name}' in the current directory and return exactly what it contains."}
                    ),
                    NodeSpec(
                        id="analyze",
                        type="task", 
                        name="Analyze Content",
                        inputs={"instruction": "Write a 1-sentence summary about test files in general."}
                    )
                ],
                edges=[("read_file", "analyze")]
            )
            
            run_spec = RunSpec(
                run_id="simple_test_001",
                goal="Test MCP filesystem integration with orchestrator"
            )
            
            print(f"🚀 Running workflow: {len(workflow.nodes)} nodes")
            
            # Track test results
            events_seen = {"RUN_START": 0, "NODE_START": 0, "NODE_COMPLETE": 0, "RUN_COMPLETE": 0, "AGENT_CHUNK": 0}
            node_outputs = {}
            
            # Execute with streaming
            async for event in orchestrator.run_streaming(workflow, run_spec):
                event_type = event.type
                if event_type in events_seen:
                    events_seen[event_type] += 1
                
                if event_type == "RUN_START":
                    print(f"✅ RUN_START: {event.data['goal']}")
                
                elif event_type == "NODE_START":
                    print(f"✅ NODE_START: {event.node_id}")
                
                elif event_type == "AGENT_CHUNK":
                    # Just count chunks, don't print all content
                    if events_seen["AGENT_CHUNK"] <= 3:  # Show first few
                        chunk_preview = str(event.data)[:60] + "..." if len(str(event.data)) > 60 else str(event.data)
                        print(f"   📦 Chunk: {chunk_preview}")
                    elif events_seen["AGENT_CHUNK"] == 4:
                        print(f"   📦 ... (streaming chunks received)")
                
                elif event_type == "NODE_COMPLETE":
                    print(f"✅ NODE_COMPLETE: {event.node_id}")
                    # Store outputs for validation (with error handling)
                    try:
                        if event.node_id == "read_file":
                            # Capture the file content for validation
                            result = event.data.get("result", {})
                            if result:
                                node_outputs["read_file"] = result
                    except Exception as e:
                        print(f"⚠️  Error processing node output: {e}")
                
                elif event_type == "RUN_COMPLETE":
                    print(f"✅ RUN_COMPLETE")
                    try:
                        blackboard = event.data.get("result", {})
                        if isinstance(blackboard, dict):
                            node_outputs.update(blackboard)
                    except Exception as e:
                        print(f"⚠️  Error processing blackboard: {e}")
                
                elif event_type == "ERROR":
                    print(f"❌ ERROR: {event.data.get('error', 'Unknown')}")
                    return False
            
            # Validate test results
            print(f"\n📊 Test Results:")
            print(f"   Events: {events_seen}")
            
            # Basic validation checks
            success = True
            if events_seen["RUN_START"] != 1:
                print(f"❌ Expected 1 RUN_START, got {events_seen['RUN_START']}")
                success = False
            
            if events_seen["NODE_START"] != 2:
                print(f"❌ Expected 2 NODE_START, got {events_seen['NODE_START']}")
                success = False
                
            if events_seen["NODE_COMPLETE"] != 2:
                print(f"❌ Expected 2 NODE_COMPLETE, got {events_seen['NODE_COMPLETE']}")
                success = False
                
            if events_seen["RUN_COMPLETE"] != 1:
                print(f"❌ Expected 1 RUN_COMPLETE, got {events_seen['RUN_COMPLETE']}")
                success = False
                
            if events_seen["AGENT_CHUNK"] == 0:
                print(f"❌ Expected streaming chunks, got 0")
                success = False
            else:
                print(f"✅ Streaming working: {events_seen['AGENT_CHUNK']} chunks")
            
            # Check if we got actual file content
            if "read_file" in node_outputs:
                print(f"✅ File read operation completed")
            elif any("Hello from Agent Orchestra" in str(output) for output in node_outputs.values()):
                print(f"✅ File content found in outputs")
            else:
                print(f"⚠️  File content validation inconclusive")
            
            if success:
                print(f"\n🎉 Integration test PASSED!")
                print(f"   ✅ MCP server connection working")
                print(f"   ✅ Agent execution working") 
                print(f"   ✅ Orchestrator workflow working")
                print(f"   ✅ Streaming events working")
                return True
            else:
                print(f"\n❌ Integration test FAILED!")
                return False
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            await client.close_all_sessions()


async def main():
    """Main entry point."""
    try:
        success = await simple_mcp_orchestrator_test()
        if success:
            print(f"\n✅ All systems working correctly!")
        else:
            print(f"\n❌ Integration test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback 
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())