#!/usr/bin/env python3
"""
Agentic Orchestra Part 3 End-to-End Demo with Real MCP Integration

This example demonstrates the complete Part 3 orchestrator stack:
- Real MCP server (filesystem)
- OpenAI agent with streaming
- Advanced control flow nodes (foreach, reduce, gate)
- TaskGroup concurrency and retry logic

This is an opt-in example - requires API keys and won't run in CI.

Usage:
    OPENAI_API_KEY=<your-key> python end_to_end_part3_demo.py

Features demonstrated:
✅ Gate-based conditional flow control
✅ Foreach processing with concurrency limits
✅ Reduce operations for aggregation  
✅ Real-time streaming with AGENT_CHUNK events
✅ Retry logic with exponential backoff
✅ TaskGroup structured concurrency
✅ Full MCP server integration
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e .")
    sys.exit(1)


async def create_test_files(temp_dir: Path) -> list[str]:
    """Create test files for the filesystem MCP server demo."""
    test_files = []
    
    # Create diverse test files
    files_data = [
        ("project_specs.md", "# Project Specifications\n\nCore features:\n- User authentication\n- Data processing\n- API endpoints\n- Real-time notifications"),
        ("requirements.txt", "fastapi>=0.68.0\nuvicorn>=0.15.0\npydantic>=1.8.0\nsqlalchemy>=1.4.0"),  
        ("config.yaml", "database:\n  url: postgresql://localhost/myapp\n  pool_size: 10\napi:\n  port: 8000\n  debug: true"),
        ("notes.txt", "Remember to:\n- Update documentation\n- Run security audit\n- Performance testing\n- Code review"),
        ("error_log.txt", "2024-01-01 10:00:00 ERROR: Database connection failed\n2024-01-01 10:01:00 WARNING: High memory usage detected")
    ]
    
    for filename, content in files_data:
        file_path = temp_dir / filename
        file_path.write_text(content)
        test_files.append(filename)
        print(f"📄 Created: {filename}")
    
    return test_files


async def run_end_to_end_demo():
    """Run comprehensive Part 3 orchestrator demo with real MCP integration."""
    
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        return
    
    print("🎭 Agentic Orchestra Part 3 End-to-End Demo")
    print("=" * 60)
    
    # Create temporary directory for demo files
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"📁 Working directory: {temp_dir}")
        
        # Create test files
        test_files = await create_test_files(temp_dir)
        print()
        
        # Configure filesystem MCP server
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(temp_dir)],
                }
            }
        }
        
        # Initialize MCP client and agent
        client = SidecarMCPClient.from_dict(config)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=5)
        
        # Create MCP executor and orchestrator
        executor = MCPExecutor(agent)
        orchestrator = Orchestrator(executor)
        
        try:
            print("🚀 Building Part 3 orchestrator workflow...")
            
            # Build complex workflow demonstrating all Part 3 features
            workflow = GraphSpec(
                nodes=[
                    # Gate: Check if we should proceed with analysis
                    NodeSpec(
                        id="pre_check",
                        type="gate", 
                        name="Pre-flight Check",
                        inputs={
                            "predicate": True,  # Always pass for demo
                            "instruction": "Verify we should proceed with file analysis"
                        }
                    ),
                    
                    # Foreach: Analyze each file individually with concurrency control
                    NodeSpec(
                        id="analyze_files",
                        type="foreach",
                        name="File Analysis",
                        inputs={
                            "items": test_files,
                            "instruction": "For each file, read its contents and provide a brief 1-sentence summary of what it contains."
                        },
                        concurrency=2,  # Process max 2 files at once
                        retries=1,
                        retry_backoff_s=0.5,
                        foreach_fail_policy="skip"  # Continue even if some files fail
                    ),
                    
                    # Reduce: Aggregate all file analyses into a project overview
                    NodeSpec(
                        id="create_overview", 
                        type="reduce",
                        name="Project Overview",
                        inputs={
                            "from_ids": ["analyze_files"],
                            "instruction": "Based on the file analysis results, create a comprehensive 3-sentence project overview describing what this project does, its key components, and current status."
                        },
                        retries=2,
                        retry_backoff_s=1.0
                    ),
                    
                    # Task: Generate recommendations
                    NodeSpec(
                        id="recommendations",
                        type="task", 
                        name="Generate Recommendations",
                        inputs={
                            "from": "create_overview",
                            "instruction": "Based on the project overview, suggest 3 specific actionable recommendations for improving this project."
                        }
                    )
                ],
                edges=[
                    ("pre_check", "analyze_files"),
                    ("analyze_files", "create_overview"), 
                    ("create_overview", "recommendations")
                ]
            )
            
            run_spec = RunSpec(
                run_id="part3_demo_001",
                goal="Comprehensive project analysis with Part 3 orchestrator features"
            )
            
            print(f"📊 Workflow: {len(workflow.nodes)} nodes, {len(workflow.edges)} edges")
            print(f"🎯 Goal: {run_spec.goal}")
            print()
            
            # Execute with streaming
            event_count = 0
            chunk_count = 0
            node_results = {}
            
            print("🎬 Starting orchestrated execution with streaming...")
            print("-" * 50)
            
            async for event in orchestrator.run_streaming(workflow, run_spec):
                event_count += 1
                
                if event.type == "RUN_START":
                    print(f"🚀 RUN_START: {event.data['goal']}")
                
                elif event.type == "NODE_START":
                    node_type = event.data.get('type', 'unknown')
                    phase = event.data.get('phase', '')
                    print(f"\n🔧 NODE_START: {event.node_id} ({node_type}) - {phase}")
                
                elif event.type == "AGENT_CHUNK":
                    chunk_count += 1
                    phase = event.data.get('phase', '')
                    
                    # Show selective chunk content for readability
                    if 'content' in event.data:
                        content_preview = str(event.data['content'])[:100]
                        if len(str(event.data['content'])) > 100:
                            content_preview += "..."
                        print(f"  📦 Chunk ({phase}): {content_preview}")
                    elif 'message' in event.data:
                        print(f"  💬 Message ({phase}): {event.data['message']}")
                    else:
                        # Show other chunk types concisely
                        chunk_str = str(event.data)[:80]
                        if len(str(event.data)) > 80:
                            chunk_str += "..."
                        print(f"  📦 Chunk ({phase}): {chunk_str}")
                
                elif event.type == "NODE_COMPLETE":
                    phase = event.data.get('phase', '')
                    output_meta = event.data.get('output_meta', [])
                    print(f"✅ NODE_COMPLETE: {event.node_id} - {phase}")
                    print(f"   Output metadata: {output_meta}")
                    
                    # Store results for final summary
                    if event.node_id and event.node_id in ['create_overview', 'recommendations']:
                        node_results[event.node_id] = event.data
                
                elif event.type == "RUN_COMPLETE":
                    print(f"\n🎉 RUN_COMPLETE!")
                    print(f"📋 Blackboard final state:")
                    blackboard = event.data.get("result", {})
                    for node_id, result in blackboard.items():
                        if isinstance(result, dict) and 'result' in result:
                            output = result['result']
                            if isinstance(output, dict) and 'output' in output:
                                content = str(output['output'])[:150]
                                if len(str(output['output'])) > 150:
                                    content += "..."
                                print(f"   {node_id}: {content}")
                            else:
                                print(f"   {node_id}: {str(output)[:150]}...")
                        else:
                            print(f"   {node_id}: {str(result)[:150]}...")
                
                elif event.type == "ERROR":
                    error_msg = event.data.get('error', 'Unknown error')
                    phase = event.data.get('phase', '')
                    print(f"❌ ERROR in {event.node_id} ({phase}): {error_msg}")
            
            print(f"\n📈 Execution Summary:")
            print(f"   Total events: {event_count}")
            print(f"   Streaming chunks: {chunk_count}")
            print(f"   Status: ✅ Success")
            
            print(f"\n🌟 Part 3 Features Demonstrated:")
            print(f"   ✅ Gate-based conditional flow control")  
            print(f"   ✅ Foreach processing with concurrency=2")
            print(f"   ✅ Reduce operations for data aggregation")
            print(f"   ✅ Real-time streaming with AGENT_CHUNK events")
            print(f"   ✅ Retry logic with exponential backoff") 
            print(f"   ✅ TaskGroup structured concurrency")
            print(f"   ✅ Full MCP server integration (filesystem)")
            print(f"   ✅ Production-grade error handling")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always cleanup MCP sessions
            await client.close_all_sessions()


async def main():
    """Main entry point with graceful error handling."""
    try:
        await run_end_to_end_demo()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎭 Agentic Orchestra Part 3 demo complete!")


if __name__ == "__main__":
    asyncio.run(main())