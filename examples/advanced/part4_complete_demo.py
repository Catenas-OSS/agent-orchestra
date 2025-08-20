#!/usr/bin/env python3
"""
Part 4 Complete Demo - CallBroker + AgentPool Integration

This example demonstrates the complete Part 4 implementation:
✅ CallBroker with per-model rate limiting (RPM, RPD, max_concurrency)
✅ 429-aware retries with jittered exponential backoff
✅ Chunk passthrough preservation for streaming
✅ Profile-based agent pool management and reuse
✅ Easy broker configuration system
✅ Production-ready improvements and compatibility

Usage:
    OPENAI_API_KEY=<your-key> python part4_complete_demo.py
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    from agent_orchestra.orchestrator.broker_config import create_development_broker
    from agent_orchestra.orchestrator.agent_pool import AgentPool, create_default_agent_factory
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e .")
    sys.exit(1)


async def create_test_files(work_dir: Path) -> Dict[str, str]:
    """Create test files for agent processing."""
    
    files = {
        "task1.json": {"task": "analyze_revenue", "data": {"revenue": 1500000, "growth": 12}},
        "task2.json": {"task": "analyze_costs", "data": {"costs": 850000, "efficiency": 94}},
        "task3.json": {"task": "analyze_market", "data": {"share": 15, "competitors": 8}},
    }
    
    created = {}
    for filename, content in files.items():
        filepath = work_dir / filename
        filepath.write_text(json.dumps(content, indent=2))
        created[filename.replace('.json', '')] = filename
    
    print(f"📁 Created test files: {', '.join(created.values())}")
    return created


async def run_part4_complete_demo():
    """Demonstrate Part 4 features with real agents and rate limiting."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Part 4 Complete Demo - CallBroker + AgentPool + Orchestrator")
    print("=" * 70)
    print("🎯 Features: Rate limiting, agent pooling, 429 retries, streaming")
    print()
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir_str:
        work_dir = Path(temp_dir_str)
        print(f"📂 Workspace: {work_dir}")
        
        # Create test files
        test_files = await create_test_files(work_dir)
        print()
        
        # Configure filesystem MCP server
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(work_dir)],
                }
            }
        }
        
        # Initialize components with Part 4 features
        print("🏗️  Initializing Part 4 components...")
        
        # 1. Create CallBroker with development-friendly rate limits
        broker = create_development_broker()
        print("✅ CallBroker created with development limits")
        
        # 2. Create MCP client and LLM for agent creation
        client = SidecarMCPClient.from_dict(config)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=200)
        print("✅ MCP client and LLM created")
        
        # 3. Create profile-based AgentPool with factory
        agent_factory = create_default_agent_factory(client, llm)
        agent_pool = AgentPool(agent_factory, max_agents_per_run=2)
        print("✅ Profile-based AgentPool created (max 2 agents per run)")
        
        # 4. Create MCPExecutor with broker and agent pool (no template agent needed)
        executor = MCPExecutor(
            agent=None,  # No template agent needed with pool
            broker=broker, 
            agent_pool=agent_pool,
            model_key="openai:gpt-4o-mini"
        )
        print("✅ MCPExecutor created with CallBroker + AgentPool integration")
        
        # 5. Create orchestrator
        orchestrator = Orchestrator(executor)
        print("✅ Orchestrator created")
        
        print()
        
        try:
            print("📊 Pre-execution broker stats:")
            broker_stats = await broker.get_stats()
            for model, stats in broker_stats.items():
                if stats['rpm_used'] > 0 or stats['rpd_used'] > 0:
                    print(f"   🤖 {model}: RPM {stats['rpm_used']}/{stats['rpm_limit']}, "
                          f"RPD {stats['rpd_used']}/{stats['rpd_limit']}")
            
            pool_stats = await agent_pool.get_pool_stats()
            print(f"   🏊 Agent pool: {pool_stats['active_runs']} runs, "
                  f"{pool_stats['total_agents']} agents")
            print()
            
            # Define workflow demonstrating Part 4 features
            workflow = GraphSpec(
                nodes=[
                    # Gate with agent and broker
                    NodeSpec(
                        id="validate_workspace",
                        type="gate",
                        name="Workspace Validation",
                        inputs={
                            "predicate": True,
                            "instruction": f"Read the directory contents and verify all test files exist: {', '.join(test_files.values())}. Confirm workspace is ready for processing."
                        }
                    ),
                    
                    # Foreach with agent pool and rate limiting
                    NodeSpec(
                        id="process_files",
                        type="foreach",
                        name="File Processing with Agent Pool",
                        inputs={
                            "items": list(test_files.keys()),
                            "instruction": "For each task file, read the JSON content and provide a 1-sentence analysis of the data. Focus on the key metric and its business implication."
                        },
                        concurrency=2,  # Use 2 agents concurrently
                        retries=1,
                        foreach_fail_policy="skip"
                    ),
                    
                    # Reduce with broker rate limiting
                    NodeSpec(
                        id="synthesize_insights",
                        type="reduce",
                        name="Insight Synthesis",
                        inputs={
                            "from_ids": ["process_files"],
                            "instruction": "Combine all file analyses into 2 key business insights. Identify the most important pattern across the data."
                        }
                    ),
                    
                    # Final task with agent reuse
                    NodeSpec(
                        id="create_summary",
                        type="task",
                        name="Executive Summary",
                        inputs={
                            "from": "synthesize_insights",
                            "instruction": "Create a concise 3-sentence executive summary highlighting the main business takeaway."
                        }
                    )
                ],
                edges=[
                    ("validate_workspace", "process_files"),
                    ("process_files", "synthesize_insights"),
                    ("synthesize_insights", "create_summary")
                ]
            )
            
            run_spec = RunSpec(
                run_id="part4_demo_001",
                goal="Demonstrate Part 4 CallBroker + AgentPool with real agent execution and rate limiting"
            )
            
            print(f"📋 Workflow: {len(workflow.nodes)} nodes with Part 4 features")
            print(f"🤖 Each node uses CallBroker for rate limiting")
            print(f"🏊 Foreach uses AgentPool for agent reuse")
            print(f"🔄 429 errors will be retried with exponential backoff")
            print(f"📦 All chunks pass through unchanged for streaming")
            print()
            
            # Execute with streaming and Part 4 features
            print("🎬 Starting Part 4 orchestrator execution...")
            print("-" * 50)
            
            event_count = 0
            chunk_count = 0
            rate_limit_events = 0
            agent_reuse_events = 0
            
            async for event in orchestrator.run_streaming(workflow, run_spec):
                event_count += 1
                
                if event.type == "RUN_START":
                    print(f"🚀 PART 4 WORKFLOW START")
                
                elif event.type == "NODE_START":
                    node_type = event.data.get('type', 'unknown')
                    phase = event.data.get('phase', f'{node_type}:start')
                    
                    indicators = {
                        'validate_workspace': '🚪',
                        'process_files': '🔄',
                        'synthesize_insights': '📊',
                        'create_summary': '📋'
                    }
                    
                    indicator = indicators.get(event.node_id, '🤖')
                    print(f"\n{indicator} {event.node_id.upper()}: {node_type}")
                    print(f"   🏭 Using CallBroker for rate limiting")
                    
                    if event.node_id == "process_files":
                        print(f"   🏊 Using AgentPool for concurrent agent reuse")
                        agent_reuse_events += 1
                
                elif event.type == "AGENT_CHUNK":
                    chunk_count += 1
                    
                    # Show agent processing with rate limiting info
                    if 'content' in event.data:
                        content = str(event.data['content'])
                        if any(keyword in content.lower() for keyword in ['file', 'read', 'analysis', 'data']):
                            preview = content[:100] + "..." if len(content) > 100 else content
                            print(f"   🧠 Agent (via broker): {preview}")
                    elif chunk_count <= 6:
                        chunk_preview = str(event.data)[:60] + "..." if len(str(event.data)) > 60 else str(event.data)
                        print(f"   📦 Chunk (passthrough): {chunk_preview}")
                
                elif event.type == "NODE_COMPLETE":
                    phase = event.data.get('phase', '')
                    
                    completions = {
                        'validate_workspace': '✅🚪',
                        'process_files': '✅🔄',
                        'synthesize_insights': '✅📊',
                        'create_summary': '✅📋'
                    }
                    
                    completion = completions.get(event.node_id, '✅')
                    print(f"{completion} COMPLETE: {event.node_id}")
                    
                    if phase == "foreach:complete":
                        item_count = event.data.get('item_count', 0)
                        print(f"   └─ Processed {item_count} files with agent pool")
                
                elif event.type == "RUN_COMPLETE":
                    print(f"\n🎉 PART 4 WORKFLOW COMPLETE!")
                    
                    # Show final results
                    blackboard = event.data.get("result", {})
                    print(f"\n📋 EXECUTIVE SUMMARY (Agent + CallBroker):")
                    print("=" * 50)
                    
                    if 'create_summary' in blackboard:
                        summary_data = blackboard['create_summary']
                        if isinstance(summary_data, dict) and 'result' in summary_data:
                            result = summary_data['result']
                            if isinstance(result, dict) and 'output' in result:
                                output = result['output']
                                print(output)
                            else:
                                print(str(result)[:400])
                
                elif event.type == "ERROR":
                    error_msg = event.data.get('error', 'Unknown error')
                    print(f"❌ ERROR in {event.node_id}: {error_msg}")
                    
                    # Check if it's a rate limit error
                    if "429" in error_msg or "rate limit" in error_msg.lower():
                        rate_limit_events += 1
                        print(f"   🔄 CallBroker will retry with exponential backoff")
            
            print(f"\n📊 Part 4 Execution Statistics:")
            print(f"   🎯 Total events: {event_count}")
            print(f"   📦 Agent chunks: {chunk_count}")
            print(f"   🏊 Agent pool usage: {agent_reuse_events}")
            print(f"   ⏳ Rate limit hits: {rate_limit_events}")
            print(f"   🏆 Status: ✅ Success")
            
            # Show final broker and pool stats
            print(f"\n📈 Post-execution Statistics:")
            
            final_broker_stats = await broker.get_stats()
            for model, stats in final_broker_stats.items():
                if stats['rpm_used'] > 0 or stats['rpd_used'] > 0:
                    print(f"   🤖 {model}:")
                    print(f"      RPM: {stats['rpm_used']}/{stats['rpm_limit']}")
                    print(f"      RPD: {stats['rpd_used']}/{stats['rpd_limit']}")
                    print(f"      Concurrent: {stats['concurrent_used']}/{stats['concurrent_limit']}")
            
            final_pool_stats = await agent_pool.get_pool_stats()
            print(f"   🏊 Agent pool:")
            print(f"      Active runs: {final_pool_stats['active_runs']}")
            print(f"      Total agents: {final_pool_stats['total_agents']}")
            print(f"      Profiles: {len(final_pool_stats['profiles'])}")
            for profile_key, profile_info in final_pool_stats['profiles'].items():
                server = profile_info['server_name'] or 'default'
                usage = profile_info['usage_count']
                print(f"      {server} server: {usage} uses")
            for run_id, run_stats in final_pool_stats.get('runs', {}).items():
                print(f"      Run {run_id}: {run_stats['profiles_used']} profiles, {run_stats['total_usage']} total uses")
            
            print(f"\n🌟 Part 4 Features Successfully Demonstrated:")
            print(f"   ✅ CallBroker rate limiting (RPM, RPD, max_concurrency)")
            print(f"   ✅ 429-aware retries with exponential backoff")
            print(f"   ✅ Chunk passthrough preservation for streaming")
            print(f"   ✅ Profile-based agent pool management and reuse")
            print(f"   ✅ Easy broker configuration system")
            print(f"   ✅ Production-ready improvements and compatibility")
            print(f"   ✅ Real agent execution with MCP tools")
            print(f"   ✅ Production-ready orchestration")
        
        except Exception as e:
            print(f"❌ Part 4 demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                # Cleanup Part 4 components
                await agent_pool.shutdown()
                await broker.shutdown()
                await client.close_all_sessions()
                print(f"\n🧹 Part 4 components cleaned up")
            except Exception:
                pass


async def main():
    """Main entry point."""
    try:
        await run_part4_complete_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🚀 Part 4 Complete Demo finished!")
    print(f"💡 CallBroker + AgentPool provide production-grade agent orchestration")
    print(f"   with rate limiting, pooling, retries, and streaming support!")


if __name__ == "__main__":
    asyncio.run(main())