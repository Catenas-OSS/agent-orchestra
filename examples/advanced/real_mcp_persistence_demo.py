#!/usr/bin/env python3
"""
Real MCP Persistence Demo - Simplified version with actual agent tool calls

This demonstrates the persistence system with real MCP agents making actual tool calls.

Requirements:
    - OpenAI API key (set OPENAI_API_KEY environment variable)
    - Node.js for MCP filesystem server (already detected)

Usage:
    OPENAI_API_KEY=<your-key> python3 real_mcp_persistence_demo.py
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator import (
        Orchestrator,
        JsonlRunStore,
        MCPExecutor,
        GraphSpec,
        NodeSpec,
        RunSpec
    )
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install python-dotenv langchain-openai")
    sys.exit(1)


def create_test_data_files(work_dir: Path) -> Dict[str, str]:
    """Create simple test data files for agent processing."""
    
    # Simple sales data
    sales_data = {
        "q1_2024": {"revenue": 150000, "customers": 45},
        "q2_2024": {"revenue": 175000, "customers": 52}
    }
    
    # Simple customer data  
    customer_data = {
        "segments": {
            "enterprise": {"count": 12, "value": 8500},
            "small_business": {"count": 85, "value": 1200}
        }
    }
    
    # Create files
    files = {
        "sales.json": sales_data,
        "customers.json": customer_data
    }
    
    created_files = {}
    for filename, data in files.items():
        filepath = work_dir / filename
        filepath.write_text(json.dumps(data, indent=2))
        created_files[filename.replace('.json', '')] = str(filepath)
        
    print(f"📁 Created test files: {', '.join(files.keys())}")
    return created_files


def create_simple_workflow(data_files: Dict[str, str]) -> GraphSpec:
    """Create a simple workflow that makes real agent tool calls."""
    
    return GraphSpec(
        nodes=[
            # Agent reads and analyzes sales file
            NodeSpec(
                id="analyze_sales",
                type="task",
                name="Analyze Sales Data",
                inputs={
                    "instruction": f"Read the sales data from {data_files['sales']} and calculate the growth rate between Q1 and Q2. Write a brief summary.",
                    "file_path": data_files['sales']
                },
                timeout_s=30
            ),
            
            # Agent reads and analyzes customer file
            NodeSpec(
                id="analyze_customers", 
                type="task",
                name="Analyze Customer Data",
                inputs={
                    "instruction": f"Read the customer data from {data_files['customers']} and identify which segment is more valuable. Calculate total value per segment.",
                    "file_path": data_files['customers']
                },
                timeout_s=30
            ),
            
            # Agent creates summary report
            NodeSpec(
                id="create_report",
                type="task",
                name="Create Business Report",
                inputs={
                    "instruction": "Create a brief business report combining the sales and customer analysis. Save it as 'business_report.txt' in the working directory.",
                    "output_file": "business_report.txt"
                },
                timeout_s=25
            )
        ],
        edges=[
            ("analyze_sales", "create_report"),
            ("analyze_customers", "create_report")
        ]
    )


async def run_demo_phase(orchestrator, workflow, run_spec, phase_name="", resume=False, interrupt_after=None):
    """Run a single phase of the workflow with progress reporting."""
    
    print(f"🎬 {phase_name.upper()} PHASE")
    print("-" * 50)
    
    start_time = time.time()
    events = []
    agent_calls = 0
    cached_nodes = 0
    completed_nodes = 0
    
    try:
        async for event in orchestrator.run_streaming(workflow, run_spec, resume=resume):
            events.append(event)
            
            if event.type == "RUN_START":
                resumed_text = " (RESUMED)" if event.data.get('resumed') else ""
                print(f"🚀 Workflow started{resumed_text}")
                
            elif event.type == "NODE_START":
                node_type = event.data.get('type', 'unknown')
                print(f"\n🔧 {event.node_id}: {node_type}")
                
            elif event.type == "AGENT_CHUNK":
                agent_calls += 1
                # Show some agent activity (limit output)
                if agent_calls <= 5:
                    # Handle different event data formats
                    if isinstance(event.data, dict):
                        content = str(event.data.get('content', ''))[:80]
                    else:
                        content = str(event.data)[:80]
                    if content.strip():
                        print(f"   🤖 {content}...")
                elif agent_calls == 6:
                    print(f"   🤖 [agent working...]")
                    
            elif event.type == "NODE_COMPLETE":
                completed_nodes += 1
                if event.data.get('resumed'):
                    cached_nodes += 1
                    print(f"   ⚡ RESUMED from checkpoint")
                else:
                    print(f"   ✅ COMPLETED with agent tools")
                
                # Check for interruption
                if interrupt_after and completed_nodes >= interrupt_after:
                    print(f"\n🚨 INTERRUPTING after {completed_nodes} nodes...")
                    break
                    
            elif event.type == "RUN_COMPLETE":
                elapsed = time.time() - start_time
                print(f"\n🎉 {phase_name.upper()} COMPLETED!")
                print(f"   ⏱️  Time: {elapsed:.2f}s")
                print(f"   📊 Events: {len(events)}")
                print(f"   🤖 Agent calls: {agent_calls}")
                print(f"   ⚡ Cached: {cached_nodes}")
                print(f"   🔄 Executed: {completed_nodes}")
                
            elif event.type == "ERROR":
                error = event.data.get('error', 'Unknown error')
                print(f"   ❌ ERROR: {error}")
                
    except Exception as e:
        print(f"\n💥 {phase_name} failed: {e}")
        raise
    
    return events, {"cached": cached_nodes, "executed": completed_nodes}


async def run_real_mcp_demo():
    """Main demo showing real MCP agents with persistence."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("💡 Get your API key from: https://platform.openai.com/api-keys")
        print("💡 Then run: OPENAI_API_KEY=<your-key> python3 real_mcp_persistence_demo.py")
        return
    
    print("🚀 Real MCP Agent Persistence Demo")
    print("=" * 60)
    print("🎯 Demonstrating actual agent tool calls with filesystem MCP server")
    print("💾 Testing durable checkpoints and intelligent resume")
    print()
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory(prefix="real_mcp_demo_") as temp_dir_str:
        work_dir = Path(temp_dir_str)
        checkpoint_dir = work_dir / "checkpoints"
        
        print(f"📂 Workspace: {work_dir}")
        print(f"💾 Checkpoints: {checkpoint_dir}")
        print()
        
        # Create test data files  
        data_files = create_test_data_files(work_dir)
        
        # Configure MCP filesystem server
        mcp_config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(work_dir)],
                }
            }
        }
        
        # Initialize MCP components
        print("🔧 Setting up real MCP agent...")
        client = SidecarMCPClient.from_dict(mcp_config)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500)
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=8)
        
        # Create persistence store
        store = JsonlRunStore(checkpoint_dir, flush_every=1)
        
        # Create broker with FREE TIER limits (3 RPM max)
        from agent_orchestra.orchestrator import BrokerConfig
        
        # Super conservative config for OpenAI free tier
        free_tier_config = BrokerConfig(
            default_rpm=2,  # Stay well below 3 RPM limit
            default_rpd=50,  # Conservative daily limit
            default_max_concurrency=1,  # Only 1 request at a time
            max_retries=5,  # More retries for rate limits
            base_delay=25.0,  # Wait 25s between retries
            max_delay=120.0,  # Max 2 min wait
            model_limits={
                "openai:gpt-4o-mini": {"rpm": 2, "rpd": 50, "max_concurrency": 1}
            }
        )
        
        broker = free_tier_config.create_broker()
        
        print("🛡️ Using FREE TIER rate limits (2 RPM, 1 concurrent request)")
        
        # Create MCP executor with rate limiting
        executor = MCPExecutor(agent=agent, default_server="filesystem", broker=broker)
        orchestrator = Orchestrator(executor, store=store)
        
        print("✅ Real MCP agent created with filesystem server")
        print("✅ Persistence store initialized") 
        print("✅ Orchestrator ready")
        print()
        
        # Create workflow
        workflow = create_simple_workflow(data_files)
        run_spec = RunSpec(
            run_id="real_mcp_demo_001",
            goal="Analyze business data using real agent tool calls"
        )
        
        print(f"📋 Workflow: {len(workflow.nodes)} nodes with real MCP tool calls")
        print()
        
        try:
            # Phase 1: Initial execution (interrupt partway through)
            events1, stats1 = await run_demo_phase(
                orchestrator, workflow, run_spec, 
                phase_name="Initial", interrupt_after=2
            )
            
            # Simulate system restart
            print("\n⏳ Simulating system restart...")
            await asyncio.sleep(1)
            
            # Phase 2: Resume from checkpoint
            fresh_client = SidecarMCPClient.from_dict(mcp_config)
            fresh_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500)
            fresh_agent = SidecarMCPAgent(llm=fresh_llm, client=fresh_client, max_steps=8)
            fresh_executor = MCPExecutor(agent=fresh_agent, default_server="filesystem", broker=broker)
            fresh_orchestrator = Orchestrator(fresh_executor, store=store)
            
            events2, stats2 = await run_demo_phase(
                fresh_orchestrator, workflow, run_spec,
                phase_name="Resume", resume=True
            )
            
            # Show results
            await show_results(work_dir, checkpoint_dir, run_spec.run_id, [stats1, stats2])
            
        except Exception as e:
            print(f"\n💥 Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                await client.close_all_sessions()
                if 'fresh_client' in locals():
                    await fresh_client.close_all_sessions()
            except:
                pass


async def show_results(work_dir: Path, checkpoint_dir: Path, run_id: str, phase_stats: list):
    """Show final results and analysis."""
    
    print("\n" + "=" * 60)
    print("📊 DEMO RESULTS")
    print("=" * 60)
    
    # Show phase statistics
    print("\n📈 Phase Statistics:")
    for i, stats in enumerate(phase_stats, 1):
        cached = stats.get('cached', 0)
        executed = stats.get('executed', 0)
        print(f"   {i}. Phase {i}:")
        print(f"      Cached nodes: {cached}")
        print(f"      Executed nodes: {executed}")
    
    # Show generated files
    print(f"\n📁 Generated Files:")
    report_files = list(work_dir.glob("*report*")) + list(work_dir.glob("*.txt"))
    if report_files:
        for file in report_files:
            size = file.stat().st_size
            print(f"   📄 {file.name}: {size:,} bytes")
            # Show content preview
            try:
                content = file.read_text()[:200]
                print(f"      Preview: {content}...")
            except:
                pass
    else:
        print("   ⚠️  No output files found")
    
    # Show checkpoint analysis
    print(f"\n💾 Checkpoint Analysis:")
    run_checkpoint_dir = checkpoint_dir / run_id
    if run_checkpoint_dir.exists():
        checkpoint_files = list(run_checkpoint_dir.iterdir())
        total_size = sum(f.stat().st_size for f in checkpoint_files)
        
        print(f"   Directory: {run_checkpoint_dir}")
        print(f"   Files: {len(checkpoint_files)} ({total_size:,} bytes total)")
        
        for file in sorted(checkpoint_files):
            size = file.stat().st_size
            print(f"   📄 {file.name}: {size:,} bytes")
        
        # Show checkpointed nodes
        if (run_checkpoint_dir / "nodes.json").exists():
            try:
                nodes_data = json.loads((run_checkpoint_dir / "nodes.json").read_text())
                print(f"\n   💾 Checkpointed nodes: {len(nodes_data)}")
                for node_id in nodes_data.keys():
                    print(f"      🔧 {node_id}")
            except:
                pass
    else:
        print("   ⚠️  No checkpoint directory found")
    
    print(f"\n🎉 Real MCP Persistence Demo Complete!")
    print("✅ Real agent tool calls with filesystem MCP server")
    print("✅ Intelligent resume skipping completed agent work")
    print("✅ Production-ready persistence for agent workflows!")


if __name__ == "__main__":
    asyncio.run(run_real_mcp_demo())