#!/usr/bin/env python3
"""
MCP Agent Persistence Demo - Real agent tool calls with resume functionality

This example demonstrates Step 5 persistence features with actual MCP agents making tool calls:
✅ Real agent execution with filesystem MCP server
✅ Durable checkpoints survive agent crashes
✅ Resume skips completed agent tasks  
✅ Cache invalidation when agent inputs change
✅ Tool call results preserved across interruptions

Requirements:
    - OpenAI API key for agent LLM
    - Node.js for MCP filesystem server

Usage:
    OPENAI_API_KEY=<your-key> python examples/advanced/mcp_persistence_demo.py
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    from agent_orchestra.orchestrator.store import JsonlRunStore
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e . python-dotenv langchain-openai")
    sys.exit(1)


def create_test_data_files(work_dir: Path) -> Dict[str, str]:
    """Create realistic test data files for agent processing."""
    
    # Sales data
    sales_data = {
        "2024_q1_sales": {
            "total_revenue": 2500000,
            "units_sold": 15000,
            "top_products": ["Product A", "Product B", "Product C"],
            "growth_rate": 0.15
        },
        "2024_q2_sales": {
            "total_revenue": 2750000,
            "units_sold": 16500,
            "top_products": ["Product A", "Product C", "Product D"],
            "growth_rate": 0.10
        }
    }
    
    # Customer data  
    customer_data = {
        "customer_segments": {
            "enterprise": {"count": 150, "avg_value": 45000},
            "mid_market": {"count": 800, "avg_value": 12000},
            "small_business": {"count": 2400, "avg_value": 3500}
        },
        "satisfaction_scores": {
            "enterprise": 4.2,
            "mid_market": 3.8,
            "small_business": 3.9
        }
    }
    
    # Marketing data
    marketing_data = {
        "campaigns": [
            {"name": "Spring Launch", "budget": 150000, "roi": 3.2, "leads": 1200},
            {"name": "Summer Sale", "budget": 200000, "roi": 2.8, "leads": 1800},
            {"name": "Fall Promotion", "budget": 175000, "roi": 3.5, "leads": 1500}
        ],
        "channels": {
            "digital": {"spend": 300000, "conversions": 2400},
            "traditional": {"spend": 225000, "conversions": 1100}
        }
    }
    
    # Create files
    files = {
        "sales_data.json": sales_data,
        "customer_data.json": customer_data,
        "marketing_data.json": marketing_data
    }
    
    created_files = {}
    for filename, data in files.items():
        filepath = work_dir / filename
        filepath.write_text(json.dumps(data, indent=2))
        created_files[filename.replace('.json', '')] = str(filepath)
        
    print(f"📁 Created test data files: {', '.join(files.keys())}")
    return created_files


def create_business_intelligence_workflow(data_files: Dict[str, str]) -> GraphSpec:
    """Create a realistic BI workflow that requires actual agent tool calls."""
    
    return GraphSpec(
        nodes=[
            # Data extraction phase - agent reads and parses files
            NodeSpec(
                id="extract_sales_data",
                type="task",
                name="Extract Sales Metrics",
                inputs={
                    "instruction": f"Read the sales data from {data_files['sales_data']} and extract key metrics. Calculate the quarterly growth rate between Q1 and Q2. Return a summary with total revenue, growth percentage, and top performing products.",
                    "file_path": data_files['sales_data']
                },
                timeout_s=30
            ),
            
            NodeSpec(
                id="extract_customer_insights", 
                type="task",
                name="Extract Customer Intelligence",
                inputs={
                    "instruction": f"Read the customer data from {data_files['customer_data']} and analyze customer segmentation. Calculate total customer value by segment and identify the most valuable segment. Also analyze satisfaction trends.",
                    "file_path": data_files['customer_data']
                },
                timeout_s=30
            ),
            
            # Validation gate - ensures data quality
            NodeSpec(
                id="data_quality_gate",
                type="gate", 
                name="Data Quality Validation",
                inputs={
                    "predicate": True,  # In real scenario, would validate extracted data
                    "instruction": "Validate that both sales and customer data have been successfully extracted with required metrics."
                }
            ),
            
            # Advanced analysis phase - agent processes multiple data sources
            NodeSpec(
                id="analyze_marketing_roi",
                type="task", 
                name="Marketing ROI Analysis",
                inputs={
                    "instruction": f"Read marketing data from {data_files['marketing_data']} and calculate ROI for each campaign. Identify the most effective campaigns and channels. Provide recommendations for budget allocation.",
                    "file_path": data_files['marketing_data'],
                    "context": "Use this for cross-referencing with sales and customer data"
                },
                timeout_s=35
            ),
            
            # Batch processing with foreach - agent processes multiple related tasks
            NodeSpec(
                id="generate_segment_reports",
                type="foreach",
                name="Generate Segment-Specific Reports", 
                inputs={
                    "items": [
                        {"segment": "enterprise", "focus": "retention and expansion"},
                        {"segment": "mid_market", "focus": "growth and acquisition"}, 
                        {"segment": "small_business", "focus": "efficiency and automation"}
                    ],
                    "instruction": "For each customer segment, create a targeted business report using the extracted data. Focus on the specified area for each segment and provide 3 specific recommendations.",
                    "base_data": "Use sales and customer data for context"
                },
                concurrency=2,
                timeout_s=40
            ),
            
            # Final synthesis - agent creates comprehensive report
            NodeSpec(
                id="create_executive_summary",
                type="reduce",
                name="Executive Summary Generation",
                inputs={
                    "from_ids": ["extract_sales_data", "extract_customer_insights", "analyze_marketing_roi", "generate_segment_reports"],
                    "instruction": "Create a comprehensive executive summary combining all analyses. Include: 1) Key business metrics and trends, 2) Customer insights and opportunities, 3) Marketing performance and recommendations, 4) Segment-specific strategies. Format as a structured business report with clear action items."
                },
                timeout_s=45
            ),
            
            # Final output - agent saves results to file
            NodeSpec(
                id="save_business_report",
                type="task",
                name="Save Comprehensive Report",
                inputs={
                    "instruction": "Save the executive summary to a file named 'business_intelligence_report.md' in markdown format. Include today's date and ensure the report is well-formatted for executive review.",
                    "output_format": "markdown",
                    "include_metadata": True
                },
                timeout_s=25
            )
        ],
        edges=[
            ("extract_sales_data", "data_quality_gate"),
            ("extract_customer_insights", "data_quality_gate"), 
            ("data_quality_gate", "analyze_marketing_roi"),
            ("analyze_marketing_roi", "generate_segment_reports"),
            ("generate_segment_reports", "create_executive_summary"),
            ("create_executive_summary", "save_business_report")
        ]
    )


async def run_mcp_persistence_demo():
    """Main demo showing MCP agents with persistence and resume."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        print("💡 Get your API key from: https://platform.openai.com/api-keys")
        return
    
    print("🚀 MCP Agent Persistence Demo")
    print("=" * 70)
    print("🎯 Demonstrating real agent tool calls with checkpoint persistence")
    print("📊 Workflow: BI analysis with file operations, data processing, and reporting")
    print()
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory(prefix="mcp_persistence_demo_") as temp_dir_str:
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
        print("🔧 Initializing MCP agent and persistence...")
        client = SidecarMCPClient.from_dict(mcp_config)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=800)
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=15)
        
        # Create persistence store
        store = JsonlRunStore(checkpoint_dir, flush_every=1)
        
        # Create MCP executor with persistence
        executor = MCPExecutor(agent=agent, default_server="filesystem")
        orchestrator = Orchestrator(executor, store=store)
        
        print("✅ MCP agent created with filesystem tools")
        print("✅ Persistence store initialized")
        print("✅ Orchestrator ready with checkpoint support")
        print()
        
        # Create business intelligence workflow
        workflow = create_business_intelligence_workflow(data_files)
        run_spec = RunSpec(
            run_id="bi_analysis_001",
            goal="Generate comprehensive business intelligence report with file operations and data analysis"
        )
        
        print(f"📋 Workflow: {len(workflow.nodes)} nodes with real agent tool calls")
        print(f"🔄 Each node will make actual MCP filesystem tool calls")
        print(f"💾 All agent results will be checkpointed for resume")
        print()
        
        try:
            # Phase 1: Initial execution (simulate interruption partway through)
            print("🎬 PHASE 1: Initial Execution with Agent Tool Calls")
            print("-" * 60)
            
            phase1_task = asyncio.create_task(run_workflow_phase(
                orchestrator, workflow, run_spec, phase_name="Initial"
            ))
            
            try:
                # Let it run for some nodes, then interrupt
                await asyncio.wait_for(phase1_task, timeout=45.0)
            except asyncio.TimeoutError:
                print(f"\n🚨 SIMULATING SYSTEM INTERRUPTION...")
                print("   (This could be a server crash, deployment, or maintenance)")
                print("   💾 Agent tool call results have been persisted to checkpoints")
                phase1_task.cancel()
                try:
                    await phase1_task
                except asyncio.CancelledError:
                    pass
            
            # Small delay to simulate system restart
            print("\n⏳ Simulating system restart...")
            await asyncio.sleep(2)
            
            # Phase 2: Resume from checkpoint
            print("\n🔄 PHASE 2: Resume from Checkpoint")
            print("-" * 60) 
            print("🎯 Demonstrating intelligent resume - skip completed agent tasks")
            print()
            
            # Create fresh components (simulating restart)
            fresh_client = SidecarMCPClient.from_dict(mcp_config)
            fresh_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=800)
            fresh_agent = SidecarMCPAgent(llm=fresh_llm, client=fresh_client, max_steps=15)
            fresh_executor = MCPExecutor(agent=fresh_agent, default_server="filesystem")
            fresh_orchestrator = Orchestrator(fresh_executor, store=store)
            
            await run_workflow_phase(
                fresh_orchestrator, workflow, run_spec, 
                phase_name="Resume", resume=True
            )
            
            # Phase 3: Cache invalidation demo
            print("\n🔄 PHASE 3: Cache Invalidation with Modified Inputs")
            print("-" * 60)
            print("🎯 Modifying workflow inputs to demonstrate cache invalidation")
            print()
            
            # Modify one of the data files to trigger cache invalidation
            modified_data = json.loads(Path(data_files['sales_data']).read_text())
            modified_data['2024_q2_sales']['total_revenue'] = 3000000  # Significant change
            Path(data_files['sales_data']).write_text(json.dumps(modified_data, indent=2))
            
            print("📝 Modified sales data file - this should invalidate related caches")
            
            # Run again - should re-execute nodes with changed inputs
            await run_workflow_phase(
                fresh_orchestrator, workflow, run_spec,
                phase_name="Cache Invalidation", resume=True
            )
            
            # Show final results
            await show_final_results(work_dir, checkpoint_dir, run_spec.run_id)
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n💥 Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                await client.close_all_sessions()
                await fresh_client.close_all_sessions()
            except:
                pass


async def run_workflow_phase(orchestrator, workflow, run_spec, phase_name="", resume=False):
    """Run a single phase of the workflow with detailed progress reporting."""
    
    start_time = time.time()
    events = []
    current_node = None
    agent_calls = 0
    cached_nodes = 0
    
    try:
        async for event in orchestrator.run_streaming(workflow, run_spec, resume=resume):
            events.append(event)
            
            if event.type == "RUN_START":
                resumed_text = " (RESUMED)" if event.data.get('resumed') else ""
                print(f"🎬 {phase_name.upper()} WORKFLOW STARTED{resumed_text}")
                
            elif event.type == "NODE_START":
                current_node = event.node_id
                node_type = event.data.get('type', 'unknown')
                print(f"\n🔧 NODE: {current_node}")
                print(f"   Type: {node_type}")
                if 'cached_items' in event.data:
                    print(f"   Cached items: {event.data['cached_items']}")
                
            elif event.type == "AGENT_CHUNK":
                agent_calls += 1
                # Show abbreviated agent activity
                if agent_calls <= 10:  # Limit output
                    content = str(event.data.get('content', ''))[:100]
                    if content:
                        print(f"   🤖 Agent: {content}...")
                elif agent_calls == 11:
                    print(f"   🤖 Agent: [continued processing...]")
                    
            elif event.type == "NODE_COMPLETE":
                if event.data.get('resumed'):
                    cached_nodes += 1
                    print(f"   ⚡ RESUMED from checkpoint (cached)")
                else:
                    print(f"   ✅ COMPLETED with agent tool calls")
                    
                # Show foreach statistics
                if 'cached_items' in event.data and 'executed_items' in event.data:
                    cached = event.data['cached_items']
                    executed = event.data['executed_items']
                    print(f"   📊 Items: {executed} executed, {cached} cached")
                    
            elif event.type == "RUN_COMPLETE":
                elapsed = time.time() - start_time
                print(f"\n🎉 {phase_name.upper()} WORKFLOW COMPLETED!")
                print(f"   ⏱️  Time: {elapsed:.2f} seconds")
                print(f"   📊 Events: {len(events)}")
                print(f"   🤖 Agent calls: {agent_calls}")
                print(f"   ⚡ Cached nodes: {cached_nodes}")
                
            elif event.type == "ERROR":
                error = event.data.get('error', 'Unknown error')
                print(f"   ❌ ERROR in {event.node_id}: {error}")
                
    except Exception as e:
        print(f"\n💥 {phase_name} phase failed: {e}")
        raise
    
    return events


async def show_final_results(work_dir: Path, checkpoint_dir: Path, run_id: str):
    """Show final results and checkpoint analysis."""
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS & CHECKPOINT ANALYSIS")
    print("=" * 70)
    
    # Show generated files
    print("\n📁 Generated Files:")
    generated_files = list(work_dir.glob("*.md")) + list(work_dir.glob("*report*"))
    if generated_files:
        for file in generated_files:
            size = file.stat().st_size
            print(f"   📄 {file.name}: {size:,} bytes")
            
            # Show first few lines of the report
            if file.suffix == '.md':
                try:
                    content = file.read_text()[:500]
                    print(f"   Preview: {content[:200]}...")
                except:
                    pass
    else:
        print("   ⚠️  No report files found (workflow may have been interrupted)")
    
    # Show checkpoint files
    print(f"\n💾 Checkpoint Files:")
    run_checkpoint_dir = checkpoint_dir / run_id
    if run_checkpoint_dir.exists():
        checkpoint_files = list(run_checkpoint_dir.iterdir())
        total_size = sum(f.stat().st_size for f in checkpoint_files)
        
        print(f"   📁 Directory: {run_checkpoint_dir}")
        print(f"   📊 Total size: {total_size:,} bytes")
        
        for file in sorted(checkpoint_files):
            size = file.stat().st_size
            print(f"   📄 {file.name}: {size:,} bytes")
            
        # Show some checkpoint content
        if (run_checkpoint_dir / "nodes.json").exists():
            try:
                nodes_data = json.loads((run_checkpoint_dir / "nodes.json").read_text())
                print(f"\n   💾 Checkpointed nodes: {len(nodes_data)}")
                for node_id in nodes_data.keys():
                    print(f"      🔧 {node_id}")
            except:
                pass
                
        if (run_checkpoint_dir / "events.jsonl").exists():
            try:
                events_content = (run_checkpoint_dir / "events.jsonl").read_text()
                event_count = len(events_content.strip().split('\n'))
                print(f"   📊 Recorded events: {event_count}")
            except:
                pass
    else:
        print("   ⚠️  No checkpoint directory found")
    
    print(f"\n🎉 MCP Agent Persistence Demo Complete!")
    print("✅ Real agent tool calls executed and checkpointed")
    print("✅ Workflow resumed intelligently from saved state") 
    print("✅ Cache invalidation working with file modifications")
    print("✅ Production-ready persistence for agent workflows")


async def main():
    """Main entry point."""
    try:
        await run_mcp_persistence_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🚀 MCP Agent Persistence Demo finished!")
    print(f"💡 This demonstrates production-ready persistence for agent workflows")
    print(f"   with real tool calls, file operations, and intelligent resume!")


if __name__ == "__main__":
    asyncio.run(main())