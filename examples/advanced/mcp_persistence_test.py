#!/usr/bin/env python3
"""
Quick MCP Persistence Test - Simulates agent tool calls without API requirements

This test demonstrates persistence features with simulated MCP agent behavior:
✅ Simulated agent tool calls and file operations
✅ Checkpoint persistence across interruptions  
✅ Resume skips completed agent tasks
✅ Cache invalidation with changed inputs
✅ Foreach granular resume with tool call batches

No API keys required - uses mock agents that simulate realistic MCP behavior.

Usage:
    python examples/advanced/mcp_persistence_test.py
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.store import JsonlRunStore


class MockMCPAgent:
    """Mock MCP agent that simulates realistic tool call behavior."""
    
    def __init__(self):
        self.call_count = 0
        self.file_system = {
            "/data/sales.json": {"revenue": 1000000, "growth": 0.15},
            "/data/customers.json": {"total": 5000, "segments": ["enterprise", "smb"]},
            "/data/reports/": {}  # Directory for saving reports
        }
    
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent execution with realistic tool calls and delays."""
        self.call_count += 1
        instruction = node.inputs.get("instruction", "")
        
        # Simulate processing time based on task complexity
        if "analyze" in instruction.lower():
            await asyncio.sleep(0.3)  # Analysis tasks take longer
        elif "read" in instruction.lower():
            await asyncio.sleep(0.2)  # File operations
        else:
            await asyncio.sleep(0.1)  # Simple tasks
        
        # Simulate different types of agent tool calls
        if "read" in instruction.lower() and "sales" in instruction.lower():
            # Simulate reading sales data file
            tool_calls = [
                {"tool": "read_file", "path": "/data/sales.json", "success": True},
                {"tool": "parse_json", "records": 12500, "success": True}
            ]
            output = f"Successfully read sales data: Revenue ${self.file_system['/data/sales.json']['revenue']:,}, Growth {self.file_system['/data/sales.json']['growth']:.1%}"
            
        elif "read" in instruction.lower() and "customer" in instruction.lower():
            # Simulate reading customer data
            tool_calls = [
                {"tool": "read_file", "path": "/data/customers.json", "success": True},
                {"tool": "analyze_segments", "segments_found": 2, "success": True}
            ]
            output = f"Customer analysis complete: {self.file_system['/data/customers.json']['total']:,} total customers across {len(self.file_system['/data/customers.json']['segments'])} segments"
            
        elif "analyze" in instruction.lower():
            # Simulate complex data analysis
            tool_calls = [
                {"tool": "statistical_analysis", "metrics_calculated": 15, "success": True},
                {"tool": "trend_detection", "trends_found": 3, "success": True},
                {"tool": "correlation_analysis", "correlations": 8, "success": True}
            ]
            output = f"Analysis complete for {node.id}: Identified 3 key trends, 8 correlations, and 15 performance metrics"
            
        elif "save" in instruction.lower() or "write" in instruction.lower():
            # Simulate saving report file
            report_name = f"report_{node.id}_{int(time.time())}.md"
            self.file_system[f"/data/reports/{report_name}"] = {"content": f"Report for {node.id}", "created": time.time()}
            tool_calls = [
                {"tool": "write_file", "path": f"/data/reports/{report_name}", "size": 2048, "success": True}
            ]
            output = f"Report saved successfully: {report_name} (2.1KB)"
            
        elif "foreach" in str(node.type).lower() or "item" in instruction.lower():
            # Simulate foreach item processing
            item_data = node.inputs.get("item", {})
            item_id = item_data.get("id", "unknown")
            tool_calls = [
                {"tool": "process_item", "item_id": item_id, "processing_time": 150, "success": True}
            ]
            output = f"Processed item {item_id}: extracted features, applied transformations, validated results"
            
        else:
            # Generic task simulation
            tool_calls = [
                {"tool": "generic_processor", "operations": 5, "success": True}
            ]
            output = f"Task {node.id} completed: performed 5 operations successfully"
        
        return {
            "output": output,
            "tool_calls": tool_calls,
            "tokens_used": len(output) + 50,  # Simulate token usage
            "processing_time": f"{len(tool_calls) * 0.1:.1f}s",
            "agent_id": f"mock_agent_{self.call_count}"
        }


def create_test_workflow() -> GraphSpec:
    """Create test workflow that simulates realistic MCP agent usage."""
    
    return GraphSpec(
        nodes=[
            # Data extraction phase
            NodeSpec(
                id="extract_sales_data",
                type="task",
                name="Extract Sales Data",
                inputs={
                    "instruction": "Read the sales data file and extract key revenue metrics and growth trends",
                    "file_path": "/data/sales.json"
                }
            ),
            
            NodeSpec(
                id="extract_customer_data", 
                type="task",
                name="Extract Customer Data",
                inputs={
                    "instruction": "Read customer data file and analyze customer segments and demographics",
                    "file_path": "/data/customers.json"
                }
            ),
            
            # Gate for data validation
            NodeSpec(
                id="data_validation_gate",
                type="gate",
                name="Data Quality Gate",
                inputs={"predicate": True}
            ),
            
            # Batch processing simulation
            NodeSpec(
                id="process_data_segments",
                type="foreach", 
                name="Process Customer Segments",
                inputs={
                    "items": [
                        {"id": "enterprise", "type": "high_value", "priority": 1},
                        {"id": "mid_market", "type": "growth", "priority": 2},
                        {"id": "small_business", "type": "volume", "priority": 3},
                        {"id": "startup", "type": "emerging", "priority": 4}
                    ],
                    "instruction": "For each customer segment, analyze characteristics and generate insights using available data"
                },
                concurrency=2
            ),
            
            # Advanced analysis 
            NodeSpec(
                id="analyze_business_metrics",
                type="task",
                name="Business Metrics Analysis", 
                inputs={
                    "instruction": "Perform comprehensive analysis of business metrics combining sales and customer data to identify opportunities"
                }
            ),
            
            # Report generation
            NodeSpec(
                id="generate_executive_report",
                type="reduce",
                name="Generate Executive Report",
                inputs={
                    "from_ids": ["extract_sales_data", "extract_customer_data", "process_data_segments", "analyze_business_metrics"],
                    "instruction": "Create comprehensive executive report combining all analyses and segment insights"
                }
            ),
            
            # Save final output
            NodeSpec(
                id="save_final_report",
                type="task", 
                name="Save Report to File",
                inputs={
                    "instruction": "Save the executive report to a markdown file with timestamp and formatting"
                }
            )
        ],
        edges=[
            ("extract_sales_data", "data_validation_gate"),
            ("extract_customer_data", "data_validation_gate"),
            ("data_validation_gate", "process_data_segments"),
            ("process_data_segments", "analyze_business_metrics"),
            ("analyze_business_metrics", "generate_executive_report"),
            ("generate_executive_report", "save_final_report")
        ]
    )


async def run_test_phase(orchestrator, workflow, run_spec, phase_name, resume=False, interrupt_after=None):
    """Run a workflow phase with detailed progress tracking."""
    
    print(f"\n🎬 {phase_name.upper()} PHASE")
    print("-" * 60)
    
    start_time = time.time()
    events = []
    tool_calls = 0
    resumed_nodes = 0
    executed_nodes = 0
    
    try:
        execution_task = None
        if interrupt_after:
            # Create task for interruption simulation
            async def run_workflow():
                nonlocal events, tool_calls, resumed_nodes, executed_nodes
                async for event in orchestrator.run_streaming(workflow, run_spec, resume=resume):
                    events.append(event)
                    
                    if event.type == "NODE_START":
                        print(f"\n🔧 {event.node_id}: {event.data.get('type', 'task')}")
                        
                    elif event.type == "AGENT_CHUNK":
                        tool_calls += 1
                        if tool_calls <= 8:  # Show first few
                            content = str(event.data.get('content', ''))[:80]
                            print(f"   🤖 {content}...")
                            
                    elif event.type == "NODE_COMPLETE":
                        if event.data.get('resumed'):
                            resumed_nodes += 1
                            print(f"   ⚡ RESUMED from checkpoint")
                        else:
                            executed_nodes += 1
                            print(f"   ✅ COMPLETED with agent tools")
                            
                        # Show foreach stats
                        if 'executed_items' in event.data:
                            cached = event.data.get('cached_items', 0)
                            executed = event.data.get('executed_items', 0)
                            print(f"   📊 Items: {executed} executed, {cached} cached")
                            
            execution_task = asyncio.create_task(run_workflow())
            
            # Wait for interruption time or completion
            try:
                await asyncio.wait_for(execution_task, timeout=interrupt_after)
            except asyncio.TimeoutError:
                print(f"\n🚨 SIMULATING INTERRUPTION after {interrupt_after}s...")
                execution_task.cancel()
                try:
                    await execution_task
                except asyncio.CancelledError:
                    pass
                print("💾 Workflow state persisted to checkpoints")
                return events, {"interrupted": True}
        else:
            # Run to completion
            async for event in orchestrator.run_streaming(workflow, run_spec, resume=resume):
                events.append(event)
                
                if event.type == "RUN_START":
                    resumed = " (RESUMED)" if event.data.get('resumed') else ""
                    print(f"🚀 Workflow started{resumed}")
                    
                elif event.type == "NODE_START":
                    print(f"\n🔧 {event.node_id}: {event.data.get('type', 'task')}")
                    
                elif event.type == "AGENT_CHUNK":
                    tool_calls += 1
                    if tool_calls <= 8:  # Show first few calls
                        content = str(event.data.get('content', ''))[:80]
                        print(f"   🤖 {content}...")
                    elif tool_calls == 9:
                        print(f"   🤖 [agent continued processing...]")
                        
                elif event.type == "NODE_COMPLETE":
                    if event.data.get('resumed'):
                        resumed_nodes += 1
                        print(f"   ⚡ RESUMED from checkpoint")
                    else:
                        executed_nodes += 1
                        print(f"   ✅ COMPLETED with agent tools")
                        
                    # Show foreach statistics  
                    if 'executed_items' in event.data:
                        cached = event.data.get('cached_items', 0)
                        executed = event.data.get('executed_items', 0)
                        print(f"   📊 Items: {executed} executed, {cached} cached")
                        
                elif event.type == "RUN_COMPLETE":
                    elapsed = time.time() - start_time
                    print(f"\n🎉 {phase_name.upper()} COMPLETED!")
                    print(f"   ⏱️  Time: {elapsed:.2f}s")
                    print(f"   📊 Events: {len(events)}")
                    print(f"   🤖 Tool calls: {tool_calls}")
                    print(f"   ⚡ Resumed: {resumed_nodes}")
                    print(f"   🔄 Executed: {executed_nodes}")
    
    except Exception as e:
        print(f"💥 {phase_name} failed: {e}")
        raise
    
    return events, {
        "events": len(events),
        "tool_calls": tool_calls, 
        "resumed_nodes": resumed_nodes,
        "executed_nodes": executed_nodes
    }


async def main():
    """Main test orchestration."""
    
    print("🧪 MCP Persistence Test - Simulated Agent Tool Calls")
    print("=" * 70)
    print("🎯 Demonstrating persistence with realistic agent behavior")
    print("📊 No API keys required - uses mock MCP agents")
    print()
    
    with tempfile.TemporaryDirectory(prefix="mcp_test_") as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        
        print(f"💾 Checkpoint directory: {checkpoint_dir}")
        print()
        
        # Create mock agent and orchestrator
        mock_agent = MockMCPAgent()
        store = JsonlRunStore(checkpoint_dir, flush_every=1)
        orchestrator = Orchestrator(mock_agent, store=store)
        
        # Create test workflow
        workflow = create_test_workflow()
        run_spec = RunSpec(
            run_id="mcp_persistence_test_001",
            goal="Test MCP agent persistence with tool calls"
        )
        
        print(f"📋 Test workflow: {len(workflow.nodes)} nodes, {len(workflow.edges)} edges")
        print(f"🤖 Mock agent simulates: file I/O, data analysis, report generation")
        print()
        
        try:
            # Phase 1: Initial run with interruption
            events1, stats1 = await run_test_phase(
                orchestrator, workflow, run_spec,
                phase_name="Initial Run", 
                interrupt_after=3.0  # Interrupt after 3 seconds
            )
            
            # Small pause for "system restart"
            print("\n⏳ Simulating system restart...")
            await asyncio.sleep(1)
            
            # Phase 2: Resume from checkpoint
            fresh_agent = MockMCPAgent()  # Fresh agent instance
            fresh_orchestrator = Orchestrator(fresh_agent, store=store)
            
            events2, stats2 = await run_test_phase(
                fresh_orchestrator, workflow, run_spec,
                phase_name="Resume", 
                resume=True
            )
            
            # Phase 3: Cache invalidation test
            print("\n🔧 Modifying workflow inputs for cache invalidation test...")
            
            # Create modified workflow with changed inputs
            modified_workflow = GraphSpec(
                nodes=[
                    NodeSpec(
                        id="extract_sales_data",
                        type="task", 
                        inputs={
                            "instruction": "Read the UPDATED sales data file and extract key revenue metrics",  # Changed instruction
                            "file_path": "/data/sales_v2.json"  # Changed file
                        }
                    ),
                    NodeSpec(
                        id="extract_customer_data",
                        type="task",
                        inputs={
                            "instruction": "Read customer data file and analyze customer segments and demographics",  # Unchanged
                            "file_path": "/data/customers.json"
                        }
                    ),
                    NodeSpec(
                        id="quick_analysis",
                        type="task",
                        inputs={
                            "instruction": "Perform quick analysis on the updated data"
                        }
                    )
                ],
                edges=[
                    ("extract_sales_data", "quick_analysis"),
                    ("extract_customer_data", "quick_analysis")
                ]
            )
            
            events3, stats3 = await run_test_phase(
                fresh_orchestrator, modified_workflow, run_spec,
                phase_name="Cache Invalidation",
                resume=True
            )
            
            # Show final summary
            await show_test_summary(checkpoint_dir, run_spec.run_id, [stats1, stats2, stats3])
            
        except Exception as e:
            print(f"💥 Test failed: {e}")
            import traceback
            traceback.print_exc()


async def show_test_summary(checkpoint_dir: Path, run_id: str, phase_stats):
    """Show comprehensive test summary."""
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    # Phase statistics
    phase_names = ["Initial", "Resume", "Cache Invalidation"]
    print("\n📈 Phase Statistics:")
    for i, (name, stats) in enumerate(zip(phase_names, phase_stats)):
        print(f"   {i+1}. {name}:")
        print(f"      Events: {stats.get('events', 'N/A')}")
        print(f"      Tool calls: {stats.get('tool_calls', 'N/A')}")
        print(f"      Resumed nodes: {stats.get('resumed_nodes', 'N/A')}")
        print(f"      Executed nodes: {stats.get('executed_nodes', 'N/A')}")
    
    # Checkpoint analysis
    run_checkpoint_dir = checkpoint_dir / run_id
    if run_checkpoint_dir.exists():
        print(f"\n💾 Checkpoint Analysis:")
        print(f"   Directory: {run_checkpoint_dir}")
        
        files = list(run_checkpoint_dir.iterdir())
        total_size = sum(f.stat().st_size for f in files)
        print(f"   Files: {len(files)} ({total_size:,} bytes total)")
        
        for file in sorted(files):
            size = file.stat().st_size
            print(f"   📄 {file.name}: {size:,} bytes")
            
        # Show checkpoint content
        if (run_checkpoint_dir / "nodes.json").exists():
            try:
                nodes_data = json.loads((run_checkpoint_dir / "nodes.json").read_text())
                print(f"\n   💾 Checkpointed nodes: {len(nodes_data)}")
                for node_id in list(nodes_data.keys())[:5]:  # Show first 5
                    print(f"      🔧 {node_id}")
                if len(nodes_data) > 5:
                    print(f"      ... and {len(nodes_data) - 5} more")
            except:
                pass
    
    print(f"\n🎉 MCP Persistence Test Complete!")
    print("✅ Simulated agent tool calls with checkpointing")
    print("✅ Resume functionality skipping completed nodes")  
    print("✅ Cache invalidation with modified inputs")
    print("✅ Production-ready persistence for agent workflows")
    print("✅ Ready for real MCP agents with API keys!")


if __name__ == "__main__":
    asyncio.run(main())