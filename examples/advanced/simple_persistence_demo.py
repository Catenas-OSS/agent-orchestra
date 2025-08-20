#!/usr/bin/env python3
"""
Simple MCP Persistence Demo

Demonstrates Agent Orchestra's durable checkpoints and resume functionality
with agent tool calls. This version uses mock agents for simplicity.
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent_orchestra.orchestrator import (
    Orchestrator, 
    JsonlRunStore,
    GraphSpec,
    NodeSpec,
    RunSpec
)

class MockMCPAgent:
    """Mock MCP agent that simulates agent tool calls"""
    
    def __init__(self):
        self.tool_calls_made = 0
    
    async def execute(self, node: NodeSpec, ctx: dict) -> dict:
        """Simulate agent execution with tool calls"""
        print(f"🤖 Mock agent executing: {node.id}")
        instruction = node.inputs.get("instruction", node.id)
        
        # Simulate tool calls based on node type
        if "file" in node.id.lower() or "extract" in node.id.lower():
            print(f"   📁 Tool call: list_files(/data/)")
            print(f"   📁 Tool call: read_file({node.id}.csv)")
            self.tool_calls_made += 2
            
        elif "analyze" in node.id.lower():
            print(f"   📊 Tool call: calculate_metrics(data)")
            print(f"   📊 Tool call: create_charts(metrics)")
            self.tool_calls_made += 2
            
        elif "report" in node.id.lower():
            print(f"   📝 Tool call: generate_report(analysis)")
            print(f"   📝 Tool call: save_file(report.pdf)")
            self.tool_calls_made += 2
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        return {
            "result": f"Processed {node.id}: {instruction}",
            "tool_calls": self.tool_calls_made,
            "status": "success"
        }

def create_business_workflow():
    """Create a realistic business intelligence workflow"""
    return GraphSpec(
        nodes=[
            NodeSpec(
                id="extract_customer_data", 
                type="task",
                inputs={"instruction": "Extract customer data from database and files"}
            ),
            NodeSpec(
                id="extract_sales_data", 
                type="task",
                inputs={"instruction": "Extract sales data and revenue metrics"}
            ),
            NodeSpec(
                id="analyze_trends", 
                type="task",
                inputs={"instruction": "Analyze trends from customer and sales data"}
            ),
            NodeSpec(
                id="generate_report", 
                type="task",
                inputs={"instruction": "Generate business intelligence report with insights"}
            ),
        ],
        edges=[
            ("extract_customer_data", "analyze_trends"),
            ("extract_sales_data", "analyze_trends"),
            ("analyze_trends", "generate_report"),
        ]
    )

async def run_demo():
    """Run the persistence demo"""
    print("🚀 Agent Orchestra - Simple Persistence Demo")
    print("=" * 50)
    
    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        store = JsonlRunStore(str(checkpoint_dir))
        agent = MockMCPAgent()
        
        # Create orchestrator with persistence
        orchestrator = Orchestrator(agent, store=store)
        
        # Create workflow
        workflow = create_business_workflow()
        run_spec = RunSpec(
            run_id="simple_demo_001",
            goal="Generate quarterly business intelligence report"
        )
        
        print("\n🎬 PHASE 1: Initial Execution")
        print("-" * 30)
        
        # Run workflow
        events1 = []
        async for event in orchestrator.run_streaming(workflow, run_spec):
            events1.append(event)
            if event.type == "NODE_COMPLETE":
                print(f"✅ {event.node_id} completed")
        
        print(f"✅ Phase 1 complete: {len(events1)} events")
        print(f"🔧 Tool calls made: {agent.tool_calls_made}")
        
        # Simulate system restart
        print("\n⏳ Simulating system restart...")
        agent2 = MockMCPAgent()
        orchestrator2 = Orchestrator(agent2, store=store)
        
        print("\n🎬 PHASE 2: Resume from Checkpoint")
        print("-" * 30)
        
        # Resume workflow
        events2 = []
        async for event in orchestrator2.run_streaming(workflow, run_spec, resume=True):
            events2.append(event)
            if event.type == "NODE_COMPLETE":
                print(f"⚡ {event.node_id} resumed from cache")
        
        print(f"✅ Phase 2 complete: {len(events2)} events")
        print(f"🔧 Tool calls made: {agent2.tool_calls_made} (should be 0 - all cached)")
        
        # Test cache invalidation
        print("\n🎬 PHASE 3: Cache Invalidation")
        print("-" * 30)
        
        # Modify goal to invalidate cache
        run_spec_modified = RunSpec(
            run_id="simple_demo_001",
            goal="Generate monthly business intelligence report"  # Changed goal
        )
        
        agent3 = MockMCPAgent()
        orchestrator3 = Orchestrator(agent3, store=store)
        
        events3 = []
        async for event in orchestrator3.run_streaming(workflow, run_spec_modified, resume=True):
            events3.append(event)
            if event.type == "NODE_COMPLETE":
                print(f"🔄 {event.node_id} re-executed (cache invalidated)")
        
        print(f"✅ Phase 3 complete: {len(events3)} events")
        print(f"🔧 Tool calls made: {agent3.tool_calls_made} (should re-execute affected nodes)")
        
        print("\n🎉 Demo Complete!")
        print("✅ Demonstrated agent persistence across restarts")
        print("✅ Showed cache invalidation with modified inputs")
        print("✅ Ready for real MCP agents with tool calls!")

if __name__ == "__main__":
    asyncio.run(run_demo())