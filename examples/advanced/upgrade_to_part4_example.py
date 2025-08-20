#!/usr/bin/env python3
"""
Upgrade to Part 4 Example - Migration Guide

This example shows how to upgrade existing Agent Orchestra code
from Parts 1-3 to use Part 4 features (CallBroker + AgentPool).

The upgrade is completely optional and backward compatible!
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor

# Part 4 imports (new, optional)
from agent_orchestra.orchestrator.broker_config import create_development_broker
from agent_orchestra.orchestrator.agent_pool import AgentPool


async def basic_orchestrator_example():
    """Example of basic orchestrator usage (Parts 1-3 style)."""
    print("📋 BEFORE: Basic Orchestrator (Parts 1-3)")
    print("-" * 40)
    
    # Simple callable executor (no agents, no rate limiting)
    async def simple_task(inputs):
        task_name = inputs.get("task", "unknown")
        return f"Completed {task_name}"
    
    # Create orchestrator the old way
    executor = CallableExecutor(simple_task)
    orchestrator = Orchestrator(executor)
    
    # Define simple workflow
    workflow = GraphSpec(
        nodes=[
            NodeSpec(id="task1", type="task", inputs={"task": "analysis"}),
            NodeSpec(id="task2", type="task", inputs={"task": "report", "from": "task1"})
        ],
        edges=[("task1", "task2")]
    )
    
    run_spec = RunSpec(run_id="basic_run", goal="Simple workflow")
    
    # Execute
    events = []
    async for event in orchestrator.run(workflow, run_spec):
        events.append(event)
        if event.type == "NODE_COMPLETE":
            print(f"✅ {event.node_id} completed")
        elif event.type == "RUN_COMPLETE":
            print(f"🎉 Workflow completed")
    
    print(f"📊 Executed {len([e for e in events if e.type == 'NODE_COMPLETE'])} tasks\n")
    return orchestrator


async def mcp_agent_example():
    """Example with MCP agents but without Part 4 features."""
    print("🤖 BEFORE: MCP Agent Orchestrator (No Part 4)")
    print("-" * 40)
    
    # Simulated MCP agent (would be real SidecarMCPAgent in practice)
    class MockMCPAgent:
        def __init__(self, name="mock"):
            self.name = name
            self.call_count = 0
        
        async def run(self, prompt):
            self.call_count += 1
            return f"Agent {self.name} result #{self.call_count}: {prompt[:30]}..."
        
        async def astream(self, prompt):
            self.call_count += 1
            yield {"chunk": 1, "content": f"Agent {self.name} processing..."}
            yield {"output": f"Agent {self.name} stream result #{self.call_count}: {prompt[:30]}..."}
    
    # Import the real MCPExecutor
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    
    agent = MockMCPAgent("original")
    executor = MCPExecutor(agent)  # No broker, no agent pool
    orchestrator = Orchestrator(executor)
    
    # Define workflow with foreach (uses regular task execution)
    workflow = GraphSpec(
        nodes=[
            NodeSpec(
                id="analyze_items",
                type="foreach",
                inputs={
                    "items": ["data1", "data2", "data3"],
                    "instruction": "Analyze each data item"
                },
                concurrency=2
            ),
            NodeSpec(
                id="summarize",
                type="reduce",
                inputs={
                    "from_ids": ["analyze_items"],
                    "instruction": "Create summary from all analyses"
                }
            )
        ],
        edges=[("analyze_items", "summarize")]
    )
    
    run_spec = RunSpec(run_id="mcp_run", goal="MCP agent workflow")
    
    # Execute with streaming
    events = []
    async for event in orchestrator.run_streaming(workflow, run_spec):
        events.append(event)
        if event.type == "NODE_START":
            print(f"🔄 Starting {event.node_id}")
        elif event.type == "AGENT_CHUNK":
            if 'content' in event.data:
                print(f"   🧠 {event.data['content']}")
        elif event.type == "NODE_COMPLETE":
            print(f"✅ {event.node_id} completed")
        elif event.type == "RUN_COMPLETE":
            print(f"🎉 MCP workflow completed")
    
    print(f"📊 Agent made {agent.call_count} calls")
    print(f"📦 Received {len([e for e in events if e.type == 'AGENT_CHUNK'])} chunks\n")
    return agent, executor


async def upgraded_part4_example():
    """Example showing how to upgrade to use Part 4 features."""
    print("🚀 AFTER: Upgraded with Part 4 Features")
    print("-" * 40)
    
    # Same mock agent as before
    class MockMCPAgent:
        def __init__(self, name="mock"):
            self.name = name
            self.call_count = 0
            # Add attributes that CallBroker can detect
            self.llm = type('LLM', (), {'model_name': 'gpt-4o-mini'})()
        
        async def run(self, prompt):
            self.call_count += 1
            # Simulate potential 429 error for demonstration
            if self.call_count == 2:
                print("   ⚠️  Simulated 429 error (CallBroker will retry)")
                raise Exception("429 Too Many Requests")
            return f"Agent {self.name} (via broker) result #{self.call_count}: {prompt[:30]}..."
        
        async def astream(self, prompt):
            self.call_count += 1
            yield {"chunk": 1, "content": f"Agent {self.name} (pooled) processing..."}
            yield {"output": f"Agent {self.name} (pooled) stream result #{self.call_count}: {prompt[:30]}..."}
    
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    
    # NEW: Create Part 4 components
    print("🏭 Creating CallBroker with development-friendly limits...")
    broker = create_development_broker()
    
    print("🏊 Creating AgentPool for agent reuse...")  
    agent_pool = AgentPool(max_agents_per_run=2)
    
    # NEW: Create executor with Part 4 features
    agent = MockMCPAgent("upgraded")
    executor = MCPExecutor(
        agent,
        broker=broker,      # NEW: Rate limiting and 429 retries
        agent_pool=agent_pool  # NEW: Agent reuse across foreach
    )
    orchestrator = Orchestrator(executor)
    
    print("✅ Orchestrator now has production-grade features!\n")
    
    # Same workflow as before - no changes needed!
    workflow = GraphSpec(
        nodes=[
            NodeSpec(
                id="analyze_items_v2",
                type="foreach",
                inputs={
                    "items": ["data1", "data2", "data3", "data4"],  # More items
                    "instruction": "Analyze each data item with rate limiting"
                },
                concurrency=3  # Higher concurrency (managed by broker)
            ),
            NodeSpec(
                id="summarize_v2", 
                type="reduce",
                inputs={
                    "from_ids": ["analyze_items_v2"],
                    "instruction": "Create summary with agent pooling"
                }
            )
        ],
        edges=[("analyze_items_v2", "summarize_v2")]
    )
    
    run_spec = RunSpec(run_id="upgraded_run", goal="Part 4 enhanced workflow")
    
    # Execute - same API as before!
    events = []
    async for event in orchestrator.run_streaming(workflow, run_spec):
        events.append(event)
        if event.type == "NODE_START":
            print(f"🔄 Starting {event.node_id} (with Part 4 features)")
        elif event.type == "AGENT_CHUNK":
            if 'content' in event.data:
                print(f"   🧠 {event.data['content']}")
        elif event.type == "NODE_COMPLETE":
            print(f"✅ {event.node_id} completed (rate limited & pooled)")
        elif event.type == "RUN_COMPLETE":
            print(f"🎉 Part 4 enhanced workflow completed")
        elif event.type == "ERROR":
            print(f"❌ Error: {event.data.get('error', 'Unknown')}")
    
    # NEW: Show Part 4 statistics
    print(f"\n📈 Part 4 Statistics:")
    
    broker_stats = await broker.get_stats()
    for model, stats in broker_stats.items():
        if stats['rpm_used'] > 0:
            print(f"   🤖 {model}: {stats['rpm_used']}/{stats['rpm_limit']} RPM used")
    
    pool_stats = await agent_pool.get_pool_stats() 
    print(f"   🏊 Agent pool: {pool_stats['total_agents']} agents, {pool_stats['active_runs']} runs")
    for run_id, run_stats in pool_stats.get('runs', {}).items():
        print(f"      Run {run_id}: {run_stats['total_usage']} total agent uses")
    
    print(f"📊 Total agent calls: {agent.call_count}")
    print(f"📦 Chunks received: {len([e for e in events if e.type == 'AGENT_CHUNK'])}")
    
    # Clean up Part 4 components
    await broker.shutdown()
    await agent_pool.shutdown()
    
    return agent, executor


async def migration_tips():
    """Show migration tips and best practices."""
    print("\n💡 MIGRATION TIPS: Upgrading to Part 4")
    print("=" * 50)
    
    print("""
🔄 ZERO-BREAKING CHANGES:
   ✅ All existing code works unchanged
   ✅ Part 4 features are completely optional
   ✅ Import statements remain the same
   ✅ API signatures are backward compatible

🚀 EASY UPGRADE PATH:
   1. Keep existing code working as-is
   2. Add CallBroker for rate limiting: 
      broker = create_development_broker()
      executor = MCPExecutor(agent, broker=broker)
   
   3. Add AgentPool for efficiency:
      pool = AgentPool()
      executor = MCPExecutor(agent, broker=broker, agent_pool=pool)
   
   4. No workflow changes needed!

⚡ BENEFITS OF UPGRADING:
   ✅ Production-grade rate limiting (RPM, RPD, concurrency)
   ✅ 429 error handling with exponential backoff
   ✅ Agent reuse across foreach operations (performance)
   ✅ Easy broker configuration (dev/staging/prod)
   ✅ Streaming chunk preservation
   ✅ Built-in statistics and monitoring

🎛️ CONFIGURATION OPTIONS:
   • Development: create_development_broker()
   • Production: create_production_broker('openai')
   • Custom: create_broker_from_config(custom_config)
   • File-based: BrokerConfig.from_file('broker.json')
   • Environment: BrokerConfig.from_env()

📈 WHEN TO UPGRADE:
   • You're hitting rate limits (429 errors)
   • You need production reliability
   • You want better performance with foreach
   • You're deploying to production
   • You want monitoring/statistics
   
🔧 WHEN TO STAY ON PARTS 1-3:
   • Simple prototyping
   • No rate limiting concerns
   • Single-agent workflows
   • Educational/learning purposes
""")


async def main():
    """Main demonstration of upgrade path."""
    print("🎯 Agent Orchestra: Parts 1-3 → Part 4 Upgrade Guide")
    print("=" * 60)
    print("Demonstrates backward compatibility and easy migration path\n")
    
    try:
        # Show original approach
        await basic_orchestrator_example()
        
        # Show MCP agent approach without Part 4
        old_agent, old_executor = await mcp_agent_example()
        
        # Show upgraded approach with Part 4
        new_agent, new_executor = await upgraded_part4_example()
        
        # Show migration tips
        await migration_tips()
        
        print("\n🌟 SUMMARY:")
        print("✅ Parts 1-3 code works unchanged")
        print("✅ Part 4 adds optional production features")
        print("✅ Migration is gradual and non-breaking")
        print("✅ You can upgrade incrementally")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🚀 Agent Orchestra upgrade demo complete!")
    print(f"💡 Part 4 provides production-ready orchestration with")
    print(f"   rate limiting, agent pooling, and reliability features!")


if __name__ == "__main__":
    asyncio.run(main())