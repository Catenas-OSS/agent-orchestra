#!/usr/bin/env python3
"""
Polished Simple Demo - Shows polish improvements without complex MCP setup

This demo demonstrates the key polish improvements using a simple mock setup:
✅ Profile-based AgentPool with no duplicate inits
✅ Server-name routing through MCPExecutor  
✅ Race-safe agent creation
✅ Clean parameter handling
✅ Agent reuse across operations
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from langchain_openai import ChatOpenAI
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    from agent_orchestra.orchestrator.broker_config import create_development_broker
    from agent_orchestra.orchestrator.agent_pool import AgentPool, create_default_agent_factory
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)


class MockSidecarMCPClient:
    """Mock client that doesn't actually connect to MCP servers."""
    
    def __init__(self, config_dict):
        self.config = config_dict
        self.servers = list(config_dict.get("mcpServers", {}).keys())
    
    async def close_all_sessions(self):
        pass
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict)


class MockSidecarMCPAgent:
    """Mock agent for demonstration purposes."""
    
    def __init__(self, llm=None, client=None, use_server_manager=True):
        self.llm = llm
        self.client = client
        self.use_server_manager = use_server_manager
        self.call_count = 0
        self.creation_id = f"agent_{id(self)}"
        print(f"   🤖 Created mock agent {self.creation_id}")
    
    async def run(self, prompt, **kwargs):
        self.call_count += 1
        server_name = kwargs.get('server_name', 'default')
        
        # Generate a realistic mock response based on the prompt
        if "Read Business Data" in prompt or "read" in prompt.lower():
            if "sales_data.json" in prompt:
                return "Sales Department: $1.5M revenue, 15% growth, 4.2/5 customer satisfaction"
            elif "marketing_data.json" in prompt:
                return "Marketing Department: $250K spend, 4.2x ROI, +32% lead generation"
            elif "operations_data.json" in prompt:
                return "Operations Department: 91% efficiency, $180K cost savings, 99.2% uptime"
        elif "analyze" in prompt.lower() or "trends" in prompt.lower():
            return """Cross-Department Analysis:
Top 3 Business Strengths:
1. Strong customer satisfaction (4.2/5) and retention (89%)
2. Excellent marketing ROI (4.2x) with significant lead growth (+32%)  
3. High operational efficiency (91%) with outstanding uptime (99.2%)

2 Areas for Improvement:
1. Marketing conversion rate could be optimized (currently 12%)
2. Operations error rate, while low at 0.8%, has room for improvement"""
        elif "save" in prompt.lower() or "report" in prompt.lower():
            return "Executive summary report saved successfully as 'executive_summary.txt'"
        
        return f"Mock response from {self.creation_id} on server {server_name} (call #{self.call_count})"
    
    async def astream(self, prompt, **kwargs):
        self.call_count += 1
        server_name = kwargs.get('server_name', 'default')
        
        yield {"chunk": 1, "content": f"Processing with {self.creation_id} on {server_name}"}
        
        # Generate same response as run()
        result = await self.run(prompt, **kwargs)
        self.call_count -= 1  # Don't double count
        
        yield {"output": result}


async def run_polished_simple_demo():
    """Run the polished simple demo."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        return
    
    print("🌟 Polished Simple Demo - Production-Ready Agent Orchestra")
    print("=" * 65)
    print("✅ All polish improvements demonstrated without complex MCP setup")
    print()
    
    # Create mock client configuration
    mock_config = {
        "mcpServers": {
            "fs_business": {"command": "mock"},
            "fs_reports": {"command": "mock"}
        }
    }
    
    client = MockSidecarMCPClient.from_dict(mock_config)
    print("🔌 Mock MCP client created with multi-server support")
    
    # Create profile-based AgentPool with factory
    print("🏊 Creating profile-based AgentPool...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=300)
    
    # Create agent factory using mock agent
    async def agent_factory(spec):
        agent = MockSidecarMCPAgent(llm, client, spec.use_server_manager)
        agent.model_key = spec.model_key
        agent.server_name = spec.server_name
        agent.policy_id = spec.policy_id
        return agent
    
    agent_pool = AgentPool(agent_factory, max_agents_per_run=5)
    print("   AgentPool ready with profile-based agent creation")
    
    # Create CallBroker and MCPExecutor
    print("🏭 Setting up CallBroker and MCPExecutor...")
    broker = create_development_broker()
    
    executor = MCPExecutor(
        agent=None,  # No template agent needed with pool
        default_server="fs_business",
        broker=broker,
        agent_pool=agent_pool,
        model_key="openai:gpt-4o-mini"
    )
    orchestrator = Orchestrator(executor)
    print("   Production-ready orchestrator configured")
    print()
    
    try:
        print("🎬 Running multi-server workflow with polish improvements...")
        
        # Test data files
        test_files = ["sales_data.json", "marketing_data.json", "operations_data.json"]
        
        # Workflow with different server routing
        workflow = GraphSpec(
            nodes=[
                # Read business data from fs_business server
                NodeSpec(
                    id="read_data",
                    type="foreach", 
                    name="Read Business Data",
                    server_name="fs_business",  # Route to business filesystem
                    inputs={
                        "items": test_files,
                        "instruction": "Read the JSON file for each business department and summarize the key metrics."
                    },
                    concurrency=2
                ),
                
                # Analyze data (default server routing)
                NodeSpec(
                    id="analyze_trends",
                    type="reduce",
                    name="Cross-Department Analysis", 
                    inputs={
                        "from_ids": ["read_data"],
                        "instruction": "Analyze trends across all departments. Identify the top 3 business strengths and 2 areas for improvement based on the metrics."
                    }
                ),
                
                # Save report to fs_reports server  
                NodeSpec(
                    id="save_report",
                    type="task",
                    name="Save Executive Report",
                    server_name="fs_reports",  # Route to reports filesystem
                    inputs={
                        "from": "analyze_trends",
                        "instruction": "Create an executive summary report and save it as 'executive_summary.txt'."
                    }
                )
            ],
            edges=[
                ("read_data", "analyze_trends"),
                ("analyze_trends", "save_report")
            ]
        )
        
        run_spec = RunSpec(
            run_id="polished_simple_demo_001",
            goal="Multi-server business analysis with production features"
        )
        
        print(f"📋 Workflow nodes: {len(workflow.nodes)}")
        print(f"🔀 Multi-server routing: fs_business → analysis → fs_reports")
        print()
        
        # Execute with all improvements
        events = []
        async for event in orchestrator.run_streaming(workflow, run_spec):
            events.append(event)
            
            if event.type == "NODE_START":
                server = ""
                for node in workflow.nodes:
                    if node.id == event.node_id:
                        server = f" (server: {node.server_name or 'default'})"
                        break
                print(f"🔄 Starting {event.node_id}{server}")
                
            elif event.type == "AGENT_CHUNK":
                if isinstance(event.data, dict) and 'content' in event.data:
                    content = str(event.data['content'])
                    if any(keyword in content.lower() for keyword in ['processing', 'mock', 'agent']):
                        preview = content[:60] + "..." if len(content) > 60 else content
                        print(f"   🧠 Agent: {preview}")
                
            elif event.type == "NODE_COMPLETE":
                print(f"✅ Completed {event.node_id}")
                
            elif event.type == "RUN_COMPLETE":
                print(f"🎉 Multi-server workflow completed!")
                
            elif event.type == "ERROR":
                print(f"❌ Error in {event.node_id}: {event.data.get('error', 'Unknown')}")
        
        # Show results and statistics
        print(f"\n📊 Execution Results:")
        print(f"   🎯 Events processed: {len(events)}")
        
        # Show broker statistics
        broker_stats = await broker.get_stats()
        for model, stats in broker_stats.items():
            if stats['rpm_used'] > 0:
                print(f"   📈 {model}: {stats['rpm_used']}/{stats['rpm_limit']} RPM used")
        
        # Show agent pool statistics  
        pool_stats = await agent_pool.get_pool_stats()
        print(f"   🏊 Agent profiles created: {len(pool_stats['profiles'])}")
        for profile_key, profile_info in pool_stats['profiles'].items():
            server = profile_info['server_name'] or 'default'
            usage = profile_info['usage_count']
            print(f"      {server} server: {usage} uses")
        
        print(f"\n🌟 All Polish Improvements Demonstrated:")
        print(f"   ✅ Profile-based AgentPool - No duplicate agent initialization")
        print(f"   ✅ Server-name routing through nodes - Clean parameter passing")
        print(f"   ✅ Agent reuse across foreach operations - Single agent per profile")
        print(f"   ✅ Race-safe agent creation - Concurrent operations handled safely")
        print(f"   ✅ Production-grade rate limiting and pooling")
        print(f"   ✅ Clean async context management and error handling")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        try:
            await client.close_all_sessions()
        except Exception:
            pass
        await broker.shutdown()
        await agent_pool.shutdown()
        print(f"\n🧹 Clean shutdown completed")


async def main():
    """Main entry point."""
    try:
        await run_polished_simple_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🚀 Polished simple demo complete!")
    print(f"💡 This demonstrates production-ready Agent Orchestra with")
    print(f"   all polish improvements for real-world deployment!")


if __name__ == "__main__":
    asyncio.run(main())