#!/usr/bin/env python3
"""
Test script for the new supervisor agent functionality with website building
"""

import asyncio
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    from agent_orchestra.orchestrator.call_broker import CallBroker
    from agent_orchestra.orchestrator.agent_pool import AgentPool
    from website_builder_workflow import create_website_workflow, simple_run
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e . && pip install langchain-openai python-dotenv")
    sys.exit(1)

async def test_supervisor_workflow():
    """Test the supervisor agent with website building workflow."""
    print("🚀 Testing Smart Agent Orchestration for Website Building")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("💡 Set it with: export OPENAI_API_KEY=your-key-here")
        return
    
    try:
        # Create configuration for MCP servers (using filesystem tools for website building)
        config: Dict[str, Any] = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                },
                "brave-search": {
                    "command": "npx", 
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"]
                }
            }
        }
        
        # Create SidecarMCPClient from configuration
        client = SidecarMCPClient.from_dict(config)
        
        # Create LLM with OpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Create agent with the client
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=10)
        
        # Create broker for rate limiting
        from agent_orchestra.orchestrator.call_broker import ModelLimits
        model_limits = {
            "openai:gpt-4o": ModelLimits(rpm=100, rpd=2000, max_concurrency=5)
        }
        broker = CallBroker(model_limits=model_limits)
        
        # Create agent factory for the pool
        async def agent_factory(spec):
            # For now, just return the same agent (in a real scenario, you'd create different agents based on spec)
            return agent
            
        # Create agent pool with factory
        agent_pool = AgentPool(factory=agent_factory)
        
        # Create executor with agent pool support
        executor = MCPExecutor(
            agent=agent,
            default_server="filesystem", 
            broker=broker,
            agent_pool=agent_pool,
            model_key="gpt-4o"
        )
        
        # Create orchestrator
        orchestrator = Orchestrator(executor=executor)
        
        # Create the workflow
        workflow = create_website_workflow()
        
        # Create a unique run with timestamp
        from agent_orchestra.orchestrator.types import RunSpec
        unique_run = RunSpec(
            run_id=f"website_build_{uuid.uuid4().hex[:8]}",
            goal="Build a modern portfolio website using smart agent orchestration",
            policy={"allow_agent_selection": True}
        )
        
        print(f"📋 Workflow created with {len(workflow.nodes)} nodes:")
        for node in workflow.nodes:
            if node.type == "supervisor":
                agent_count = len(node.available_agents or {})
                print(f"  🧠 {node.id} (supervisor) - {agent_count} available agents")
            else:
                print(f"  📝 {node.id} ({node.type})")
        
        print(f"\n🔗 Workflow edges: {workflow.edges}")
        print("\n▶️ Starting workflow execution...")
        print("-" * 60)
        
        # Run the workflow with streaming
        event_count = 0
        async for event in orchestrator.run_streaming(workflow, unique_run):
            event_count += 1
            
            if event.type == "RUN_START":
                print(f"🏁 Run started: {event.data.get('goal', 'No goal specified')}")
                
            elif event.type == "NODE_START": 
                node_id = event.node_id
                print(f"\n🔄 Starting node: {node_id}")
                if node_id == "website_supervisor":
                    print("  🧠 Supervisor agent analyzing requirements and selecting specialists...")
                
            elif event.type == "AGENT_CHUNK":
                # Show streaming output from agents
                chunk_data = event.data
                if isinstance(chunk_data, dict) and "text" in chunk_data:
                    print(f"💭 {chunk_data['text'][:100]}{'...' if len(chunk_data['text']) > 100 else ''}")
                elif isinstance(chunk_data, str):
                    print(f"💭 {chunk_data[:100]}{'...' if len(chunk_data) > 100 else ''}")
                    
            elif event.type == "NODE_COMPLETE":
                node_id = event.node_id
                phase = event.data.get("phase", "complete")
                print(f"\n✅ Node completed: {node_id} ({phase})")
                
                if node_id == "website_supervisor":
                    agents_called = event.data.get("agents_called", 0)
                    print(f"  🤖 Supervisor called {agents_called} specialized agents")
                    
            elif event.type == "RUN_COMPLETE":
                print(f"\n🎉 Workflow completed successfully!")
                print(f"📊 Total events processed: {event_count}")
                
            elif event.type == "ERROR":
                error_msg = event.data.get("error", "Unknown error")
                print(f"\n❌ Error in {event.node_id}: {error_msg}")
        
        print("\n" + "=" * 60)
        print("✨ Smart Agent Orchestration Test Complete!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_supervisor_parsing():
    """Test the supervisor's agent call parsing logic."""
    print("\n🧪 Testing Supervisor Agent Call Parsing")
    print("-" * 40)
    
    # Import the orchestrator to test parsing
    from agent_orchestra.orchestrator.core import Orchestrator
    
    orchestrator = Orchestrator(executor=None)  # Just for testing parsing
    
    # Test supervisor output
    test_output = """
I need to create a portfolio website for a graphic designer. Based on the requirements, I'll call the following specialized agents:

CALL_AGENT: ui_designer
TASK: Create a modern, visually appealing design for a graphic designer's portfolio website
INPUT: Target audience: potential clients, Style: clean and professional, Focus: showcase creative work
---

CALL_AGENT: content_writer  
TASK: Write engaging copy for the portfolio website
INPUT: Brand voice: creative but professional, Sections: about, portfolio, services, contact
---

CALL_AGENT: frontend_developer
TASK: Implement the portfolio website with responsive design
INPUT: Design specifications from UI designer, Content from writer, Mobile-first approach
---

FINAL_RESULT: Created a complete portfolio website with modern design, engaging content, and responsive implementation. The site effectively showcases the designer's work with professional presentation and optimal user experience.
"""
    
    # Test parsing
    calls = orchestrator._parse_agent_calls(test_output)
    
    print(f"📋 Parsed {len(calls)} agent calls:")
    for i, call in enumerate(calls, 1):
        print(f"{i}. Agent: {call.get('agent_id', 'Unknown')}")
        print(f"   Task: {call.get('task', 'No task')[:60]}...")
        print(f"   Input: {call.get('input', 'No input')[:60]}...")
        print()
    
    # Test final result extraction  
    lines = test_output.split("\n")
    final_result = ""
    capture = False
    for line in lines:
        if line.strip().startswith("FINAL_RESULT:"):
            final_result = line.replace("FINAL_RESULT:", "").strip()
            capture = True
        elif capture and line.strip():
            final_result += "\n" + line.strip()
    
    print(f"🎯 Final Result: {final_result[:100]}...")
    print("\n✅ Parsing test completed!")

if __name__ == "__main__":
    print("🌐 Smart Agent Orchestration - Website Builder Test")
    print("This demonstrates the new supervisor agent functionality")
    print()
    
    # Run parsing test first
    asyncio.run(test_supervisor_parsing())
    
    # Then run full workflow test
    print("\n" + "=" * 60)
    asyncio.run(test_supervisor_workflow())