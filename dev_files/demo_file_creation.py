#!/usr/bin/env python3
"""
Demo: Supervisor Agents Creating Real Website Files using MCP Tools
This shows the supervisor orchestrating agents that create actual HTML, CSS, JS files
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
    from agent_orchestra.orchestrator.call_broker import CallBroker, ModelLimits
    from agent_orchestra.orchestrator.agent_pool import AgentPool
    from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e . && pip install langchain-openai python-dotenv")
    sys.exit(1)

async def demo_website_file_creation():
    """Demonstrate supervisor agents creating real website files."""
    print("🌐 Smart Agent Orchestration - Real File Creation Demo")
    print("=" * 65)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("💡 Set it with: export OPENAI_API_KEY=your-key-here")
        return
    
    try:
        # Create a project directory for the website
        project_dir = Path("/tmp/lovable_website_demo")
        project_dir.mkdir(exist_ok=True)
        print(f"📁 Creating website in: {project_dir}")
        
        # Create MCP configuration with filesystem access to project directory
        config: Dict[str, Any] = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(project_dir)]
                }
            }
        }
        
        # Create SidecarMCPClient and Agent
        client = SidecarMCPClient.from_dict(config)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=15)  # More steps for file creation
        
        # Create broker and agent pool
        model_limits = {
            "openai:gpt-4o": ModelLimits(rpm=100, rpd=2000, max_concurrency=5)
        }
        broker = CallBroker(model_limits=model_limits)
        
        async def agent_factory(spec):
            return agent
            
        agent_pool = AgentPool(factory=agent_factory)
        
        # Create executor 
        executor = MCPExecutor(
            agent=agent,
            default_server="filesystem", 
            broker=broker,
            agent_pool=agent_pool,
            model_key="gpt-4o"
        )
        
        # Create orchestrator
        orchestrator = Orchestrator(executor=executor)
        
        # Define website building agents with file creation focus
        file_creation_agents = {
            "ui_designer": {
                "description": "Creates design specs, wireframes, and CSS styling for beautiful websites",
                "capabilities": ["responsive design", "CSS frameworks", "design systems", "wireframes"],
                "server": "filesystem"
            },
            "frontend_developer": {
                "description": "Creates HTML, CSS, and JavaScript files for complete web interfaces",
                "capabilities": ["HTML5 structure", "CSS3 styling", "JavaScript interactivity", "responsive implementation"],
                "server": "filesystem"
            },
            "content_creator": {
                "description": "Creates website content, copy, and text files",
                "capabilities": ["copywriting", "content strategy", "SEO optimization", "content files"],
                "server": "filesystem"
            }
        }
        
        # Create workflow focused on file creation
        workflow = GraphSpec(
            nodes=[
                # Project initialization
                NodeSpec(
                    id="project_init",
                    type="task",
                    name="Initialize Website Project Structure",
                    inputs={
                        "project_type": "Modern portfolio website for a creative professional",
                        "target_directory": str(project_dir),
                        "requirements": "Clean design, mobile responsive, portfolio showcase, contact form"
                    },
                    server_name="filesystem"
                ),
                
                # File creation supervisor
                NodeSpec(
                    id="file_creation_supervisor", 
                    type="supervisor",
                    name="Website File Creation Supervisor",
                    inputs={
                        "project_setup": "project_init",
                        "goal": "Create all necessary files for a complete portfolio website"
                    },
                    available_agents=file_creation_agents,
                    max_agent_calls=3,  # UI designer, frontend dev, content creator
                    server_name="filesystem"
                ),
                
                # Verification
                NodeSpec(
                    id="file_verification",
                    type="task",
                    name="Verify Created Files",
                    inputs={
                        "created_files": "file_creation_supervisor",
                        "project_dir": str(project_dir)
                    },
                    server_name="filesystem"
                )
            ],
            edges=[
                ("project_init", "file_creation_supervisor"),
                ("file_creation_supervisor", "file_verification")
            ]
        )
        
        # Create unique run
        run = RunSpec(
            run_id=f"file_creation_{uuid.uuid4().hex[:8]}",
            goal="Create real website files using supervisor agent orchestration",
            policy={"create_files": True, "use_filesystem": True}
        )
        
        print(f"🏗️  Workflow: {len(workflow.nodes)} nodes")
        print(f"   📝 project_init (creates folder structure)")
        print(f"   🧠 file_creation_supervisor (orchestrates {len(file_creation_agents)} file-creating agents)")
        print(f"   ✅ file_verification (checks created files)")
        print()
        print("🚀 Starting file creation workflow...")
        print("-" * 65)
        
        # Execute workflow
        created_files = []
        async for event in orchestrator.run_streaming(workflow, run):
            
            if event.type == "RUN_START":
                print(f"🏁 Started: {event.data.get('goal', 'File creation run')}")
                
            elif event.type == "NODE_START": 
                node_id = event.node_id
                print(f"\n🔄 Executing: {node_id}")
                if node_id == "file_creation_supervisor":
                    print("   🧠 Supervisor selecting agents to create website files...")
                
            elif event.type == "AGENT_CHUNK":
                # Show file creation activity
                chunk_data = event.data
                if isinstance(chunk_data, dict) and "text" in chunk_data:
                    text = chunk_data["text"]
                    if any(keyword in text.lower() for keyword in ["created", "file", "html", "css", "js", "index"]):
                        print(f"📄 {text[:80]}{'...' if len(text) > 80 else ''}")
                elif isinstance(chunk_data, str) and any(keyword in chunk_data.lower() for keyword in ["created", "file"]):
                    print(f"📄 {chunk_data[:80]}{'...' if len(chunk_data) > 80 else ''}")
                    
            elif event.type == "NODE_COMPLETE":
                node_id = event.node_id
                phase = event.data.get("phase", "complete")
                print(f"✅ Completed: {node_id} ({phase})")
                
                if node_id == "file_creation_supervisor":
                    agents_called = event.data.get("agents_called", 0)
                    print(f"   🤖 Supervisor orchestrated {agents_called} file-creating agents")
                    
            elif event.type == "RUN_COMPLETE":
                print(f"\n🎉 File creation workflow completed!")
                
                # Check what files were actually created
                if project_dir.exists():
                    created_files = list(project_dir.rglob("*"))
                    print(f"📁 Files created in {project_dir}:")
                    for file_path in sorted(created_files):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            print(f"   📄 {file_path.name} ({size} bytes)")
                
            elif event.type == "ERROR":
                error_msg = event.data.get("error", "Unknown error")
                print(f"❌ Error in {event.node_id}: {error_msg}")
        
        # Final file summary
        print("\n" + "=" * 65)
        if created_files:
            print("✨ Real Website Files Created Successfully!")
            print(f"📍 Location: {project_dir}")
            print("\n🎯 This demonstrates supervisor agents using MCP tools to:")
            print("   • Create actual HTML, CSS, and JavaScript files")
            print("   • Organize files into proper project structure") 
            print("   • Generate real, usable website code")
            print("   • Coordinate multiple specialized file-creating agents")
        else:
            print("⚠️  No files detected - check MCP server connectivity")
        
        print(f"\n🌐 Ready to serve your lovable website from {project_dir}!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 File Creation Demo - Supervisor Agents Making Real Websites")
    print("This shows agents using MCP filesystem tools to create actual files")
    print()
    
    asyncio.run(demo_website_file_creation())