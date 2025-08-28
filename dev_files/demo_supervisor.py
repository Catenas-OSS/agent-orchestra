#!/usr/bin/env python3
"""
Clean demonstration of Smart Agent Orchestration without external MCP dependencies
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
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
    from agent_orchestra.orchestrator.executors import CallableExecutor
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e . && pip install langchain-openai python-dotenv")
    sys.exit(1)

class WebsiteBuilderExecutor:
    """Mock executor that simulates website building agents without external dependencies."""
    
    def __init__(self):
        self.llm = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
            except Exception:
                pass
    
    async def execute(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a node with mock website building logic."""
        
        if node.type == "task":
            if "requirements" in node.id:
                return {"output": "Modern portfolio website for freelance graphic designer. Target audience: potential clients. Requirements: clean design, mobile responsive, portfolio showcase, contact form."}
            elif "final_review" in node.id:
                return {"output": "Website successfully completed with modern design, responsive layout, engaging content, and all functionality working properly. Ready for deployment!"}
            else:
                return {"output": f"Task '{node.name}' completed successfully."}
                
        elif node.type == "supervisor":
            # Note: The supervisor logic is actually handled by the core orchestrator, not this executor
            # This is just a fallback for demonstration
            print("⚠️  Supervisor node detected - this should be handled by the core orchestrator")
            
            # Return a simple supervisor response for demonstration
            return {"output": """
CALL_AGENT: ui_designer
TASK: Create a modern, clean UI design for a graphic designer's portfolio website
INPUT: Target audience: potential clients, Style: minimalist and professional, Focus: showcase creative work prominently
---

CALL_AGENT: content_writer  
TASK: Write engaging copy for the portfolio website
INPUT: Brand voice: creative but professional, Sections needed: about, portfolio, services, contact, SEO focus: graphic design services
---

CALL_AGENT: frontend_developer
TASK: Implement responsive portfolio website with modern features
INPUT: Design from UI designer, Content from writer, Features: image gallery, contact form, mobile-first approach
---

FINAL_RESULT: Created a complete professional portfolio website with modern UI design, engaging content, responsive implementation, and thorough quality testing. The website effectively showcases the designer's work with excellent user experience and performance.
"""}
        
        return {"output": f"Node {node.id} completed"}

async def demo_supervisor_orchestration():
    """Demonstrate the supervisor agent orchestration system."""
    print("🌐 Smart Agent Orchestration - Website Builder Demo")
    print("=" * 60)
    
    # Check for OpenAI API key
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if has_openai:
        print("🤖 Using real OpenAI GPT-4o for supervisor decisions")
    else:
        print("🎭 Using mock responses (set OPENAI_API_KEY for real LLM)")
    print()
    
    try:
        # Create executor
        executor = CallableExecutor(WebsiteBuilderExecutor().execute)
        orchestrator = Orchestrator(executor=executor)
        
        # Define specialized agents for website building
        available_agents = {
            "ui_designer": {
                "description": "Creates beautiful, modern UI designs and layouts for websites",
                "capabilities": ["responsive design", "color schemes", "typography", "user experience"],
                "server": "design"
            },
            "frontend_developer": {
                "description": "Implements HTML, CSS, and JavaScript for web interfaces",
                "capabilities": ["HTML5", "CSS3", "JavaScript", "responsive frameworks", "accessibility"],
                "server": "development"
            },
            "content_writer": {
                "description": "Creates engaging, SEO-optimized content for websites",
                "capabilities": ["copywriting", "SEO", "brand voice", "content strategy"],
                "server": "content"
            },
            "backend_developer": {
                "description": "Builds server-side functionality and APIs",
                "capabilities": ["APIs", "databases", "server logic", "authentication", "deployment"],
                "server": "development"
            },
            "quality_assurance": {
                "description": "Tests websites for bugs, performance, and usability issues",
                "capabilities": ["testing", "performance analysis", "accessibility audit", "cross-browser compatibility"],
                "server": "testing"
            }
        }
        
        # Create workflow with supervisor agent
        workflow = GraphSpec(
            nodes=[
                # Requirements gathering
                NodeSpec(
                    id="requirements",
                    type="task",
                    name="Gather Website Requirements",
                    inputs={
                        "user_request": "Create a modern portfolio website for a freelance graphic designer"
                    }
                ),
                
                # Smart supervisor agent
                NodeSpec(
                    id="website_supervisor",
                    type="supervisor",
                    name="Website Building Supervisor",
                    inputs={
                        "requirements": "requirements",
                        "project_type": "portfolio website",
                        "target_audience": "potential clients looking for graphic design services"
                    },
                    available_agents=available_agents,
                    max_agent_calls=4,
                    server_name="coordination"
                ),
                
                # Final review
                NodeSpec(
                    id="final_review",
                    type="task", 
                    name="Review and Finalize Website",
                    inputs={
                        "supervisor_output": "website_supervisor"
                    }
                )
            ],
            edges=[
                ("requirements", "website_supervisor"),
                ("website_supervisor", "final_review")
            ]
        )
        
        # Create unique run
        run = RunSpec(
            run_id=f"demo_build_{uuid.uuid4().hex[:8]}",
            goal="Demonstrate smart agent orchestration for website building",
            policy={"demo_mode": True}
        )
        
        print(f"📋 Created workflow with {len(workflow.nodes)} nodes:")
        print(f"   📝 requirements (task)")
        print(f"   🧠 website_supervisor (supervisor) - {len(available_agents)} available agents")
        print(f"   📝 final_review (task)")
        print()
        print("🚀 Executing smart agent orchestration workflow...")
        print("-" * 60)
        
        # Execute workflow
        event_count = 0
        async for event in orchestrator.run_streaming(workflow, run):
            event_count += 1
            
            if event.type == "RUN_START":
                print(f"🏁 Started: {event.data.get('goal', 'Demo run')}")
                
            elif event.type == "NODE_START": 
                node_id = event.node_id
                print(f"\n🔄 Executing: {node_id}")
                if node_id == "website_supervisor":
                    print("   🧠 Supervisor analyzing requirements and selecting specialists...")
                
            elif event.type == "NODE_COMPLETE":
                node_id = event.node_id
                phase = event.data.get("phase", "complete")
                print(f"✅ Completed: {node_id} ({phase})")
                
                if node_id == "website_supervisor":
                    agents_called = event.data.get("agents_called", 0)
                    print(f"   🤖 Supervisor orchestrated {agents_called} specialized agents")
                    
            elif event.type == "RUN_COMPLETE":
                print(f"\n🎉 Workflow completed successfully!")
                print(f"📊 Total events: {event_count}")
                
            elif event.type == "ERROR":
                error_msg = event.data.get("error", "Unknown error")
                print(f"❌ Error in {event.node_id}: {error_msg}")
        
        print("\n" + "=" * 60)
        print("✨ Smart Agent Orchestration Demo Complete!")
        print("\n🌟 Key Features Demonstrated:")
        print("   • Supervisor agent dynamically chooses specialized agents")
        print("   • Multiple agents coordinate to build complete websites")  
        print("   • Real-time streaming shows orchestration decisions")
        print("   • Scalable architecture supports any number of specialists")
        print("\n🎯 Perfect for building lovable websites with AI! 🌐")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Smart Agent Orchestration Demo")
    print("Showcasing intelligent multi-agent website building")
    print()
    
    asyncio.run(demo_supervisor_orchestration())