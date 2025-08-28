#!/usr/bin/env python3
"""
Website Builder Workflow - Demonstrates Smart Agent Orchestration

This workflow showcases the new supervisor agent that can dynamically choose
and orchestrate specialized agents to build a complete website.
"""

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec

# Define specialized agents for website building
available_agents = {
    "ui_designer": {
        "description": "Creates beautiful, modern UI designs and layouts for websites",
        "capabilities": ["responsive design", "color schemes", "typography", "user experience"],
        "server": "filesystem"
    },
    "frontend_developer": {
        "description": "Implements HTML, CSS, and JavaScript for web interfaces",
        "capabilities": ["HTML5", "CSS3", "JavaScript", "responsive frameworks", "accessibility"],
        "server": "filesystem"
    },
    "content_writer": {
        "description": "Creates engaging, SEO-optimized content for websites",
        "capabilities": ["copywriting", "SEO", "brand voice", "content strategy"],
        "server": "brave-search"
    },
    "backend_developer": {
        "description": "Builds server-side functionality and APIs",
        "capabilities": ["APIs", "databases", "server logic", "authentication", "deployment"],
        "server": "filesystem"
    },
    "quality_assurance": {
        "description": "Tests websites for bugs, performance, and usability issues",
        "capabilities": ["testing", "performance analysis", "accessibility audit", "cross-browser compatibility"],
        "server": "filesystem"
    }
}

def create_website_workflow():
    """Create a workflow that uses supervisor agent to build websites intelligently."""
    
    nodes = [
        # Requirements gathering node
        NodeSpec(
            id="requirements",
            type="task",
            name="Gather Website Requirements",
            inputs={
                "user_request": "Create a modern portfolio website for a freelance graphic designer"
            }
        ),
        
        # Smart supervisor agent that chooses appropriate agents
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
            max_agent_calls=4,  # Can call up to 4 agents as needed
            server_name="filesystem"
        ),
        
        # Final review and integration
        NodeSpec(
            id="final_review",
            type="task", 
            name="Review and Finalize Website",
            inputs={
                "supervisor_output": "website_supervisor"
            }
        )
    ]
    
    edges = [
        ("requirements", "website_supervisor"),
        ("website_supervisor", "final_review")
    ]
    
    return GraphSpec(nodes=nodes, edges=edges)

def create_complex_ecommerce_workflow():
    """Create a more complex workflow for an e-commerce website."""
    
    nodes = [
        # Business analysis
        NodeSpec(
            id="business_analysis",
            type="task",
            name="Analyze E-commerce Business Requirements", 
            inputs={
                "business_type": "Online clothing boutique",
                "target_market": "Young professionals aged 25-35",
                "budget": "Medium ($10k-50k)",
                "timeline": "3 months"
            }
        ),
        
        # First supervisor: Planning and Design
        NodeSpec(
            id="design_supervisor",
            type="supervisor", 
            name="Design and Planning Supervisor",
            inputs={
                "business_analysis": "business_analysis",
                "phase": "design_and_planning"
            },
            available_agents={
                "ui_designer": available_agents["ui_designer"],
                "content_writer": available_agents["content_writer"],
                "business_analyst": {
                    "description": "Analyzes business requirements and creates technical specifications",
                    "capabilities": ["requirements analysis", "technical specs", "user stories", "wireframes"],
                    "server": "claude"
                }
            },
            max_agent_calls=3,
            server_name="filesystem"
        ),
        
        # Second supervisor: Development 
        NodeSpec(
            id="development_supervisor",
            type="supervisor",
            name="Development Supervisor", 
            inputs={
                "design_specs": "design_supervisor",
                "phase": "development"
            },
            available_agents={
                "frontend_developer": available_agents["frontend_developer"],
                "backend_developer": available_agents["backend_developer"],
                "database_architect": {
                    "description": "Designs and implements database schemas for e-commerce systems",
                    "capabilities": ["database design", "data modeling", "performance optimization", "security"],
                    "server": "claude"
                }
            },
            max_agent_calls=3,
            server_name="filesystem"
        ),
        
        # Third supervisor: Testing and Deployment
        NodeSpec(
            id="deployment_supervisor", 
            type="supervisor",
            name="Testing and Deployment Supervisor",
            inputs={
                "developed_site": "development_supervisor",
                "phase": "testing_and_deployment"
            },
            available_agents={
                "quality_assurance": available_agents["quality_assurance"],
                "devops_engineer": {
                    "description": "Handles deployment, monitoring, and infrastructure setup",
                    "capabilities": ["CI/CD", "cloud deployment", "monitoring", "security", "scaling"],
                    "server": "claude"
                },
                "performance_tester": {
                    "description": "Tests website performance, load handling, and optimization",
                    "capabilities": ["load testing", "performance optimization", "SEO audit", "analytics setup"],
                    "server": "claude"
                }
            },
            max_agent_calls=3,
            server_name="filesystem"
        ),
        
        # Final integration
        NodeSpec(
            id="project_completion",
            type="task",
            name="Complete E-commerce Project",
            inputs={
                "design_output": "design_supervisor", 
                "development_output": "development_supervisor",
                "deployment_output": "deployment_supervisor"
            }
        )
    ]
    
    edges = [
        ("business_analysis", "design_supervisor"),
        ("design_supervisor", "development_supervisor"), 
        ("development_supervisor", "deployment_supervisor"),
        ("deployment_supervisor", "project_completion")
    ]
    
    return GraphSpec(nodes=nodes, edges=edges)

# Create module-level exports for CLI
from agent_orchestra.orchestrator.executors import CallableExecutor

# Realistic demo executor that simulates actual work
import asyncio
import random

class StreamingMockExecutor:
    """Enhanced mock executor that streams chunks with cost data to simulate real LLM behavior."""
    
    def __init__(self):
        self.total_cost = 0
        self.total_tokens = 0
    
    async def __call__(self, node, ctx):
        """Execute node with realistic streaming simulation."""
        return await self.mock_executor_with_streaming(node, ctx)
    
    async def mock_executor_with_streaming(self, node, ctx):
        """Simulate realistic AI agent work with streaming, timing, cost, and tokens."""
        
        node_name = node.name or node.id
        print(f"🤖 Starting: {node_name}")
        
        # Check if we have a streaming callback in context
        stream_callback = getattr(ctx, 'stream_callback', None) if hasattr(ctx, 'stream_callback') else None
        
        # Simulate different work durations based on node type
        if "supervisor" in node.id:
            # Supervisors think longer and use more tokens
            work_time = random.uniform(4, 10)
            prompt_tokens = random.randint(800, 1500)
            completion_tokens = random.randint(400, 900)
            
            # Stream thinking chunks for supervisors
            thinking_chunks = [
                "Analyzing the website requirements and available specialist agents...",
                "Evaluating which agents would be most suitable for this portfolio website project...",
                "The UI Designer agent can handle the visual design and layout...",
                "Frontend Developer agent will implement the responsive code...",
                "Content Writer agent can create compelling copy...", 
                "Quality Assurance agent will test and validate the final result..."
            ]
            
        else:
            # Regular tasks work normally
            work_time = random.uniform(2, 6)
            prompt_tokens = random.randint(200, 600)
            completion_tokens = random.randint(150, 400)
            
            # Stream working chunks for regular tasks
            thinking_chunks = [
                f"Processing {node_name} requirements...",
                f"Applying best practices for {node_name}...",
                f"Finalizing {node_name} implementation..."
            ]
        
        # Calculate realistic costs (approximate GPT-4 pricing)
        total_tokens = prompt_tokens + completion_tokens
        cost = (prompt_tokens * 0.00003) + (completion_tokens * 0.00006)
        
        # Stream chunks during execution
        chunk_interval = work_time / len(thinking_chunks)
        for i, chunk_text in enumerate(thinking_chunks):
            await asyncio.sleep(chunk_interval)
            
            # Simulate streaming chunk with partial cost/token data
            chunk_data = {
                "text": chunk_text,
                "usage": {
                    "prompt_tokens": prompt_tokens if i == len(thinking_chunks) - 1 else 0,
                    "completion_tokens": int(completion_tokens * (i + 1) / len(thinking_chunks)),
                    "total_tokens": int(total_tokens * (i + 1) / len(thinking_chunks))
                },
                "cost": cost * (i + 1) / len(thinking_chunks) if i == len(thinking_chunks) - 1 else 0
            }
            
            # Send chunk if callback available
            if stream_callback:
                await stream_callback("AGENT_CHUNK", node.id, chunk_data)
            
            print(f"💭 {chunk_text}")
        
        # Simulate different outputs based on node
        if "requirements" in node.id:
            output = "Requirements gathered: Modern portfolio site with dark theme, responsive design, and smooth animations"
        elif "supervisor" in node.id:
            output = "Coordinated 4 specialist agents: UI Designer created wireframes, Frontend Developer built responsive layout, Content Strategist wrote compelling copy, Technical Architect optimized performance"
        elif "review" in node.id:
            output = "Final review complete: Website meets all requirements with professional quality suitable for enterprise clients"
        else:
            output = f"Completed {node_name} with professional results"
        
        # Send final output chunk
        if stream_callback:
            final_chunk = {
                "output": output,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                "cost": cost
            }
            await stream_callback("AGENT_CHUNK", node.id, final_chunk)
        
        print(f"✅ Finished: {node_name}")
        
        # Update global tracking
        self.total_cost += cost
        self.total_tokens += total_tokens
        
        # Return comprehensive result with cost and token data
        return {
            "output": output,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens
            },
            "cost": cost
        }

# Create the streaming mock executor instance
streaming_mock_executor = StreamingMockExecutor()

# Export the simple workflow by default
graph_spec = create_website_workflow()
import uuid
import time
timestamp = int(time.time())
run_spec = RunSpec(
    run_id=f"website_build_{timestamp}_{uuid.uuid4().hex[:6]}",
    goal="Build a modern portfolio website using smart agent orchestration"
)
executor = CallableExecutor(streaming_mock_executor)

# Also create the complex workflow
simple_run = run_spec
complex_run = RunSpec(
    run_id="ecommerce_build_001", 
    goal="Build a complete e-commerce website with multiple specialized supervisor agents"
)

if __name__ == "__main__":
    print("🌐 Website Builder Workflows Created!")
    print("\n📋 Available workflows:")
    print("1. Simple Portfolio Website (1 supervisor, 4 agents)")
    print("2. Complex E-commerce Website (3 supervisors, 8+ agents)")
    print("\n🤖 Smart Agent Orchestration Features:")
    print("- Supervisor agents that choose appropriate specialists")
    print("- Dynamic agent selection based on project needs")
    print("- Multi-stage workflows with specialized supervisors")
    print("- Automatic load balancing across different agent types")
    
    # Export workflows
    simple_workflow = create_website_workflow()
    complex_workflow = create_complex_ecommerce_workflow()
    
    print(f"\n✅ Simple workflow: {len(simple_workflow.nodes)} nodes, {len(simple_workflow.edges)} edges")
    print(f"✅ Complex workflow: {len(complex_workflow.nodes)} nodes, {len(complex_workflow.edges)} edges")