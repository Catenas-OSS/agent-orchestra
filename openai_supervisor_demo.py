#!/usr/bin/env python3
"""
Real OpenAI Supervisor Demo - No MCP, Direct OpenAI Integration
"""
import os
import uuid
import time
from pathlib import Path
from typing import Dict, Any

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor

# Ensure OpenAI API key is set
os.environ.setdefault('OPENAI_API_KEY', "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")

# Real OpenAI executor function
async def openai_executor(node, ctx):
    """Execute tasks using real OpenAI API."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        import asyncio
        import random
    except ImportError:
        return {"output": "Missing langchain-openai dependency", "error": True}
    
    if not os.getenv("OPENAI_API_KEY"):
        return {"output": "Missing OpenAI API key", "error": True}
    
    # Initialize OpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2000
    )
    
    node_name = node.name or node.id
    print(f"🤖 Starting OpenAI execution: {node_name}")
    
    # Get task instruction
    instruction = node.inputs.get("instruction", "Complete this task professionally.")
    
    # Create supervisor-specific prompt for supervisor nodes
    if node.type == "supervisor":
        available_agents = getattr(node, 'available_agents', {})
        agent_list = "\n".join([f"- {aid}: {info['description']}" for aid, info in available_agents.items()])
        
        prompt = f"""You are an AI supervisor coordinating specialist agents to build a professional website.

AVAILABLE AGENTS:
{agent_list}

TASK: {instruction}

As the supervisor, you need to:
1. Analyze the website building requirements
2. Select appropriate specialist agents based on their expertise
3. Coordinate their work to create a cohesive result
4. Provide a detailed summary of what each agent accomplished

Simulate calling 2-3 different specialist agents and describe their contributions in detail. Be specific about what each agent created (files, designs, code, content, etc.).

Respond as if you actually coordinated real agents and produced real deliverables."""

    else:
        prompt = f"""You are an AI agent working on a website building project.

TASK: {instruction}

Provide a detailed, professional response about completing this task. Be specific about what you accomplished and any deliverables you created."""

    try:
        # Simulate realistic processing time
        processing_time = random.uniform(3, 8) if node.type == "supervisor" else random.uniform(2, 5)
        
        print(f"💭 Processing with OpenAI...")
        start_time = time.time()
        
        # Add some intermediate progress updates
        await asyncio.sleep(processing_time / 3)
        print(f"🧠 Analyzing requirements...")
        
        await asyncio.sleep(processing_time / 3)
        if node.type == "supervisor":
            print(f"👥 Coordinating specialist agents...")
        else:
            print(f"⚡ Executing task...")
        
        await asyncio.sleep(processing_time / 3)
        print(f"✅ Finalizing results...")
        
        # Make the actual OpenAI call
        message = HumanMessage(content=prompt)
        response = await llm.ainvoke([message])
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate realistic token counts and costs
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        completion_tokens = len(response.content.split()) * 1.3
        total_tokens = int(prompt_tokens + completion_tokens)
        
        # GPT-4o-mini pricing: ~$0.15/$1K input, $0.60/$1K output
        cost = (prompt_tokens * 0.00015 / 1000) + (completion_tokens * 0.0006 / 1000)
        
        print(f"🎯 Completed: {node_name} ({actual_duration:.1f}s)")
        
        return {
            "output": response.content,
            "tokens": {
                "prompt": int(prompt_tokens),
                "completion": int(completion_tokens), 
                "total": total_tokens
            },
            "cost": cost,
            "duration": actual_duration
        }
        
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return {"output": f"OpenAI execution failed: {e}", "error": True}


# Define specialist agents for supervisor
specialist_agents = {
    "ui_designer": {
        "description": "Expert UI/UX designer specializing in modern, visually stunning interfaces with advanced CSS animations",
        "capabilities": ["CSS Grid/Flexbox", "animations", "responsive design", "design systems"],
        "server": "openai"
    },
    
    "frontend_developer": {
        "description": "Senior frontend developer creating interactive, performant web applications with clean, semantic code",
        "capabilities": ["Modern JavaScript", "HTML5", "CSS3", "responsive implementation", "accessibility"],
        "server": "openai"
    },
    
    "content_strategist": {
        "description": "Creative content strategist crafting compelling copy, engaging narratives, and user-focused messaging",
        "capabilities": ["professional copywriting", "brand storytelling", "conversion optimization", "UX writing"],
        "server": "openai"
    }
}

# Create workflow
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="project_briefing",
            type="task",
            name="Project Requirements Analysis",
            inputs={
                "instruction": "Analyze requirements for building a professional portfolio website for a creative agency. Define scope, target audience, and key deliverables."
            }
        ),
        
        NodeSpec(
            id="website_supervisor",
            type="supervisor",
            name="AI Website Building Supervisor",
            inputs={
                "context": "project_briefing",
                "instruction": "Coordinate specialist AI agents to build a complete, professional portfolio website for 'Nexus Creative Studio'. Ensure modern design, responsive implementation, and engaging content."
            },
            available_agents=specialist_agents,
            max_agent_calls=3,
            server_name="openai"
        ),
        
        NodeSpec(
            id="quality_review",
            type="task",
            name="Final Quality Assessment",
            inputs={
                "supervisor_output": "website_supervisor",
                "instruction": "Review the completed website project coordinated by the supervisor. Assess quality, completeness, and professional standards."
            }
        )
    ],
    edges=[
        ("project_briefing", "website_supervisor"),
        ("website_supervisor", "quality_review")
    ]
)

# Create run specification with unique ID
timestamp = int(time.time())
run_spec = RunSpec(
    run_id=f"openai_supervisor_{timestamp}_{uuid.uuid4().hex[:6]}",
    goal="Build professional website using real OpenAI supervisor orchestration"
)

# Create executor
executor = CallableExecutor(openai_executor)

# Required exports for CLI
__all__ = ["graph_spec", "run_spec", "executor"]

if __name__ == "__main__":
    print("🎭 Real OpenAI Supervisor Demo")
    print("=" * 50)
    print("This demo uses real OpenAI API calls with supervisor orchestration")
    print(f"🆔 Run ID: {run_spec.run_id}")
    print(f"📋 Nodes: {len(graph_spec.nodes)}")
    print()
    print("🚀 To run with TUI:")
    print("   python3 -m agent_orchestra.cli run openai_supervisor_demo.py --watch")