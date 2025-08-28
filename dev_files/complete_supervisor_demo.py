#!/usr/bin/env python3

import sys
import uuid
from pathlib import Path

_src_path = str(Path(__file__).parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.sidecar.sidecar_client import SidecarMCPClient
from agent_orchestra.sidecar.sidecar_agent import SidecarMCPAgent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

config = {
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./output"]
        }
    }
}

client = SidecarMCPClient.from_dict(config)
agent = SidecarMCPAgent(
    llm=llm,
    client=client,
    sidecar_run_context={"task": "website_builder"}
)

executor = MCPExecutor(
    agent=agent,
    default_server="filesystem",
    broker=None,
    agent_pool=None,
    model_key="openai:gpt-4o-mini"
)

specialist_agents = {
    "ui_designer": {
        "description": "Expert UI/UX designer specializing in modern, visually stunning interfaces with advanced CSS animations and micro-interactions",
        "capabilities": ["CSS Grid/Flexbox mastery", "CSS animations", "responsive design", "design systems", "modern color theory"],
        "server": "filesystem"
    },
    "frontend_developer": {
        "description": "Senior frontend developer who creates interactive, performant web applications with clean, semantic code", 
        "capabilities": ["Modern JavaScript (ES2023)", "HTML5 semantic markup", "CSS custom properties", "responsive implementation", "accessibility (WCAG)"],
        "server": "filesystem"
    },
    "content_strategist": {
        "description": "Creative content strategist who crafts compelling copy, engaging narratives, and user-focused messaging",
        "capabilities": ["copywriting", "storytelling", "user journey mapping", "brand voice", "conversion optimization"],
        "server": "filesystem"
    },
    "technical_architect": {
        "description": "Technical architect ensuring code quality, performance optimization, and best practices implementation",
        "capabilities": ["code structure", "performance optimization", "SEO implementation", "progressive enhancement", "cross-browser compatibility"],
        "server": "filesystem"
    }
}

graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="website_supervisor",
            type="supervisor", 
            name="Elite Website Development Team Supervisor",
            inputs={
                "brief": """Create a stunning, professional portfolio website for a creative agency called 'Nexus Creative Studio'. 

REQUIREMENTS:
- Modern, visually striking design with smooth animations and micro-interactions
- Dark theme with vibrant accent colors (electric blue, neon green)
- Hero section with animated background, compelling headline, and call-to-action
- Services section showcasing web design, branding, digital marketing
- Portfolio grid with hover effects and project showcases
- Team section with member photos and bios
- Contact form with validation and smooth interactions
- Mobile-responsive design that looks perfect on all devices
- High-performance, semantic HTML with proper SEO structure
- Custom CSS animations, transitions, and interactive elements
- Professional JavaScript functionality (smooth scroll, form handling, animations)

STYLE DIRECTION: 
Think high-end digital agency - sleek, modern, creative but professional. Inspired by agencies like Pentagram, IDEO, or R/GA. Use contemporary typography, generous whitespace, and subtle but impressive visual effects.

TECHNICAL REQUIREMENTS:
- Semantic HTML5 structure
- CSS Grid and Flexbox layouts
- CSS custom properties for theming
- Vanilla JavaScript with modern ES6+ features
- Progressive enhancement principles
- Accessibility best practices (WCAG 2.1 AA)
- Optimized performance and loading

Each specialist should collaborate to create something that would impress potential enterprise clients. This is a showcase piece that demonstrates cutting-edge web development capabilities."""
            },
            available_agents=specialist_agents,
            max_agent_calls=6,
            server_name="filesystem"
        )
    ],
    edges=[]
)

run_spec = RunSpec(
    run_id=f"website_build_{uuid.uuid4().hex[:6]}",
    goal="Build website using supervised agent coordination"
)

__all__ = ["graph_spec", "run_spec", "executor"]