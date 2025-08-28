#!/usr/bin/env python3

import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.sidecar.sidecar_client import SidecarMCPClient
from agent_orchestra.sidecar.sidecar_agent import SidecarMCPAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    from langchain_openai import ChatOpenAI
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY required")
    
    output_dir = Path("./website_output")
    output_dir.mkdir(exist_ok=True)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=4000)
    
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(output_dir.absolute())]
            }
        }
    }
    
    client = SidecarMCPClient.from_dict(config)
    agent = SidecarMCPAgent(
        llm=llm,
        client=client,
        sidecar_run_context={
            "task": "website_building",
            "project": "nexus_creative_studio"
        }
    )
    
    executor = MCPExecutor(
        agent=agent,
        default_server="filesystem",
        model_key="openai:gpt-4o-mini"
    )
    
    return agent, executor

def create_workflow():
    specialist_agents = {
        "ui_designer": {
            "description": "Expert UI/UX designer who creates stunning, modern web interfaces with advanced CSS animations and responsive design. Creates complete HTML and CSS files that showcase cutting-edge design trends and user experience principles.",
            "capabilities": [
                "Modern HTML5 semantic structure creation",
                "Advanced CSS Grid and Flexbox layouts",
                "CSS animations, transitions, and micro-interactions",
                "Responsive design across all device sizes",
                "Modern typography and color theory",
                "Accessibility-compliant design implementation"
            ],
            "server": "filesystem"
        },
        "frontend_developer": {
            "description": "Senior frontend developer who builds interactive, high-performance web applications with clean, maintainable code. Specializes in modern JavaScript, progressive enhancement, and creating seamless user experiences.",
            "capabilities": [
                "Modern JavaScript (ES2023+) implementation",
                "Interactive UI components and animations",
                "Performance optimization and code splitting",
                "Cross-browser compatibility and testing",
                "Progressive Web App features",
                "Advanced DOM manipulation and event handling"
            ],
            "server": "filesystem"
        },
        "content_strategist": {
            "description": "Creative content strategist who crafts compelling copy, engaging narratives, and conversion-focused messaging. Expert in brand voice development and user journey optimization.",
            "capabilities": [
                "Professional copywriting and brand storytelling",
                "SEO-optimized content creation",
                "User experience and conversion copywriting",
                "Content architecture and information hierarchy",
                "Brand voice consistency and messaging",
                "Call-to-action optimization"
            ],
            "server": "filesystem"
        },
        "technical_architect": {
            "description": "Technical architect ensuring code quality, performance optimization, and scalable architecture. Focuses on best practices, maintainability, and enterprise-level standards.",
            "capabilities": [
                "Code architecture and organizational patterns",
                "Performance optimization and Core Web Vitals",
                "SEO technical implementation and meta optimization",
                "Security best practices and vulnerability prevention",
                "Code quality standards and maintainability",
                "Cross-platform compatibility and testing"
            ],
            "server": "filesystem"
        }
    }
    
    project_brief = """Create a stunning, professional portfolio website for a creative agency called 'Nexus Creative Studio'.

PROJECT REQUIREMENTS:
Build a high-end digital agency website that demonstrates cutting-edge web development capabilities and would impress enterprise clients. This should be a showcase piece that highlights modern web technologies and exceptional design.

DESIGN SPECIFICATIONS:
- Modern, visually striking design with smooth animations and subtle micro-interactions
- Dark theme with vibrant accent colors (electric blue #00D4FF, neon green #39FF14)
- Contemporary typography with excellent hierarchy and readability
- Generous whitespace and clean, minimalist aesthetic
- Professional visual effects that enhance user experience

REQUIRED SECTIONS:
1. Hero Section: Compelling headline, professional tagline, animated call-to-action
2. Services: Web Design & Development, Brand Identity, Digital Marketing (with icons and descriptions)
3. Portfolio Grid: 6-8 project showcases with hover effects and detailed views
4. Team Section: 4-5 team member profiles with roles and brief bios
5. Contact Section: Professional contact form with validation and company details

TECHNICAL REQUIREMENTS:
- Semantic HTML5 structure with proper document outline
- Advanced CSS with Grid, Flexbox, and custom properties
- Vanilla JavaScript with modern ES6+ features and smooth animations
- Mobile-first responsive design (320px, 768px, 1024px, 1440px breakpoints)
- WCAG 2.1 AA accessibility compliance
- Optimized performance and fast loading times
- SEO-friendly structure with proper meta tags and schema markup

DELIVERABLES:
- index.html (complete website structure)
- styles.css (comprehensive styling with animations)
- script.js (interactive functionality and animations)
- All files should be production-ready and well-organized

The final website should feel professional, trustworthy, and showcase technical excellence suitable for enterprise clients."""
    
    graph_spec = GraphSpec(
        nodes=[
            NodeSpec(
                id="website_supervisor",
                type="supervisor",
                name="Elite Website Development Team Supervisor",
                inputs={"brief": project_brief},
                available_agents=specialist_agents,
                max_agent_calls=6,
                server_name="filesystem",
                timeout_s=300
            )
        ],
        edges=[]
    )
    
    timestamp = int(__import__('time').time())
    run_spec = RunSpec(
        run_id=f"website_{timestamp}_{uuid.uuid4().hex[:6]}",
        goal="Build website using supervisor orchestration"
    )
    
    return graph_spec, run_spec

try:
    graph_spec, run_spec = create_workflow()
    
    def init_with_timeout():
        return setup_environment()
    
    try:
        with ThreadPoolExecutor() as executor_pool:
            future = executor_pool.submit(init_with_timeout)
            agent, executor = future.result(timeout=10.0)
    except (FuturesTimeoutError, TimeoutError):
        from openai_supervisor_demo import executor as fallback_executor
        executor = fallback_executor
        agent = None
    except Exception:
        from openai_supervisor_demo import executor as fallback_executor
        executor = fallback_executor
        agent = None
        
except Exception as e:
    logger.error(f"Failed: {e}")
    raise

__all__ = ["graph_spec", "run_spec", "executor"]