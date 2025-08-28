#!/usr/bin/env python3
"""
AI-Powered Website Builder with Supervisor Agent Orchestration

This example demonstrates how to use Agent Orchestra's supervisor capabilities
to coordinate multiple specialized AI agents in building a complete, professional
website. The supervisor intelligently delegates tasks to specialist agents based
on their expertise and capabilities.

Features:
- Multi-agent coordination with intelligent task delegation
- Real-time collaboration between UI designers, developers, and content strategists
- Automatic file generation and project structure creation
- Professional-grade output suitable for enterprise clients

Requirements:
- OpenAI API key (set OPENAI_API_KEY environment variable)
- Node.js (for MCP filesystem server)
- Agent Orchestra package

Usage:
    python examples/supervisor_website_builder.py

The generated website will be saved to the ./website_output/ directory.
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Agent Orchestra imports
from agent_orchestra.orchestrator.types import NodeSpec, GraphSpec, RunSpec
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.sidecar.sidecar_client import SidecarMCPClient
from agent_orchestra.sidecar.sidecar_agent import SidecarMCPAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_environment() -> tuple[SidecarMCPAgent, MCPExecutor]:
    """
    Initialize the AI agent and executor for website building.
    
    Returns:
        Tuple of (agent, executor) ready for orchestration
        
    Raises:
        ImportError: If required dependencies are not installed
        EnvironmentError: If OpenAI API key is not set
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain_openai is required. Install it with: pip install langchain-openai"
        ) from e
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required. "
            "Get your API key from https://platform.openai.com/api-keys"
        )
    
    # Setup output directory
    output_dir = Path("./website_output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Website output directory: {output_dir.absolute()}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,  # Slight creativity for design decisions
        max_tokens=4000   # Ensure adequate response length
    )
    
    # Configure MCP filesystem server
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(output_dir.absolute())]
            }
        }
    }
    
    # Create client and agent with timeout handling
    logger.info("Creating MCP client...")
    try:
        import asyncio
        # Try to create the client with a reasonable timeout
        client = SidecarMCPClient.from_dict(config)
        logger.info("MCP client created successfully")
        
        agent = SidecarMCPAgent(
            llm=llm,
            client=client,
            sidecar_run_context={
                "task": "website_building",
                "project": "nexus_creative_studio",
                "quality_level": "enterprise"
            }
        )
        logger.info("MCP agent created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create MCP client/agent: {e}")
        raise EnvironmentError(f"MCP setup failed: {e}. Check if Node.js and npx are available.")
    
    # Create executor with verbose logging
    logger.info("Creating MCPExecutor...")
    executor = MCPExecutor(
        agent=agent,
        default_server="filesystem",
        broker=None,
        agent_pool=None,
        model_key="openai:gpt-4o-mini"
    )
    logger.info("MCPExecutor created successfully")
    
    logger.info("Environment setup complete")
    return agent, executor


def define_specialist_agents() -> Dict[str, Dict[str, Any]]:
    """
    Define the specialist agent profiles for website development.
    
    Returns:
        Dictionary mapping agent IDs to their capabilities and descriptions
    """
    return {
        "ui_designer": {
            "description": (
                "Expert UI/UX designer specializing in modern, visually stunning interfaces "
                "with advanced CSS animations and micro-interactions. Creates designs that "
                "balance aesthetics with usability and accessibility."
            ),
            "capabilities": [
                "CSS Grid/Flexbox mastery",
                "CSS animations and transitions", 
                "Responsive design patterns",
                "Design systems and tokens",
                "Modern color theory and typography",
                "User experience optimization"
            ],
            "server": "filesystem"
        },
        
        "frontend_developer": {
            "description": (
                "Senior frontend developer who creates interactive, performant web applications "
                "with clean, semantic code. Specializes in modern JavaScript and progressive "
                "enhancement techniques."
            ),
            "capabilities": [
                "Modern JavaScript (ES2023+)",
                "HTML5 semantic markup",
                "CSS custom properties and modern features",
                "Responsive implementation",
                "Web accessibility (WCAG 2.1 AA)",
                "Performance optimization"
            ],
            "server": "filesystem"
        },
        
        "content_strategist": {
            "description": (
                "Creative content strategist who crafts compelling copy, engaging narratives, "
                "and user-focused messaging. Expert in conversion optimization and brand voice "
                "development."
            ),
            "capabilities": [
                "Professional copywriting",
                "Brand storytelling",
                "User journey mapping",
                "Conversion optimization",
                "SEO content strategy",
                "Microcopy and UX writing"
            ],
            "server": "filesystem"
        },
        
        "technical_architect": {
            "description": (
                "Technical architect ensuring code quality, performance optimization, and "
                "best practices implementation. Focuses on maintainable, scalable solutions "
                "and cross-browser compatibility."
            ),
            "capabilities": [
                "Code architecture and organization",
                "Performance optimization techniques",
                "SEO technical implementation", 
                "Progressive enhancement strategies",
                "Cross-browser compatibility",
                "Code quality and maintainability"
            ],
            "server": "filesystem"
        }
    }


def create_project_specification() -> str:
    """
    Generate the detailed project specification for the website build.
    
    Returns:
        Comprehensive project brief for the supervisor agent
    """
    return """Create a stunning, professional portfolio website for a creative agency called 'Nexus Creative Studio'.

PROJECT OVERVIEW:
Build a high-end digital agency website that showcases cutting-edge web development capabilities and would impress enterprise clients. This is a showcase piece demonstrating modern web technologies and design excellence.

DESIGN REQUIREMENTS:
- Modern, visually striking design with smooth animations and micro-interactions
- Dark theme with vibrant accent colors (electric blue #00D4FF, neon green #39FF14)
- Contemporary typography with excellent hierarchy and readability
- Generous whitespace and clean, minimalist aesthetic
- Subtle but impressive visual effects and animations

CONTENT SECTIONS:
1. Hero Section:
   - Animated background or particle effects
   - Compelling headline about digital transformation
   - Professional tagline and call-to-action button
   - Smooth scroll indicator

2. Services Section:
   - Web Design & Development
   - Brand Identity & Strategy  
   - Digital Marketing & Growth
   - Each with icons, descriptions, and hover effects

3. Portfolio Grid:
   - 6-8 fictional client projects
   - Hover effects revealing project details
   - Mix of web, branding, and digital marketing work
   - Modal or detailed view functionality

4. Team Section:
   - 4-5 team member profiles
   - Professional headshot placeholders
   - Roles: Creative Director, Lead Developer, UX Designer, etc.
   - Brief, engaging bios

5. Contact Section:
   - Professional contact form with validation
   - Office location and contact details
   - Social media links
   - Google Maps integration placeholder

TECHNICAL SPECIFICATIONS:
- Semantic HTML5 structure with proper document outline
- CSS Grid and Flexbox for advanced layouts
- CSS custom properties for consistent theming
- Vanilla JavaScript with modern ES6+ features
- Mobile-first responsive design (breakpoints: 320px, 768px, 1024px, 1440px)
- Progressive enhancement principles
- WCAG 2.1 AA accessibility compliance
- Optimized performance and fast loading times
- SEO-friendly structure with proper meta tags

STYLE INSPIRATION:
Think high-end digital agencies like Pentagram, IDEO, R/GA, or Wieden+Kennedy. The website should feel:
- Professional yet creative
- Cutting-edge but not overwhelming  
- Trustworthy for enterprise clients
- Showcasing technical excellence

DELIVERABLES:
- index.html (main page)
- styles.css (comprehensive stylesheet)
- script.js (interactive functionality)
- Any additional assets or files needed
- Clean, well-organized file structure

QUALITY STANDARDS:
All code should be production-ready, well-commented, and follow modern web development best practices. The final result should be something a creative agency would proudly deploy as their own website.

Each specialist agent should collaborate effectively, building upon each other's work to create a cohesive, impressive final product. The supervisor should coordinate the team to ensure all requirements are met and the quality exceeds expectations."""


def create_workflow() -> tuple[GraphSpec, RunSpec]:
    """
    Create the orchestration workflow for website building.
    
    Returns:
        Tuple of (graph_spec, run_spec) for the orchestrator
    """
    specialist_agents = define_specialist_agents()
    project_brief = create_project_specification()
    
    # Create workflow graph
    graph_spec = GraphSpec(
        nodes=[
            NodeSpec(
                id="website_supervisor",
                type="supervisor", 
                name="Elite Website Development Team Supervisor",
                inputs={"brief": "Create a simple modern website with HTML, CSS, and JS files. Make it professional and responsive."},
                available_agents=specialist_agents,
                max_agent_calls=2,  # Reduce complexity to prevent hanging
                server_name="filesystem",
                timeout_s=60   # 1-minute timeout to prevent hanging
            )
        ],
        edges=[]  # Single supervisor node for this workflow
    )
    
    # Create run specification with timestamp for uniqueness
    import time
    timestamp = int(time.time())
    run_spec = RunSpec(
        run_id=f"nexus_website_{timestamp}_{uuid.uuid4().hex[:6]}",
        goal="Build professional agency website using supervised multi-agent coordination"
    )
    
    return graph_spec, run_spec


# Initialize workflow components at module level for CLI usage
def _initialize_workflow():
    """Initialize workflow components, with fallback for missing dependencies."""
    try:
        # Setup environment
        agent, executor = setup_environment()
        
        # Create workflow
        graph_spec, run_spec = create_workflow()
        
        return agent, executor, graph_spec, run_spec
        
    except Exception as e:
        # Fallback for when dependencies aren't available
        logger.error(f"Workflow initialization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if __name__ == "__main__":
            print(f"⚠️ Warning: Could not initialize workflow components: {e}")
            print("This is expected if OpenAI API key is not set or dependencies are missing.")
        
        # Create minimal fallback components
        from agent_orchestra.orchestrator.executors import CallableExecutor
        
        async def mock_executor(node, ctx):
            return {"output": f"Mock execution of {node.id}"}
        
        import time
        timestamp = int(time.time())
        graph_spec = GraphSpec(nodes=[], edges=[])
        run_spec = RunSpec(run_id=f"fallback_{timestamp}_{uuid.uuid4().hex[:6]}", goal="Fallback workflow")
        executor = CallableExecutor(mock_executor)
        
        return None, executor, graph_spec, run_spec

# Initialize components directly - fix the hanging issue at the source
logger.info("Initializing workflow components...")

try:
    # Create workflow first
    graph_spec, run_spec = create_workflow()
    
    # Initialize MCP components with timeout handling
    import signal
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    
    def init_with_timeout():
        """Initialize MCP components with a timeout."""
        try:
            return setup_environment()
        except Exception as e:
            logger.error(f"MCP setup failed: {e}")
            raise
    
    # Try to initialize with a 10-second timeout
    try:
        with ThreadPoolExecutor() as executor_pool:
            future = executor_pool.submit(init_with_timeout)
            agent, executor = future.result(timeout=10.0)
        logger.info("MCP components initialized successfully")
        
    except (FuturesTimeoutError, TimeoutError):
        logger.warning("MCP initialization timed out, using fallback")
        # Use the OpenAI-only executor as fallback
        from openai_supervisor_demo import executor as fallback_executor
        executor = fallback_executor
        agent = None
        
    except Exception as e:
        logger.warning(f"MCP initialization failed, using fallback: {e}")
        # Use the OpenAI-only executor as fallback
        from openai_supervisor_demo import executor as fallback_executor
        executor = fallback_executor
        agent = None
    
    logger.info("Workflow initialization complete")
    
except Exception as e:
    logger.error(f"Failed to create workflow: {e}")
    raise


def main() -> None:
    """
    Main execution function for the website builder demo.
    """
    print("🎭 Agent Orchestra - AI Website Builder Demo")
    print("=" * 60)
    print("Building a professional website using coordinated AI agents...")
    print()
    
    if globals().get('agent'):
        print("✅ Setup complete! Ready to build website.")
        print(f"📁 Output directory: {Path('./website_output').absolute()}")
        print(f"🆔 Run ID: {run_spec.run_id}")
        print()
        
        print("🚀 To run this demo:")
        print("   python -m agent_orchestra.cli_py run examples/quickstart/supervisor_website_builder.py --watch")
        print()
        print("📖 Or import in your own code:")
        print("   from examples.quickstart.supervisor_website_builder import graph_spec, run_spec, executor")
    else:
        print("❌ Setup failed. Please check your OpenAI API key and dependencies.")


if __name__ == "__main__":
    main()


# Export required components for CLI usage
__all__ = ["graph_spec", "run_spec", "executor"]