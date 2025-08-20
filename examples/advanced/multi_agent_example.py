#!/usr/bin/env python3
"""
Multi-Agentic Orchestrator Example

Shows how to use different agents for different tasks within a single orchestrator workflow.
Each agent has different MCP servers and capabilities.

This is a conceptual example - full implementation would require MultiAgentExecutor.
"""

import asyncio
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec

# Conceptual multi-agent workflow
def create_multi_agent_workflow():
    """Create workflow using different agents for different capabilities."""
    
    workflow = GraphSpec(
        nodes=[
            # Gate: Use validation agent with business rules
            NodeSpec(
                id="business_validation",
                type="gate",
                server_name="validation_agent",  # Agent with business logic tools
                inputs={
                    "predicate": True,
                    "instruction": "Validate business requirements and location"
                }
            ),
            
            # Foreach: Use web scraping agent for parallel searches  
            NodeSpec(
                id="web_research",
                type="foreach",
                server_name="web_agent",  # Agent with Playwright/browser tools
                inputs={
                    "items": ["OpenTable", "Yelp", "Google Maps"],
                    "instruction": "Search each platform for restaurants"
                },
                concurrency=2
            ),
            
            # Task: Use data analysis agent for processing
            NodeSpec(
                id="analyze_data",
                type="task", 
                server_name="data_agent",  # Agent with pandas/analysis tools
                inputs={
                    "from": "web_research",
                    "instruction": "Analyze restaurant data for patterns and ratings"
                }
            ),
            
            # Reduce: Use LLM agent for intelligent summary
            NodeSpec(
                id="create_summary",
                type="reduce",
                server_name="llm_agent",  # Agent optimized for text generation
                inputs={
                    "from_ids": ["analyze_data"],
                    "instruction": "Create personalized restaurant recommendations"
                }
            ),
            
            # Task: Use booking agent with calendar/reservation tools
            NodeSpec(
                id="booking_assistance", 
                type="task",
                server_name="booking_agent",  # Agent with calendar/booking APIs
                inputs={
                    "from": "create_summary",
                    "instruction": "Help with reservation booking and scheduling"
                }
            )
        ],
        edges=[
            ("business_validation", "web_research"),
            ("web_research", "analyze_data"), 
            ("analyze_data", "create_summary"),
            ("create_summary", "booking_assistance")
        ]
    )
    
    return workflow

# Agent configurations for different capabilities
agent_configs = {
    "validation_agent": {
        "mcpServers": {
            "business_rules": {
                "command": "python", 
                "args": ["business_rules_server.py"]
            }
        }
    },
    
    "web_agent": {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"]
            }
        }
    },
    
    "data_agent": {
        "mcpServers": {
            "pandas": {
                "command": "python",
                "args": ["pandas_mcp_server.py"]
            }
        }
    },
    
    "llm_agent": {
        "mcpServers": {
            # Could be empty - pure LLM reasoning
        }
    },
    
    "booking_agent": {
        "mcpServers": {
            "calendar": {
                "command": "python", 
                "args": ["calendar_mcp_server.py"]
            },
            "reservations": {
                "command": "python",
                "args": ["booking_api_server.py"] 
            }
        }
    }
}

print("🎭 Multi-Agentic Orchestrator Workflow Example")
print("=" * 50)

workflow = create_multi_agent_workflow()
print(f"📊 Nodes: {len(workflow.nodes)}")
print(f"🔗 Edges: {len(workflow.edges)}")

print(f"\n🤖 Agent Specializations:")
for node in workflow.nodes:
    agent = node.server_name or "default"
    print(f"   {node.id} ({node.type}) → {agent}")

print(f"\n💡 This shows how different agents can handle different parts")
print(f"   of a complex workflow based on their specialized tools!")