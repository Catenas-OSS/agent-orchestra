#!/usr/bin/env python3
"""
Enhanced Restaurant Search Example - With Sidecar Features

Same restaurant search, but with enhanced telemetry, policy enforcement,
and observability features provided by Agent Orchestra.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for local development  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent_orchestra import SidecarMCPAgent, SidecarMCPClient


class RestaurantTelemetry:
    """Telemetry collector for restaurant search operations."""
    
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.search_count: int = 0
    
    def emit(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        event_type = event['event_type']
        data = event.get('data', {})
        
        if event_type == 'agent_run_start':
            self.search_count += 1
            print(f"🔍 Starting restaurant search #{self.search_count}")
        elif event_type == 'agent_run_success':
            print(f"✅ Restaurant search completed successfully")
        elif event_type == 'session_created':
            print(f"🌐 Connected to {data.get('server_name')} server")
        elif event_type in ['tool_call_start', 'tool_call_success']:
            tool_name = data.get('tool_name', 'unknown')
            print(f"🔧 Tool: {tool_name}")


async def main() -> None:
    # Load environment variables
    load_dotenv()

    # Enhanced configuration with sidecar features
    config: Dict[str, Any] = {
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": ["@playwright/mcp@latest"],
          "env": {
            "DISPLAY": ":1"
          }
        }
      },
      # Sidecar enhancements
      "sidecar": {
        "policy": {
          # Allow web automation tools but block dangerous operations
          "allowed_tools": ["page_goto", "page_screenshot", "page_content", "click"],
          "disallowed_tools": ["file_delete", "system_exec"]
        },
        "run_context": {
          "task": "restaurant_search",
          "location": "san_francisco",
          "session_id": "rest_search_001"
        }
      }
    }

    # Initialize telemetry
    telemetry = RestaurantTelemetry()

    # Create enhanced SidecarMCPClient with telemetry
    client = SidecarMCPClient.from_dict(config, telemetry=telemetry)

    print("🎭 Agent Orchestra - Enhanced Restaurant Search")
    print("=" * 50)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Create enhanced agent with sidecar telemetry
    agent = SidecarMCPAgent(
        llm=llm, 
        client=client, 
        max_steps=30,
        sidecar_telemetry=telemetry,
        sidecar_run_context={"enhanced_search": True}
    )

    print("🤖 Agent initialized with enhanced features")
    print(f"📊 Telemetry: Active")
    print(f"🔒 Policy: Tool restrictions enforced")
    print()

    # Run the query (same API as original)
    result = await agent.run(
        "Find the best restaurant in San Francisco",
    )
    
    print(f"\n🍽️ Restaurant Search Result:")
    print(f"{result}")
    
    # Show enhanced insights
    print(f"\n📈 Search Analytics:")
    print(f"   Total searches: {telemetry.search_count}")
    print(f"   Events captured: {len(telemetry.events)}")
    print(f"   Event types: {len(set(e['event_type'] for e in telemetry.events))}")

if __name__ == "__main__":
    asyncio.run(main())