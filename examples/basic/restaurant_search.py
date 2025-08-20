#!/usr/bin/env python3
"""
Restaurant Search Example - Agent Orchestra Version

This is the exact same example as the original mcp-use version,
but using Agent Orchestra sidecar imports for enhanced features.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent_orchestra.sidecar import SidecarMCPAgent, SidecarMCPClient

async def main() -> None:
    # Load environment variables
    load_dotenv()

    # Create configuration dictionary (exactly the same as mcp-use)
    config: Dict[str, Any] = {
      "mcpServers": {
        "playwright": {
          "command": "npx",
          "args": ["@playwright/mcp@latest"],
          "env": {
            "DISPLAY": ":1"
          }
        }
      }
    }

    # Create SidecarMCPClient from configuration dictionary (drop-in replacement)
    client = SidecarMCPClient.from_dict(config)

    # Create LLM (exactly the same)
    llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client (drop-in replacement)
    agent = SidecarMCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query (exactly the same API)
    result = await agent.run(
        "Find the best restaurant in San Francisco",
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())