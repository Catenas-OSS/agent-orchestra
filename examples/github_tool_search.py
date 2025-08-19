#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agent_orchestra import SidecarMCPAgent, SidecarMCPClient

load_dotenv()  # 🔑 Load API keys

# 1️⃣ Configure the GitHub MCP server
CONFIG = { # type: ignore
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["@modelcontextprotocol/server-github@latest"],
            "env": {
                "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")
            }
        }
    }
    
}

async def main():
    # 2️⃣ Create MCP client and LLM
    client = SidecarMCPClient.from_dict(CONFIG) # type: ignore
    llm = ChatOpenAI(model="gpt-4o")
    
    # 3️⃣ Wire the LLM to the client
    agent = SidecarMCPAgent(llm=llm, client=client, max_steps=1)
    
    try:
        print("🔍 Listing all available tools from GitHub MCP server...\n")
        
        # 4️⃣ Ask the agent to list available tools
        result = await agent.run("List all available tools and their descriptions from the GitHub MCP server") # type: ignore
        print("📊 Available GitHub MCP Tools:")
        print(result) # type: ignore
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 5️⃣ Always clean up running MCP sessions
        await client.close_all_sessions() # type: ignore

if __name__ == "__main__":
    asyncio.run(main())
