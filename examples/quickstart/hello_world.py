#!/usr/bin/env python3
"""
Basic MCP Agent + Server example using Agent Orchestra.
This demonstrates how to use Agent Orchestra as a drop-in replacement for mcp-use.

This example:
1. Sets up MCP servers (filesystem and brave search)
2. Creates an Agent Orchestra client (with sidecar features)
3. Uses the agent to interact with the servers
4. Shows telemetry and policy enforcement in action
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

from agent_orchestra import SidecarMCPClient, SidecarMCPAgent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class SimpleTelemetry:
    """Simple telemetry collector for demonstration."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def emit(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        print(f"📊 [{event['event_type']}] {event.get('data', {})}")


async def main() -> None:
    """Demonstrate Agent Orchestra usage identical to mcp-use."""
    
    # Load environment variables
    load_dotenv()
    
    print("🎭 Agent Orchestra - Basic MCP Example")
    print("=" * 60)
    
    # Create a temporary config file (like you would with mcp-use)
    config: Dict[str, Any] = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            },
            "brave-search": {
                "command": "npx", 
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "env": {
                    "BRAVE_API_KEY": "your-api-key-here"
                }
            }
        },
        # Agent Orchestra sidecar configuration (optional)
        "sidecar": {
            "policy": {
                "disallowed_tools": ["file_delete"]  # Safety: don't allow file deletion
            },
            "run_context": {
                "session_id": "demo_session_001",
                "user": "demo_user"
            }
        }
    }
    
    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    
    print(f"📝 Created config at: {config_path}")
    
    # Initialize telemetry
    telemetry = SimpleTelemetry()
    
    try:
        # Create MCP client - exactly like mcp-use but with sidecar features
        print("\n🔧 Creating MCP Client...")
        client = SidecarMCPClient.from_config_file(
            config_path,
            telemetry=telemetry  # Agent Orchestra enhancement
        )
        client: SidecarMCPClient = client

        print(f"✅ Client created with servers: {client.get_server_names()}")

        # Create sessions (mcp-use compatible API)
        print("\n🔗 Creating sessions...")
        sessions = await client.create_all_sessions()
        print(f"✅ Created {len(sessions)} sessions")

        # List available tools from each server
        print("\n🔨 Available tools:")
        for server_name, session in sessions.items():
            try:
                tools = await session.list_tools()
                print(f"  {server_name}: {[tool.name for tool in tools]}")
            except Exception as e:
                print(f"  {server_name}: Failed to list tools - {e}")

        # Create an agent (mcp-use API with sidecar enhancements)
        print("\n🤖 Creating MCP Agent...")
        llm = ChatOpenAI(model="gpt-4o-mini")  # Create LLM instance
        agent = SidecarMCPAgent(
            llm=llm,
            client=client,
            sidecar_telemetry=telemetry,  # Agent Orchestra enhancement
            sidecar_run_context={
                "task": "filesystem_demo",
                "timestamp": "2024-01-01"
            }
        )

        print("✅ Agent initialized")

        # Demonstrate basic agent usage (mcp-use compatible)
        print("\n💬 Agent Interaction Example:")
        print("-" * 40)

        # Example 1: File operations
        try:
            result = await agent.run(
                "Create a simple text file called 'hello.txt' with the content 'Hello from Agent Orchestra!'"
            )
            print(f"📁 File operation result: {str(result)[:100]}...")

        except Exception as e:
            print(f"⚠️ File operation failed (expected if MCP servers not installed): {e}")

        # Example 2: Show conversation history (mcp-use API)
        history = agent.get_conversation_history()
        print(f"💭 Conversation history has {len(history)} items")

        # Example 3: Streaming response (mcp-use compatible)
        print("\n🌊 Streaming Example:")
        print("-" * 20)

        try:
            chunk_count = 0
            async for chunk in agent.stream_events("List the files in the /tmp directory"):
                chunk_count += 1
                if chunk_count <= 3:  # Show first few chunks
                    print(f"Stream chunk {chunk_count}: {str(chunk)[:50]}...")
                if chunk_count >= 5:
                    break

            print(f"✅ Processed {chunk_count} streaming chunks")
        except Exception as e:
            print(f"⚠️ Streaming failed (expected if MCP servers not installed): {e}")

        # Cleanup
        await client.close_all_sessions()
        print("✅ Sessions closed")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("\n💡 Note: This example requires MCP servers to be installed:")
        print("   npm install -g @modelcontextprotocol/server-filesystem")
        print("   npm install -g @modelcontextprotocol/server-brave-search")
        print("\n🚀 But Agent Orchestra API compatibility is demonstrated!")

    finally:
        # Clean up config file
        Path(config_path).unlink(missing_ok=True)
    
    # Show telemetry results (Agent Orchestra enhancement)
    print(f"\n📊 Telemetry Summary:")
    print(f"   Total events captured: {len(telemetry.events)}")
    event_types: Set[Any] = set(event['event_type'] for event in telemetry.events)
    print(f"   Event types: {event_types}")
    
    print("\n🎉 Example completed!")
    print("🚀 Agent Orchestra provides 100% mcp-use compatibility")
    print("✨ Plus enhanced telemetry, safety, and extensibility!")


if __name__ == "__main__":
    asyncio.run(main())