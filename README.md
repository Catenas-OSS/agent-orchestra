# Agent Orchestra

A sidecar library that enhances [mcp-use](https://github.com/anthropics/mcp-use) with enterprise features while maintaining 100% API compatibility.

## Features

- 🔄 **Drop-in replacement** for mcp-use with identical API
- 📊 **Enhanced telemetry** - Track all MCP interactions and agent operations  
- 🔒 **Policy enforcement** - Control tool access with safety policies
- 🎯 **Production ready** - Built for enterprise deployment and observability
- 🔧 **Zero breaking changes** - Existing mcp-use code works unchanged

## Installation

```bash
pip install mcp-use  # Required dependency
pip install agent-orchestra
```

## Quick Start

```python
import asyncio
from agent_orchestra import SidecarMCPClient, SidecarMCPAgent

async def main():
    # Same configuration as mcp-use
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            }
        }
    }
    
    # Drop-in replacement with optional enhancements
    client = SidecarMCPClient.from_dict(config)
    agent = SidecarMCPAgent(client=client, model="gpt-4o")
    
    async with client:
        result = await agent.run("List files in the directory")
        print(result)

asyncio.run(main())
```

## Enhanced Features

### Telemetry & Observability
```python
class MyTelemetry:
    def emit(self, event):
        print(f"Event: {event['event_type']}")
        # Send to your monitoring system

client = SidecarMCPClient.from_dict(config, telemetry=MyTelemetry())
```

### Policy Enforcement
```python
config = {
    "mcpServers": {...},
    "sidecar": {
        "policy": {
            "disallowed_tools": ["file_delete", "system_exec"],
            "allowed_tools": ["file_read", "file_list"]
        }
    }
}
```

## Migration from mcp-use

Agent Orchestra is designed as a **zero-effort migration**:

1. Change imports:
   ```python
   # Before
   from mcp_use import MCPClient, MCPAgent
   
   # After  
   from agent_orchestra import SidecarMCPClient, SidecarMCPAgent
   ```

2. All existing code works unchanged
3. Optionally add sidecar features

## Examples

- [Basic Usage](examples/basic_usage.py) - Simple getting started example
- [Policy Enforcement](examples/policy_enforcement.py) - Safety and tool control
- [Real MCP Servers](examples/basic_mcp_example.py) - Complete working example

## Requirements

- Python 3.8+
- [mcp-use](https://github.com/anthropics/mcp-use) 
- MCP servers (install with npm)

## License

MIT
