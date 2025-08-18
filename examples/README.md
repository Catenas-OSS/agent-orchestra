# Agent Orchestra Examples

This directory contains example configurations demonstrating Agent Orchestra capabilities with mcp-use integration.

## Quick Start

```bash
# Basic hello world workflow
python -m agent_orchestra.cli run examples/hello_world.json \
  --tools examples/simple_working_tools.yaml \
  --agents examples/agents/

# Demo with custom configurations  
python examples/demo.py
```

## Example Files

### Workflows
- `hello_world.json` - 5-node workflow demonstrating basic orchestration
- `simple_graph.json` - Minimal 2-node workflow for testing

### Tools Configuration
- `simple_working_tools.yaml` - Basic STDIO MCP server setup
- `http_tools.yaml` - HTTP/SSE transport examples
- `tools_with_readiness.yaml` - Server health check configurations

### Agent Configurations (`agents/`)
- `executor@v1.yaml` - General-purpose execution agent
- `planner@v1.yaml` - Strategic planning agent  
- `researcher@v1.yaml` - Research and analysis agent
- `writer@v1.yaml` - Content generation agent
- `judge@v1.yaml` - Evaluation and scoring agent
- `simple@v1.yaml` - Basic agent for testing
- `restricted@v1.yaml` - Agent with tool access controls
- `simple_no_manager@v1.yaml` - Agent without Server Manager

## Transport Types

### STDIO (Default)
```yaml
mcpServers:
  my_server:
    command: "npx"
    args: ["@modelcontextprotocol/server-everything"]
```

### HTTP/SSE  
```yaml
mcpServers:
  api_server:
    url: "https://api.example.com/mcp"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    timeout_s: 30
```

## Agent Features

- **Server Manager**: Dynamic tool discovery with semantic search
- **Tool Access Control**: Allow/disallow lists with precedence rules
- **Streaming Support**: Real-time execution monitoring
- **Caching**: Performance optimization for repeated operations
- **Readiness Probing**: Health checks with configurable timeouts

## Environment Variables

Set these before running:
```bash
export OPENAI_API_KEY="your-key-here"
export API_TOKEN="your-api-token"  # If using HTTP examples
```