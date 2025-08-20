# Agent Orchestra Examples

This directory contains examples showing how to use Agent Orchestra as a drop-in replacement for mcp-use with enhanced enterprise features.

## Prerequisites

```bash
# Install dependencies
pip install mcp-use
npm install -g @modelcontextprotocol/server-filesystem
```

## Examples

### 📋 `basic_usage.py`
**Simple getting started example**

Shows basic Agent Orchestra usage with telemetry and policy features.

```bash
python3 basic_usage.py
```

### 🔒 `policy_enforcement.py` 
**Policy and safety demonstration**

Shows how Agent Orchestra enforces tool usage policies while maintaining mcp-use compatibility.

```bash
python3 policy_enforcement.py
```

### 🏗️ `basic_mcp_example.py`
**Real MCP servers example**

Complete example using actual MCP servers with enhanced features.

```bash
python3 basic_mcp_example.py
```

### 🧪 `full_compatibility_test.py`
**Compatibility validation**

Comprehensive test validating 100% API compatibility with mcp-use.

```bash
python3 full_compatibility_test.py
```

### 🎭 `end_to_end_part3_demo.py`
**Part 3 Orchestrator Features Demo** *(opt-in, requires API key)*

Complete demonstration of Part 3 advanced orchestrator features with real MCP integration:
- Gate-based conditional flow control
- Foreach processing with concurrency limits  
- Reduce operations for data aggregation
- Real-time streaming with AGENT_CHUNK events
- Retry logic with exponential backoff
- TaskGroup structured concurrency

```bash
OPENAI_API_KEY=<your-key> python3 end_to_end_part3_demo.py
```

### 🧪 `simple_mcp_orchestrator_test.py`  
**MCP Integration Test** *(opt-in, requires API key)*

Focused integration test verifying MCPExecutor + Orchestrator + streaming works correctly with real MCP servers.

```bash
OPENAI_API_KEY=<your-key> python3 simple_mcp_orchestrator_test.py
```

## Key Features

### Drop-in Compatibility
```python
# Same API as mcp-use
from agent_orchestra import SidecarMCPClient, SidecarMCPAgent

client = SidecarMCPClient.from_dict(config)
agent = SidecarMCPAgent(client=client, model="gpt-4o")
result = await agent.run("Your query")
```

### Enhanced Features
```python
# Add telemetry and policies
client = SidecarMCPClient.from_dict(config,
    telemetry=my_telemetry,
    policy={"disallowed_tools": ["dangerous_tool"]}
)
```

## Migration from mcp-use

1. **Change imports:**
   ```python
   # Before
   from mcp_use import MCPClient, MCPAgent
   
   # After  
   from agent_orchestra import SidecarMCPClient, SidecarMCPAgent
   ```

2. **Optionally add sidecar config:**
   ```python
   config["sidecar"] = {
       "policy": {"disallowed_tools": ["risky_tool"]},
       "run_context": {"environment": "production"}
   }
   ```

3. **All existing code works unchanged**