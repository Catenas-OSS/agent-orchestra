# 🎭 Agent Orchestra

**Universal Multi-Agent Orchestrator** - The easiest way to create and orchestrate multi-agent fleets using MCP-USE.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- **🔌 Full MCP-USE Integration**: Complete integration with [mcp-use](https://github.com/mcp-use/mcp-use) including:
  - **🌐 All Transport Types**: STDIO, HTTP, and SSE support
  - **🧠 Server Manager**: Dynamic tool discovery with semantic search
  - **🎯 Tool Access Control**: Allow/disallow lists with precedence rules
  - **⚡ Streaming Support**: Real-time execution with `agent.astream()`
  - **🏥 Health Checks**: Configurable readiness probing and timeouts
- **📊 DAG Execution**: Async directed acyclic graph execution with topological sorting
- **🔄 Deterministic Replay**: Full checkpoint and replay capability for debugging
- **💰 Budget Management**: Token, cost, and time budgets with automatic limits
- **🛡️ Security & Governance**: HITL (Human-in-the-Loop) approvals and policy controls
- **📈 Observability**: OpenTelemetry spans and JSONL event logging
- **🔧 Plugin System**: Extensible adapter system via entry points
- **⚡ Scalable**: Support for 10k+ concurrent agents with bounded concurrency

## 🚀 Quick Start

### Installation

**From GitHub (recommended for now):**

```bash
pip install git+https://github.com/your-username/agent-orchestra.git
```

**For development:**

```bash
git clone https://github.com/your-username/agent-orchestra.git
cd agent-orchestra
pip install -e .[dev]
```

**From PyPI (coming soon):**

```bash
pip install agent-orchestra
```

### Hello World Example

1. **Set up your environment**:

```bash
# Install with mcp-use support
pip install "agent-orchestra[mcp-use]"

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

2. **Run the included example**:

```bash
python -m agent_orchestra.cli run examples/hello_world.json \
  --tools examples/simple_working_tools.yaml \
  --agents examples/agents/
```

3. **Or use programmatically**:

```python
import asyncio
from agent_orchestra import Orchestrator

async def main():
    orchestrator = Orchestrator()
    result = await orchestrator.run(
        graph="examples/hello_world.json",
        tools_file="examples/simple_working_tools.yaml", 
        agents_path="examples/agents/"
    )
    print(f"Success: {result.success}")
    print(f"Outputs: {result.outputs}")

asyncio.run(main())
```

### MCP Transport Examples

**STDIO Transport (most common):**
```yaml
# simple_working_tools.yaml
mcpServers:
  everything:
    command: "npx"
    args: ["@modelcontextprotocol/server-everything"]
    timeout_s: 30
```

**HTTP/SSE Transport:**
```yaml
# http_tools.yaml
mcpServers:
  api_server:
    url: "https://api.example.com/mcp"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
    timeout_s: 30
```

## 📚 Core Concepts

### 🔗 Graph Specification

Agent Orchestra uses JSON graphs to define workflows:

- **Nodes**: Individual agents or processing units
- **Edges**: Dependencies between nodes
- **Context**: Initial data and configuration
- **Metadata**: Graph information and settings

### 🤖 Node Types

- **`mcp_agent`**: Wraps an MCP-USE agent
- **`composite`**: Shards work across multiple sub-agents
- **`function`**: Executes Python functions

### 🔌 Adapters

Adapters provide clean interfaces to different agent backends:

- **`mcp_use`**: Integration with mcp-use package (default)
- **Custom adapters**: Extensible via entry points

### 📋 Event System

All executions generate structured events:

- **Run lifecycle**: start, end, error
- **Node lifecycle**: schedule, start, complete, error
- **Policy events**: budget exceeded, HITL required
- **System events**: checkpoints, tool usage

## 🛠️ CLI Commands

### Basic Operations

```bash
# Run a graph
catenas run graph.json

# Run with custom context
catenas run graph.json --context '{"key": "value"}'

# Tail events in real-time
catenas tail <run_id>

# Resume from checkpoint
catenas resume <checkpoint_id>
```

### Monitoring & Debugging

```bash
# Inspect a run
catenas inspect <run_id>

# List checkpoints
catenas list-checkpoints

# Filter events by type
catenas tail <run_id> --filter "node.complete"
```

### HITL Approvals

```bash
# Approve a pending request
catenas approve <run_id> <node_id> --reason "Looks good"

# Deny a request
catenas deny <run_id> <node_id> --reason "Security concern"
```

## 🔧 Configuration

### Budget Limits

```python
from agent_orchestra import Budget

budget = Budget(
    max_tokens=1000000,  # 1M tokens
    max_cost=50.0,       # $50
    max_children=1000,   # 1000 sub-agents
    max_time=3600        # 1 hour
)

orchestrator = Orchestrator(budget=budget)
```

### Security Policies

```python
from agent_orchestra import Policy, PolicyRule, PolicyAction

policy = Policy()
policy.set_allowed_tools({"web_search", "analysis"})
policy.set_allowed_domains({"example.com", "trusted-site.org"})

policy.add_rule(PolicyRule(
    name="block_file_ops",
    condition="tool:file_write",
    action=PolicyAction.DENY,
    reason="File operations not allowed"
))
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI (catenas)  │    │   Orchestrator   │    │   MCP Agents    │
│                 │───▶│                  │───▶│                 │
│ • run           │    │ • DAG execution  │    │ • mcp-use       │
│ • tail          │    │ • Checkpointing  │    │ • Custom tools  │
│ • inspect       │    │ • Event logging  │    │ • Web, analysis │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Storage Layer  │
                       │                  │
                       │ • JSONL events   │
                       │ • JSON checkpts  │
                       │ • SQLite (future)│
                       └──────────────────┘
```

## 🧪 Examples

Check out the `examples/` directory:

- **`hello_world.json`**: Simple 5-node workflow
- **`demo.py`**: Programmatic usage example
- **`complex_research.json`**: Multi-stage research pipeline

## 🔍 Development

### Running Tests

```bash
pytest
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
```

### Building

```bash
python -m build
```

## 📦 Package Structure

```
src/agent_orchestra/
├── __init__.py          # Public API
├── orchestrator.py      # Main engine
├── graph.py            # DAG specification
├── nodes.py            # Node abstractions
├── events.py           # Event system
├── checkpointer.py     # State persistence
├── policy.py           # Security & budgets
├── cli.py              # CLI interface
└── adapters/
    ├── base.py         # Adapter protocol
    └── mcp_use.py      # MCP-USE integration
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [mcp-use](https://github.com/mcp-use/mcp-use) for the excellent MCP agent framework
- The broader MCP (Model Context Protocol) community
- All contributors and users of Agent Orchestra
