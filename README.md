# Agent Orchestra: Production-Ready Multi-Agent Orchestration

**Agent Orchestra** is a production-grade, open-source framework for building sophisticated multi-agent workflows with enterprise-level features. It provides advanced orchestration, rate limiting, agent pooling, and comprehensive observability for real-world AI applications.

## 🚀 Getting Started

### Installation

```bash
pip install agent-orchestra
```

### Your First Workflow

Create a Python file (e.g., `my_workflow.py`) and define your workflow:

```python
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor

# Define the workflow graph
workflow = GraphSpec(
    nodes=[
        NodeSpec(id="hello", type="task", inputs={"text": "Hello"}),
        NodeSpec(id="world", type="task", inputs={"text": "World"}),
    ],
    edges=[("hello", "world")]
)

# Define a simple executor function
async def my_executor(node, ctx):
    print(f"{node.inputs['text']}")
    return {"status": "ok"}

# Define the run specification
run_spec = RunSpec(run_id="my_first_run", goal="A simple hello world workflow")

# Define the executor
executor = CallableExecutor(my_executor)
```

### Run Your Workflow

```bash
# Run with the Agent Orchestra CLI
python -m agent_orchestra.cli_py run my_workflow.py

# Or with live TUI to watch execution in real-time
python -m agent_orchestra.cli_py run my_workflow.py --watch
```

### Quick Start with Examples

```bash
# Try the supervisor website builder (creates a professional website!)
python -m agent_orchestra.cli_py run examples/quickstart/supervisor_website_builder.py --watch

# Or start with the basics
python examples/quickstart/hello_world.py
```

## ✨ Key Features

### 🎭 **Multi-Agent Orchestration**
- **Supervisor Agents**: Coordinate multiple specialists intelligently
- **Agent Pools**: Reuse agents efficiently across workflows  
- **MCP Integration**: Native Model Context Protocol support
- **Real-time Streaming**: Live workflow execution with progress tracking

### 🏭 **Production-Ready**
- **Rate Limiting & Retries**: Built-in resilience and resource management
- **Persistent State**: SQLite-backed checkpointing and resume
- **Comprehensive Logging**: Detailed execution traces and debugging
- **Enterprise Features**: Policy enforcement, telemetry, observability

### 🚀 **Developer Experience**  
- **Python-First**: Define workflows directly in Python code
- **Live TUI**: Beautiful terminal interface for monitoring
- **Rich Examples**: From "Hello World" to complex multi-agent systems
- **CLI Tools**: Simple commands for running and managing workflows

## 📚 Examples & Learning Path

### 🌟 Showcase: AI Website Builder
Our flagship example demonstrates Agent Orchestra's full capabilities:

```bash
python -m agent_orchestra.cli_py run examples/quickstart/supervisor_website_builder.py --watch
```

**What it does**: A supervisor agent coordinates 4 specialists (UI Designer, Frontend Developer, Content Strategist, Technical Architect) to build a professional website for "Nexus Creative Studio". The result is enterprise-quality HTML, CSS, and JavaScript that would impress any client.

### 📖 Learning Path

1. **[Hello World](examples/quickstart/hello_world.py)** - Basic MCP setup
2. **[Restaurant Search](examples/quickstart/restaurant_search.py)** - Single-agent workflow
3. **[Orchestration Basics](examples/quickstart/orchestration_basics.py)** - Multi-node workflows  
4. **[Supervisor Website Builder](examples/quickstart/supervisor_website_builder.py)** - ⭐ **Multi-agent coordination**
5. **[Advanced Examples](examples/advanced/)** - Production patterns, streaming, persistence

### 🎯 Quick Commands

```bash
# See all examples
ls examples/quickstart/
ls examples/advanced/

# Run any example
python -m agent_orchestra.cli_py run examples/quickstart/hello_world.py

# View execution with live TUI
python -m agent_orchestra.cli_py run examples/advanced/multi_agent_example.py --watch
```

See **[examples/README.md](examples/README.md)** for comprehensive documentation.

## 🏗️ Use Cases

**Agent Orchestra excels at:**

- **Multi-Agent Coordination**: Supervisor agents managing specialist teams
- **Complex Workflows**: Multi-step processes with branching and loops  
- **Production AI Applications**: Rate limiting, retries, persistence, observability
- **Creative Tasks**: Content generation, design, development coordination
- **Data Processing**: Parallel processing with result aggregation
- **Tool Integration**: Connecting multiple AI models and external APIs

## 🛠️ Development

### Setup

```bash
git clone https://github.com/Catenas-OSS/agent-orchestra
cd agent-orchestra
pip install -e .[dev]
```

### Run Tests

```bash
pytest
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
