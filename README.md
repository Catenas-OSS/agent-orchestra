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

### Run from the Command Line

Use the `catenas` CLI to run your workflow:

```bash
catenas run my_workflow.py
```

Or, run with a live TUI to watch the execution in real-time:

```bash
catenas run my_workflow.py --watch
```

## Key Features

*   **Python-First Workflows:** Define your workflows directly in Python.
*   **Live TUI:** A terminal-based user interface to monitor your workflows in real-time.
*   **Extensible Executor Model:** Bring your own execution logic.
*   **SQLite Persistence:** Automatically persist runs to a local SQLite database.
*   **Run Management:** List, show, and tail runs from the CLI.

## Development

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
