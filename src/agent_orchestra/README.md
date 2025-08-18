# 🎭 Agent Orchestra Core

**Core Implementation of the Universal Multi-Agent Orchestrator** - Python package containing the core orchestration engine, graph execution, and MCP-USE integration.

## 📁 Module Structure

```
src/agent_orchestra/
├── __init__.py          # Public API exports
├── orchestrator.py      # Main orchestration engine
├── graph.py            # DAG specification and validation
├── nodes.py            # Node abstractions and execution
├── events.py           # Event system and JSONL logging
├── checkpointer.py     # State persistence and replay
├── policy.py           # Security, budgets, and HITL
├── cli.py              # CLI interface (catenas command)
├── tools_loader.py     # Tools configuration loader
└── adapters/
    ├── __init__.py     # Adapter system initialization
    ├── base.py         # Adapter protocol and contracts
    └── mcp_use.py      # MCP-USE integration adapter
```

## 🔧 Core Components

### 🎯 Orchestrator (`orchestrator.py`)

The main engine that coordinates multi-agent workflow execution:

- **`Orchestrator`**: Primary class for running graph workflows
- **`RunResult`**: Standardized execution results with metrics
- **DAG Execution**: Async topological sorting and parallel execution
- **Resource Management**: Token, cost, and time budget tracking

```python
from agent_orchestra import Orchestrator

orchestrator = Orchestrator()
result = await orchestrator.run("workflow.json")
```

### 📊 Graph System (`graph.py`)

JSON-based workflow specification and validation:

- **`GraphSpec`**: Pydantic model for graph definitions (frozen contract)
- **`NodeConfig`**: Individual node configuration schema
- **`Graph`**: Runtime graph representation with execution state
- **Validation**: Schema validation and dependency checking

```python
# Graph specification format
{
  "nodes": {
    "agent1": {
      "type": "mcp_agent",
      "adapter": "mcp_use",
      "config": {...}
    }
  },
  "edges": {
    "agent1": ["agent2"]
  },
  "ctx": {...}
}
```

### 🤖 Node System (`nodes.py`)

Abstraction layer for different agent types:

- **`Node`**: Abstract base class for all node types
- **`CompositeNode`**: Sharding work across multiple sub-agents
- **`NodeResult`**: Standardized execution results
- **Factory Functions**: Dynamic node creation based on type

### 📝 Event System (`events.py`)

Comprehensive event logging and observability:

- **`Event`**: Structured event schema (frozen contract)
- **`EventType`**: Enumeration of all event types
- **`EventEmitter`**: Async event publishing
- **`JSONLSink`**: JSONL file logging for replay
- **OpenTelemetry**: Span integration for distributed tracing

```python
# Event types include:
- run.start/end/error
- node.schedule/start/complete/error
- policy.block, budget.exceeded
- checkpoint.saved, tool.usage
```

### 💾 Checkpointing (`checkpointer.py`)

Deterministic state persistence and replay:

- **`CheckpointState`**: Complete execution state snapshot
- **`Checkpointer`**: Save/load checkpoint management
- **Deterministic Context**: RNG seeding and time injection
- **Full Replay**: Exact reproduction of previous runs

### 🛡️ Policy & Security (`policy.py`)

Resource limits, security controls, and human oversight:

- **`Budget`**: Token, cost, children, and time limits
- **`Policy`**: Tool allowlists, domain restrictions, custom rules
- **`HITLManager`**: Human-in-the-Loop approval workflows
- **`PolicyAction`**: Allow, deny, or require approval decisions

```python
budget = Budget(
    max_tokens=1000000,
    max_cost=50.0,
    max_time=3600
)

policy = Policy()
policy.set_allowed_tools({"web_search", "analysis"})
```

### 🔌 Adapter System (`adapters/`)

Pluggable backend integrations:

- **`BaseAdapter`**: Protocol defining adapter interface (frozen contract)
- **`AdapterResult`**: Standardized result format with usage metrics
- **`UsageMetrics`**: Token usage, cost, and latency tracking
- **Registry**: Dynamic adapter loading via entry points

### 💻 CLI Interface (`cli.py`)

Command-line interface for orchestration management:

- **`catenas run`**: Execute graph workflows
- **`catenas tail`**: Real-time event streaming
- **`catenas resume`**: Checkpoint-based resumption
- **`catenas inspect`**: Run analysis and debugging
- **HITL Commands**: Approve/deny pending requests

## 🔄 Execution Flow

```
1. Load GraphSpec from JSON
2. Create Orchestrator with policies/budgets
3. Initialize adapters and event system
4. Execute DAG with topological sorting
5. Stream events to JSONL for observability
6. Save checkpoints for resumption
7. Return RunResult with metrics
```

## 📋 Key Contracts (Frozen APIs)

These interfaces are stable and won't change without major version bumps:

- **GraphSpec**: JSON schema for workflow definitions
- **Event/EventType**: Event system structure
- **BaseAdapter**: Adapter protocol for backends
- **AdapterResult/UsageMetrics**: Standardized result formats
- **CheckpointState**: State persistence format

## 🔧 Extension Points

### Custom Adapters

```python
from agent_orchestra.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    async def execute(self, config, inputs, context):
        # Custom implementation
        return AdapterResult(...)
```

### Custom Node Types

```python
from agent_orchestra.nodes import Node

class MyNode(Node):
    async def execute(self, inputs, context, trace_id, span_id):
        # Custom node logic
        return NodeResult(...)
```

### Policy Rules

```python
policy.add_rule(PolicyRule(
    name="block_dangerous_tools",
    condition="tool:file_delete",
    action=PolicyAction.REQUIRE_APPROVAL
))
```

## 🏗️ Architecture Patterns

- **Event-Driven**: All operations emit structured events
- **Plugin System**: Adapters loaded via entry points
- **Async/Await**: Full async support for scalability
- **Immutable Contracts**: Frozen schemas for stability
- **Observability-First**: OpenTelemetry and JSONL logging
- **Deterministic**: Reproducible execution via checkpoints

## 🧪 Testing

The module includes comprehensive test coverage:

- `test_graph.py`: Graph validation and execution
- `test_events.py`: Event system functionality
- `test_adapters.py`: Adapter protocol compliance
- `test_policy.py`: Policy and budget enforcement

Run tests with:
```bash
pytest tests/
```

## 🔍 Development Guidelines

### Adding New Node Types

1. Extend the `Node` base class
2. Update the `create_node` factory function
3. Add validation to `NodeConfig.validate_type`
4. Document the new type in graph specifications

### Creating Custom Adapters

1. Implement the `BaseAdapter` protocol
2. Register via entry points in `pyproject.toml`
3. Follow the `AdapterResult` format exactly
4. Include comprehensive usage metrics

### Event System Extensions

1. Add new event types to `EventType` enum
2. Maintain backward compatibility
3. Include all required fields in event data
4. Update documentation for new events

## 📊 Performance Considerations

- **Bounded Concurrency**: Configurable max concurrent nodes
- **Memory Management**: Streaming events to disk
- **Checkpoint Frequency**: Configurable checkpoint intervals
- **Budget Enforcement**: Proactive resource monitoring
- **Adapter Pooling**: Reuse adapter instances where possible

## 🛠️ Usage Examples

### Basic Orchestration

```python
from agent_orchestra import Orchestrator, Budget

orchestrator = Orchestrator(
    budget=Budget(max_tokens=100000),
    max_concurrency=5
)

result = await orchestrator.run("workflow.json")
print(f"Success: {result.success}")
print(f"Total tokens: {result.total_tokens}")
```

### Event Streaming

```python
async for event in orchestrator.astream("workflow.json"):
    if event.type == EventType.NODE_COMPLETE:
        print(f"Node {event.data['node_id']} completed")
```

### Checkpoint Resume

```python
checkpointer = Checkpointer("./checkpoints")
checkpoint = checkpointer.load("checkpoint_id")
result = await orchestrator.resume(checkpoint)
```

This core module provides the foundation for scalable, observable, and secure multi-agent orchestration with full deterministic replay capabilities.