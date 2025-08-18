# Agent Orchestra - Claude Assistant Documentation

## Project Overview

**Agent Orchestra** is a Universal Multi-Agent Orchestrator that provides the easiest way to create and orchestrate multi-agent fleets using MCP-USE (Model Context Protocol). The project enables users to define complex workflows as directed acyclic graphs (DAGs) and execute them with full observability, checkpointing, and policy controls.

### Key Features
- 🔌 **MCP-USE Integration**: Seamless integration with mcp-use for agent management
- 📊 **DAG Execution**: Async directed acyclic graph execution with topological sorting
- 🔄 **Deterministic Replay**: Full checkpoint and replay capability for debugging
- 💰 **Budget Management**: Token, cost, and time budgets with automatic limits
- 🛡️ **Security & Governance**: HITL (Human-in-the-Loop) approvals and policy controls
- 📈 **Observability**: OpenTelemetry spans and JSONL event logging
- 🔧 **Plugin System**: Extensible adapter system via entry points
- ⚡ **Scalable**: Support for 10k+ concurrent agents with bounded concurrency

## Project Structure

```
src/agent_orchestra/
├── __init__.py          # Public API exports
├── orchestrator.py      # Main orchestration engine
├── graph.py            # DAG specification and validation
├── nodes.py            # Node abstractions and implementations
├── events.py           # Event system and logging
├── checkpointer.py     # State persistence and replay
├── policy.py           # Security policies and budget management
├── cli.py              # Command-line interface
└── adapters/
    ├── base.py         # Adapter protocol and registry
    └── mcp_use.py      # MCP-USE integration adapter
```

## Code Quality Standards

### Current Documentation Status ✅

The codebase follows **excellent documentation standards** with:

- **Comprehensive module docstrings**: Every module has detailed docstrings explaining purpose and scope
- **Complete class documentation**: All classes include docstrings with clear descriptions
- **Full method documentation**: All public methods have Google-style docstrings with Args, Returns, and Raises sections
- **Type safety**: Full type annotations throughout with strict mypy configuration
- **Consistent style**: Professional formatting and naming conventions

### Documentation Examples

**Module Level** (from `orchestrator.py:1-4`):
```python
"""Main orchestration engine.

Provides the core Orchestrator class with run/astream/resume capabilities.
"""
```

**Class Level** (from `orchestrator.py:48`):
```python
class Orchestrator:
    """Main orchestration engine for multi-agent workflows."""
```

**Method Level** (from `orchestrator.py:81-100`):
```python
async def run(
    self,
    graph: Graph | dict[str, Any] | str | Path,
    ctx: dict[str, Any] | None = None,
    run_id: str | None = None
) -> RunResult:
    """Execute a graph and return the final result.
    
    Args:
        graph: Graph specification as Graph object, dict, or file path.
        ctx: Additional context to merge with graph context.
        run_id: Custom run ID, generates UUID if not provided.
        
    Returns:
        RunResult with execution metrics and outputs.
        
    Raises:
        ValueError: If graph specification is invalid.
        RuntimeError: If execution fails unexpectedly.
    """
```

### Type Safety Implementation ✅

The project uses **strict type checking** with comprehensive type annotations:

- **Pydantic models** for data validation (GraphSpec, NodeConfig)
- **Union types** for flexible parameters (`str | Path`, `dict | None`)
- **Generic collections** with proper typing (`dict[str, Any]`, `list[str]`)
- **Optional types** clearly marked (`str | None`)
- **Return type annotations** on all functions
- **Strict mypy configuration** in pyproject.toml

## Development Guidelines

### Adding New Features

1. **Follow existing patterns**: Use the established docstring and type annotation style
2. **Maintain contracts**: GraphSpec and Event schemas are frozen contracts - changes require major version bump
3. **Add comprehensive tests**: Include unit tests for new functionality
4. **Update documentation**: Ensure all new code includes proper docstrings

### Code Quality Checks

Run these commands before committing:

```bash
# Type checking
mypy src/

# Linting  
ruff check src/

# Tests
pytest

# Build verification
python -m build
```

### Security Considerations

- **No secrets in code**: Never include API keys, tokens, or credentials
- **Input validation**: All external inputs must be validated through Pydantic models
- **Resource limits**: All operations respect budget constraints
- **Policy enforcement**: Tool usage and domain access controlled by policies

## Architecture Overview

The system follows a **layered architecture**:

1. **CLI Layer** (`cli.py`): User interface and command processing
2. **Orchestration Layer** (`orchestrator.py`): Core execution engine
3. **Graph Layer** (`graph.py`): DAG specification and validation
4. **Node Layer** (`nodes.py`): Individual agent abstractions
5. **Adapter Layer** (`adapters/`): Integration with external agent systems
6. **Infrastructure Layer**: Events, checkpointing, and policies

### Key Components

- **Orchestrator**: Main engine that executes graphs with full observability
- **Graph/GraphSpec**: JSON schema for defining workflows (frozen contract)
- **Event System**: Structured logging with OpenTelemetry integration
- **Checkpointer**: State persistence for deterministic replay
- **Policy/Budget**: Resource management and security controls
- **Adapters**: Pluggable interfaces to different agent backends

## Common Tasks

### Working with Documentation

When modifying the codebase:

1. **Always maintain docstrings** - follow the Google style format
2. **Update type annotations** - ensure all parameters and returns are typed
3. **Validate with mypy** - run `mypy src/` to catch type issues
4. **Keep contracts stable** - GraphSpec and Event schemas should not change without major version bump

### Testing

- Unit tests in `tests/` directory
- Use `pytest-asyncio` for async testing
- Mock external dependencies appropriately
- Test both success and error paths

### Performance Considerations

- Use async/await for I/O operations
- Respect concurrency limits (`max_concurrency`)
- Implement proper resource cleanup
- Monitor budget consumption

## Frozen Contracts

These APIs should **NOT** change without major version bumps:

- **GraphSpec schema** (`graph.py`): JSON structure for defining workflows
- **Event schema** (`events.py`): Event types and structure for logging
- **Public API exports** (`__init__.py`): Main classes and functions

## Extension Points

The system is designed for extensibility:

- **Custom Adapters**: Implement `BaseAdapter` protocol for new agent backends
- **Custom Nodes**: Extend node types beyond `mcp_agent`, `composite`, `function`
- **Custom Policies**: Add new policy rules and budget constraints
- **Custom Events**: Extend event types for domain-specific logging

## Best Practices

1. **Use the existing patterns** - follow established code style and structure
2. **Maintain type safety** - always include proper type annotations  
3. **Document thoroughly** - include comprehensive docstrings for all public APIs
4. **Test comprehensively** - add tests for new functionality
5. **Respect budgets** - ensure all operations consume appropriate resources
6. **Handle errors gracefully** - provide meaningful error messages and recovery

This codebase represents a **high-quality, production-ready** system with excellent documentation standards, comprehensive type safety, and thoughtful architecture design.