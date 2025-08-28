# Agent Orchestra Examples

This directory contains examples demonstrating Agent Orchestra's capabilities, organized from basic to advanced usage patterns.

## Quick Start Examples (`quickstart/`)

Perfect for getting started with Agent Orchestra:

- **[hello_world.py](quickstart/hello_world.py)** - Basic Agent Orchestra setup and usage
- **[restaurant_search.py](quickstart/restaurant_search.py)** - Simple single-agent workflow  
- **[orchestration_basics.py](quickstart/orchestration_basics.py)** - Multi-node workflow orchestration
- **[supervisor_website_builder.py](quickstart/supervisor_website_builder.py)** - ⭐ **Supervisor agent coordinating multiple specialists**

## Advanced Examples (`advanced/`)

For complex orchestration patterns and production use cases:

- **[multi_agent_example.py](advanced/multi_agent_example.py)** - Coordinating multiple agents with different capabilities
- **[streaming_orchestration.py](advanced/streaming_orchestration.py)** - Real-time streaming workflow execution
- **[broker_config_example.py](advanced/broker_config_example.py)** - Advanced broker configuration and rate limiting
- **[persistence_and_resume.py](advanced/persistence_and_resume.py)** - Workflow persistence and resumption patterns

## Integration Examples (`integrations/`)

Connecting Agent Orchestra with external services:

- **[github_tool_search.py](integrations/github_tool_search.py)** - GitHub API integration example

## Workflow Graphs (`graphs/`)

JSON workflow definitions for common patterns:

- `simple.json` - Basic linear workflow
- `complex.json` - Multi-branch conditional workflow  
- `foreach.json` - Parallel processing patterns
- `file_processing.json` - File manipulation workflows
- `mcp_demo.json` - MCP server integration patterns

## Prerequisites

1. **Python 3.8+** with Agent Orchestra installed
2. **API Keys** - Set required environment variables:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export BRAVE_API_KEY="your-key-here"  # For search examples
   ```
3. **Node.js** - Required for MCP filesystem servers
4. **Dependencies** - Install as needed:
   ```bash
   pip install langchain-openai  # For LLM examples
   ```

## Running Examples

### Command Line Interface

```bash
# Run any example with the CLI
python -m agent_orchestra.cli_py run examples/quickstart/supervisor_website_builder.py --watch

# Run with specific configuration
python -m agent_orchestra.cli_py run examples/advanced/multi_agent_example.py --config config.json
```

### Direct Python Execution

```bash
# Most examples can be run directly
python examples/quickstart/hello_world.py

# Or imported for programmatic use
python -c "from examples.quickstart.supervisor_website_builder import graph_spec, run_spec, executor"
```

## Example Categories Explained

### 🚀 Quickstart
- **Purpose**: Learn core concepts quickly
- **Complexity**: Low to medium  
- **Time to run**: < 5 minutes each
- **Best for**: New users, demos, proof of concepts

### 🔧 Advanced
- **Purpose**: Production-ready patterns
- **Complexity**: High
- **Time to run**: 5-30 minutes each  
- **Best for**: Building real applications, complex workflows

### 🔌 Integrations
- **Purpose**: Connect with external services
- **Complexity**: Medium
- **Requirements**: API keys, external accounts
- **Best for**: Real-world applications, tool chains

## Recommended Learning Path

1. **Start here**: `quickstart/hello_world.py`
2. **Learn orchestration**: `quickstart/orchestration_basics.py`
3. **See the magic**: `quickstart/supervisor_website_builder.py` ⭐
4. **Scale up**: `advanced/multi_agent_example.py`
5. **Add persistence**: `advanced/persistence_and_resume.py`

## Contributing Examples

When adding new examples:

1. **Follow naming conventions**: Use descriptive, action-oriented names
2. **Include comprehensive docstrings**: Explain purpose, requirements, and usage
3. **Add error handling**: Graceful failure with helpful messages
4. **Test thoroughly**: Ensure examples work across different environments
5. **Update this README**: Add your example to the appropriate section

## Getting Help

- **Documentation**: See project README and docs/ directory
- **Issues**: Report problems on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions

---

*Start with the quickstart examples and work your way up! Each example builds on concepts from the previous ones.*