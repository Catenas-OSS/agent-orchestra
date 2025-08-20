# 🚀 Agentic Orchestra Release Checklist

This checklist is intended to be used as a guide for preparing a new release of Agentic Orchestra.

## ✅ Pre-Release Validation

### Core Integration Tests
- [ ] **mcp-use Integration**: All transport types working (STDIO, HTTP, SSE)
- [ ] **Server Manager**: Semantic tool search and discovery working
- [ ] **Agent Caching**: Comprehensive cache key validation
- [ ] **Tool Access Control**: Allow/disallow list precedence working
- [ ] **Streaming Support**: `agent.astream()` with fallback implemented
- [ ] **Readiness Probing**: Health checks with configurable timeouts
- [ ] **Resource Lifecycle**: Proper cleanup with `close_all_sessions()`

### Configuration Validation
- [ ] **All Example Configs**: YAML/JSON files parse correctly
- [ ] **Agent Specs**: All agents in `examples/agents.json` load successfully
- [ ] **Transport Configs**: HTTP, STDIO, and mixed configurations work
- [ ] **Schema Validation**: Tools/agents validation working

### Documentation
- [ ] **README Updated**: Reflects all new features and changes.
- [ ] **Examples Documentation**: `examples/README.md` is up-to-date.
- [ ] **API Documentation**: All public APIs are documented with docstrings.

### Code Quality
- [ ] **Dependencies**: `pyproject.toml` is up-to-date.
- [ ] **Gitignore**: Covers all temporary/generated files.
- [ ] **Type Safety**: No critical type errors in core modules (`mypy src/`).
- [ ] **Linting**: No linting errors (`ruff check src/`).
- [ ] **Tests**: All tests pass (`pytest`).

### Community & Documentation
- [ ] **CONTRIBUTING.md**: Guidelines are up-to-date.
- [ ] **CODE_OF_CONDUCT.md**: Is present and linked correctly.

## 🏗️ Implementation Status

### ✅ Complete Features
1.  **Full MCP Transport Support**
    -   STDIO, HTTP/SSE, and mixed configurations.
2.  **Server Manager Integration**
    -   Dynamic tool discovery and management tools.
3.  **Agent Intelligence**
    -   Tool access control with allow/deny lists.
    -   Streaming support with async generators.
4.  **Production Features**
    -   Event-driven observability.
    -   Stateful checkpointing and replay.

### ⚠️ Known Limitations
- **MCPAgent `allowed_tools` Parameter**: Not yet supported by mcp-use library (fallback implemented).
- **Network Dependencies**: Requires internet for OpenAI API calls.
- **Node.js Dependencies**: STDIO transport requires `npx` for many servers.

## 📋 Release Package Contents

### Core Files
```
src/agent_orchestra/
├── __init__.py
├── orchestrator/
│   ├── __init__.py
│   ├── agent_pool.py
│   ├── broker_config.py
│   ├── call_broker.py
│   ├── core.py
│   ├── executors_mcp.py
│   ├── executors.py
│   ├── fs_utils.py
│   ├── types.py
│   └── utils.py
└── sidecar/
    ├── __init__.py
    ├── sidecar_adapter.py
    ├── sidecar_agent.py
    ├── sidecar_client.py
    └── sidecar_session.py
```

### Examples
```
examples/
├── README.md
├── agents.json
├── basic/
├── advanced/
└── integrations/
```

## 🚀 Next Steps for Release

1.  **Final Testing**: Run full workflow with real API keys.
2.  **Version Tagging**: Tag release in git (`git tag -a vX.Y.Z -m "Release vX.Y.Z"`).
3.  **PyPI Upload**: Build and upload to Python Package Index (`python -m build && twine upload dist/*`).
4.  **Documentation**: Update any final links/references.
5.  **Announcement**: Prepare release notes highlighting new features and changes.
