"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_telemetry():
    """Mock telemetry collector."""
    telemetry = Mock()
    telemetry.emit = Mock()
    return telemetry


@pytest.fixture
def sample_config():
    """Sample MCP configuration for testing."""
    return {
        "mcpServers": {
            "test_server": {
                "command": "echo",
                "args": ["test"]
            },
            "another_server": {
                "command": "python",
                "args": ["-m", "some_module"]
            }
        }
    }


@pytest.fixture
def sidecar_config():
    """Sample configuration with sidecar block."""
    return {
        "mcpServers": {
            "test_server": {
                "command": "echo", 
                "args": ["test"]
            }
        },
        "sidecar": {
            "policy": {
                "allowed_tools": ["safe_tool"],
                "disallowed_tools": ["dangerous_tool"]
            },
            "run_context": {
                "trace_id": "test-trace-123"
            }
        }
    }


@pytest.fixture
def mock_tool_info():
    """Mock tool information for adapter testing."""
    return [
        {
            "name": "file_read",
            "description": "Read a file from disk",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                }
            }
        },
        {
            "name": "file_delete", 
            "description": "Delete a file permanently",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                }
            }
        },
        {
            "name": "web_fetch",
            "description": "Fetch content from a URL",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                }
            }
        }
    ]