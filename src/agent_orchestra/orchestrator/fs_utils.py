"""
Filesystem utilities for Agent Orchestra.

Provides safe path handling for MCP filesystem servers.
"""

from pathlib import Path
from typing import Dict, Any


def fs_args(root: Path, rel: str) -> Dict[str, str]:
    """
    Create safe filesystem arguments for MCP server calls.
    
    Args:
        root: The root directory for the MCP filesystem server
        rel: Relative path within the root
        
    Returns:
        Dictionary with safe path argument for MCP calls
        
    Raises:
        AssertionError: If the resolved path is outside the root
    """
    # Sanity check: ensure target stays inside root
    resolved_path = (root / rel).resolve()
    root_resolved = root.resolve()
    
    if not str(resolved_path).startswith(str(root_resolved)):
        raise ValueError(f"Path '{rel}' resolves to '{resolved_path}' which is outside root '{root_resolved}'")
    
    return {"path": rel}


def ensure_relative_path(path: str) -> str:
    """
    Ensure a path is relative (not absolute).
    
    Args:
        path: Input path (may be absolute or relative)
        
    Returns:
        Relative path string
    """
    p = Path(path)
    if p.is_absolute():
        # Convert absolute to relative by taking just the name
        return p.name
    return str(p)


def copy_files_to_root(files: Dict[str, Any], root: Path) -> Dict[str, str]:
    """
    Copy files into the filesystem root and return relative paths.
    
    Args:
        files: Dictionary of filename -> content
        root: Root directory to copy files into
        
    Returns:
        Dictionary mapping original names to relative paths
    """
    import json
    
    copied_files = {}
    
    for filename, content in files.items():
        # Ensure filename is safe
        safe_filename = Path(filename).name
        target_path = root / safe_filename
        
        # Write content to file
        if isinstance(content, dict):
            target_path.write_text(json.dumps(content, indent=2))
        else:
            target_path.write_text(str(content))
        
        copied_files[filename] = safe_filename
    
    return copied_files


def validate_mcp_filesystem_config(root: Path) -> Dict[str, Any]:
    """
    Create a validated MCP filesystem server configuration.
    
    Args:
        root: Root directory for the filesystem server
        
    Returns:
        MCP server configuration dictionary
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    
    return {
        "command": "npx",
        "args": [
            "-y", 
            "@modelcontextprotocol/server-filesystem",
            "--stdio",
            "--root",
            str(root)
        ]
    }


def create_multi_server_config(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a multi-server MCP configuration.
    
    Args:
        configs: Dictionary mapping server names to their configurations
        
    Returns:
        Complete MCP configuration with mcpServers section
        
    Example:
        configs = {
            "fs_sales": {"root": "/tmp/sales"},
            "fs_ops": {"root": "/tmp/ops"},
            "playwright": {"type": "playwright"}
        }
    """
    mcp_servers = {}
    
    for server_name, config in configs.items():
        if config.get("type") == "playwright":
            mcp_servers[server_name] = {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-playwright", "--stdio"]
            }
        elif "root" in config:
            # Filesystem server
            root = Path(config["root"]).resolve()
            root.mkdir(parents=True, exist_ok=True)
            mcp_servers[server_name] = validate_mcp_filesystem_config(root)
        else:
            # Custom server config
            mcp_servers[server_name] = config
    
    return {"mcpServers": mcp_servers}