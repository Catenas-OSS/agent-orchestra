from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, Optional

from .types import NodeSpec, RunSpec


def _stable(obj: Any) -> Any:
    """Ensure non-serializable types become strings for stable hashing."""
    if isinstance(obj, dict):
        return {k: _stable(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_stable(v) for v in obj]
    try:
        json.dumps(obj)  # Check if serializable
        return obj
    except Exception:
        return str(obj)


def _hash_server_config(server_name: Optional[str], server_configs: Optional[Dict[str, Any]] = None) -> str:
    """Compute hash of server configuration for cache invalidation."""
    if not server_name or not server_configs:
        return "no_server_config"
    
    server_config = server_configs.get(server_name, {})
    if not server_config:
        return "empty_server_config"
    
    # Extract key configuration that affects execution
    config_data = {
        "command": server_config.get("command"),
        "args": server_config.get("args", []),
        "env": server_config.get("env", {}),
        "cwd": server_config.get("cwd"),
    }
    
    # Remove None values and sort for stable hashing
    config_data = {k: v for k, v in config_data.items() if v is not None}
    stable_config = _stable(config_data)
    
    config_str = json.dumps(stable_config, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


def node_signature(
    node: NodeSpec, 
    run_spec: RunSpec, 
    server_configs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate stable cache key for a node based on all inputs that affect execution.
    
    Args:
        node: Node specification containing execution parameters
        run_spec: Run specification containing global context
        server_configs: MCP server configurations (mcpServers dict)
    
    Returns:
        64-character hex string uniquely identifying this node's execution context
    """
    payload: Dict[str, Any] = {
        "node_id": node.id,
        "type": node.type,
        "inputs": _stable(node.inputs),
        "timeout_s": node.timeout_s,
        "retries": node.retries,
        "retry_backoff_s": node.retry_backoff_s,
        "retry_on_timeout": node.retry_on_timeout,
        "concurrency": node.concurrency,
        "foreach_fail_policy": node.foreach_fail_policy,
        "server_name": node.server_name,
        "goal": run_spec.goal,
        "policy": _stable(run_spec.policy),
        "hints": _stable(run_spec.hints),
        "cache_ttl_s": run_spec.cache_ttl_s,
        "max_in_flight": run_spec.max_in_flight,
        "orchestrator_version": "ao@5",
    }
    
    # Enhanced signature factors (your suggested improvements)
    payload.update({
        "model_key": getattr(run_spec, "model_key", None),
        "policy_id": getattr(run_spec, "policy_id", None),
        "server_cfg_hash": _hash_server_config(node.server_name, server_configs),
    })
    
    # Create stable JSON representation
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def foreach_item_signature(
    node: NodeSpec,
    run_spec: RunSpec, 
    item_index: int,
    item_data: Any,
    server_configs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate signature for a specific foreach item to enable granular resume.
    
    Args:
        node: Foreach node specification
        run_spec: Run specification
        item_index: Index of item in the foreach list
        item_data: The specific item data being processed
        server_configs: MCP server configurations
    
    Returns:
        64-character hex string for this specific foreach item
    """
    # Create a virtual sub-node for this specific item
    item_payload: Dict[str, Any] = {
        "parent_node_id": node.id,
        "item_index": item_index,
        "item_data": _stable(item_data),
        "type": "foreach_item",
        "timeout_s": node.timeout_s,
        "retries": node.retries,
        "retry_backoff_s": node.retry_backoff_s,
        "retry_on_timeout": node.retry_on_timeout,
        "server_name": node.server_name,
        "goal": run_spec.goal,
        "policy": _stable(run_spec.policy),
        "hints": _stable(run_spec.hints),
        "orchestrator_version": "ao@5",
    }
    
    # Enhanced factors
    item_payload.update({
        "model_key": getattr(run_spec, "model_key", None),
        "policy_id": getattr(run_spec, "policy_id", None),
        "server_cfg_hash": _hash_server_config(node.server_name, server_configs),
    })
    
    raw = json.dumps(item_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()