"""Graph specification and execution logic.

Defines the JSON schema for graphs and provides DAG execution capabilities.
This module contains frozen contracts that should not change without major version bumps.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class NodeConfig(BaseModel):
    """Configuration for a single node in the graph."""

    type: str = Field(..., description="Node type: mcp_agent, composite, or function")
    adapter: str = Field(default="mcp_use", description="Adapter to use for this node")
    config: dict[str, Any] = Field(default_factory=dict, description="Node-specific configuration")
    timeout: int | None = Field(default=None, description="Timeout in seconds")
    retries: int = Field(default=0, description="Number of retries on failure")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed_types = {'mcp_agent', 'composite', 'function'}
        if v not in allowed_types:
            raise ValueError(f"Node type must be one of {allowed_types}")
        return v


class GraphSpec(BaseModel):
    """JSON specification for a graph (frozen contract)."""

    nodes: dict[str, NodeConfig] = Field(..., description="Node definitions by ID")
    edges: dict[str, list[str]] = Field(..., description="Adjacency list: from_node -> [to_nodes]")
    ctx: dict[str, Any] = Field(default_factory=dict, description="Initial context/inputs")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Graph metadata")

    @field_validator('edges')
    @classmethod
    def validate_edges(cls, v: dict[str, list[str]], info) -> dict[str, list[str]]:
        """Validate that all edge references point to valid nodes."""
        if not hasattr(info, 'data') or 'nodes' not in info.data:
            return v

        nodes = info.data['nodes']
        for from_node, to_nodes in v.items():
            if from_node not in nodes:
                raise ValueError(f"Edge from unknown node: {from_node}")
            for to_node in to_nodes:
                if to_node not in nodes:
                    raise ValueError(f"Edge to unknown node: {to_node}")
        return v

    def validate_dag(self) -> None:
        """Validate that the graph is a DAG (no cycles)."""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            if node in rec_stack:
                raise ValueError(f"Cycle detected involving node: {node}")
            if node in visited:
                return True

            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.edges.get(node, []):
                if not dfs(neighbor):
                    return False

            rec_stack.remove(node)
            return True

        for node in self.nodes:
            if node not in visited:
                dfs(node)

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order."""
        self.validate_dag()

        in_degree = {node: 0 for node in self.nodes}
        for _, to_nodes in self.edges.items():
            for to_node in to_nodes:
                in_degree[to_node] += 1

        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in self.edges.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")

        return result

    def get_dependencies(self, node: str) -> set[str]:
        """Get all nodes that this node depends on."""
        deps = set()
        for from_node, to_nodes in self.edges.items():
            if node in to_nodes:
                deps.add(from_node)
                deps.update(self.get_dependencies(from_node))
        return deps

    def get_dependents(self, node: str) -> set[str]:
        """Get all nodes that depend on this node."""
        return set(self.edges.get(node, []))


@dataclass
class Graph:
    """Runtime graph representation with execution state."""

    spec: GraphSpec
    run_id: str
    trace_id: str
    created_at: float = field(default_factory=lambda: __import__('time').time())

    def __post_init__(self) -> None:
        """Validate graph on creation."""
        self.spec.validate_dag()

    @classmethod
    def from_json(cls, json_str: str, run_id: str, trace_id: str) -> "Graph":
        """Create graph from JSON string."""
        data = json.loads(json_str)
        spec = GraphSpec(**data)
        return cls(spec=spec, run_id=run_id, trace_id=trace_id)

    @classmethod
    def from_file(cls, path: str | Path, run_id: str, trace_id: str) -> "Graph":
        """Create graph from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read(), run_id=run_id, trace_id=trace_id)

    def to_json(self) -> str:
        """Serialize graph spec to JSON."""
        return self.spec.model_dump_json(indent=2)

    def get_node_config(self, node_id: str) -> NodeConfig:
        """Get configuration for a specific node."""
        if node_id not in self.spec.nodes:
            raise ValueError(f"Node not found: {node_id}")
        return self.spec.nodes[node_id]

    def get_execution_order(self) -> list[str]:
        """Get nodes in execution order (topological sort)."""
        return self.spec.topological_sort()

