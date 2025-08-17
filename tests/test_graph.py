"""Tests for graph module."""

import pytest
from pydantic import ValidationError

from agent_orchestra.graph import GraphSpec, Graph, NodeConfig


def test_node_config_validation():
    """Test NodeConfig validation."""
    # Valid config
    config = NodeConfig(type="mcp_agent", adapter="mcp_use")
    assert config.type == "mcp_agent"
    assert config.adapter == "mcp_use"
    
    # Invalid type
    with pytest.raises(ValidationError):
        NodeConfig(type="invalid_type")


def test_graph_spec_basic():
    """Test basic GraphSpec functionality."""
    spec = GraphSpec(
        nodes={
            "node1": NodeConfig(type="mcp_agent"),
            "node2": NodeConfig(type="mcp_agent")
        },
        edges={"node1": ["node2"]},
        ctx={"test": "value"}
    )
    
    assert "node1" in spec.nodes
    assert "node2" in spec.nodes
    assert spec.edges["node1"] == ["node2"]
    assert spec.ctx["test"] == "value"


def test_graph_spec_edge_validation():
    """Test edge validation in GraphSpec."""
    # Valid edges
    spec = GraphSpec(
        nodes={
            "a": NodeConfig(type="mcp_agent"),
            "b": NodeConfig(type="mcp_agent")
        },
        edges={"a": ["b"]}
    )
    assert spec.edges == {"a": ["b"]}
    
    # Invalid edge - references non-existent node
    with pytest.raises(ValidationError):
        GraphSpec(
            nodes={"a": NodeConfig(type="mcp_agent")},
            edges={"a": ["nonexistent"]}
        )


def test_dag_validation():
    """Test DAG cycle detection."""
    # Valid DAG
    spec = GraphSpec(
        nodes={
            "a": NodeConfig(type="mcp_agent"),
            "b": NodeConfig(type="mcp_agent"),
            "c": NodeConfig(type="mcp_agent")
        },
        edges={"a": ["b"], "b": ["c"]}
    )
    spec.validate_dag()  # Should not raise
    
    # Cycle detection
    spec_with_cycle = GraphSpec(
        nodes={
            "a": NodeConfig(type="mcp_agent"),
            "b": NodeConfig(type="mcp_agent")
        },
        edges={"a": ["b"], "b": ["a"]}
    )
    
    with pytest.raises(ValueError, match="Cycle detected"):
        spec_with_cycle.validate_dag()


def test_topological_sort():
    """Test topological sorting."""
    spec = GraphSpec(
        nodes={
            "a": NodeConfig(type="mcp_agent"),
            "b": NodeConfig(type="mcp_agent"),
            "c": NodeConfig(type="mcp_agent"),
            "d": NodeConfig(type="mcp_agent")
        },
        edges={"a": ["b", "c"], "b": ["d"], "c": ["d"]}
    )
    
    topo_order = spec.topological_sort()
    
    # 'a' should come before 'b' and 'c'
    assert topo_order.index("a") < topo_order.index("b")
    assert topo_order.index("a") < topo_order.index("c")
    
    # 'b' and 'c' should come before 'd'
    assert topo_order.index("b") < topo_order.index("d")
    assert topo_order.index("c") < topo_order.index("d")


def test_dependencies():
    """Test dependency tracking."""
    spec = GraphSpec(
        nodes={
            "a": NodeConfig(type="mcp_agent"),
            "b": NodeConfig(type="mcp_agent"),
            "c": NodeConfig(type="mcp_agent")
        },
        edges={"a": ["b"], "b": ["c"]}
    )
    
    # 'a' has no dependencies
    assert spec.get_dependencies("a") == set()
    
    # 'b' depends on 'a'
    assert spec.get_dependencies("b") == {"a"}
    
    # 'c' depends on 'a' and 'b'
    assert spec.get_dependencies("c") == {"a", "b"}


def test_graph_creation():
    """Test Graph creation and methods."""
    spec = GraphSpec(
        nodes={
            "test": NodeConfig(type="mcp_agent")
        },
        edges={},
        ctx={"test_ctx": True}
    )
    
    graph = Graph(spec=spec, run_id="test-run", trace_id="test-trace")
    
    assert graph.run_id == "test-run"
    assert graph.trace_id == "test-trace"
    assert graph.spec.ctx["test_ctx"] is True
    
    # Test JSON serialization
    json_str = graph.to_json()
    assert "test_ctx" in json_str
    
    # Test getting node config
    node_config = graph.get_node_config("test")
    assert node_config.type == "mcp_agent"
    
    # Test execution order
    order = graph.get_execution_order()
    assert "test" in order


def test_graph_from_json():
    """Test creating graph from JSON."""
    json_data = """
    {
        "nodes": {
            "example": {
                "type": "mcp_agent",
                "adapter": "mcp_use",
                "config": {}
            }
        },
        "edges": {},
        "ctx": {"example": true}
    }
    """
    
    graph = Graph.from_json(json_data, run_id="json-test", trace_id="json-trace")
    
    assert graph.run_id == "json-test"
    assert "example" in graph.spec.nodes
    assert graph.spec.ctx["example"] is True