#!/usr/bin/env python3
"""
Test supervisor TUI integration manually
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_orchestra.tui.model import RunTUIModel, NodeState
from agent_orchestra.orchestrator.types import NodeSpec

def test_supervisor_integration():
    """Test that our supervisor TUI integration works."""
    
    # Create a mock run model
    model = RunTUIModel("test_run", goal="Test supervisor integration")
    
    # Create a supervisor NodeSpec
    supervisor_spec = NodeSpec(
        id="test_supervisor",
        type="supervisor",
        name="Test Supervisor Agent",
        available_agents={
            "writer": {
                "description": "Creates written content",
                "capabilities": ["writing", "editing"],
                "server": "filesystem"
            },
            "analyst": {
                "description": "Analyzes data and trends", 
                "capabilities": ["analysis", "reporting"],
                "server": "filesystem"
            }
        },
        max_agent_calls=2
    )
    
    # Create NodeState with supervisor info
    node_state = NodeState(
        id=supervisor_spec.id,
        name=supervisor_spec.name,
        node_type=supervisor_spec.type,
        available_agents=supervisor_spec.available_agents,
        max_agent_calls=supervisor_spec.max_agent_calls
    )
    
    print("✅ Supervisor NodeState created successfully")
    print(f"   Node type: {node_state.node_type}")
    print(f"   Is supervisor: {node_state.is_supervisor}")
    print(f"   Available agents: {len(node_state.available_agents)}")
    print(f"   Max agent calls: {node_state.max_agent_calls}")
    
    # Test event handling
    model.nodes[supervisor_spec.id] = node_state
    
    # Simulate NODE_START event
    model.apply_event("NODE_START", supervisor_spec.id, {
        "node_type": "supervisor",
        "available_agents": supervisor_spec.available_agents,
        "max_agent_calls": supervisor_spec.max_agent_calls
    })
    
    print("✅ NODE_START event processed successfully")
    
    # Simulate supervisor decision
    model.apply_event("AGENT_CHUNK", supervisor_spec.id, {
        "text": "AGENT_CALL: writer - Need compelling product description"
    })
    
    print("✅ Supervisor decision event processed successfully")
    print(f"   Decisions recorded: {len(node_state.supervisor_decisions)}")
    
    # Simulate completion with agents called
    model.apply_event("NODE_COMPLETE", supervisor_spec.id, {
        "agents_called": ["writer", "analyst"],
        "output_summary": "Successfully coordinated writer and analyst to complete project"
    })
    
    print("✅ NODE_COMPLETE event processed successfully")
    print(f"   Agents called: {len(node_state.agents_called)}")
    print(f"   Final status: {node_state.status}")
    
    print("\n🧠 SUPERVISOR TUI INTEGRATION TEST PASSED!")
    return True

if __name__ == "__main__":
    test_supervisor_integration()