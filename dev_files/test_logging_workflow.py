#!/usr/bin/env python3
"""
Test workflow to verify the comprehensive logging system and TUI integration.
This workflow intentionally generates various types of log messages to test the new logging infrastructure.
"""

import asyncio
from typing import Dict, Any
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import MockExecutor
from agent_orchestra.tui.logging import get_system_logger

# Test the logging system
system_logger = get_system_logger()
system_logger.info("test_workflow", "Initializing test workflow for logging verification")

# Create a simple workflow that will generate logs
graph_spec = GraphSpec(
    nodes=[
        NodeSpec(
            id="test_node_1",
            name="Log Generation Test",
            task="Generate various log messages to test the logging infrastructure",
            depends_on=[]
        ),
        NodeSpec(
            id="test_node_2", 
            name="System Integration Test",
            task="Test system integration with comprehensive logging",
            depends_on=["test_node_1"]
        )
    ],
    edges=[("test_node_1", "test_node_2")]
)

# Custom executor that generates test logs
class LoggingTestExecutor(MockExecutor):
    """Test executor that generates comprehensive logs for TUI testing."""
    
    def __init__(self):
        super().__init__()
        self.system_logger = get_system_logger()
    
    async def execute_node(self, node_id: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute node with comprehensive logging."""
        self.system_logger.info("executor", f"Starting node execution: {node_id}")
        
        # Simulate various types of operations with logging
        if node_id == "test_node_1":
            self.system_logger.info("executor", "Testing basic logging functionality", node_id=node_id)
            self.system_logger.debug("executor", "Debug message for node execution", node_id=node_id)
            self.system_logger.warning("executor", "Warning: This is a test warning", node_id=node_id)
            
            # Simulate some work
            await asyncio.sleep(0.5)
            
            self.system_logger.info("executor", "Node 1 processing complete", node_id=node_id)
            return {"result": "Logging test complete", "status": "success"}
        
        elif node_id == "test_node_2":
            self.system_logger.info("executor", "Testing system integration", node_id=node_id)
            
            # Test different log levels
            self.system_logger.debug("executor", "Detailed debug information for integration", node_id=node_id)
            self.system_logger.info("executor", "System integration proceeding normally", node_id=node_id)
            self.system_logger.warning("executor", "Test warning for integration verification", node_id=node_id)
            
            # Simulate longer processing
            await asyncio.sleep(1.0)
            
            # Test error logging (but don't actually fail)
            self.system_logger.info("executor", "Integration test completed successfully", node_id=node_id)
            return {"result": "Integration test passed", "metrics": {"processed": 2, "warnings": 2}}
        
        else:
            self.system_logger.error("executor", f"Unknown node: {node_id}", node_id=node_id)
            return {"error": f"Unknown node: {node_id}"}

# Create executor instance
executor = LoggingTestExecutor()
system_logger.info("test_workflow", "Created LoggingTestExecutor")

# Create run specification
run_spec = RunSpec(
    run_id=f"log_test_run_{int(asyncio.get_event_loop().time())}",
    goal="Test comprehensive logging system integration with TUI"
)

system_logger.info("test_workflow", f"Created run spec: {run_spec.run_id}")
system_logger.info("test_workflow", "Test workflow setup complete - ready for execution")