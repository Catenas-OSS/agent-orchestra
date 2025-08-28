"""
Python workflow loader for Agent Orchestra CLI.

Loads Python workflow files and extracts orchestrator components.
"""

from __future__ import annotations
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent_orchestra.orchestrator.types import GraphSpec, RunSpec
from agent_orchestra.orchestrator.store_factory import create_store
from .logging import get_system_logger


@dataclass
class WorkflowResult:
    """Result of loading a Python workflow."""
    graph_spec: GraphSpec
    run_spec: RunSpec
    executor: Any
    store: Any
    namespace: Dict[str, Any]


def load_workflow(workflow_path: Path) -> WorkflowResult:
    """
    Load a Python workflow file and extract orchestrator components.
    
    Args:
        workflow_path: Path to the Python workflow file
        
    Returns:
        WorkflowResult containing the loaded components
        
    Raises:
        Exception: If the workflow file is invalid or missing required components
    """
    system_logger = get_system_logger()
    system_logger.info("workflow_loader", f"Loading workflow: {workflow_path.name}")
    
    if not workflow_path.exists():
        system_logger.error("workflow_loader", f"File not found: {workflow_path}")
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    
    if not workflow_path.suffix == '.py':
        system_logger.error("workflow_loader", f"Invalid file type: {workflow_path.suffix}")
        raise ValueError(f"Workflow file must be a Python file (.py), got: {workflow_path.suffix}")
    
    # Add the workflow directory to sys.path so imports work
    workflow_dir = workflow_path.parent
    if str(workflow_dir) not in sys.path:
        sys.path.insert(0, str(workflow_dir))
        system_logger.debug("workflow_loader", f"Added to sys.path: {workflow_dir}")
    
    try:
        # Execute the Python file to get its namespace
        system_logger.info("workflow_loader", "Executing workflow file")
        # Set up a proper namespace with __file__ and other builtins
        # Use a special __name__ to prevent if __name__ == "__main__" blocks from running
        namespace = {
            '__file__': str(workflow_path.absolute()),
            '__name__': '__workflow_loader__',
            '__builtins__': __builtins__,
        }
        
        with open(workflow_path, 'r') as f:
            code = compile(f.read(), str(workflow_path), 'exec')
            exec(code, namespace)
        
        # Extract components from the namespace
        system_logger.info("workflow_loader", "Extracting workflow components")
        # First try to extract from global scope
        try:
            graph_spec = _extract_graph_spec(namespace, workflow_path)
            run_spec = _extract_run_spec(namespace, workflow_path)
            executor = _extract_executor(namespace, workflow_path)
            store = _extract_store(namespace, workflow_path)
            
            system_logger.info("workflow_loader", f"Successfully extracted: graph({len(graph_spec.nodes)} nodes), executor({type(executor).__name__})")
        except ValueError as e:
            # If not found in global scope, this might be a demo script
            # Try to run it and see if it creates a workflow 
            if 'main' in namespace and callable(namespace['main']):
                system_logger.error("workflow_loader", "Detected demo script with main() function")
                raise ValueError(f"This appears to be a demo script with a main() function. "
                                f"The CLI expects workflow definition files with GraphSpec, RunSpec, and executor at module level. "
                                f"Original error: {e}")
            else:
                raise e
        
        return WorkflowResult(
            graph_spec=graph_spec,
            run_spec=run_spec,
            executor=executor,
            store=store,
            namespace=namespace
        )
        
    except Exception as e:
        system_logger.error("workflow_loader", f"Error loading workflow: {str(e)}")
        raise Exception(f"Error loading workflow from {workflow_path}: {e}") from e
    
    finally:
        # Clean up sys.path
        if str(workflow_dir) in sys.path:
            sys.path.remove(str(workflow_dir))


def _extract_graph_spec(namespace: Dict[str, Any], workflow_path: Path) -> GraphSpec:
    """Extract GraphSpec from the workflow namespace."""
    
    # Look for a variable named 'workflow', 'graph_spec', or 'graph'
    for var_name in ['workflow', 'graph_spec', 'graph']:
        if var_name in namespace:
            obj = namespace[var_name]
            if isinstance(obj, GraphSpec):
                return obj
    
    # Look for any GraphSpec instance in the namespace
    for name, obj in namespace.items():
        if isinstance(obj, GraphSpec):
            return obj
    
    raise ValueError(f"No GraphSpec found in workflow {workflow_path}. "
                     f"Expected variable named 'workflow', 'graph_spec', 'graph', "
                     f"or any GraphSpec instance.")


def _extract_run_spec(namespace: Dict[str, Any], workflow_path: Path) -> RunSpec:
    """Extract RunSpec from the workflow namespace."""
    
    # Look for a variable named 'run_spec', 'run', or 'spec'
    for var_name in ['run_spec', 'run', 'spec']:
        if var_name in namespace:
            obj = namespace[var_name]
            if isinstance(obj, RunSpec):
                return obj
    
    # Look for any RunSpec instance in the namespace
    for name, obj in namespace.items():
        if isinstance(obj, RunSpec):
            return obj
    
    # If no RunSpec found, create a default one
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    goal = f"Execute workflow from {workflow_path.name}"
    
    return RunSpec(run_id=run_id, goal=goal)


def _extract_executor(namespace: Dict[str, Any], workflow_path: Path) -> Any:
    """Extract executor from the workflow namespace or create a default one."""
    
    # Look for a variable named 'executor'
    if 'executor' in namespace:
        return namespace['executor']
    
    # Look for any object with 'execute' or 'run' methods that might be an executor
    for name, obj in namespace.items():
        if hasattr(obj, 'execute') and hasattr(obj, 'run'):
            return obj
    
    # Look for specific executor types
    executor_types = ['MCPExecutor', 'LocalExecutor', 'Executor']
    for name, obj in namespace.items():
        if any(executor_type in str(type(obj)) for executor_type in executor_types):
            return obj
    
    # If no executor found, create a default MCPExecutor
    system_logger.warning("workflow_loader", f"No executor found in {workflow_path.name}, creating default MCPExecutor")
    
    try:
        from .orchestrator.executors_mcp import MCPExecutor
        from .orchestrator.broker_config import create_development_broker
        from .orchestrator.agent_pool import AgentPool, create_default_agent_factory
        from .sidecar.sidecar_client import SidecarMCPClient
        
        # Create minimal MCP client
        client = SidecarMCPClient.from_dict({"mcpServers": {}})
        
        # Create default components
        broker = create_development_broker()
        agent_factory = create_default_agent_factory(client, None)  # No LLM for now
        agent_pool = AgentPool(agent_factory, max_agents_per_run=3)
        
        executor = MCPExecutor(
            agent=None,
            default_server="default",
            broker=broker,
            agent_pool=agent_pool,
            model_key="openai:gpt-4o-mini"
        )
        
        return executor
        
    except ImportError as e:
        raise ValueError(f"No executor found in workflow {workflow_path} and cannot create default: {e}")


def _extract_store(namespace: Dict[str, Any], workflow_path: Path) -> Any:
    """Extract store from the workflow namespace or create default."""
    
    # Look for a variable named 'store' or 'store_instance'
    for var_name in ['store', 'store_instance']:
        if var_name in namespace:
            return namespace[var_name]
    
    # Look for any store-like object
    for name, obj in namespace.items():
        if hasattr(obj, 'save_run') and hasattr(obj, 'save_event'):
            return obj
    
    # Create default SQLite store
    return create_store("sqlite", ".ao_runs/ao.sqlite3")