"""
Comprehensive tests for SQLiteRunStore implementation.
Ensures drop-in compatibility with JsonlRunStore and validates SQLite-specific features.
"""

import asyncio
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
import pytest
from unittest.mock import AsyncMock

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec, Event
from agent_orchestra.orchestrator.executors import CallableExecutor
from agent_orchestra.orchestrator.store_sqlite import SQLiteRunStore
from agent_orchestra.orchestrator.store import SavedNode, SavedForeachItem
from agent_orchestra.orchestrator.signature import node_signature, foreach_item_signature


# Test fixtures and utilities
@pytest.fixture
def tmp_sqlite_store(tmp_path: Path):
    """Create temporary SQLiteRunStore for testing."""
    db_path = tmp_path / "test_runs.sqlite3"
    store = SQLiteRunStore(db_path)
    return store


@pytest.fixture
def mock_executor():
    """Create mock executor for testing."""
    executor = AsyncMock()
    executor.execute_node = AsyncMock()
    return executor


@pytest.fixture  
def simple_workflow():
    """Create simple test workflow."""
    return GraphSpec(
        nodes=[
            NodeSpec(id="task1", type="task", inputs={"data": "value1"}),
            NodeSpec(id="task2", type="task", inputs={"data": "value2"}),
            NodeSpec(id="task3", type="task", inputs={"data": "value3"}),
        ],
        edges=[
            ("task1", "task2"),
            ("task2", "task3"),
        ]
    )


@pytest.fixture
def foreach_workflow():
    """Create workflow with foreach node."""
    return GraphSpec(
        nodes=[
            NodeSpec(id="setup", type="task", inputs={"data": "initial"}),
            NodeSpec(
                id="process_items", 
                type="foreach",
                inputs={
                    "items": [
                        {"id": 1, "value": "a"},
                        {"id": 2, "value": "b"}, 
                        {"id": 3, "value": "c"}
                    ]
                }
            ),
            NodeSpec(id="summary", type="reduce", inputs={"from_ids": ["process_items"]}),
        ],
        edges=[
            ("setup", "process_items"),
            ("process_items", "summary"),
        ]
    )


# Basic functionality tests
@pytest.mark.asyncio
async def test_sqlite_basic_persistence_and_resume(tmp_sqlite_store, simple_workflow):
    """Test basic persistence and resume functionality."""
    async def executor_func(node, ctx):
        return {"result": f"processed_{node.id}"}
    
    executor = CallableExecutor(executor_func)
    
    orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    run_spec = RunSpec(run_id="test_run_001", goal="Test basic persistence")
    
    # Phase 1: Initial execution
    events1 = []
    async for event in orchestrator.run_streaming(simple_workflow, run_spec):
        events1.append(event)
    
    # Verify run status in database
    status = await tmp_sqlite_store.get_run_status("test_run_001")
    assert status is not None
    assert status['status'] == 'complete'
    assert status['goal'] == 'Test basic persistence'
    
    # Phase 2: Resume (should skip all nodes)
    fresh_orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    
    events2 = []
    async for event in fresh_orchestrator.run_streaming(simple_workflow, run_spec, resume=True):
        events2.append(event)
    
    # Should have minimal events (just resume confirmations)
    resume_events = [e for e in events2 if e.data.get('resumed')]
    assert len(resume_events) >= 3  # All nodes should be resumed
    
    # Verify checkpoint was loaded correctly
    checkpoint = await tmp_sqlite_store.load_checkpoint("test_run_001")
    assert len(checkpoint) == 3
    assert "task1" in checkpoint
    assert "task2" in checkpoint
    assert "task3" in checkpoint


@pytest.mark.asyncio
async def test_sqlite_cache_invalidation_on_input_change(tmp_sqlite_store, simple_workflow):
    """Test cache invalidation when node inputs change."""
    async def executor_func(node, ctx):
        return {"result": f"processed_{node.id}_{node.inputs.get('data', 'default')}"}
    
    executor = CallableExecutor(executor_func)
    
    orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    run_spec = RunSpec(run_id="test_run_002", goal="Test cache invalidation")
    
    # Phase 1: Initial execution
    events1 = []
    async for event in orchestrator.run_streaming(simple_workflow, run_spec):
        events1.append(event)
    
    # Phase 2: Modify workflow inputs
    modified_workflow = GraphSpec(
        nodes=[
            NodeSpec(id="task1", type="task", inputs={"data": "CHANGED_VALUE"}),  # Modified
            NodeSpec(id="task2", type="task", inputs={"data": "value2"}),
            NodeSpec(id="task3", type="task", inputs={"data": "value3"}),
        ],
        edges=simple_workflow.edges
    )
    
    events2 = []
    async for event in orchestrator.run_streaming(modified_workflow, run_spec, resume=True):
        events2.append(event)
    
    # Should re-execute nodes affected by the change
    node_complete_events = [e for e in events2 if e.type == "NODE_COMPLETE" and not e.data.get('resumed')]
    assert len(node_complete_events) >= 1  # At least task1 should re-execute


@pytest.mark.asyncio
async def test_sqlite_foreach_granular_resume(tmp_sqlite_store, foreach_workflow):
    """Test foreach granular resume functionality."""
    call_count = 0
    
    async def mock_executor_func(node, ctx):
        nonlocal call_count
        call_count += 1
        
        if node.id == "setup":
            return {"result": "setup_complete"}
        elif node.id.startswith("process_items:"):  # Foreach sub-nodes have format "process_items:0", "process_items:1", etc.
            # For foreach sub-nodes, the item data is in node.inputs["item"]
            item = node.inputs.get("item", {})
            return {"result": f"processed_item_{item.get('id', 'unknown')}"}
        elif node.id == "summary":
            return {"result": "summary_complete"}
        
    executor = CallableExecutor(mock_executor_func)
    orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    run_spec = RunSpec(run_id="test_run_003", goal="Test foreach granular resume")
    
    # Phase 1: Initial execution
    events1 = []
    async for event in orchestrator.run_streaming(foreach_workflow, run_spec):
        events1.append(event)
    
    initial_call_count = call_count
    
    # Verify foreach items were saved
    foreach_items = await tmp_sqlite_store.load_foreach_items("test_run_003", "process_items")
    assert len(foreach_items) == 3  # Should have 3 items
    assert 0 in foreach_items
    assert 1 in foreach_items
    assert 2 in foreach_items
    
    # Phase 2: Resume (should skip completed foreach items)
    call_count = 0  # Reset counter
    fresh_orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    
    events2 = []
    async for event in fresh_orchestrator.run_streaming(foreach_workflow, run_spec, resume=True):
        events2.append(event)
    
    # Should make minimal new calls (everything cached)
    assert call_count == 0


@pytest.mark.asyncio
async def test_sqlite_event_sequence_continuity(tmp_sqlite_store, simple_workflow):
    """Test event sequence continuity across resume sessions."""
    executor = CallableExecutor(lambda node, inputs, run_spec: {"result": f"processed_{node.id}"})
    orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    run_spec = RunSpec(run_id="test_run_004", goal="Test event sequence")
    
    # Phase 1: Initial execution
    events1 = []
    async for event in orchestrator.run_streaming(simple_workflow, run_spec):
        events1.append(event)
    
    last_seq_1 = await tmp_sqlite_store.get_last_event_seq("test_run_004")
    
    # Phase 2: Resume
    fresh_orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    
    events2 = []
    async for event in fresh_orchestrator.run_streaming(simple_workflow, run_spec, resume=True):
        events2.append(event)
    
    last_seq_2 = await tmp_sqlite_store.get_last_event_seq("test_run_004")
    
    # Sequence should continue from where it left off
    assert last_seq_2 >= last_seq_1


@pytest.mark.asyncio
async def test_sqlite_concurrent_operations(tmp_sqlite_store):
    """Test concurrent read/write operations don't cause conflicts."""
    run_spec = RunSpec(run_id="test_run_005", goal="Test concurrent access")
    
    async def write_events():
        """Simulate writing events concurrently."""
        for i in range(10):
            event = Event(
                type="NODE_START",
                run_id="test_run_005",
                node_id=f"node_{i}",
                data={"test": f"data_{i}"},
                event_seq=i
            )
            await tmp_sqlite_store.append_event("test_run_005", event)
            await asyncio.sleep(0.01)  # Small delay
    
    async def read_status():
        """Simulate reading status concurrently."""
        for _ in range(10):
            status = await tmp_sqlite_store.get_run_status("test_run_005")
            await asyncio.sleep(0.01)
    
    # Start run
    await tmp_sqlite_store.start_run("test_run_005", run_spec)
    
    # Run concurrent operations
    await asyncio.gather(write_events(), read_status())
    
    # Verify final state
    status = await tmp_sqlite_store.get_run_status("test_run_005")
    assert status is not None
    
    last_seq = await tmp_sqlite_store.get_last_event_seq("test_run_005")
    assert last_seq >= 9


@pytest.mark.asyncio
async def test_sqlite_gate_pruning_persistence(tmp_sqlite_store):
    """Test gate pruning state persistence."""
    run_id = "test_run_006"
    pruned_nodes = {"node_a", "node_b", "node_c"}
    
    # Create run first (required for foreign key)
    run_spec = RunSpec(run_id=run_id, goal="Test gate pruning")
    await tmp_sqlite_store.start_run(run_id, run_spec)
    
    # Save gate pruning state
    await tmp_sqlite_store.save_gate_pruning(run_id, pruned_nodes)
    
    # Load and verify
    loaded_pruned = await tmp_sqlite_store.load_gate_pruning(run_id)
    assert loaded_pruned == pruned_nodes
    
    # Test empty case
    empty_pruned = await tmp_sqlite_store.load_gate_pruning("nonexistent_run")
    assert empty_pruned == set()


@pytest.mark.asyncio 
async def test_sqlite_json_serialization_safety(tmp_sqlite_store):
    """Test that complex objects are properly serialized."""
    run_spec = RunSpec(run_id="test_run_007", goal="Test JSON serialization")
    await tmp_sqlite_store.start_run("test_run_007", run_spec)
    
    # Create event with complex data that caused issues in Step 5
    complex_event = Event(
        type="AGENT_CHUNK",
        run_id="test_run_007", 
        node_id="test_node",
        data={
            "content": "test content",
            "complex_object": {"nested": {"deep": "value"}},
            "list_data": [1, 2, {"nested": "item"}]
        },
        event_seq=1
    )
    
    # Should not raise JSON serialization errors
    await tmp_sqlite_store.append_event("test_run_007", complex_event)
    
    # Verify event was stored
    last_seq = await tmp_sqlite_store.get_last_event_seq("test_run_007")
    assert last_seq == 1


@pytest.mark.asyncio
async def test_sqlite_run_statistics(tmp_sqlite_store):
    """Test run statistics gathering."""
    run_spec = RunSpec(run_id="test_run_008", goal="Test statistics")
    await tmp_sqlite_store.start_run("test_run_008", run_spec)
    
    # Add various events
    events = [
        Event(type="RUN_START", run_id="test_run_008", event_seq=1),
        Event(type="NODE_START", run_id="test_run_008", node_id="node1", event_seq=2),
        Event(type="NODE_COMPLETE", run_id="test_run_008", node_id="node1", event_seq=3),
        Event(type="NODE_START", run_id="test_run_008", node_id="node2", event_seq=4),
        Event(type="NODE_COMPLETE", run_id="test_run_008", node_id="node2", event_seq=5),
        Event(type="RUN_COMPLETE", run_id="test_run_008", event_seq=6),
    ]
    
    for event in events:
        await tmp_sqlite_store.append_event("test_run_008", event)
    
    # Add some node results
    await tmp_sqlite_store.save_node_result("test_run_008", SavedNode(
        node_id="node1", signature="sig1", result={"output": "result1"}
    ))
    await tmp_sqlite_store.save_node_result("test_run_008", SavedNode(
        node_id="node2", signature="sig2", result={"output": "result2"}
    ))
    
    # Add foreach items
    await tmp_sqlite_store.save_foreach_item("test_run_008", SavedForeachItem(
        node_id="node1", item_index=0, signature="foreach_sig1", result={"item": "result1"}
    ))
    await tmp_sqlite_store.save_foreach_item("test_run_008", SavedForeachItem(
        node_id="node1", item_index=1, signature="foreach_sig2", result={"item": "result2"}
    ))
    
    # Get statistics
    stats = await tmp_sqlite_store.get_run_statistics("test_run_008")
    
    assert stats['event_counts']['NODE_START'] == 2
    assert stats['event_counts']['NODE_COMPLETE'] == 2
    assert stats['completed_nodes'] == 2
    assert stats['completed_foreach_items'] == 2


@pytest.mark.asyncio
async def test_sqlite_list_runs(tmp_sqlite_store):
    """Test listing runs with filtering."""
    # Create multiple runs with different statuses
    run_specs = [
        RunSpec(run_id="run_complete_1", goal="Completed run 1"),
        RunSpec(run_id="run_complete_2", goal="Completed run 2"),
        RunSpec(run_id="run_error_1", goal="Error run 1"),
        RunSpec(run_id="run_running_1", goal="Running run 1"),
    ]
    
    for spec in run_specs:
        await tmp_sqlite_store.start_run(spec.run_id, spec)
    
    # Mark different statuses
    await tmp_sqlite_store.mark_run_complete("run_complete_1")
    await tmp_sqlite_store.mark_run_complete("run_complete_2")
    await tmp_sqlite_store.mark_run_error("run_error_1", "Test error")
    # run_running_1 stays in running status
    
    # Test listing all runs
    all_runs = await tmp_sqlite_store.list_runs()
    assert len(all_runs) == 4
    
    # Test filtering by status
    completed_runs = await tmp_sqlite_store.list_runs(status="complete")
    assert len(completed_runs) == 2
    assert all(run['status'] == 'complete' for run in completed_runs)
    
    error_runs = await tmp_sqlite_store.list_runs(status="error")
    assert len(error_runs) == 1
    assert error_runs[0]['error'] == "Test error"
    
    running_runs = await tmp_sqlite_store.list_runs(status="running")
    assert len(running_runs) == 1


@pytest.mark.asyncio
async def test_sqlite_database_constraints(tmp_sqlite_store):
    """Test database constraints and foreign key relationships."""
    # Test that we can't insert node_results without a run
    with pytest.raises(Exception):  # Should violate foreign key constraint
        def _bad_insert():
            with tmp_sqlite_store._get_connection() as conn:
                conn.execute("""
                    INSERT INTO node_results (run_id, node_id, signature, result_json, completed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("nonexistent_run", "test_node", "sig", "{}", time.time()))
        
        await asyncio.to_thread(_bad_insert)


@pytest.mark.asyncio
async def test_sqlite_database_schema_validation(tmp_sqlite_store):
    """Test that database schema is created correctly."""
    def _check_schema():
        with tmp_sqlite_store._get_connection() as conn:
            # Check that all expected tables exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = {
                'runs', 'node_results', 'events', 'gate_pruning', 'broker_metrics'
            }
            assert expected_tables.issubset(set(tables))
            
            # Check that indexes exist
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='index'
                AND name NOT LIKE 'sqlite_%'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            
            expected_indexes = {
                'idx_events_run_seq', 'idx_nodes_lookup', 'idx_nodes_sub_lookup',
                'idx_runs_status', 'idx_events_type'
            }
            assert expected_indexes.issubset(set(indexes))
    
    await asyncio.to_thread(_check_schema)


@pytest.mark.asyncio
async def test_sqlite_wal_mode_performance(tmp_sqlite_store):
    """Test that WAL mode is properly configured for concurrent access."""
    def _check_wal():
        with tmp_sqlite_store._get_connection() as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.upper() == "WAL"
            
            cursor = conn.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]
            assert fk_enabled == 1
    
    await asyncio.to_thread(_check_wal)


# Integration tests
@pytest.mark.asyncio
async def test_sqlite_drop_in_compatibility_with_orchestrator(tmp_sqlite_store, simple_workflow):
    """Test that SQLiteRunStore is a perfect drop-in replacement for JsonlRunStore."""
    async def executor_func(node, ctx):
        return {"result": f"processed_{node.id}", "timestamp": time.time()}
    
    executor = CallableExecutor(executor_func)
    
    # Test the full orchestrator lifecycle
    orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    run_spec = RunSpec(run_id="compatibility_test", goal="Test drop-in compatibility")
    
    # Should work exactly like JsonlRunStore
    events = []
    async for event in orchestrator.run_streaming(simple_workflow, run_spec):
        events.append(event)
    
    # Verify all expected event types
    event_types = {event.type for event in events}
    assert "RUN_START" in event_types
    assert "NODE_START" in event_types
    assert "NODE_COMPLETE" in event_types
    assert "RUN_COMPLETE" in event_types
    
    # Verify resume works identically
    resume_events = []
    fresh_orchestrator = Orchestrator(executor, store=tmp_sqlite_store)
    async for event in fresh_orchestrator.run_streaming(simple_workflow, run_spec, resume=True):
        resume_events.append(event)
    
    # Should have resume markers
    resumed_count = sum(1 for event in resume_events if event.data.get('resumed'))
    assert resumed_count >= 3  # All 3 nodes should be resumed


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])