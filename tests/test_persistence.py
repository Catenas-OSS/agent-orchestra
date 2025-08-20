"""
Comprehensive tests for orchestrator persistence and resume functionality.
Tests checkpointing, resume logic, cache invalidation, and production safety features.
"""

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
import pytest
from unittest.mock import AsyncMock

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor
from agent_orchestra.orchestrator.store import JsonlRunStore, SavedNode
from agent_orchestra.orchestrator.signature import node_signature, foreach_item_signature


# Test fixtures and utilities
@pytest.fixture
def tmp_store(tmp_path: Path):
    """Create temporary JsonlRunStore for testing."""
    store_path = tmp_path / "test_runs"
    store = JsonlRunStore(store_path, flush_every=1)  # Flush every event for testing
    return store


@pytest.fixture
def simple_executor():
    """Create simple test executor."""
    async def simple_task(node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Simple task that adds 1 to input x."""
        x = node.inputs.get("x", 0)
        await asyncio.sleep(0.01)  # Small delay to simulate work
        return {"result": x + 1}
    
    return CallableExecutor(simple_task)


@pytest.fixture
def orchestrator_with_store(simple_executor, tmp_store):
    """Create orchestrator with persistence."""
    return Orchestrator(simple_executor, store=tmp_store)


@pytest.fixture
def orchestrator_without_store(simple_executor):
    """Create orchestrator without persistence (control)."""
    return Orchestrator(simple_executor, store=None)


# Utility functions
async def collect_events(event_generator):
    """Collect all events from an async generator."""
    events = []
    async for event in event_generator:
        events.append(event)
    return events


def create_simple_graph():
    """Create simple test graph: A -> B -> C."""
    return GraphSpec(
        nodes=[
            NodeSpec(id="A", type="task", inputs={"x": 1}),
            NodeSpec(id="B", type="task", inputs={"x": 2}),
            NodeSpec(id="C", type="task", inputs={"x": 3}),
        ],
        edges=[("A", "B"), ("B", "C")]
    )


def create_foreach_graph():
    """Create graph with foreach node."""
    return GraphSpec(
        nodes=[
            NodeSpec(
                id="process_items",
                type="foreach", 
                inputs={"items": [{"x": 1}, {"x": 2}, {"x": 3}]},
                concurrency=2
            ),
            NodeSpec(id="final", type="task", inputs={"x": 10}),
        ],
        edges=[("process_items", "final")]
    )


def create_gate_graph():
    """Create graph with gate node."""
    return GraphSpec(
        nodes=[
            NodeSpec(id="gate", type="gate", inputs={"predicate": False}),
            NodeSpec(id="after_gate", type="task", inputs={"x": 5}),
            NodeSpec(id="parallel", type="task", inputs={"x": 6}),
        ],
        edges=[("gate", "after_gate")]  # parallel is independent
    )


class TestBasicPersistence:
    """Test basic persistence functionality."""
    
    @pytest.mark.asyncio
    async def test_events_and_results_persisted(self, orchestrator_with_store, tmp_store):
        """Test that events and results are saved to disk."""
        graph = create_simple_graph()
        run_spec = RunSpec(run_id="test_run_001", goal="test persistence")
        
        # Execute workflow
        events = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec)
        )
        
        # Verify events were generated
        assert len(events) > 0
        assert events[0].type == "RUN_START"
        assert events[-1].type == "RUN_COMPLETE"
        
        # Verify files were created
        run_dir = tmp_store.root / "test_run_001"
        assert run_dir.exists()
        assert (run_dir / "events.jsonl").exists()
        assert (run_dir / "nodes.json").exists()
        assert (run_dir / "meta.json").exists()
        assert (run_dir / "status.json").exists()
        
        # Verify node results were saved
        checkpoint = await tmp_store.load_checkpoint("test_run_001")
        assert len(checkpoint) == 3  # A, B, C
        assert "A" in checkpoint
        assert "B" in checkpoint 
        assert "C" in checkpoint
        
        # Verify signatures are present
        for node_id, saved_node in checkpoint.items():
            assert saved_node.signature
            assert len(saved_node.signature) == 64  # SHA256 hex
    
    @pytest.mark.asyncio
    async def test_orchestrator_without_store_works(self, orchestrator_without_store):
        """Test backward compatibility - orchestrator works without store."""
        graph = create_simple_graph()
        run_spec = RunSpec(run_id="no_store_run", goal="test no persistence")
        
        # Should work normally without persistence
        events = await collect_events(
            orchestrator_without_store.run_streaming(graph, run_spec)
        )
        
        assert len(events) > 0
        assert events[0].type == "RUN_START"
        assert events[-1].type == "RUN_COMPLETE"


class TestResumeLogic:
    """Test workflow resume functionality."""
    
    @pytest.mark.asyncio
    async def test_resume_skips_completed_nodes(self, orchestrator_with_store, tmp_store):
        """Test that resume skips completed nodes with matching signatures."""
        graph = create_simple_graph()
        run_spec = RunSpec(run_id="resume_test_001", goal="test resume")
        
        # First run - complete workflow
        events1 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec)
        )
        
        # Verify first run completed
        assert events1[-1].type == "RUN_COMPLETE"
        
        # Second run with resume=True - should skip all nodes
        events2 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec, resume=True)
        )
        
        # Verify second run also completed
        assert events2[-1].type == "RUN_COMPLETE"
        
        # Count resumed nodes
        resumed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and e.data.get("resumed")]
        assert len(resumed_events) == 3  # All nodes should be resumed
        
        # Verify no task execution events (no AGENT_CHUNK)
        agent_chunks = [e for e in events2 if e.type == "AGENT_CHUNK"]
        assert len(agent_chunks) == 0
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_input_change(self, simple_executor, tmp_store):
        """Test that changed inputs trigger re-execution."""
        run_spec = RunSpec(run_id="cache_test_001", goal="test cache invalidation")
        
        # First run with input x=1
        graph1 = GraphSpec(
            nodes=[NodeSpec(id="A", type="task", inputs={"x": 1})],
            edges=[]
        )
        
        orchestrator1 = Orchestrator(simple_executor, store=tmp_store)
        events1 = await collect_events(
            orchestrator1.run_streaming(graph1, run_spec)
        )
        
        # Second run with changed input x=999 
        graph2 = GraphSpec(
            nodes=[NodeSpec(id="A", type="task", inputs={"x": 999})],  # Changed input
            edges=[]
        )
        
        orchestrator2 = Orchestrator(simple_executor, store=tmp_store)
        events2 = await collect_events(
            orchestrator2.run_streaming(graph2, run_spec, resume=True)
        )
        
        # Should not be resumed due to signature mismatch
        resumed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and e.data.get("resumed")]
        assert len(resumed_events) == 0
        
        # Should have task execution events
        agent_chunks = [e for e in events2 if e.type == "AGENT_CHUNK"]
        assert len(agent_chunks) > 0  # Re-executed due to signature change
    
    @pytest.mark.asyncio
    async def test_partial_resume_after_failure(self, tmp_store):
        """Test resume after partial workflow failure."""
        
        # Create executor that fails on node B
        call_count = {"count": 0}
        
        async def failing_task(node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
            call_count["count"] += 1
            if node.id == "B" and call_count["count"] <= 2:  # Fail B twice, then succeed
                raise Exception("Simulated failure")
            
            x = node.inputs.get("x", 0)
            return {"result": x + 1}
        
        failing_executor = CallableExecutor(failing_task)
        orchestrator = Orchestrator(failing_executor, store=tmp_store)
        
        graph = create_simple_graph()
        run_spec = RunSpec(run_id="partial_resume_001", goal="test partial resume")
        
        # First run - should fail at B
        events1 = await collect_events(
            orchestrator.run_streaming(graph, run_spec)
        )
        
        # Verify it failed
        error_events = [e for e in events1 if e.type == "ERROR"]
        assert len(error_events) > 0
        
        # Check that A was completed and saved
        checkpoint = await tmp_store.load_checkpoint("partial_resume_001")
        assert "A" in checkpoint
        assert "B" not in checkpoint  # B failed, not saved
        
        # Second run with resume - should skip A, retry B, execute C
        events2 = await collect_events(
            orchestrator.run_streaming(graph, run_spec, resume=True)
        )
        
        # Verify completion
        assert events2[-1].type == "RUN_COMPLETE"
        
        # A should be resumed, B and C should be executed
        resumed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and e.data.get("resumed")]
        assert len(resumed_events) == 1  # Only A resumed
        
        completed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and not e.data.get("resumed")]
        assert len(completed_events) == 2  # B and C executed


class TestGatePruning:
    """Test gate node pruning and resume behavior."""
    
    @pytest.mark.asyncio 
    async def test_gate_pruning_persisted_and_resumed(self, orchestrator_with_store, tmp_store):
        """Test gate pruning state is persisted and resumed correctly."""
        graph = create_gate_graph()
        run_spec = RunSpec(run_id="gate_test_001", goal="test gate pruning")
        
        # First run - gate blocks, should prune after_gate
        events1 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec)
        )
        
        # Verify gate blocked
        gate_events = [e for e in events1 if e.node_id == "gate" and e.type == "NODE_COMPLETE"]
        assert len(gate_events) == 1
        assert gate_events[0].data.get("phase") == "gate:blocked"
        
        # Verify after_gate was not executed (pruned)
        after_gate_events = [e for e in events1 if e.node_id == "after_gate"]
        assert len(after_gate_events) == 0
        
        # Verify parallel was executed (independent)
        parallel_events = [e for e in events1 if e.node_id == "parallel"]
        assert len(parallel_events) == 2  # START + COMPLETE
        
        # Verify pruning state was saved
        pruned_nodes = await tmp_store.load_gate_pruning("gate_test_001")
        assert "after_gate" in pruned_nodes
        
        # Resume run - should maintain pruning
        events2 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec, resume=True)
        )
        
        # Verify completion without deadlock
        assert events2[-1].type == "RUN_COMPLETE"
        
        # Verify gate and parallel resumed, after_gate still pruned
        resumed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and e.data.get("resumed")]
        resumed_node_ids = [e.node_id for e in resumed_events]
        assert "gate" in resumed_node_ids
        assert "parallel" in resumed_node_ids
        assert "after_gate" not in resumed_node_ids


class TestForeachResume:
    """Test foreach node resume functionality."""
    
    @pytest.mark.asyncio
    async def test_foreach_basic_resume(self, orchestrator_with_store):
        """Test basic foreach resume functionality."""
        graph = create_foreach_graph()
        run_spec = RunSpec(run_id="foreach_test_001", goal="test foreach resume")
        
        # First run - complete
        events1 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec)
        )
        assert events1[-1].type == "RUN_COMPLETE"
        
        # Resume run - should skip foreach and final
        events2 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec, resume=True)
        )
        
        # Both nodes should be resumed
        resumed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and e.data.get("resumed")]
        assert len(resumed_events) == 2


class TestEventSequenceContinuity:
    """Test event sequence continuity across resume sessions."""
    
    @pytest.mark.asyncio
    async def test_event_sequence_monotonic(self, orchestrator_with_store, tmp_store):
        """Test that event sequences are monotonic across resume."""
        graph = create_simple_graph()
        run_spec = RunSpec(run_id="sequence_test_001", goal="test event sequence")
        
        # First run
        events1 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec)
        )
        max_seq1 = max(e.event_seq for e in events1)
        
        # Resume run  
        events2 = await collect_events(
            orchestrator_with_store.run_streaming(graph, run_spec, resume=True)
        )
        min_seq2 = min(e.event_seq for e in events2)
        
        # Second run sequences should continue from first run
        assert min_seq2 > max_seq1
        
        # Verify sequences are monotonic within each run
        seq1 = [e.event_seq for e in events1]
        assert seq1 == sorted(seq1)
        
        seq2 = [e.event_seq for e in events2]
        assert seq2 == sorted(seq2)


class TestFileSafety:
    """Test file safety and crash resilience."""
    
    @pytest.mark.asyncio
    async def test_atomic_writes(self, tmp_store):
        """Test atomic write operations."""
        run_id = "atomic_test_001"
        
        # Start run to create directory
        await tmp_store.start_run(run_id, RunSpec(run_id=run_id, goal="test"))
        
        # Save node result
        saved_node = SavedNode("test_node", "test_sig", {"result": "test"})
        await tmp_store.save_node_result(run_id, saved_node)
        
        # Verify no .tmp files left behind
        run_dir = tmp_store.root / run_id
        tmp_files = list(run_dir.glob("*.tmp"))
        assert len(tmp_files) == 0
        
        # Verify content was saved correctly
        nodes_file = run_dir / "nodes.json"
        assert nodes_file.exists()
        
        data = json.loads(nodes_file.read_text())
        assert "test_node" in data
        assert data["test_node"]["signature"] == "test_sig"
    
    @pytest.mark.asyncio
    async def test_file_locking(self, tmp_store):
        """Test file locking prevents concurrent access."""
        run_id = "lock_test_001"
        run_spec = RunSpec(run_id=run_id, goal="test locking")
        
        # Start first run
        await tmp_store.start_run(run_id, run_spec)
        
        # Second store should fail to start same run
        tmp_store2 = JsonlRunStore(tmp_store.root)
        
        with pytest.raises(RuntimeError, match="locked by another orchestrator"):
            await tmp_store2.start_run(run_id, run_spec)
    
    @pytest.mark.asyncio
    async def test_corrupted_checkpoint_handling(self, tmp_store):
        """Test graceful handling of corrupted checkpoint files."""
        run_id = "corrupt_test_001"
        
        # Create run directory with corrupted nodes.json
        run_dir = tmp_store.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Write invalid JSON
        (run_dir / "nodes.json").write_text("invalid json content")
        
        # Should return empty checkpoint on corruption
        checkpoint = await tmp_store.load_checkpoint(run_id)
        assert checkpoint == {}


class TestLargeChunkHandling:
    """Test handling of large agent chunks."""
    
    @pytest.mark.asyncio
    async def test_large_chunk_redaction(self, tmp_path):
        """Test that oversized chunks are redacted in events."""
        small_max_size = 100  # Very small for testing
        store = JsonlRunStore(tmp_path, max_chunk_size=small_max_size)
        
        run_spec = RunSpec(run_id="chunk_test_001", goal="test chunks")
        await store.start_run(run_spec.run_id, run_spec)
        
        # Create event with large content
        from agent_orchestra.orchestrator.types import Event
        large_content = "x" * (small_max_size + 100)  # Exceeds limit
        
        event = Event(
            type="AGENT_CHUNK",
            run_id=run_spec.run_id,
            node_id="test_node",
            data={"content": large_content},
            event_seq=1
        )
        
        # Append event
        await store.append_event(run_spec.run_id, event)
        
        # Read back and verify redaction
        events_file = store.root / run_spec.run_id / "events.jsonl"
        content = events_file.read_text()
        
        # Should contain redaction message, not original content
        assert "REDACTED" in content
        assert large_content not in content


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_persistence_overhead(self, simple_executor, tmp_store):
        """Test that persistence adds minimal overhead."""
        graph = GraphSpec(
            nodes=[NodeSpec(id=f"node_{i}", type="task", inputs={"x": i}) for i in range(50)],
            edges=[(f"node_{i}", f"node_{i+1}") for i in range(49)]
        )
        run_spec = RunSpec(run_id="perf_test_001", goal="performance test")
        
        # Test with persistence
        orch_with_store = Orchestrator(simple_executor, store=tmp_store)
        
        start_time = time.time()
        events_with_store = await collect_events(
            orch_with_store.run_streaming(graph, run_spec)
        )
        time_with_store = time.time() - start_time
        
        # Test without persistence  
        orch_without_store = Orchestrator(simple_executor, store=None)
        
        start_time = time.time()
        events_without_store = await collect_events(
            orch_without_store.run_streaming(graph, RunSpec(run_id="perf_test_002", goal="no persistence"))
        )
        time_without_store = time.time() - start_time
        
        # Both should complete successfully
        assert events_with_store[-1].type == "RUN_COMPLETE"
        assert events_without_store[-1].type == "RUN_COMPLETE"
        
        # Persistence overhead should be reasonable (less than 50% increase)
        overhead_ratio = time_with_store / time_without_store
        assert overhead_ratio < 1.5, f"Persistence overhead too high: {overhead_ratio:.2f}x"
    
    @pytest.mark.asyncio
    async def test_large_workflow_scalability(self, simple_executor, tmp_store):
        """Test scalability with large workflows."""
        # Create larger workflow (100 nodes)
        num_nodes = 100
        graph = GraphSpec(
            nodes=[NodeSpec(id=f"task_{i}", type="task", inputs={"x": i}) for i in range(num_nodes)],
            edges=[(f"task_{i}", f"task_{i+1}") for i in range(num_nodes-1)]
        )
        
        orchestrator = Orchestrator(simple_executor, store=tmp_store)
        run_spec = RunSpec(run_id="scale_test_001", goal="scalability test")
        
        # Should complete without issues
        start_time = time.time()
        events = await collect_events(
            orchestrator.run_streaming(graph, run_spec)
        )
        elapsed = time.time() - start_time
        
        # Verify completion
        assert events[-1].type == "RUN_COMPLETE"
        
        # Should complete in reasonable time (less than 30s for 100 nodes)
        assert elapsed < 30.0
        
        # Verify all nodes were checkpointed
        checkpoint = await tmp_store.load_checkpoint("scale_test_001")
        assert len(checkpoint) == num_nodes


# Integration tests with realistic scenarios
class TestIntegrationScenarios:
    """Integration tests with realistic workflow patterns."""
    
    @pytest.mark.asyncio
    async def test_data_processing_pipeline(self, tmp_store):
        """Test realistic data processing pipeline with resume."""
        
        # Simulate data processing tasks
        processing_stats = {"files_processed": 0}
        
        async def data_processor(node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
            """Simulate data processing with different behaviors per node type."""
            
            if "extract" in node.id:
                processing_stats["files_processed"] += 1
                return {"extracted_records": 1000, "file_size": "10MB"}
            
            elif "transform" in node.id:
                # Simulate transformation
                await asyncio.sleep(0.02)
                return {"transformed_records": 950, "cleaning_applied": True}
            
            elif "validate" in node.id:
                return {"validation_passed": True, "error_count": 5}
            
            elif "load" in node.id:
                return {"loaded_records": 945, "load_time": "2.3s"}
            
            else:
                return {"status": "completed"}
        
        processor_executor = CallableExecutor(data_processor)
        orchestrator = Orchestrator(processor_executor, store=tmp_store)
        
        # Create realistic data pipeline
        pipeline = GraphSpec(
            nodes=[
                NodeSpec(id="extract_source", type="task", inputs={"source": "database"}),
                NodeSpec(id="validate_input", type="gate", inputs={"predicate": True}),
                NodeSpec(id="transform_data", type="task", inputs={"rules": "clean_format"}),
                NodeSpec(id="validate_output", type="task", inputs={"schema": "target_schema"}),
                NodeSpec(id="load_target", type="task", inputs={"target": "warehouse"}),
            ],
            edges=[
                ("extract_source", "validate_input"),
                ("validate_input", "transform_data"),
                ("transform_data", "validate_output"), 
                ("validate_output", "load_target"),
            ]
        )
        
        run_spec = RunSpec(run_id="pipeline_001", goal="Process daily data batch")
        
        # First run - complete pipeline
        events1 = await collect_events(
            orchestrator.run_streaming(pipeline, run_spec)
        )
        
        assert events1[-1].type == "RUN_COMPLETE"
        assert processing_stats["files_processed"] == 1
        
        # Reset stats
        processing_stats["files_processed"] = 0
        
        # Resume run - should skip all completed steps
        events2 = await collect_events(
            orchestrator.run_streaming(pipeline, run_spec, resume=True)
        )
        
        assert events2[-1].type == "RUN_COMPLETE"
        assert processing_stats["files_processed"] == 0  # No reprocessing
        
        # Verify all nodes were resumed
        resumed_events = [e for e in events2 if e.type == "NODE_COMPLETE" and e.data.get("resumed")]
        assert len(resumed_events) == 5  # All nodes resumed


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_persistence.py -v
    pytest.main([__file__, "-v"])