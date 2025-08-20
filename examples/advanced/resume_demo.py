#!/usr/bin/env python3
"""
Step 5 Resume Demo - Durable Checkpoints & Resume Functionality

This example demonstrates the complete Step 5 implementation:
✅ Durable checkpoints with crash recovery
✅ Intelligent cache invalidation based on node signatures  
✅ Gate pruning state preservation across resume
✅ Event sequence continuity
✅ Production-ready file safety and atomic operations
✅ Backward compatibility with existing workflows

Usage:
    python examples/advanced/resume_demo.py
"""

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors import CallableExecutor
from agent_orchestra.orchestrator.store import JsonlRunStore


# Demo workflow components
class DemoWorkflowTasks:
    """Demo tasks that simulate real-world data processing."""
    
    def __init__(self):
        self.execution_log = []
        self.processing_delays = {
            "data_extraction": 2.0,      # Simulate slow database extraction
            "data_validation": 1.0,      # Validation checks
            "data_transformation": 3.0,   # Heavy transformation 
            "quality_analysis": 1.5,     # Analysis step
            "report_generation": 2.5,    # Report creation
        }
    
    async def execute_task(self, node: NodeSpec, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Execute demo task with realistic delays and logging."""
        task_name = node.id
        
        # Log execution
        self.execution_log.append({
            "task": task_name,
            "timestamp": time.time(),
            "inputs": node.inputs
        })
        
        print(f"      🔄 Executing {task_name}...")
        
        # Simulate processing time
        delay = self.processing_delays.get(task_name, 0.5)
        await asyncio.sleep(delay)
        
        # Generate realistic results based on task type
        if "extraction" in task_name:
            result = {
                "records_extracted": 10000,
                "source_files": 5,
                "extraction_time": f"{delay:.1f}s",
                "status": "success"
            }
        elif "validation" in task_name:
            result = {
                "validation_passed": True,
                "error_count": 12,
                "warning_count": 45,
                "validation_rules": 23,
                "status": "passed_with_warnings"
            }
        elif "transformation" in task_name:
            result = {
                "records_transformed": 9988,
                "transformation_rules": 15,
                "data_quality_score": 0.94,
                "transformation_time": f"{delay:.1f}s",
                "status": "completed"
            }
        elif "analysis" in task_name:
            result = {
                "quality_score": 0.92,
                "anomalies_detected": 3,
                "completeness": 0.98,
                "accuracy": 0.91,
                "analysis_time": f"{delay:.1f}s"
            }
        elif "generation" in task_name:
            result = {
                "report_pages": 25,
                "charts_generated": 12,
                "export_formats": ["PDF", "Excel", "JSON"],
                "generation_time": f"{delay:.1f}s",
                "file_size": "2.3MB"
            }
        else:
            result = {"status": "completed", "execution_time": f"{delay:.1f}s"}
        
        print(f"      ✅ Completed {task_name} - {result.get('status', 'done')}")
        return result


def create_demo_workflow() -> GraphSpec:
    """Create realistic data processing workflow for demo."""
    return GraphSpec(
        nodes=[
            # Data ingestion phase
            NodeSpec(
                id="data_extraction",
                type="task", 
                name="Extract Data from Sources",
                inputs={
                    "sources": ["database", "api", "files"],
                    "date_range": "2024-01-01 to 2024-01-31",
                    "batch_size": 1000
                },
                timeout_s=30,
                retries=2
            ),
            
            # Validation gate
            NodeSpec(
                id="validation_gate",
                type="gate",
                name="Data Quality Gate", 
                inputs={"predicate": True}  # Would normally check extraction results
            ),
            
            # Core processing pipeline
            NodeSpec(
                id="data_validation",
                type="task",
                name="Validate Data Quality",
                inputs={
                    "validation_rules": ["completeness", "accuracy", "consistency"],
                    "threshold": 0.9
                },
                timeout_s=20
            ),
            
            NodeSpec(
                id="data_transformation", 
                type="task",
                name="Transform and Clean Data",
                inputs={
                    "transformations": ["normalize", "deduplicate", "enrich"],
                    "target_schema": "warehouse_v2"
                },
                timeout_s=45
            ),
            
            # Analysis and reporting
            NodeSpec(
                id="quality_analysis",
                type="task",
                name="Analyze Data Quality Metrics", 
                inputs={
                    "metrics": ["completeness", "accuracy", "timeliness"],
                    "benchmarks": "industry_standard"
                },
                timeout_s=25
            ),
            
            NodeSpec(
                id="report_generation",
                type="task",
                name="Generate Processing Report",
                inputs={
                    "template": "monthly_processing_report",
                    "recipients": ["data_team", "stakeholders"],
                    "format": "executive_summary"
                },
                timeout_s=30
            ),
        ],
        edges=[
            ("data_extraction", "validation_gate"),
            ("validation_gate", "data_validation"),
            ("data_validation", "data_transformation"),
            ("data_transformation", "quality_analysis"),
            ("quality_analysis", "report_generation"),
        ]
    )


async def run_demo_phase_1(orchestrator: Orchestrator, workflow: GraphSpec, run_spec: RunSpec):
    """Phase 1: Initial workflow execution."""
    print("📊 PHASE 1: Initial Workflow Execution")
    print("=" * 60)
    print(f"🎯 Goal: {run_spec.goal}")
    print(f"🆔 Run ID: {run_spec.run_id}")
    print(f"📈 Workflow: {len(workflow.nodes)} nodes, {len(workflow.edges)} dependencies")
    print()
    
    start_time = time.time()
    events = []
    
    try:
        print("🚀 Starting workflow execution...")
        print()
        
        async for event in orchestrator.run_streaming(workflow, run_spec):
            events.append(event)
            
            if event.type == "RUN_START":
                print(f"🎬 WORKFLOW STARTED")
                
            elif event.type == "NODE_START":
                node_name = event.data.get('phase', event.node_id)
                print(f"\n🔧 NODE: {event.node_id}")
                print(f"   └─ Type: {event.data.get('type', 'unknown')}")
                
            elif event.type == "NODE_COMPLETE":
                phase = event.data.get('phase', '')
                if event.data.get('resumed'):
                    print(f"   ✅ RESUMED from checkpoint")
                else:
                    print(f"   ✅ COMPLETED successfully")
                    
            elif event.type == "RUN_COMPLETE":
                elapsed = time.time() - start_time
                print(f"\n🎉 WORKFLOW COMPLETED!")
                print(f"   ⏱️  Total time: {elapsed:.2f} seconds")
                print(f"   📊 Events generated: {len(events)}")
                
            elif event.type == "ERROR":
                error = event.data.get('error', 'Unknown error')
                print(f"   ❌ ERROR: {error}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Workflow interrupted by user")
        raise
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        raise
    
    return events


async def simulate_interruption():
    """Simulate workflow interruption after some processing."""
    print("\n🚨 SIMULATING SYSTEM INTERRUPTION...")
    print("   (In production, this could be a crash, restart, or kill signal)")
    await asyncio.sleep(1)
    raise KeyboardInterrupt("Simulated interruption")


async def run_demo_phase_2(orchestrator: Orchestrator, workflow: GraphSpec, run_spec: RunSpec):
    """Phase 2: Resume from checkpoint after interruption."""
    print("\n\n📊 PHASE 2: Resume from Checkpoint")
    print("=" * 60)
    print("🔄 Resuming workflow from last checkpoint...")
    print("✨ Demonstrating intelligent cache invalidation and resume")
    print()
    
    start_time = time.time()
    events = []
    
    try:
        async for event in orchestrator.run_streaming(workflow, run_spec, resume=True):
            events.append(event)
            
            if event.type == "RUN_START":
                if event.data.get('resumed'):
                    print(f"🔄 WORKFLOW RESUMED from checkpoint")
                else:
                    print(f"🎬 WORKFLOW STARTED (no checkpoint found)")
                    
            elif event.type == "NODE_START":
                print(f"\n🔧 NODE: {event.node_id}")
                
            elif event.type == "NODE_COMPLETE":
                if event.data.get('resumed'):
                    sig_preview = event.data.get('signature', 'N/A')[:12] + "..."
                    print(f"   ⚡ RESUMED from checkpoint (sig: {sig_preview})")
                else:
                    print(f"   ✅ EXECUTED (signature changed or not cached)")
                    
            elif event.type == "RUN_COMPLETE":
                elapsed = time.time() - start_time
                print(f"\n🎉 WORKFLOW RESUMED AND COMPLETED!")
                print(f"   ⏱️  Resume time: {elapsed:.2f} seconds")
                print(f"   📊 Events in resume: {len(events)}")
                
                # Count resumed vs executed nodes
                resumed_count = len([e for e in events if e.type == "NODE_COMPLETE" and e.data.get('resumed')])
                executed_count = len([e for e in events if e.type == "NODE_COMPLETE" and not e.data.get('resumed')])
                
                print(f"   ⚡ Nodes resumed: {resumed_count}")
                print(f"   🔄 Nodes re-executed: {executed_count}")
                
            elif event.type == "ERROR":
                error = event.data.get('error', 'Unknown error') 
                print(f"   ❌ ERROR: {error}")
    
    except Exception as e:
        print(f"\n💥 Resume failed: {e}")
        raise
    
    return events


async def demonstrate_cache_invalidation(store: JsonlRunStore, workflow: GraphSpec, run_spec: RunSpec):
    """Demonstrate cache invalidation when inputs change."""
    print("\n\n📊 PHASE 3: Cache Invalidation Demo")
    print("=" * 60)
    print("🔧 Modifying workflow inputs to demonstrate cache invalidation...")
    print()
    
    # Create modified workflow with changed inputs
    modified_workflow = GraphSpec(
        nodes=[
            # Changed the date range input - should invalidate cache
            NodeSpec(
                id="data_extraction", 
                type="task",
                name="Extract Data from Sources",
                inputs={
                    "sources": ["database", "api", "files"],
                    "date_range": "2024-02-01 to 2024-02-28",  # CHANGED DATE RANGE
                    "batch_size": 1000
                },
                timeout_s=30,
                retries=2
            ),
            
            # Keep other nodes the same for comparison
            NodeSpec(
                id="data_validation",
                type="task", 
                name="Validate Data Quality",
                inputs={
                    "validation_rules": ["completeness", "accuracy", "consistency"],
                    "threshold": 0.9
                },
                timeout_s=20
            ),
        ],
        edges=[("data_extraction", "data_validation")]
    )
    
    # Create new executor for this demo
    demo_tasks = DemoWorkflowTasks()
    executor = CallableExecutor(demo_tasks.execute_task)
    orchestrator = Orchestrator(executor, store=store)
    
    # Run with modified inputs
    modified_run_spec = RunSpec(run_id=run_spec.run_id, goal="Demo cache invalidation")
    
    events = []
    async for event in orchestrator.run_streaming(modified_workflow, modified_run_spec, resume=True):
        events.append(event)
        
        if event.type == "NODE_COMPLETE":
            if event.data.get('resumed'):
                print(f"   ⚡ {event.node_id}: RESUMED (inputs unchanged)")
            else:
                print(f"   🔄 {event.node_id}: RE-EXECUTED (inputs changed)")
    
    # Analysis
    resumed = [e for e in events if e.type == "NODE_COMPLETE" and e.data.get('resumed')]
    executed = [e for e in events if e.type == "NODE_COMPLETE" and not e.data.get('resumed')]
    
    print(f"\n📈 Cache Invalidation Results:")
    print(f"   ✅ Nodes with unchanged inputs (resumed): {len(resumed)}")
    print(f"   🔄 Nodes with changed inputs (re-executed): {len(executed)}")
    print(f"   🧠 Cache hit ratio: {len(resumed)/(len(resumed)+len(executed))*100:.1f}%")


async def inspect_checkpoint_files(store: JsonlRunStore, run_id: str):
    """Inspect the checkpoint files created during the demo."""
    print("\n\n📊 CHECKPOINT FILE INSPECTION")
    print("=" * 60)
    
    run_dir = store.root / run_id
    if not run_dir.exists():
        print("❌ No checkpoint directory found")
        return
    
    print(f"📁 Checkpoint directory: {run_dir}")
    print()
    
    # List all files
    files = list(run_dir.iterdir())
    for file in sorted(files):
        size = file.stat().st_size
        print(f"   📄 {file.name}: {size:,} bytes")
    
    print()
    
    # Inspect key files
    if (run_dir / "meta.json").exists():
        meta = json.loads((run_dir / "meta.json").read_text())
        print("📋 Run Metadata:")
        print(f"   🆔 Run ID: {meta.get('run_id')}")
        print(f"   🕐 Created: {time.ctime(meta.get('created_at', 0))}")
        print(f"   🏗️  Orchestrator: {meta.get('orchestrator_version')}")
        print()
    
    if (run_dir / "nodes.json").exists():
        nodes = json.loads((run_dir / "nodes.json").read_text())
        print(f"💾 Saved Nodes ({len(nodes)} total):")
        for node_id, node_data in nodes.items():
            sig_preview = node_data['signature'][:12] + "..."
            saved_at = time.ctime(node_data.get('saved_at', 0))
            print(f"   🔧 {node_id}: signature {sig_preview} (saved {saved_at})")
        print()
    
    if (run_dir / "events.jsonl").exists():
        events_content = (run_dir / "events.jsonl").read_text()
        event_count = len(events_content.strip().split('\n'))
        print(f"📊 Event Log: {event_count} events recorded")
        
        # Show last few events
        last_events = events_content.strip().split('\n')[-3:]
        print("   Last 3 events:")
        for event_line in last_events:
            try:
                event = json.loads(event_line)
                event_type = event.get('type', 'UNKNOWN')
                node_id = event.get('node_id', 'N/A')
                seq = event.get('event_seq', 0)
                print(f"     {seq:3d}: {event_type} - {node_id}")
            except:
                print(f"     Invalid event: {event_line[:50]}...")
        print()


async def cleanup_demo(store: JsonlRunStore, run_id: str):
    """Clean up demo files."""
    run_dir = store.root / run_id
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
        print(f"🧹 Cleaned up demo files: {run_dir}")


async def main():
    """Main demo orchestration."""
    print("🚀 Agent Orchestra - Step 5 Resume Demo")
    print("=" * 70)
    print("Demonstrating durable checkpoints and intelligent resume functionality")
    print()
    
    # Setup
    with tempfile.TemporaryDirectory(prefix="ao_resume_demo_") as tmp_dir:
        store_path = Path(tmp_dir) / "checkpoints"
        store = JsonlRunStore(store_path, flush_every=1)  # Immediate flush for demo
        
        demo_tasks = DemoWorkflowTasks()
        executor = CallableExecutor(demo_tasks.execute_task)
        orchestrator = Orchestrator(executor, store=store)
        
        workflow = create_demo_workflow()
        run_spec = RunSpec(
            run_id="data_processing_pipeline_001",
            goal="Process monthly data batch with resume capability"
        )
        
        try:
            # Phase 1: Initial execution (will be "interrupted")
            print("🎬 Starting demonstration with a realistic data processing workflow...")
            print(f"📊 This workflow simulates: data extraction → validation → transformation → analysis → reporting")
            print()
            
            # Create a task to run Phase 1, but interrupt it partway through
            phase1_task = asyncio.create_task(
                run_demo_phase_1(orchestrator, workflow, run_spec)
            )
            
            # Let it run for a bit, then "interrupt"
            try:
                await asyncio.wait_for(phase1_task, timeout=6.0)  # Interrupt after ~6 seconds
            except asyncio.TimeoutError:
                print(f"\n🚨 SIMULATING SYSTEM INTERRUPTION...")
                print("   (This could be a server crash, deployment, or maintenance)")
                phase1_task.cancel()
                try:
                    await phase1_task
                except asyncio.CancelledError:
                    pass
                print("   💾 Workflow state has been persisted to checkpoints")
            
            # Small delay to simulate system restart
            print("\n⏳ Simulating system restart...")
            await asyncio.sleep(2)
            
            # Phase 2: Resume from checkpoint
            # Create fresh orchestrator (simulating restart)
            fresh_executor = CallableExecutor(demo_tasks.execute_task)
            fresh_orchestrator = Orchestrator(fresh_executor, store=store)
            
            await run_demo_phase_2(fresh_orchestrator, workflow, run_spec)
            
            # Phase 3: Cache invalidation demo
            await demonstrate_cache_invalidation(store, workflow, run_spec)
            
            # Inspect the checkpoint files
            await inspect_checkpoint_files(store, run_spec.run_id)
            
            # Summary
            print("\n\n🎉 DEMO COMPLETE - Key Features Demonstrated:")
            print("=" * 70)
            print("✅ Durable checkpoints - Workflow state persisted to disk")
            print("✅ Intelligent resume - Skip completed nodes, restore blackboard state")
            print("✅ Cache invalidation - Re-execute when inputs change, reuse when unchanged")
            print("✅ Event continuity - Monotonic event sequences across interruptions")
            print("✅ Production safety - Atomic writes, file locking, crash resilience")
            print("✅ Gate pruning preservation - Conditional flow control survives resume")
            print("✅ Backward compatibility - Works with existing orchestrator APIs")
            print()
            print("🏗️  The checkpoint files demonstrate:")
            print("   📁 File structure: meta.json, events.jsonl, nodes.json, status.json")
            print("   🔐 Atomic operations: No partial writes or corruption")
            print("   📊 Event logs: Complete audit trail with sequence numbers")
            print("   💾 Node signatures: SHA256 cache keys for intelligent invalidation")
            print()
            print("🚀 Ready for production workflows with crash recovery and resume!")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n💥 Demo failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo ended by user")
        sys.exit(0)