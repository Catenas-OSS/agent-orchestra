
"""
Restored Python-first CLI for Agent Orchestra.
Supports catenas run <workflow.py> with optional --watch TUI.
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
except ImportError:
    print("CLI dependencies not installed. Run: pip install 'agent-orchestra[cli]'")
    sys.exit(1)

from .workflow_loader import load_workflow
from .orchestrator.core import Orchestrator
from .orchestrator.store_factory import create_store
from .orchestrator.types import RunSpec
from .logging import get_system_logger, init_logging

# Optional TUI imports
try:
    from .tui.model import RunTUIModel, NodeState
    from .tui.main import ProfessionalUrwidTUI
    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False

app = typer.Typer(help="Agent Orchestra - Python-first workflow orchestration")
console = Console()

# Global flag to track if we're in TUI mode
_tui_mode_active = False

# Initialize logging system early to capture all output  
_early_logger = None

def safe_console_print(*args, **kwargs):
    """Print to console only if not in TUI mode."""
    if not _tui_mode_active:
        console.print(*args, **kwargs)


@app.command()
def run(
    workflow: Path = typer.Argument(..., help="Python workflow file"),
    goal: str = typer.Option("", help="Run goal/description"),
    resume: Optional[str] = typer.Option(None, help="Resume from run ID"),
    watch: bool = typer.Option(False, "--watch", help="Show live TUI dashboard"),
    store: str = typer.Option("sqlite", help="Store type: sqlite or jsonl"),
    store_path: Optional[str] = typer.Option(None, help="Store path (default: .ao_runs/)"),
):
    """Execute a Python workflow with optional live TUI dashboard."""
    
    # Set global TUI mode flag
    global _tui_mode_active
    _tui_mode_active = watch
    
    # Initialize logging VERY early - before any imports or operations
    system_logger = init_logging(tui_mode=watch)
    if watch:
        # In TUI mode, capture ALL output immediately
        system_logger.enable_tui_mode()
    
    system_logger.info("cli", f"Starting workflow execution: {workflow.name}")
    
    try:
        # Load workflow
        system_logger.info("cli", "Loading workflow definition")
        workflow_result = load_workflow(workflow)
        graph_spec = workflow_result.graph_spec
        run_spec = workflow_result.run_spec
        executor = workflow_result.executor
        store_instance = workflow_result.store
        system_logger.info("cli", f"Loaded graph with {len(graph_spec.nodes)} nodes")
        
        # Override goal if provided
        if goal:
            run_spec = RunSpec(run_spec.run_id, goal)
            system_logger.info("cli", f"Goal override: {goal}")
        
        # Handle resume
        if resume:
            run_spec = RunSpec(resume, run_spec.goal)
            system_logger.info("cli", f"Resuming run: {resume}")
        
        # Create orchestrator
        system_logger.info("cli", "Initializing orchestrator")
        orchestrator = Orchestrator(executor, store=store_instance)
        
        # Run with or without TUI
        if watch:
            if not TUI_AVAILABLE:
                safe_console_print("[red]TUI not available. Install with: pip install urwid rich typer[/red]")
                safe_console_print("Running in plain mode instead...")
                system_logger.info("cli", "TUI unavailable, falling back to plain mode")
                asyncio.run(_run_plain(orchestrator, graph_spec, run_spec))
            else:
                system_logger.info("cli", "Starting TUI mode")
                asyncio.run(_run_with_professional_tui(orchestrator, graph_spec, run_spec))
        else:
            system_logger.info("cli", "Starting plain mode")
            asyncio.run(_run_plain(orchestrator, graph_spec, run_spec))
            
    except Exception as e:
        system_logger.error("cli", f"Workflow execution failed: {str(e)}")
        safe_console_print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def _run_with_professional_tui(orchestrator, graph_spec, run_spec):
    """Run workflow with the new professional Rich-based TUI dashboard."""
    
    if not TUI_AVAILABLE:
        # Fallback to plain mode if TUI not available
        await _run_plain(orchestrator, graph_spec, run_spec)
        return
    
    system_logger = get_system_logger()
    system_logger.info("tui", "Initializing TUI dashboard")
    
    # Initialize enhanced TUI model
    model = RunTUIModel(run_spec.run_id, goal=run_spec.goal)
    
    # Add nodes from graph spec (preserve topological order)
    for node in graph_spec.nodes:
        node_state = NodeState(
            id=node.id,
            name=node.name,
            node_type=node.type,
            server=getattr(node, 'server_name', None),
            inputs=getattr(node, 'inputs', {})  # Capture inputs from workflow definition
        )
        
        # Add supervisor-specific fields if it's a supervisor node
        if node.type == "supervisor":
            node_state.available_agents = getattr(node, 'available_agents', {})
            node_state.max_agent_calls = getattr(node, 'max_agent_calls', 5)
        
        model.nodes[node.id] = node_state
    
    # Store DAG edges for mini-map
    model.dag_edges = [(edge[0], edge[1]) for edge in graph_spec.edges]
    
    try:
        # Use new Professional Urwid TUI (truly interactive)
        system_logger.info("tui", "Creating TUI interface")
        tui = ProfessionalUrwidTUI(orchestrator, model)
        
        # Run the Urwid TUI with streaming events
        system_logger.info("tui", "Starting workflow execution with TUI")
        run_stream = orchestrator.run_streaming(graph_spec, run_spec)
        await tui.run(run_stream)
        system_logger.info("tui", "TUI session completed")
            
    except KeyboardInterrupt:
        system_logger.info("tui", "User interrupted TUI session")
        # Don't print to console in TUI mode - it interferes with the interface
        pass
    except Exception as e:
        system_logger.error("tui", f"TUI error: {str(e)}")
        # Don't print to console in TUI mode - log the error instead
        system_logger.critical("tui", f"Critical TUI failure, falling back to plain mode: {str(e)}")
        # Fallback to plain execution
        system_logger.info("tui", "Falling back to plain execution")
        await _run_plain(orchestrator, graph_spec, run_spec)


async def _run_plain(orchestrator, graph_spec, run_spec):
    """Run workflow with basic console output."""
    
    safe_console_print(f"🚀 Starting workflow: {run_spec.goal}")
    safe_console_print(f"📋 Nodes: {len(graph_spec.nodes)}")
    safe_console_print("")
    
    try:
        node_count = 0
        async for event in orchestrator.run_streaming(graph_spec, run_spec):
            event_type = event.type
            node_id = getattr(event, 'node_id', None)
            data = getattr(event, 'data', {})
            
            if event_type == "RUN_START":
                safe_console_print("▶️  Execution started")
                
            elif event_type == "NODE_START":
                node_count += 1
                node_name = next((n.name for n in graph_spec.nodes if n.id == node_id), node_id)
                safe_console_print(f"[{node_count}/{len(graph_spec.nodes)}] 🤖 {node_name}")
                
            elif event_type == "AGENT_CHUNK":
                if isinstance(data, dict) and data.get("text"):
                    text = data["text"][:80] + ("..." if len(data["text"]) > 80 else "")
                    safe_console_print(f"   💭 {text}")
                    
            elif event_type == "NODE_COMPLETE":
                output = data.get("output_summary", "")
                tokens = data.get("tokens", {})
                cost = data.get("cost", 0)
                
                safe_console_print(f"   ✅ Completed")
                if tokens and tokens.get("total", 0) > 0:
                    safe_console_print(f"   📊 {tokens.get('total', 0):,} tokens | ${cost:.4f}")
                if output:
                    preview = output[:100] + ("..." if len(output) > 100 else "")
                    safe_console_print(f"   🎯 {preview}")
                safe_console_print("")
                
            elif event_type == "RUN_COMPLETE":
                safe_console_print("🎉 Workflow completed!")
                break
                
            elif event_type == "ERROR":
                error = data.get("error", str(data))
                safe_console_print(f"❌ Error: {error}")
                
    except KeyboardInterrupt:
        safe_console_print("\n⏹️  Interrupted by user")
    except Exception as e:
        safe_console_print(f"💥 Execution failed: {e}")
        raise


@app.command()
def ui(
    run_id: str = typer.Argument(..., help="Run ID to monitor"),
    store: str = typer.Option("sqlite", help="Store type: sqlite or jsonl"),
    store_path: Optional[str] = typer.Option(None, help="Store path"),
):
    """Monitor an existing run with TUI dashboard."""
    
    # Create store instance
    if store_path is None:
        store_path = ".ao_runs/ao.sqlite3" if store == "sqlite" else ".ao_runs/"
    
    store_instance = create_store(store, store_path)
    
    # TODO: Implement monitoring of existing run from store
    # This would:
    # 1. Load existing events from store
    # 2. Replay them to build initial model state  
    # 3. Poll for new events periodically
    # 4. Show TUI dashboard
    pass


@app.command()
def ls(
    store: str = typer.Option("sqlite", help="Store type: sqlite|jsonl|auto"),
    db: Optional[str] = typer.Option(None, help="SQLite database path"),
    root: Optional[str] = typer.Option(None, help="JSONL root directory"),
    limit: int = typer.Option(20, help="Max runs to list")
):
    """List recent runs."""
    if store == "sqlite" or (store == "auto" and os.getenv("AO_STORE", "sqlite") == "sqlite"):
        _list_sqlite_runs(db, limit)
    else:
        _list_jsonl_runs(root, limit)


def _list_sqlite_runs(db: Optional[str], limit: int):
    """List runs from SQLite database."""
    import sqlite3
    
    db_path = Path(db or ".ao_runs/ao.sqlite3")
    if not db_path.exists():
        safe_console_print("[dim]No SQLite database found. Run a workflow first.[/]")
        return
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute('''
                SELECT run_id, status, goal, datetime(created_at, 'unixepoch') as created,
                       datetime(updated_at, 'unixepoch') as updated
                FROM runs 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
    except Exception as e:
        safe_console_print(f"[red]Error reading database: {e}[/]")
        return
    
    if not rows:
        safe_console_print("[dim]No runs found.[/]")
        return
    
    # Create table
    table = Table(title="Recent Runs")
    table.add_column("Run ID", style="bold")
    table.add_column("Status")
    table.add_column("Goal")
    table.add_column("Created")
    table.add_column("Updated")
    
    for row in rows:
        run_id, status, goal, created, updated = row
        
        # Style status
        status_style = {
            "complete": "green",
            "running": "yellow",
            "error": "red",
            "canceled": "blue"
        }.get(status, "white")
        
        table.add_row(
            run_id,
            f"[{status_style}]{status}[/]",
            goal or "[dim]No goal[/]",
            created.split('.')[0] if created else "-",  # Remove microseconds
            updated.split('.')[0] if updated else "-"
        )
    
    safe_console_print(table)


def _list_jsonl_runs(root: Optional[str], limit: int):
    """List runs from JSONL directory."""
    root_path = Path(root or ".ao_runs")
    if not root_path.exists():
        safe_console_print("[dim]No JSONL runs directory found.[/]")
        return
    
    # Get run directories sorted by modification time
    run_dirs = []
    for item in root_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            meta_file = item / "meta.json"
            if meta_file.exists():
                run_dirs.append((item.name, item.stat().st_mtime))
    
    run_dirs.sort(key=lambda x: x[1], reverse=True)
    run_dirs = run_dirs[:limit]
    
    if not run_dirs:
        safe_console_print("[dim]No JSONL runs found.[/]")
        return
    
    table = Table(title="Recent JSONL Runs")
    table.add_column("Run ID", style="bold")
    table.add_column("Goal")
    table.add_column("Modified")
    
    for run_id, mtime in run_dirs:
        try:
            meta_path = root_path / run_id / "meta.json"
            meta = json.loads(meta_path.read_text())
            goal = meta.get("run_spec", {}).get("goal", "No goal")
            modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            table.add_row(run_id, goal, modified)
        except Exception:
            table.add_row(run_id, "[dim]Error reading meta[/]", "-")
    
    safe_console_print(table)


@app.command()
def show(
    run_id: str,
    store: str = typer.Option("sqlite", help="Store type: sqlite|jsonl"),
    db: Optional[str] = typer.Option(None, help="SQLite database path"),
    limit: int = typer.Option(50, help="Max events to show")
):
    """Show events for a specific run."""
    if store == "sqlite":
        _show_sqlite_events(run_id, db, limit)
    else:
        _show_jsonl_events(run_id, limit)


def _show_sqlite_events(run_id: str, db: Optional[str], limit: int):
    """Show events from SQLite database."""
    import sqlite3
    
    db_path = Path(db or ".ao_runs/ao.sqlite3")
    if not db_path.exists():
        safe_console_print(f"[red]Database not found: {db_path}[/]")
        return
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute('''
                SELECT seq, type, node_id, data_json, datetime(ts, 'unixepoch') as timestamp
                FROM events 
                WHERE run_id = ? 
                ORDER BY seq DESC 
                LIMIT ?
            ''', (run_id, limit))
            
            rows = list(reversed(cursor.fetchall()))  # Show in chronological order
    except Exception as e:
        safe_console_print(f"[red]Error reading events: {e}[/]")
        return
    
    if not rows:
        safe_console_print(f"[dim]No events found for run {run_id}[/]")
        return
    
    safe_console_print(f"[bold]Events for run: {run_id}[/]\n")
    
    for seq, event_type, node_id, data_json, timestamp in rows:
        # Parse data for display
        try:
            data = json.loads(data_json) if data_json else {}
            data_preview = _format_event_data(data)
        except:
            data_preview = data_json[:50] if data_json else ""
        
        # Format timestamp
        time_str = timestamp.split('.')[0] if timestamp else "-"
        
        # Color code by event type
        type_style = {
            "RUN_START": "green",
            "NODE_START": "yellow",
            "NODE_COMPLETE": "green",
            "ERROR": "red",
            "RUN_COMPLETE": "green bold"
        }.get(event_type, "white")
        
        safe_console_print(f"[dim]{seq:04d}[/] [{type_style}]{event_type}[/] {node_id or '-':15} {time_str} {data_preview}")


def _show_jsonl_events(run_id: str, limit: int):
    """Show events from JSONL files."""
    events_file = Path(".ao_runs") / run_id / "events.jsonl"
    if not events_file.exists():
        safe_console_print(f"[red]Events file not found: {events_file}[/]")
        return
    
    try:
        lines = events_file.read_text().strip().split('\n')
        lines = [line for line in lines if line.strip()][-limit:]  # Last N lines
        
        safe_console_print(f"[bold]Events for run: {run_id}[/]\n")
        
        for line in lines:
            try:
                event = json.loads(line)
                event_type = event.get('type', 'UNKNOWN')
                node_id = event.get('node_id', '-')
                data_preview = _format_event_data(event.get('data', {}))
                
                type_style = {
                    "RUN_START": "green",
                    "NODE_START": "yellow", 
                    "NODE_COMPLETE": "green",
                    "ERROR": "red",
                    "RUN_COMPLETE": "green bold"
                }.get(event_type, "white")
                
                safe_console_print(f"[{type_style}]{event_type}[/] {node_id:15} {data_preview}")
            except Exception as e:
                safe_console_print(f"[dim]Invalid event: {line[:50]}...[/]")
                
    except Exception as e:
        safe_console_print(f"[red]Error reading events file: {e}[/]")


def _format_event_data(data: Dict[str, Any]) -> str:
    """Format event data for display."""
    if not data:
        return ""
    
    # Show key fields
    parts = []
    if "resumed" in data and data["resumed"]:
        parts.append("[cyan]resumed[/]")
    if "error" in data:
        parts.append(f"[red]error: {str(data['error'])[:50]}[/]")
    if "phase" in data:
        parts.append(f"phase: {data['phase']}")
    if "goal" in data:
        parts.append(f"goal: {data['goal'][:30]}")
    
    return " | ".join(parts)


@app.command()
def tail(
    run_id: str,
    store: str = typer.Option("sqlite", help="Store type: sqlite|jsonl"),
    db: Optional[str] = typer.Option(None, help="SQLite database path"),
    refresh: float = typer.Option(0.5, help="Refresh interval in seconds")
):
    """Attach to an existing run and follow events with live TUI."""
    if store != "sqlite":
        safe_console_print("[yellow]Tail is optimized for SQLite store[/]")
        return
    
    asyncio.run(_tail_run(run_id, db, refresh))


async def _tail_run(run_id: str, db: Optional[str], refresh: float):
    """Tail a run with live TUI."""
    import sqlite3
    
    db_path = Path(db or ".ao_runs/ao.sqlite3")
    if not db_path.exists():
        safe_console_print(f"[red]Database not found: {db_path}[/]")
        return
    
    model = RunTUIModel(run_id=run_id)
    last_seq = 0
    
    def _fetch_new_events():
        nonlocal last_seq
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute('''
                SELECT seq, type, node_id, data_json
                FROM events 
                WHERE run_id = ? AND seq > ? 
                ORDER BY seq ASC
            ''', (run_id, last_seq))
            
            rows = cursor.fetchall()
            
        if rows:
            last_seq = rows[-1][0]
        
        return rows
    
    # Initial load of run info
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("SELECT goal, status FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            model.goal = row[0] or ""
            model.run_status = row[1] or "unknown"
    
    with Live(model.render(), console=console, refresh_per_second=4) as live:
        try:
            while True:
                # Fetch and process new events
                for seq, event_type, node_id, data_json in _fetch_new_events():
                    try:
                        data = json.loads(data_json) if data_json else {}
                    except:
                        data = {}
                    
                    # Update model based on event type (same logic as run command)
                    if event_type == "NODE_START":
                        model.update_node_status(node_id, "running")
                        model.add_chunk(node_id, "Started")
                    elif event_type == "NODE_COMPLETE":
                        is_resumed = data.get("resumed", False)
                        status = "resumed" if is_resumed else "complete"
                        model.update_node_status(node_id, status)
                        model.add_chunk(node_id, "Completed" if not is_resumed else "Resumed")
                    elif event_type == "ERROR":
                        model.update_node_status(node_id, "error")
                        model.add_error(node_id, data.get("error", "Unknown error"))
                    elif event_type == "RUN_COMPLETE":
                        model.run_status = "complete" 
                        model.add_chunk("SYSTEM", "Run completed")
                        live.update(model.render())
                        await asyncio.sleep(2)  # Show final state
                        return
                
                live.update(model.render())
                await asyncio.sleep(refresh)
                
        except KeyboardInterrupt:
            safe_console_print("\n[yellow]Tail stopped by user[/]")


@app.command()
def resume(
    run_id: str,
    workflow: Path = typer.Argument(..., help="Python workflow file to resume"),
    store: str = typer.Option("sqlite", help="Store type: sqlite|jsonl|auto"),
    db: Optional[str] = typer.Option(None, help="SQLite database path"),
    root: Optional[str] = typer.Option(None, help="JSONL root directory"),
):
    """Resume a run from checkpoint with live TUI.
    
    Note: You must provide the same workflow file that was used to start the original run.
    The resume will pick up from where the run left off using stored checkpoints.
    """
    # Verify the run exists
    goal = f"Resumed: {run_id}"
    
    if store == "sqlite":
        try:
            import sqlite3
            db_path = Path(db or ".ao_runs/ao.sqlite3")
            if db_path.exists():
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.execute("SELECT goal FROM runs WHERE run_id = ?", (run_id,))
                    row = cursor.fetchone()
                    if not row:
                        safe_console_print(f"[red]Run {run_id} not found in database[/]")
                        raise typer.Exit(1)
                    if row[0]:
                        goal = f"Resume: {row[0]}"
        except Exception as e:
            safe_console_print(f"[red]Error checking run: {e}[/]")
            raise typer.Exit(1)
    
    # Call run function directly with resume parameter
    run(
        workflow=workflow,
        goal=goal,
        resume=run_id,  # Pass run_id as resume parameter
        watch=True,     # Default to TUI for resume
        store=store,
        store_path=db or root
    )


if __name__ == "__main__":
    app()
