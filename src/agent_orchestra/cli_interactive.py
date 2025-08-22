"""
Agent Orchestra Interactive CLI - Claude Code style interface

Features:
- Scrollable logs panel
- Node selection and detailed views  
- Graph visualization
- Interactive navigation with keyboard controls
- Clean organized layout
"""

from __future__ import annotations
import asyncio
import json
import time
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

try:
    import typer
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.align import Align
except ImportError:
    print("CLI dependencies not installed. Run: pip install 'agentic-orchestra[cli]'")
    sys.exit(1)

# Cross-platform keyboard input handling
try:
    import termios
    import tty
    import select
    
    def get_key():
        """Get single key press (Unix)."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0.1):
                key = sys.stdin.read(1)
                if key == '\x1b':  # ESC sequence
                    key = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.1) else key
                return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None
            
except ImportError:
    # Windows fallback
    try:
        import msvcrt
        def get_key():
            """Get single key press (Windows)."""
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8')
            return None
    except ImportError:
        def get_key():
            """Fallback - no keyboard input."""
            return None

from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.orchestrator.agent_pool import AgentPool, create_default_agent_factory
from agent_orchestra.orchestrator.broker_config import create_development_broker
from agent_orchestra.orchestrator.store_factory import create_store
from agent_orchestra.sidecar.sidecar_client import SidecarMCPClient

console = Console()
app = typer.Typer(help="Agent Orchestra Interactive CLI - Claude Code style")


@dataclass
class LogEntry:
    timestamp: datetime
    node_id: str
    level: str  # INFO, AGENT, ERROR, SYSTEM
    message: str
    data: Optional[Dict] = None


@dataclass 
class NodeState:
    id: str
    name: str
    status: str  # queued, running, complete, error
    server_name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[LogEntry] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []


class InteractiveRunner:
    def __init__(self, run_id: str, graph_spec: GraphSpec):
        self.run_id = run_id
        self.graph_spec = graph_spec
        self.nodes: Dict[str, NodeState] = {}
        self.global_logs: List[LogEntry] = []
        self.selected_node: Optional[str] = None
        self.run_status = "running"
        self.start_time = datetime.now()
        self.keyboard_thread = None
        self.should_exit = False
        self.needs_update = True  # Track if UI needs refresh
        self.last_render_time = time.time()
        
        # Initialize nodes
        for node in graph_spec.nodes:
            self.nodes[node.id] = NodeState(
                id=node.id,
                name=node.name or node.id,
                status="queued",
                server_name=getattr(node, 'server_name', 'default')
            )
            
        # Start with first node selected
        if self.nodes:
            self.selected_node = list(self.nodes.keys())[0]
    
    def add_log(self, node_id: str, level: str, message: str, data: Optional[Dict] = None):
        """Add log entry to both global and node-specific logs."""
        entry = LogEntry(
            timestamp=datetime.now(),
            node_id=node_id,
            level=level,
            message=message,
            data=data
        )
        
        self.global_logs.append(entry)
        if node_id in self.nodes:
            self.nodes[node_id].logs.append(entry)
        
        # Keep logs bounded
        if len(self.global_logs) > 1000:
            self.global_logs = self.global_logs[-800:]
            
        # Mark for update
        self.needs_update = True
    
    def update_node_status(self, node_id: str, status: str, **kwargs):
        """Update node status and timing."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        old_status = node.status
        node.status = status
        
        if status == "running" and node.started_at is None:
            node.started_at = datetime.now()
        elif status in ["complete", "error"] and node.completed_at is None:
            node.completed_at = datetime.now()
            if node.started_at:
                node.duration = (node.completed_at - node.started_at).total_seconds()
        
        if status == "error" and "error" in kwargs:
            node.error = kwargs["error"]
            
        # Mark for update only if status actually changed
        if old_status != status:
            self.needs_update = True
    
    def handle_keyboard_input(self):
        """Handle keyboard input for navigation."""
        while not self.should_exit:
            key = get_key()
            if key:
                if key == '\x03':  # Ctrl+C
                    self.should_exit = True
                    break
                elif key == '[A' or key == 'k':  # Up arrow or k
                    self.select_previous_node()
                    self.needs_update = True
                elif key == '[B' or key == 'j':  # Down arrow or j
                    self.select_next_node()
                    self.needs_update = True
                elif key == '\r' or key == ' ':  # Enter or Space
                    # Toggle between global and node-specific logs
                    pass
                elif key == 'q':
                    self.should_exit = True
                    break
            time.sleep(0.05)  # Small delay to prevent high CPU usage
    
    def select_previous_node(self):
        """Select previous node in list."""
        if not self.nodes:
            return
            
        node_ids = list(self.nodes.keys())
        if self.selected_node in node_ids:
            current_idx = node_ids.index(self.selected_node)
            new_idx = (current_idx - 1) % len(node_ids)
            self.selected_node = node_ids[new_idx]
        else:
            self.selected_node = node_ids[0]
    
    def select_next_node(self):
        """Select next node in list."""
        if not self.nodes:
            return
            
        node_ids = list(self.nodes.keys())
        if self.selected_node in node_ids:
            current_idx = node_ids.index(self.selected_node)
            new_idx = (current_idx + 1) % len(node_ids)
            self.selected_node = node_ids[new_idx]
        else:
            self.selected_node = node_ids[0]
    
    def start_keyboard_thread(self):
        """Start keyboard input handling in background thread."""
        if self.keyboard_thread is None:
            self.keyboard_thread = threading.Thread(
                target=self.handle_keyboard_input,
                daemon=True
            )
            self.keyboard_thread.start()
    
    def stop_keyboard_thread(self):
        """Stop keyboard input handling."""
        self.should_exit = True
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1)
    
    def create_layout(self) -> Layout:
        """Create the main layout with panels."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )
        
        layout["left"].split_column(
            Layout(name="graph", ratio=1),
            Layout(name="nodes", ratio=1)
        )
        
        layout["right"].split_column(
            Layout(name="details", ratio=1), 
            Layout(name="logs", ratio=2)
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render header with run info."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        status_color = {
            "running": "yellow",
            "complete": "green", 
            "error": "red",
            "canceled": "red"
        }.get(self.run_status, "white")
        
        completed_count = sum(1 for node in self.nodes.values() if node.status == "complete")
        total_count = len(self.nodes)
        
        header_text = Text()
        header_text.append(f"Run: {self.run_id}  ", style="bold cyan")
        header_text.append(f"Status: ", style="dim")
        header_text.append(f"{self.run_status}  ", style=f"bold {status_color}")
        header_text.append(f"Progress: {completed_count}/{total_count}  ", style="dim")
        header_text.append(f"Elapsed: {elapsed:.1f}s", style="dim")
        
        return Panel(Align.center(header_text), title="Agent Orchestra", border_style="blue")
    
    def render_graph(self) -> Panel:
        """Render graph structure as tree."""
        tree = Tree("🔗 Workflow Graph")
        
        # Build adjacency list
        edges = {edge[0]: edge[1] for edge in self.graph_spec.edges}
        
        # Find root nodes (no incoming edges)
        all_targets = set(edge[1] for edge in self.graph_spec.edges)
        root_nodes = [node.id for node in self.graph_spec.nodes if node.id not in all_targets]
        
        def add_node_to_tree(parent_tree, node_id: str, visited: set):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if not node:
                return
                
            # Style based on status
            status_style = {
                "queued": "dim",
                "running": "yellow", 
                "complete": "green",
                "error": "red"
            }.get(node.status, "white")
            
            icon = {"queued": "⏳", "running": "🔄", "complete": "✅", "error": "❌"}.get(node.status, "⚪")
            
            node_text = f"{icon} {node.name}"
            if node_id == self.selected_node:
                node_text = f"[bold underline]{node_text}[/]"
                
            branch = parent_tree.add(Text(node_text, style=status_style))
            
            # Add children
            next_node = edges.get(node_id)
            if next_node:
                add_node_to_tree(branch, next_node, visited)
        
        # Add all root nodes
        visited = set()
        for root in root_nodes:
            add_node_to_tree(tree, root, visited)
        
        return Panel(tree, title="Graph", border_style="green")
    
    def render_nodes(self) -> Panel:
        """Render nodes list."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Node", style="cyan", width=15)
        table.add_column("Status", width=8)
        table.add_column("Server", width=8, style="dim")
        table.add_column("Duration", width=8, style="dim")
        
        for node in self.nodes.values():
            status_style = {
                "queued": "dim",
                "running": "yellow",
                "complete": "green", 
                "error": "red"
            }.get(node.status, "white")
            
            duration_str = f"{node.duration:.1f}s" if node.duration else "-"
            if node.status == "running" and node.started_at:
                current_duration = (datetime.now() - node.started_at).total_seconds()
                duration_str = f"{current_duration:.1f}s"
            
            # Highlight selected node
            node_name = node.name[:13] + ".." if len(node.name) > 15 else node.name
            if node.id == self.selected_node:
                node_name = f"[bold underline]{node_name}[/]"
            
            table.add_row(
                node_name,
                Text(node.status, style=status_style),
                node.server_name[:6] + ".." if len(node.server_name) > 8 else node.server_name,
                duration_str
            )
        
        return Panel(table, title="Nodes", border_style="cyan")
    
    def render_details(self) -> Panel:
        """Render details for selected node."""
        if not self.selected_node or self.selected_node not in self.nodes:
            return Panel(
                Align.center(Text("Select a node to view details", style="dim")),
                title="Node Details",
                border_style="yellow"
            )
        
        node = self.nodes[self.selected_node]
        
        details = Table(show_header=False, box=None)
        details.add_column("Key", style="bold", width=12)
        details.add_column("Value")
        
        details.add_row("ID:", node.id)
        details.add_row("Name:", node.name)
        details.add_row("Status:", Text(node.status, style={
            "queued": "dim", "running": "yellow", "complete": "green", "error": "red"
        }.get(node.status, "white")))
        details.add_row("Server:", node.server_name)
        
        if node.started_at:
            details.add_row("Started:", node.started_at.strftime("%H:%M:%S"))
        if node.completed_at:
            details.add_row("Completed:", node.completed_at.strftime("%H:%M:%S"))
        if node.duration:
            details.add_row("Duration:", f"{node.duration:.1f}s")
        if node.error:
            details.add_row("Error:", Text(node.error[:50] + "..." if len(node.error) > 50 else node.error, style="red"))
        
        details.add_row("Logs:", f"{len(node.logs)} entries")
        
        return Panel(details, title=f"Details: {node.name}", border_style="yellow")
    
    def render_logs(self) -> Panel:
        """Render logs panel with scrolling."""
        if self.selected_node and self.selected_node in self.nodes:
            # Show node-specific logs
            logs = self.nodes[self.selected_node].logs[-20:]  # Last 20 entries
            title = f"Logs: {self.nodes[self.selected_node].name}"
        else:
            # Show global logs
            logs = self.global_logs[-20:]  # Last 20 entries
            title = "Global Logs"
        
        log_text = Text()
        for entry in logs:
            time_str = entry.timestamp.strftime("%H:%M:%S")
            level_style = {
                "INFO": "dim",
                "AGENT": "cyan",
                "ERROR": "red", 
                "SYSTEM": "green"
            }.get(entry.level, "white")
            
            log_text.append(f"[{time_str}] ", style="dim")
            log_text.append(f"{entry.level}: ", style=level_style)
            log_text.append(f"{entry.message}\n")
        
        return Panel(log_text, title=title, border_style="blue", height=15)
    
    def render_footer(self) -> Panel:
        """Render footer with controls."""
        footer_text = Text()
        footer_text.append("Controls: ", style="bold")
        footer_text.append("↑/↓ or j/k Navigate  ", style="dim")
        footer_text.append("Space Toggle Logs  ", style="dim") 
        footer_text.append("q or Ctrl+C Exit", style="dim")
        
        return Panel(Align.center(footer_text), border_style="dim")
    
    def render(self) -> Layout:
        """Render complete interface."""
        layout = self.create_layout()
        
        layout["header"].update(self.render_header())
        layout["graph"].update(self.render_graph())
        layout["nodes"].update(self.render_nodes())
        layout["details"].update(self.render_details())
        layout["logs"].update(self.render_logs())
        layout["footer"].update(self.render_footer())
        
        return layout


# Copy MCP integration functions from cli_mcp.py
def _load_graph(path: Path) -> tuple[GraphSpec, Optional[Dict]]:
    """Load graph specification from JSON file, including embedded MCP config."""
    try:
        data = json.loads(path.read_text())
        nodes = [NodeSpec(**n) for n in data.get("nodes", [])]
        edges = [tuple(e) for e in data.get("edges", [])]
        graph_spec = GraphSpec(nodes=nodes, edges=edges)
        
        meta = data.get("_meta", {})
        embedded_config = meta.get("mcp_config")
        
        if embedded_config:
            console.print("[green]✓[/] Found embedded MCP configuration in graph file")
            
        return graph_spec, embedded_config
    except Exception as e:
        console.print(f"[red]Error loading graph from {path}: {e}[/]")
        raise typer.Exit(1)


def _load_mcp_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load MCP configuration from file or create default."""
    # Default configuration for demo/development
    return {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path.cwd())]
            }
        },
        "sidecar": {
            "policy": {
                "disallowed_tools": []
            },
            "run_context": {
                "session_id": "interactive_cli_session"
            }
        }
    }


async def _create_mcp_executor(
    config: Optional[Dict] = None,
    model_key: str = "gpt-4o-mini",
    default_server: str = "filesystem"
) -> MCPExecutor:
    """Create MCPExecutor with proper MCP client and agent pool."""
    
    # Load MCP configuration
    if config is None:
        config = _load_mcp_config()
    
    # Create MCP client
    from langchain_openai import ChatOpenAI
    
    client = SidecarMCPClient.from_dict(config)
    llm = ChatOpenAI(model=model_key, temperature=0)
    
    # Create agent factory and pool
    agent_factory = create_default_agent_factory(client, llm)
    agent_pool = AgentPool(agent_factory, max_agents_per_run=5)
    
    # Create broker for rate limiting
    broker = create_development_broker()
    
    # Create MCPExecutor
    executor = MCPExecutor(
        agent=None,  # Use agent_pool instead
        default_server=default_server,
        broker=broker,
        agent_pool=agent_pool,
        model_key=model_key
    )
    
    return executor


@app.command()
def run(
    graph: Path = typer.Argument(..., help="Path to workflow JSON file", exists=True, readable=True),
    run_id: Optional[str] = typer.Option(None, help="Run ID (auto-generated if not provided)"),
    goal: str = typer.Option("", help="Run goal/description"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model to use"),
):
    """Execute a workflow with interactive Claude Code style interface."""
    
    # Load graph and embedded config
    try:
        graph_spec, embedded_config = _load_graph(graph)
    except Exception:
        return
    
    rid = run_id or f"run-{int(time.time())}"
    
    console.print(f"[bold]Starting interactive workflow:[/] {graph}")
    console.print(f"[dim]Run ID: {rid}[/]")
    console.print()
    
    # Execute with interactive interface
    asyncio.run(_run_interactive(
        graph_spec=graph_spec,
        run_spec=RunSpec(rid, goal),
        embedded_config=embedded_config,
        model_key=model
    ))


async def _run_interactive(
    graph_spec: GraphSpec,
    run_spec: RunSpec,
    embedded_config: Optional[Dict],
    model_key: str
):
    """Execute workflow with interactive interface."""
    
    # Create MCP executor
    mcp_config = embedded_config if embedded_config else _load_mcp_config()
    executor = await _create_mcp_executor(mcp_config, model_key, "filesystem")
    
    # Create store and orchestrator
    store_instance = create_store("sqlite", ".ao_runs/ao.sqlite3")
    orchestrator = Orchestrator(executor, store=store_instance)
    
    # Set run context for agent pool
    executor.set_run_context(run_spec.run_id)
    
    # Create interactive runner
    runner = InteractiveRunner(run_spec.run_id, graph_spec)
    
    # Start the interactive interface with proper screen clearing
    with Live(runner.render(), refresh_per_second=2, console=console, screen=True) as live:
        
        # Start keyboard input handling
        runner.start_keyboard_thread()
        
        # Add initial log
        runner.add_log("SYSTEM", "SYSTEM", f"🚀 Starting workflow execution...")
        
        try:
            # Process workflow events
            async for event in orchestrator.run_streaming(graph_spec, run_spec):
                
                # Check for exit signal from keyboard
                if runner.should_exit:
                    break
                event_type = event.type
                node_id = getattr(event, 'node_id', None)
                data = getattr(event, 'data', {})
                
                if event_type == "RUN_START":
                    runner.add_log("SYSTEM", "SYSTEM", f"Run started: {run_spec.run_id}")
                
                elif event_type == "NODE_START":
                    if node_id:
                        runner.update_node_status(node_id, "running")
                        server_name = getattr([n for n in graph_spec.nodes if n.id == node_id][0], 'server_name', 'default')
                        runner.add_log(node_id, "INFO", f"Started on server: {server_name}")
                
                elif event_type == "AGENT_CHUNK":
                    if node_id and data:
                        # Extract meaningful content
                        if isinstance(data, tuple) and len(data) >= 2:
                            content = str(data[1])[:100] if data[1] else ""
                        else:
                            content = str(data)[:100]
                        
                        if content and len(content.strip()) > 10:
                            runner.add_log(node_id, "AGENT", content)
                
                elif event_type == "NODE_COMPLETE":
                    if node_id:
                        runner.update_node_status(node_id, "complete")
                        runner.add_log(node_id, "INFO", "✅ Completed successfully")
                
                elif event_type == "ERROR":
                    if node_id:
                        error_msg = str(data) if data else "Unknown error"
                        runner.update_node_status(node_id, "error", error=error_msg)
                        runner.add_log(node_id, "ERROR", f"❌ Error: {error_msg}")
                
                elif event_type == "RUN_COMPLETE":
                    runner.run_status = "complete"
                    runner.add_log("SYSTEM", "SYSTEM", "🎉 Workflow completed successfully!")
                    live.update(runner.render())
                    await asyncio.sleep(3)  # Show final state
                    return
                
                # Update display only when needed
                if runner.needs_update:
                    live.update(runner.render())
                    runner.needs_update = False
                
        except KeyboardInterrupt:
            runner.run_status = "canceled"
            runner.add_log("SYSTEM", "ERROR", "❌ Execution canceled by user")
            live.update(runner.render())
            console.print("\n[yellow]Execution canceled by user[/]")
        except Exception as e:
            runner.run_status = "error"
            runner.add_log("SYSTEM", "ERROR", f"❌ Execution failed: {e}")
            live.update(runner.render())
            console.print(f"\n[red]Execution failed: {e}[/]")
        finally:
            # Clean up keyboard thread
            runner.stop_keyboard_thread()


if __name__ == "__main__":
    app()