"""CLI interface for Agent Orchestra (catenas command).

Provides command-line interface for running, monitoring, and managing orchestrations.
"""

import asyncio
import json
import sys
from pathlib import Path

import click

from agent_orchestra import Orchestrator, __version__
from agent_orchestra.agents_loader import AgentsLoaderError, validate_agents_config
from agent_orchestra.events import EventType, read_events_from_jsonl
from agent_orchestra.policy import HITLManager
from agent_orchestra.tools_loader import ToolsLoaderError, validate_tools_config


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Agent Orchestra CLI - Universal Multi-Agent Orchestrator.
    
    Main CLI entry point that provides commands for running, monitoring,
    and managing multi-agent orchestrations.
    """
    pass


@main.command()
@click.argument("graph_file", type=click.Path(exists=True, path_type=Path))
@click.option("--run-id", help="Custom run ID")
@click.option("--context", "-c", help="Additional context as JSON string")
@click.option("--tools", type=click.Path(exists=True, path_type=Path),
              help="Tools configuration file (.yaml/.yml/.json)")
@click.option("--agents", type=click.Path(exists=True, path_type=Path),
              help="Agents directory or file (.yaml/.yml/.json)")
@click.option("--checkpoint-dir", type=click.Path(path_type=Path),
              default="./checkpoints", help="Checkpoint directory")
@click.option("--event-dir", type=click.Path(path_type=Path),
              default="./events", help="Event log directory")
@click.option("--max-concurrency", type=int, default=10,
              help="Maximum concurrent node executions")
def run(
    graph_file: Path,
    run_id: str | None,
    context: str | None,
    tools: Path | None,
    agents: Path | None,
    checkpoint_dir: Path,
    event_dir: Path,
    max_concurrency: int
) -> None:
    """Run a graph from a JSON specification file.
    
    Executes a multi-agent orchestration from a graph specification file,
    with optional tools configuration, context, and execution parameters.
    
    Args:
        graph_file: Path to the JSON graph specification file
        run_id: Optional custom run identifier for this execution
        context: Additional context as JSON string to merge with graph context
        tools: Optional path to tools configuration file (.yaml/.yml/.json)
        checkpoint_dir: Directory to store execution checkpoints
        event_dir: Directory to store event logs
        max_concurrency: Maximum number of nodes to execute concurrently
        
    Raises:
        SystemExit: If validation fails, context is invalid JSON, or execution fails
    """

    async def _run() -> None:
        # Validate tools config if provided
        if tools:
            try:
                click.echo(f"Validating tools config: {tools}")
                validate_tools_config(tools)
                click.echo("✅ Tools config validation passed")
            except ToolsLoaderError as e:
                click.echo(f"❌ Tools config validation failed: {e}", err=True)
                sys.exit(1)

        # Validate agents config if provided
        if agents:
            try:
                click.echo(f"Validating agents config: {agents}")
                validate_agents_config(agents)
                click.echo("✅ Agents config validation passed")
            except AgentsLoaderError as e:
                click.echo(f"❌ Agents config validation failed: {e}", err=True)
                sys.exit(1)

        # Parse additional context
        ctx = {}
        if context:
            try:
                ctx = json.loads(context)
            except json.JSONDecodeError:
                click.echo(f"Error: Invalid JSON in context: {context}", err=True)
                sys.exit(1)

        # Create orchestrator
        orchestrator = Orchestrator(
            checkpoint_dir=checkpoint_dir,
            event_dir=event_dir,
            max_concurrency=max_concurrency
        )

        # Validate graph against agents requirements
        _validate_graph_with_agents(graph_file, agents)
        
        # Load and run graph
        try:
            click.echo(f"Loading graph from {graph_file}")
            result = await orchestrator.run(
                graph=graph_file,
                ctx=ctx,
                run_id=run_id,
                tools_file=tools,
                agents_path=agents
            )

            if result.success:
                click.echo("✅ Run completed successfully")
                click.echo(f"Run ID: {result.run_id}")
                click.echo(f"Total tokens: {result.total_tokens}")
                click.echo(f"Total cost: ${result.total_cost:.4f}")
                click.echo(f"Total time: {result.total_time:.2f}s")

                if result.outputs:
                    click.echo("\nOutputs:")
                    click.echo(json.dumps(result.outputs, indent=2))
            else:
                click.echo(f"❌ Run failed: {result.error}", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_run())


@main.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
def validate(files: tuple[Path, ...]) -> None:
    """Validate configuration files (tools.yaml, graph.json, etc.).
    
    Validates one or more configuration files, automatically detecting
    file types and applying appropriate validation rules.
    
    Args:
        files: Tuple of file paths to validate
        
    Raises:
        SystemExit: If any validation fails (exits with code 1)
    """

    exit_code = 0

    for file_path in files:
        click.echo(f"Validating {file_path}")

        try:
            # Handle directories (agents directories)
            if file_path.is_dir():
                try:
                    from agent_orchestra.agents_loader import load_agents_config
                    result = load_agents_config(file_path)
                    
                    # Count agents by provider
                    provider_counts = {}
                    version_counts = {}
                    
                    for agent in result.agents_list:
                        model_id = agent.get("model", "")
                        if ":" in model_id:
                            provider = model_id.split(":", 1)[0]
                            provider_counts[provider] = provider_counts.get(provider, 0) + 1
                        
                        agent_id = agent.get("id", "")
                        if "@v" in agent_id:
                            version = agent_id.split("@v", 1)[1]
                            version_counts[f"v{version}"] = version_counts.get(f"v{version}", 0) + 1
                    
                    click.echo(f"✅ {file_path}: Valid agents directory")
                    click.echo(f"   Agents: {len(result.agents_list)} total")
                    
                    if provider_counts:
                        providers_str = ", ".join(f"{provider}: {count}" for provider, count in sorted(provider_counts.items()))
                        click.echo(f"   Providers: {providers_str}")
                    
                    if version_counts:
                        versions_str = ", ".join(f"{version}: {count}" for version, count in sorted(version_counts.items()))
                        click.echo(f"   Versions: {versions_str}")
                    
                    continue
                except AgentsLoaderError as e:
                    click.echo(f"❌ {file_path}: Invalid agents directory: {e}", err=True)
                    exit_code = 1
                    continue
            
            # Handle files
            if file_path.suffix.lower() in {'.yaml', '.yml'}:
                # YAML file - try as tools config, then agents config
                tools_valid = False
                agents_valid = False
                
                try:
                    validate_tools_config(file_path)
                    click.echo(f"✅ {file_path}: Valid tools configuration")
                    tools_valid = True
                except ToolsLoaderError:
                    pass
                
                if not tools_valid:
                    try:
                        validate_agents_config(file_path)
                        click.echo(f"✅ {file_path}: Valid agents configuration")
                        agents_valid = True
                    except AgentsLoaderError:
                        pass
                
                if not tools_valid and not agents_valid:
                    click.echo(f"❌ {file_path}: Not a valid tools or agents configuration", err=True)
                    exit_code = 1
                    
            elif file_path.suffix.lower() == '.json':
                # JSON file - try as tools config, then agents config, then generic JSON
                tools_valid = False
                agents_valid = False
                
                try:
                    validate_tools_config(file_path)
                    click.echo(f"✅ {file_path}: Valid tools configuration")
                    tools_valid = True
                except ToolsLoaderError:
                    pass
                
                if not tools_valid:
                    try:
                        validate_agents_config(file_path)
                        click.echo(f"✅ {file_path}: Valid agents configuration")
                        agents_valid = True
                    except AgentsLoaderError:
                        pass
                
                if not tools_valid and not agents_valid:
                    # Try as generic JSON
                    try:
                        with open(file_path) as f:
                            json.load(f)
                        click.echo(f"✅ {file_path}: Valid JSON")
                    except json.JSONDecodeError as e:
                        click.echo(f"❌ {file_path}: Invalid JSON: {e}", err=True)
                        exit_code = 1
            else:
                click.echo(f"⚠️  {file_path}: Unknown file type, skipping validation")

        except ToolsLoaderError as e:
            click.echo(f"❌ {file_path}: {e}", err=True)
            exit_code = 1
        except Exception as e:
            click.echo(f"❌ {file_path}: Validation error: {e}", err=True)
            exit_code = 1

    if exit_code == 0:
        click.echo("✅ All files validated successfully")
    else:
        sys.exit(exit_code)


@main.command()
@click.argument("run_id")
@click.option("--event-dir", type=click.Path(path_type=Path),
              default="./events", help="Event log directory")
@click.option("--follow", "-f", is_flag=True, help="Follow new events")
@click.option("--filter", "event_filter", help="Filter by event type")
def tail(run_id: str, event_dir: Path, follow: bool, event_filter: str | None) -> None:
    """Tail events from a running or completed orchestration.
    
    Displays events from an orchestration run in real-time or from
    a completed run's event log file.
    
    Args:
        run_id: Unique identifier for the orchestration run
        event_dir: Directory containing event log files
        follow: Whether to follow new events (TODO: not yet implemented)
        event_filter: Optional event type filter to apply
        
    Raises:
        SystemExit: If event file not found or reading fails
    """

    async def _tail() -> None:
        event_file = event_dir / f"{run_id}.jsonl"

        if not event_file.exists():
            click.echo(f"❌ Event file not found: {event_file}", err=True)
            sys.exit(1)

        try:
            async for event in read_events_from_jsonl(event_file):
                # Apply filter if specified
                if event_filter and event.type.value != event_filter:
                    continue

                # Format event for display
                timestamp = event.timestamp
                event_type = event.type.value
                node_id = event.node_id or "-"

                click.echo(f"[{timestamp:.3f}] {event_type:20} {node_id:15} {event.payload}")

                # TODO: Implement follow mode with file watching
                if follow:
                    pass  # Would need inotify/polling here

        except Exception as e:
            click.echo(f"❌ Error reading events: {e}", err=True)
            sys.exit(1)

    asyncio.run(_tail())


@main.command()
@click.argument("checkpoint_id")
@click.option("--checkpoint-dir", type=click.Path(path_type=Path),
              default="./checkpoints", help="Checkpoint directory")
@click.option("--event-dir", type=click.Path(path_type=Path),
              default="./events", help="Event log directory")
def resume(checkpoint_id: str, checkpoint_dir: Path, event_dir: Path) -> None:
    """Resume execution from a checkpoint.
    
    Restores and continues execution of an orchestration from a previously
    saved checkpoint, maintaining all state and progress.
    
    Args:
        checkpoint_id: Unique identifier of the checkpoint to resume from
        checkpoint_dir: Directory containing checkpoint files
        event_dir: Directory to store event logs for resumed execution
        
    Raises:
        SystemExit: If checkpoint not found or resume fails
    """

    async def _resume() -> None:
        orchestrator = Orchestrator(
            checkpoint_dir=checkpoint_dir,
            event_dir=event_dir
        )

        try:
            click.echo(f"Resuming from checkpoint {checkpoint_id}")
            result = await orchestrator.resume(checkpoint_id)

            if result.success:
                click.echo("✅ Resume completed successfully")
                click.echo(f"Run ID: {result.run_id}")
                click.echo(f"Total tokens: {result.total_tokens}")
                click.echo(f"Total cost: ${result.total_cost:.4f}")
                click.echo(f"Total time: {result.total_time:.2f}s")
            else:
                click.echo(f"❌ Resume failed: {result.error}", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_resume())


@main.command()
@click.argument("run_id")
@click.argument("node_id")
@click.option("--reason", required=True, help="Reason for approval")
@click.option("--approver", help="Approver name", default="cli-user")
def approve(run_id: str, node_id: str, reason: str, approver: str) -> None:
    """Approve a pending HITL request.
    
    Approves a human-in-the-loop (HITL) request that is blocking
    execution of a specific node in an orchestration.
    
    Args:
        run_id: Unique identifier for the orchestration run
        node_id: Identifier of the node with pending approval
        reason: Human-readable reason for the approval
        approver: Name/identifier of the person approving
        
    Raises:
        SystemExit: If approval request not found
    """

    # This is a simplified implementation
    # In production, this would connect to a shared approval system
    hitl_manager = HITLManager()

    approval_id = f"{run_id}_{node_id}"

    if hitl_manager.approve(approval_id, approver, reason):
        click.echo(f"✅ Approved {approval_id}")
    else:
        click.echo(f"❌ Approval not found: {approval_id}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id")
@click.argument("node_id")
@click.option("--reason", required=True, help="Reason for denial")
@click.option("--approver", help="Approver name", default="cli-user")
def deny(run_id: str, node_id: str, reason: str, approver: str) -> None:
    """Deny a pending HITL request.
    
    Denies a human-in-the-loop (HITL) request, which will typically
    cause the associated node execution to fail or be skipped.
    
    Args:
        run_id: Unique identifier for the orchestration run
        node_id: Identifier of the node with pending approval
        reason: Human-readable reason for the denial
        approver: Name/identifier of the person denying
        
    Raises:
        SystemExit: If approval request not found
    """

    hitl_manager = HITLManager()

    approval_id = f"{run_id}_{node_id}"

    if hitl_manager.deny(approval_id, approver, reason):
        click.echo(f"❌ Denied {approval_id}")
    else:
        click.echo(f"❌ Approval not found: {approval_id}", err=True)
        sys.exit(1)


@main.command()
@click.argument("run_id")
@click.option("--span-id", help="Specific span to inspect")
@click.option("--event-dir", type=click.Path(path_type=Path),
              default="./events", help="Event log directory")
def inspect(run_id: str, span_id: str | None, event_dir: Path) -> None:
    """Inspect a run with detailed event information.
    
    Provides detailed analysis and timeline view of an orchestration run,
    including event summary, timeline, and payload inspection.
    
    Args:
        run_id: Unique identifier for the orchestration run
        span_id: Optional specific span to filter events by
        event_dir: Directory containing event log files
        
    Raises:
        SystemExit: If event file not found or no events found
    """

    async def _inspect() -> None:
        event_file = event_dir / f"{run_id}.jsonl"

        if not event_file.exists():
            click.echo(f"❌ Event file not found: {event_file}", err=True)
            sys.exit(1)

        events = []
        async for event in read_events_from_jsonl(event_file):
            if span_id is None or event.span_id == span_id:
                events.append(event)

        if not events:
            click.echo(f"❌ No events found for run {run_id}", err=True)
            sys.exit(1)

        # Group events by type
        event_summary = {}
        for event in events:
            event_type = event.type.value
            if event_type not in event_summary:
                event_summary[event_type] = 0
            event_summary[event_type] += 1

        click.echo(f"Run ID: {run_id}")
        if span_id:
            click.echo(f"Span ID: {span_id}")
        click.echo(f"Total events: {len(events)}")
        click.echo("\nEvent summary:")
        for event_type, count in sorted(event_summary.items()):
            click.echo(f"  {event_type}: {count}")

        # Show timeline
        click.echo("\nTimeline:")
        for event in sorted(events, key=lambda e: e.timestamp):
            timestamp = event.timestamp
            event_type = event.type.value
            node_id = event.node_id or "-"

            click.echo(f"  [{timestamp:.3f}] {event_type:20} {node_id}")

            # Show payload for important events
            if event.type in [EventType.NODE_START, EventType.NODE_COMPLETE, EventType.NODE_ERROR]:
                if event.payload:
                    payload_str = json.dumps(event.payload, indent=4)
                    click.echo(f"    {payload_str}")

    asyncio.run(_inspect())


@main.command()
@click.option("--checkpoint-dir", type=click.Path(path_type=Path),
              default="./checkpoints", help="Checkpoint directory")
def list_checkpoints(checkpoint_dir: Path) -> None:
    """List all available checkpoints.
    
    Displays information about all checkpoints stored in the checkpoint
    directory, including metadata and execution statistics.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
    """

    from agent_orchestra.checkpointer import Checkpointer

    checkpointer = Checkpointer(checkpoint_dir)
    checkpoints = checkpointer.list_checkpoints()

    if not checkpoints:
        click.echo("No checkpoints found")
        return

    click.echo(f"Found {len(checkpoints)} checkpoints:")
    click.echo()

    for checkpoint in checkpoints:
        click.echo(f"Checkpoint ID: {checkpoint.checkpoint_id}")
        click.echo(f"  Run ID: {checkpoint.run_id}")
        click.echo(f"  Timestamp: {checkpoint.timestamp}")
        click.echo(f"  Completed nodes: {len(checkpoint.completed_nodes)}")
        click.echo(f"  Failed nodes: {len(checkpoint.failed_nodes)}")
        click.echo(f"  Tokens used: {checkpoint.total_tokens}")
        click.echo(f"  Cost: ${checkpoint.total_cost:.4f}")
        click.echo()


@main.command()
def create_example() -> None:
    """Create an example graph file.
    
    Generates a sample orchestration graph file demonstrating
    basic multi-agent workflow patterns and configuration.
    """

    example_graph = {
        "nodes": {
            "search": {
                "type": "mcp_agent",
                "adapter": "mcp_use",
                "config": {
                    "agent_config": {
                        "mcpServers": {
                            "browser": {
                                "command": "npx",
                                "args": ["@playwright/mcp@latest"]
                            }
                        }
                    },
                    "tool": "web_search",
                    "max_steps": 5
                }
            },
            "analyze": {
                "type": "mcp_agent",
                "adapter": "mcp_use",
                "config": {
                    "agent_config": {
                        "mcpServers": {
                            "analysis": {
                                "command": "python",
                                "args": ["analysis_server.py"]
                            }
                        }
                    },
                    "tool": "analyze",
                    "max_steps": 3
                }
            },
            "summarize": {
                "type": "mcp_agent",
                "adapter": "mcp_use",
                "config": {
                    "agent_config": "{}",
                    "tool": "summarize",
                    "max_steps": 2
                }
            }
        },
        "edges": {
            "search": ["analyze"],
            "analyze": ["summarize"]
        },
        "ctx": {
            "query": "What are the latest trends in AI research?",
            "max_results": 5
        }
    }

    output_file = Path("example_graph.json")
    with open(output_file, 'w') as f:
        json.dump(example_graph, f, indent=2)

    click.echo(f"Created example graph: {output_file}")
    click.echo("Run with: catenas run example_graph.json")


def _validate_graph_with_agents(graph_file: Path, agents_path: Path | None) -> None:
    """Validate graph against agents requirements."""
    import json
    
    try:
        with open(graph_file) as f:
            graph_data = json.load(f)
    except Exception as e:
        click.echo(f"❌ Failed to load graph file: {e}", err=True)
        sys.exit(1)
    
    nodes = graph_data.get("nodes", {})
    
    for node_id, node_spec in nodes.items():
        if node_spec.get("type") == "mcp_agent":
            config = node_spec.get("config", {})
            
            if agents_path:
                # --agents provided: require config.agent_id
                if "agent_id" not in config:
                    click.echo(f"❌ Node '{node_id}' type=mcp_agent requires config.agent_id when --agents is provided", err=True)
                    sys.exit(1)
            else:
                # --agents absent: allow config.model (with provider prefix), warn about inline fallback
                if "model" not in config:
                    click.echo(f"❌ Node '{node_id}' type=mcp_agent requires config.model when --agents is not provided", err=True)
                    sys.exit(1)
                
                model = config["model"]
                if ":" not in model:
                    click.echo(f"❌ Node '{node_id}' config.model must have provider prefix (e.g., 'openai:gpt-4o-mini')", err=True)
                    sys.exit(1)
                
                click.echo(f"⚠️  WARNING: Node '{node_id}' using inline fallback model '{model}'. Consider using --agents with agent_id instead.")


if __name__ == "__main__":
    main()

