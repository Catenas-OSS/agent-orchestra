"""
TUI state model - pure state, no Rich dependencies, easy to test.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Literal, Dict, Optional, Any, List
import time
import hashlib


@dataclass
class ToolRow:
    """A single tool execution entry for tool trace."""
    name: str
    started_at: float
    ended_at: Optional[float] = None
    ok: bool = True
    args_preview: str = ""
    diff_preview: Optional[str] = None
    bytes_in: Optional[int] = None
    bytes_out: Optional[int] = None
    path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class InstructionFingerprint:
    """Agent instructions with fingerprinting for change detection."""
    task_title: Optional[str] = None
    task_body: str = ""
    system: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    
    def __post_init__(self):
        """Generate fingerprint from instruction content."""
        content = f"{self.task_body}|{self.system}|{len(self.tools)}|{str(self.policy)}"
        self.fingerprint = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class NodeState:
    """Enhanced state for a single node with full observability."""
    id: str
    name: Optional[str] = None
    server: Optional[str] = None
    model: Optional[str] = None
    status: Literal["pending", "running", "complete", "resumed", "error", "skipped"] = "pending"
    
    # Timing
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    
    # Attempts and retries
    attempt: int = 1
    max_attempts: int = 1
    
    # For foreach nodes
    items_done: int = 0
    items_total: Optional[int] = None
    
    # Logs and chunks (bounded)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=300))
    
    # Tool executions
    tool_trace: List[ToolRow] = field(default_factory=list)
    
    # Instructions and context
    instructions: Optional[InstructionFingerprint] = None
    resolved_inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Output and artifacts
    output_summary: str = ""
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    tokens_used: Dict[str, int] = field(default_factory=dict)  # prompt, completion, total
    cost: Optional[float] = None
    
    # Policy and safety
    redacted: bool = True  # Default to redacted for privacy
    policy_hits: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get node execution duration in seconds."""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None
    
    @property  
    def is_foreach(self) -> bool:
        """Check if this is a foreach node."""
        return self.items_total is not None


@dataclass
class MetricsState:
    """Real-time metrics from broker and agent pool."""
    broker_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # per-model stats
    pool_stats: Dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class RunTUIModel:
    """Enhanced state model for the professional TUI dashboard."""
    run_id: str
    goal: str = ""
    
    # Node state (now using enhanced NodeState)
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    
    # Global logs and events (bounded)
    global_chunks: deque[str] = field(default_factory=lambda: deque(maxlen=2000))
    global_errors: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    system_events: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    
    # Metrics and performance
    metrics: MetricsState = field(default_factory=MetricsState)
    
    # Run-level state
    status: Literal["idle", "running", "done", "error"] = "idle"
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    
    # UI state
    redaction_enabled: bool = True  # Privacy by default
    selected_node_id: Optional[str] = None
    last_update_ts: float = field(default_factory=time.time)
    
    # DAG structure (for mini-map)
    dag_edges: List[tuple[str, str]] = field(default_factory=list)
    
    @property
    def elapsed_time(self) -> float:
        """Get total run elapsed time."""
        if self.started_at:
            end_time = self.ended_at or time.time()
            return end_time - self.started_at
        return 0.0
    
    @property
    def progress_summary(self) -> Dict[str, int]:
        """Get run progress summary."""
        total = len(self.nodes)
        done = len([n for n in self.nodes.values() if n.status in ["complete", "resumed"]])
        running = len([n for n in self.nodes.values() if n.status == "running"])
        error = len([n for n in self.nodes.values() if n.status == "error"])
        
        return {
            "total": total,
            "done": done, 
            "running": running,
            "error": error,
            "pending": total - done - running - error
        }
    
    def apply_event(self, event_type: str, node_id: Optional[str], data: Dict[str, Any]) -> None:
        """Apply an orchestrator event to update the enhanced model state."""
        self.last_update_ts = time.time()
        
        if event_type == "RUN_START":
            self.status = "running"
            self.started_at = time.time()
            if "goal" in data:
                self.goal = data["goal"]
            self.system_events.append(f"🚀 Run started: {self.goal[:50]}...")
        
        elif event_type == "NODE_START" and node_id:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.status = "running"
                node.started_at = data.get("started_at", time.time())
                node.attempt = data.get("attempt", 1)
                node.max_attempts = data.get("max_attempts", 1)
                node.server = data.get("server_name")
                node.model = data.get("model")
                
                self.system_events.append(f"▶️ {node.name or node_id} started")
        
        elif event_type == "AGENT_INSTRUCTIONS" and node_id:
            # Handle new AGENT_INSTRUCTIONS event
            if node_id in self.nodes:
                instructions = InstructionFingerprint(
                    task_title=data.get("task", {}).get("title"),
                    task_body=data.get("task", {}).get("body", ""),
                    system=data.get("system", ""),
                    tools=data.get("tools", []),
                    policy=data.get("policy", {}),
                    context=data.get("context", {}),
                )
                self.nodes[node_id].instructions = instructions
        
        elif event_type == "TOOL_START" and node_id:
            # Handle new TOOL_START event
            if node_id in self.nodes:
                tool_row = ToolRow(
                    name=data.get("name", "unknown"),
                    started_at=data.get("started_at", time.time()),
                    args_preview=str(data.get("args_preview", ""))[:100]
                )
                self.nodes[node_id].tool_trace.append(tool_row)
        
        elif event_type == "TOOL_END" and node_id:
            # Handle new TOOL_END event - update the last tool row
            if node_id in self.nodes and self.nodes[node_id].tool_trace:
                last_tool = self.nodes[node_id].tool_trace[-1]
                if last_tool.name == data.get("name"):
                    last_tool.ended_at = data.get("ended_at", time.time())
                    last_tool.ok = data.get("ok", True)
                    last_tool.bytes_in = data.get("bytes_in")
                    last_tool.bytes_out = data.get("bytes_out")
                    last_tool.path = data.get("path")
                    last_tool.diff_preview = data.get("diff_preview")
                    if not last_tool.ok:
                        last_tool.error_message = data.get("error", "Unknown error")
        
        elif event_type == "RETRY_ATTEMPT" and node_id:
            # Handle new RETRY_ATTEMPT event
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.attempt = data.get("attempt", node.attempt + 1)
                reason = data.get("reason", "unknown")
                delay = data.get("next_delay_s", 0)
                self.system_events.append(f"🔄 {node.name or node_id} retry #{node.attempt} ({reason}, delay: {delay}s)")
        
        elif event_type == "AGENT_CHUNK" and node_id:
            # Extract meaningful text from the chunk
            text = ""
            if isinstance(data, dict):
                text = data.get("text") or data.get("message") or str(data)[:200]
                
                # Extract token usage from chunks (MCP agents include this)
                if "usage" in data and node_id in self.nodes:
                    usage = data["usage"]
                    self.nodes[node_id].tokens_used = {
                        "prompt": usage.get("prompt_tokens", 0),
                        "completion": usage.get("completion_tokens", 0), 
                        "total": usage.get("total_tokens", 0)
                    }
                    # Update global metrics
                    self.metrics.total_tokens_in += usage.get("prompt_tokens", 0)
                    self.metrics.total_tokens_out += usage.get("completion_tokens", 0)
                
                # Extract cost from chunks
                if "cost" in data and node_id in self.nodes:
                    cost = data["cost"]
                    self.nodes[node_id].cost = cost
                    self.metrics.total_cost += cost
                
                # Extract step information (detailed agent reasoning)
                if "step" in data and node_id in self.nodes:
                    step_info = data["step"]
                    step_text = f"STEP: {step_info}"
                    self.nodes[node_id].logs.append(step_text)
                    text = step_text  # Use step info as the main text
                
                # Extract tool calls
                if "tool_call" in data and node_id in self.nodes:
                    tool_call = data["tool_call"]
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = str(tool_call.get("args", {}))[:100]
                    
                    # Create tool row
                    tool_row = ToolRow(
                        name=tool_name,
                        started_at=time.time(),
                        args_preview=tool_args
                    )
                    self.nodes[node_id].tool_trace.append(tool_row)
                    text = f"TOOL: {tool_name}({tool_args})"
                
                # Extract final output
                if "output" in data and node_id in self.nodes:
                    output = data["output"]
                    self.nodes[node_id].output_summary = str(output)[:200]
                    text = f"OUTPUT: {output}"
            else:
                text = str(data)[:200]
            
            if text.strip():
                self.global_chunks.append(f"[{node_id}] {text}")
                if node_id in self.nodes:
                    self.nodes[node_id].logs.append(text)
            
            # Handle foreach progress
            if isinstance(data, dict) and data.get("phase") == "foreach:item":
                if "index" in data and node_id in self.nodes:
                    self.nodes[node_id].items_done = max(
                        self.nodes[node_id].items_done, 
                        data["index"] + 1
                    )
        
        elif event_type == "NODE_COMPLETE" and node_id:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.status = "resumed" if data.get("resumed") else "complete"
                node.ended_at = data.get("ended_at", time.time())
                
                # Enhanced completion data
                node.output_summary = data.get("output_summary", "")
                node.artifacts = data.get("artifacts", [])
                
                # Token and cost tracking
                if "tokens" in data:
                    node.tokens_used = data["tokens"]
                    self.metrics.total_tokens_in += data["tokens"].get("prompt", 0)
                    self.metrics.total_tokens_out += data["tokens"].get("completion", 0)
                
                if "cost" in data:
                    node.cost = data["cost"]
                    self.metrics.total_cost += data["cost"]
                
                # Set items_total if available  
                if "items_total" in data:
                    node.items_total = data["items_total"]
                
                self.system_events.append(f"✅ {node.name or node_id} completed")
        
        elif event_type == "ERROR":
            self.status = "error"
            error_msg = data.get("error", str(data))
            timestamp = time.strftime('%H:%M:%S')
            self.global_errors.append(f"{timestamp} {error_msg}")
            
            if node_id and node_id in self.nodes:
                self.nodes[node_id].status = "error"
                self.nodes[node_id].ended_at = time.time()
            
            self.system_events.append(f"❌ Error: {error_msg[:50]}...")
        
        elif event_type == "RUN_COMPLETE":
            self.status = "done"
            self.ended_at = time.time()
            self.system_events.append(f"🏁 Run completed in {self.elapsed_time:.1f}s")
    
    def apply_metrics(self, broker_stats: Dict[str, Any], pool_stats: Dict[str, Any]) -> None:
        """Apply broker and pool stats to the metrics."""
        self.metrics.broker_stats = broker_stats
        self.metrics.pool_stats = pool_stats  
        self.metrics.last_updated = time.time()
        self.last_update_ts = time.time()
    
    def toggle_redaction(self) -> None:
        """Toggle redaction mode for privacy."""
        self.redaction_enabled = not self.redaction_enabled
        for node in self.nodes.values():
            node.redacted = self.redaction_enabled
    
    def get_node_by_index(self, index: int) -> Optional[NodeState]:
        """Get node by list index (for UI navigation)."""
        nodes = list(self.nodes.values())
        if 0 <= index < len(nodes):
            return nodes[index]
        return None
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the enhanced state."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status,
            "elapsed_time": self.elapsed_time,
            "progress": self.progress_summary,
            "last_update_ts": self.last_update_ts,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "name": node.name,
                    "server": node.server,
                    "model": node.model,
                    "status": node.status,
                    "started_at": node.started_at,
                    "ended_at": node.ended_at,
                    "duration": node.duration,
                    "attempt": node.attempt,
                    "max_attempts": node.max_attempts,
                    "items_done": node.items_done,
                    "items_total": node.items_total,
                    "output_summary": node.output_summary,
                    "tokens_used": node.tokens_used,
                    "cost": node.cost,
                    "tool_count": len(node.tool_trace),
                    "artifact_count": len(node.artifacts),
                }
                for node_id, node in self.nodes.items()
            },
            "recent_chunks": list(self.global_chunks)[-10:],
            "recent_errors": list(self.global_errors)[-5:],
            "recent_system_events": list(self.system_events)[-10:],
            "metrics": {
                "total_cost": self.metrics.total_cost,
                "total_tokens_in": self.metrics.total_tokens_in,
                "total_tokens_out": self.metrics.total_tokens_out,
                "broker_stats": self.metrics.broker_stats,
                "pool_stats": self.metrics.pool_stats,
            },
            "ui_state": {
                "redaction_enabled": self.redaction_enabled,
                "selected_node_id": self.selected_node_id,
            }
        }