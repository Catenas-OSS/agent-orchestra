from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

EventType = Literal["RUN_START", "NODE_START", "NODE_COMPLETE", "RUN_COMPLETE", "ERROR", "AGENT_CHUNK"]

@dataclass(frozen=True)
class Event:
    type: EventType
    run_id: str
    node_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict) # type: ignore
    event_seq: int = 0  # Monotonic sequence for deterministic concurrent logging

@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: Literal["task", "foreach", "reduce", "gate"]
    name: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict) # type: ignore
    timeout_s: Optional[float] = None
    retries: int = 0
    retry_backoff_s: float = 0.5
    retry_on_timeout: bool = True  # Whether timeouts count as failed attempts for retry
    concurrency: Optional[int] = None  # For foreach: max concurrent items
    foreach_fail_policy: Literal["fail_fast", "skip"] = "fail_fast"  # How to handle failed foreach items
    server_name: Optional[str] = None  # reserved for later

@dataclass(frozen=True)
class GraphSpec:
    nodes: List[NodeSpec]
    edges: List[Tuple[str, str]]  # (from_id, to_id)

@dataclass
class RunSpec:
    run_id: str
    goal: str
    policy: Dict[str, Any] = field(default_factory=dict) # type: ignore
    hints: Dict[str, Any] = field(default_factory=dict) # type: ignore
    cache_ttl_s: Optional[int] = None
    max_in_flight: Optional[int] = None  # Global limit on concurrent node execution