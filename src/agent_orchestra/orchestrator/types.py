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

@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: Literal["task"]  # future: foreach | reduce | gate | parent_agent
    name: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict) # type: ignore
    timeout_s: Optional[float] = None
    retries: int = 0
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