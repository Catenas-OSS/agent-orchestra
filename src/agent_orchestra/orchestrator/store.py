from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Set

from .types import Event


@dataclass
class SavedNode:
    """Represents a saved node result with its signature for cache validation."""
    node_id: str
    signature: str
    result: Dict[str, Any]


@dataclass 
class SavedForeachItem:
    """Represents a saved foreach item result for granular resume."""
    node_id: str
    item_index: int
    signature: str
    result: Any


class RunStore(Protocol):
    """Abstract interface for orchestrator run persistence."""
    
    async def start_run(self, run_id: str, run_spec: Any) -> None:
        """Initialize storage for a new run."""
        ...
    
    async def append_event(self, run_id: str, event: Event) -> None:
        """Append event to run's event log."""
        ...
    
    async def save_node_result(self, run_id: str, saved: SavedNode) -> None:
        """Save node execution result with signature."""
        ...
    
    async def save_foreach_item(self, run_id: str, item: SavedForeachItem) -> None:
        """Save individual foreach item result."""
        ...
    
    async def load_checkpoint(self, run_id: str) -> Dict[str, SavedNode]:
        """Load all saved node results for resume."""
        ...
    
    async def load_foreach_items(self, run_id: str, node_id: str) -> Dict[int, SavedForeachItem]:
        """Load saved foreach item results for a node."""
        ...
    
    async def save_gate_pruning(self, run_id: str, pruned_nodes: Set[str]) -> None:
        """Save gate pruning state."""
        ...
    
    async def load_gate_pruning(self, run_id: str) -> Set[str]:
        """Load gate pruning state."""
        ...
    
    async def get_last_event_seq(self, run_id: str) -> int:
        """Get the last event sequence number for continuity."""
        ...
    
    async def mark_run_complete(self, run_id: str) -> None:
        """Mark run as successfully completed."""
        ...
    
    async def mark_run_error(self, run_id: str, error: str) -> None:
        """Mark run as failed with error message."""
        ...
    
    async def mark_run_canceled(self, run_id: str) -> None:
        """Mark run as canceled."""
        ...


def _atomic_write(path: Path, content: str) -> None:
    """Write file atomically using temp file + rename."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)  # Atomic rename
    finally:
        # Clean up temp file if something went wrong
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For complex objects, convert to string representation
        try:
            # Try to get a meaningful representation
            if hasattr(obj, '__dict__'):
                return f"<{obj.__class__.__name__}: {str(obj)[:200]}>"
            else:
                return f"<{obj.__class__.__name__}: {str(obj)[:200]}>"
        except Exception:
            return f"<{obj.__class__.__name__}: [non-serializable]>"


def _redact_large_chunks(event_data: Dict[str, Any], max_size: int = 10000) -> Dict[str, Any]:
    """Redact oversized content in event data to control disk usage."""
    if not isinstance(event_data, dict):
        return event_data
    
    result = {}
    for key, value in event_data.items():
        if isinstance(value, str) and len(value) > max_size:
            result[key] = f"[REDACTED: {len(value)} chars, exceeds {max_size}]"
        elif isinstance(value, dict):
            # Recursively check nested dicts
            str_repr = str(value)
            if len(str_repr) > max_size:
                result[key] = f"[REDACTED: large dict with {len(value)} keys]"
            else:
                result[key] = _redact_large_chunks(value, max_size)
        else:
            result[key] = value
    return result


class JsonlRunStore:
    """JSONL-based run store with crash safety and file locking."""
    
    def __init__(
        self, 
        root: str | Path = ".ao_runs",
        flush_every: int = 10,
        max_chunk_size: int = 10000
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.max_chunk_size = max_chunk_size
        
        # Per-run locks for concurrent access protection
        self._locks: Dict[str, asyncio.Lock] = {}
        self._event_counters: Dict[str, int] = {}
    
    def _run_dir(self, run_id: str) -> Path:
        """Get directory for run, creating if needed."""
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _get_lock(self, run_id: str) -> asyncio.Lock:
        """Get or create lock for run."""
        if run_id not in self._locks:
            self._locks[run_id] = asyncio.Lock()
        return self._locks[run_id]
    
    async def _acquire_file_lock(self, run_id: str) -> None:
        """Acquire file-based lock to prevent concurrent orchestrators."""
        lock_file = self._run_dir(run_id) / ".lock"
        
        # Simple file-based locking
        if lock_file.exists():
            # Check if lock is stale (older than 1 hour)
            try:
                mtime = lock_file.stat().st_mtime
                if time.time() - mtime > 3600:  # 1 hour
                    lock_file.unlink()  # Remove stale lock
                else:
                    raise RuntimeError(f"Run {run_id} is locked by another orchestrator")
            except OSError:
                pass  # File disappeared, continue
        
        # Create lock file
        await asyncio.to_thread(lock_file.write_text, str(os.getpid()))
    
    async def _release_file_lock(self, run_id: str) -> None:
        """Release file-based lock."""
        lock_file = self._run_dir(run_id) / ".lock"
        try:
            lock_file.unlink()
        except Exception:
            pass  # Lock file might not exist
    
    async def start_run(self, run_id: str, run_spec: Any) -> None:
        """Initialize storage for a new run."""
        async with self._get_lock(run_id):
            await self._acquire_file_lock(run_id)
            
            run_dir = self._run_dir(run_id)
            meta_file = run_dir / "meta.json"
            
            if not meta_file.exists():
                # Convert run_spec to dict safely
                if hasattr(run_spec, "__dict__"):
                    spec_data = run_spec.__dict__.copy()
                else:
                    spec_data = {"run_spec": str(run_spec)}
                
                meta_data = {
                    "run_id": run_id,
                    "run_spec": spec_data,
                    "created_at": time.time(),
                    "orchestrator_version": "ao@5"
                }
                
                await asyncio.to_thread(
                    _atomic_write, 
                    meta_file, 
                    json.dumps(meta_data, indent=2)
                )
    
    async def append_event(self, run_id: str, event: Event) -> None:
        """Append event to run's event log with periodic flushing."""
        async with self._get_lock(run_id):
            events_file = self._run_dir(run_id) / "events.jsonl"
            
            # Make event JSON serializable and redact large chunks
            event_dict = event.__dict__.copy()
            event_dict = _make_json_serializable(event_dict)
            if "data" in event_dict:
                event_dict["data"] = _redact_large_chunks(event_dict["data"], self.max_chunk_size)
            
            event_line = json.dumps(event_dict) + "\n"
            
            # Append to file
            await asyncio.to_thread(
                lambda: events_file.open("a", encoding="utf-8").write(event_line)
            )
            
            # Periodic flushing for crash safety
            self._event_counters[run_id] = self._event_counters.get(run_id, 0) + 1
            if self._event_counters[run_id] % self.flush_every == 0:
                await asyncio.to_thread(self._flush_events_file, events_file)
    
    def _flush_events_file(self, events_file: Path) -> None:
        """Force flush and fsync events file."""
        try:
            with events_file.open("a", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass  # Best effort
    
    async def save_node_result(self, run_id: str, saved: SavedNode) -> None:
        """Save node result with atomic update."""
        async with self._get_lock(run_id):
            nodes_file = self._run_dir(run_id) / "nodes.json"
            
            # Load existing data
            if nodes_file.exists():
                content = await asyncio.to_thread(nodes_file.read_text, "utf-8")
                data = json.loads(content)
            else:
                data = {}
            
            # Update with new node result  
            data[saved.node_id] = {
                "signature": saved.signature,
                "result": saved.result,
                "saved_at": time.time()
            }
            
            # Atomic write
            await asyncio.to_thread(
                _atomic_write, 
                nodes_file,
                json.dumps(data, indent=2)
            )
    
    async def save_foreach_item(self, run_id: str, item: SavedForeachItem) -> None:
        """Save individual foreach item result."""
        async with self._get_lock(run_id):
            items_file = self._run_dir(run_id) / f"foreach_{item.node_id}.json"
            
            # Load existing data
            if items_file.exists():
                content = await asyncio.to_thread(items_file.read_text, "utf-8")
                data = json.loads(content)
            else:
                data = {}
            
            # Update with new item result
            data[str(item.item_index)] = {
                "signature": item.signature,
                "result": item.result,
                "saved_at": time.time()
            }
            
            # Atomic write
            await asyncio.to_thread(
                _atomic_write,
                items_file, 
                json.dumps(data, indent=2)
            )
    
    async def load_checkpoint(self, run_id: str) -> Dict[str, SavedNode]:
        """Load all saved node results."""
        nodes_file = self._run_dir(run_id) / "nodes.json"
        
        if not nodes_file.exists():
            return {}
        
        try:
            content = await asyncio.to_thread(nodes_file.read_text, "utf-8")
            data = json.loads(content)
            
            checkpoint: Dict[str, SavedNode] = {}
            for node_id, node_data in data.items():
                checkpoint[node_id] = SavedNode(
                    node_id=node_id,
                    signature=node_data["signature"],
                    result=node_data["result"]
                )
            
            return checkpoint
        except Exception:
            return {}  # Return empty on any error
    
    async def load_foreach_items(self, run_id: str, node_id: str) -> Dict[int, SavedForeachItem]:
        """Load saved foreach item results."""
        items_file = self._run_dir(run_id) / f"foreach_{node_id}.json"
        
        if not items_file.exists():
            return {}
        
        try:
            content = await asyncio.to_thread(items_file.read_text, "utf-8")
            data = json.loads(content)
            
            items: Dict[int, SavedForeachItem] = {}
            for index_str, item_data in data.items():
                index = int(index_str)
                items[index] = SavedForeachItem(
                    node_id=node_id,
                    item_index=index,
                    signature=item_data["signature"], 
                    result=item_data["result"]
                )
            
            return items
        except Exception:
            return {}
    
    async def save_gate_pruning(self, run_id: str, pruned_nodes: Set[str]) -> None:
        """Save gate pruning state."""
        async with self._get_lock(run_id):
            pruning_file = self._run_dir(run_id) / "gate_pruning.json"
            
            data = {
                "pruned_nodes": list(pruned_nodes),
                "saved_at": time.time()
            }
            
            await asyncio.to_thread(
                _atomic_write,
                pruning_file,
                json.dumps(data, indent=2)
            )
    
    async def load_gate_pruning(self, run_id: str) -> Set[str]:
        """Load gate pruning state."""
        pruning_file = self._run_dir(run_id) / "gate_pruning.json"
        
        if not pruning_file.exists():
            return set()
        
        try:
            content = await asyncio.to_thread(pruning_file.read_text, "utf-8")
            data = json.loads(content)
            return set(data.get("pruned_nodes", []))
        except Exception:
            return set()
    
    async def get_last_event_seq(self, run_id: str) -> int:
        """Get last event sequence number for continuity."""
        events_file = self._run_dir(run_id) / "events.jsonl"
        
        if not events_file.exists():
            return 0
        
        try:
            content = await asyncio.to_thread(events_file.read_text, "utf-8")
            lines = content.strip().split('\n')
            
            # Find the last valid event sequence
            last_seq = 0
            for line in reversed(lines):
                if line.strip():
                    try:
                        event_data = json.loads(line)
                        if "event_seq" in event_data:
                            return event_data["event_seq"]
                    except Exception:
                        continue
            
            return last_seq
        except Exception:
            return 0
    
    async def mark_run_complete(self, run_id: str) -> None:
        """Mark run as completed."""
        await self._mark_run_status(run_id, "complete", None)
    
    async def mark_run_error(self, run_id: str, error: str) -> None:
        """Mark run as failed."""
        await self._mark_run_status(run_id, "error", error)
    
    async def mark_run_canceled(self, run_id: str) -> None:
        """Mark run as canceled."""
        await self._mark_run_status(run_id, "canceled", None)
    
    async def _mark_run_status(self, run_id: str, status: str, error: Optional[str]) -> None:
        """Internal method to mark run status."""
        async with self._get_lock(run_id):
            status_file = self._run_dir(run_id) / "status.json"
            
            data = {
                "status": status,
                "timestamp": time.time()
            }
            if error:
                data["error"] = error
            
            await asyncio.to_thread(
                _atomic_write,
                status_file,
                json.dumps(data, indent=2)
            )
            
            # Release file lock when run is complete
            await self._release_file_lock(run_id)