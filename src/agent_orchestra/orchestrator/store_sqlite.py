"""
SQLite-based RunStore implementation for Agent Orchestra.

Provides drop-in replacement for JsonlRunStore with:
- Fast queries and analytics capabilities
- WAL mode for concurrent access
- Atomic transactions and crash safety
- Foundation for UI/HITL workflows
"""

from __future__ import annotations
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set
from dataclasses import asdict

from .store import SavedNode, SavedForeachItem, _make_json_serializable
from .types import Event


class SQLiteRunStore:
    """SQLite-based run store with concurrent access and analytics capabilities."""
    
    def __init__(
        self, 
        db_path: str | Path = ".ao_runs/ao.sqlite3",
        busy_timeout: int = 3000
    ):
        """Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database file
            busy_timeout: SQLite busy timeout in milliseconds
        """
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout = busy_timeout
        
        # Initialize database schema
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection with optimized settings."""
        conn = sqlite3.connect(str(self.path), timeout=self.busy_timeout / 1000.0)
        
        # Configure SQLite for performance and safety
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout}")
        
        return conn
    
    def _init_db(self) -> None:
        """Initialize database schema if not exists."""
        with self._get_connection() as conn:
            # Create tables with proper constraints and indexes
            conn.executescript("""
                -- Main runs table
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    goal TEXT,
                    status TEXT NOT NULL CHECK(status IN ('running','complete','error','canceled')) DEFAULT 'running',
                    error TEXT
                );
                
                -- Node execution results with foreach sub_id support
                CREATE TABLE IF NOT EXISTS node_results (
                    run_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    sub_id TEXT,  -- NULL for normal nodes, "F:0", "F:1", etc. for foreach items
                    signature TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    completed_at REAL NOT NULL,
                    PRIMARY KEY (run_id, node_id, sub_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                
                -- Event stream with sequence continuity
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    ts REAL NOT NULL,
                    type TEXT NOT NULL,
                    node_id TEXT,
                    data_json TEXT,
                    PRIMARY KEY (run_id, seq),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                
                -- Gate pruning state storage
                CREATE TABLE IF NOT EXISTS gate_pruning (
                    run_id TEXT PRIMARY KEY,
                    pruned_nodes_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                
                -- Future: CallBroker metrics
                CREATE TABLE IF NOT EXISTS broker_metrics (
                    run_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    calls INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    rate_limits INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL NOT NULL,
                    PRIMARY KEY (run_id, model_key),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                
                -- Performance indexes
                CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);
                CREATE INDEX IF NOT EXISTS idx_nodes_lookup ON node_results(run_id, node_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_sub_lookup ON node_results(run_id, node_id, sub_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, type);
            """)
    
    async def _acquire_file_lock(self, run_id: str) -> None:
        """Acquire file-based lock for run (compatibility with JsonlRunStore)."""
        # For SQLite, we use database-level locking, but provide this for compatibility
        pass
    
    async def _release_file_lock(self, run_id: str) -> None:
        """Release file-based lock (compatibility with JsonlRunStore)."""
        # For SQLite, we use database-level locking, but provide this for compatibility
        pass
    
    async def start_run(self, run_id: str, run_spec: Any) -> None:
        """Initialize storage for a new run."""
        now = time.time()
        goal = getattr(run_spec, 'goal', None) if run_spec else None
        
        def _insert():
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO runs (run_id, created_at, updated_at, goal, status)
                    VALUES (?, ?, ?, ?, 'running')
                    ON CONFLICT(run_id) DO UPDATE SET 
                        updated_at = excluded.updated_at,
                        status = 'running'
                """, (run_id, now, now, goal))
        
        await asyncio.to_thread(_insert)
    
    async def append_event(self, run_id: str, event: Event) -> None:
        """Append event to run's event log."""
        now = time.time()
        
        # Convert event to serializable format
        event_dict = _make_json_serializable(event.__dict__.copy())
        
        def _insert():
            with self._get_connection() as conn:
                # Insert event
                conn.execute("""
                    INSERT INTO events (run_id, seq, ts, type, node_id, data_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    event_dict.get('event_seq', 0),
                    now,
                    event_dict['type'],
                    event_dict.get('node_id'),
                    json.dumps(event_dict.get('data', {}))
                ))
                
                # Update run timestamp
                conn.execute("""
                    UPDATE runs SET updated_at = ? WHERE run_id = ?
                """, (now, run_id))
        
        await asyncio.to_thread(_insert)
    
    async def save_node_result(self, run_id: str, saved: SavedNode) -> None:
        """Save node execution result with signature."""
        now = time.time()
        
        def _insert():
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO node_results (run_id, node_id, sub_id, signature, result_json, completed_at)
                    VALUES (?, ?, NULL, ?, ?, ?)
                    ON CONFLICT(run_id, node_id, sub_id) DO UPDATE SET
                        signature = excluded.signature,
                        result_json = excluded.result_json,
                        completed_at = excluded.completed_at
                """, (
                    run_id, 
                    saved.node_id, 
                    saved.signature, 
                    json.dumps(_make_json_serializable(saved.result)), 
                    now
                ))
        
        await asyncio.to_thread(_insert)
    
    async def save_foreach_item(self, run_id: str, item: SavedForeachItem) -> None:
        """Save individual foreach item result."""
        now = time.time()
        sub_id = f"F:{item.item_index}"
        
        def _insert():
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO node_results (run_id, node_id, sub_id, signature, result_json, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, node_id, sub_id) DO UPDATE SET
                        signature = excluded.signature,
                        result_json = excluded.result_json,
                        completed_at = excluded.completed_at
                """, (
                    run_id, 
                    item.node_id, 
                    sub_id,
                    item.signature, 
                    json.dumps(_make_json_serializable(item.result)), 
                    now
                ))
        
        await asyncio.to_thread(_insert)
    
    async def load_checkpoint(self, run_id: str) -> Dict[str, SavedNode]:
        """Load all saved node results for resume."""
        def _load():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT node_id, signature, result_json
                    FROM node_results 
                    WHERE run_id = ? AND sub_id IS NULL
                """, (run_id,))
                
                return {
                    node_id: SavedNode(
                        node_id=node_id,
                        signature=signature,
                        result=json.loads(result_json)
                    )
                    for node_id, signature, result_json in cursor.fetchall()
                }
        
        return await asyncio.to_thread(_load)
    
    async def load_foreach_items(self, run_id: str, node_id: str) -> Dict[int, SavedForeachItem]:
        """Load saved foreach item results for a node."""
        def _load():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT sub_id, signature, result_json
                    FROM node_results 
                    WHERE run_id = ? AND node_id = ? AND sub_id LIKE 'F:%'
                """, (run_id, node_id))
                
                result = {}
                for sub_id, signature, result_json in cursor.fetchall():
                    # Parse sub_id like "F:3" to get index 3
                    index = int(sub_id.split(':')[1])
                    result[index] = SavedForeachItem(
                        node_id=node_id,
                        item_index=index,
                        signature=signature,
                        result=json.loads(result_json)
                    )
                
                return result
        
        return await asyncio.to_thread(_load)
    
    async def save_gate_pruning(self, run_id: str, pruned_nodes: Set[str]) -> None:
        """Save gate pruning state."""
        now = time.time()
        
        def _save():
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO gate_pruning (run_id, pruned_nodes_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        pruned_nodes_json = excluded.pruned_nodes_json,
                        updated_at = excluded.updated_at
                """, (run_id, json.dumps(list(pruned_nodes)), now))
        
        await asyncio.to_thread(_save)
    
    async def load_gate_pruning(self, run_id: str) -> Set[str]:
        """Load gate pruning state."""
        def _load():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT pruned_nodes_json FROM gate_pruning WHERE run_id = ?
                """, (run_id,))
                
                row = cursor.fetchone()
                if row:
                    return set(json.loads(row[0]))
                return set()
        
        return await asyncio.to_thread(_load)
    
    async def get_last_event_seq(self, run_id: str) -> int:
        """Get the last event sequence number for continuity."""
        def _get():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id = ?
                """, (run_id,))
                
                return cursor.fetchone()[0]
        
        return await asyncio.to_thread(_get)
    
    async def mark_run_complete(self, run_id: str) -> None:
        """Mark run as successfully completed."""
        now = time.time()
        
        def _update():
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE runs SET status = 'complete', updated_at = ? WHERE run_id = ?
                """, (now, run_id))
        
        await asyncio.to_thread(_update)
    
    async def mark_run_error(self, run_id: str, error: str) -> None:
        """Mark run as failed with error message."""
        now = time.time()
        
        def _update():
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE runs SET status = 'error', error = ?, updated_at = ? WHERE run_id = ?
                """, (error, now, run_id))
        
        await asyncio.to_thread(_update)
    
    async def mark_run_canceled(self, run_id: str) -> None:
        """Mark run as canceled."""
        now = time.time()
        
        def _update():
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE runs SET status = 'canceled', updated_at = ? WHERE run_id = ?
                """, (now, run_id))
        
        await asyncio.to_thread(_update)
    
    # Analytics and UI helper methods
    
    async def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run status and metadata."""
        def _get():
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT run_id, created_at, updated_at, goal, status, error
                    FROM runs WHERE run_id = ?
                """, (run_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'run_id': row[0],
                        'created_at': row[1],
                        'updated_at': row[2], 
                        'goal': row[3],
                        'status': row[4],
                        'error': row[5]
                    }
                return None
        
        return await asyncio.to_thread(_get)
    
    async def list_runs(self, status: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
        """List runs with optional status filter."""
        def _list():
            with self._get_connection() as conn:
                if status:
                    cursor = conn.execute("""
                        SELECT run_id, created_at, updated_at, goal, status, error
                        FROM runs WHERE status = ?
                        ORDER BY updated_at DESC LIMIT ?
                    """, (status, limit))
                else:
                    cursor = conn.execute("""
                        SELECT run_id, created_at, updated_at, goal, status, error
                        FROM runs ORDER BY updated_at DESC LIMIT ?
                    """, (limit,))
                
                return [
                    {
                        'run_id': row[0],
                        'created_at': row[1],
                        'updated_at': row[2],
                        'goal': row[3],
                        'status': row[4],
                        'error': row[5]
                    }
                    for row in cursor.fetchall()
                ]
        
        return await asyncio.to_thread(_list)
    
    async def get_run_statistics(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive run statistics."""
        def _stats():
            with self._get_connection() as conn:
                # Event counts by type
                cursor = conn.execute("""
                    SELECT type, COUNT(*) FROM events 
                    WHERE run_id = ? GROUP BY type
                """, (run_id,))
                event_counts = dict(cursor.fetchall())
                
                # Node completion stats
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM node_results 
                    WHERE run_id = ? AND sub_id IS NULL
                """, (run_id,))
                completed_nodes = cursor.fetchone()[0]
                
                # Foreach item stats
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM node_results 
                    WHERE run_id = ? AND sub_id LIKE 'F:%'
                """, (run_id,))
                completed_foreach_items = cursor.fetchone()[0]
                
                return {
                    'event_counts': event_counts,
                    'completed_nodes': completed_nodes,
                    'completed_foreach_items': completed_foreach_items
                }
        
        return await asyncio.to_thread(_stats)