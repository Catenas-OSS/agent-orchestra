"""Deterministic checkpointing and replay system.

Provides state persistence with full deterministic replay capability.
"""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckpointState:
    """Complete state snapshot for deterministic replay."""

    run_id: str
    trace_id: str
    checkpoint_id: str
    timestamp: float

    # Graph state
    graph_spec: dict[str, Any]
    node_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_nodes: list[str] = field(default_factory=list)
    failed_nodes: list[str] = field(default_factory=list)

    # Execution context
    context: dict[str, Any] = field(default_factory=dict)

    # Determinism state
    rng_seed: int = field(default_factory=lambda: int(time.time() * 1000000) % (2**32))
    injected_time: float = field(default_factory=time.time)

    # Usage tracking
    total_tokens: int = 0
    total_cost: float = 0.0
    total_children: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "CheckpointState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def compute_hash(self) -> str:
        """Compute deterministic hash of the checkpoint state."""
        # Create a stable representation for hashing
        stable_dict = {
            "graph_spec": self.graph_spec,
            "node_states": self.node_states,
            "completed_nodes": sorted(self.completed_nodes),
            "failed_nodes": sorted(self.failed_nodes),
            "context": self.context,
            "rng_seed": self.rng_seed,
            "injected_time": self.injected_time
        }

        stable_json = json.dumps(stable_dict, sort_keys=True, default=str)
        return hashlib.sha256(stable_json.encode()).hexdigest()


class Checkpointer:
    """Manages checkpointing and replay functionality."""

    def __init__(self, checkpoint_dir: str | Path = "./checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        run_id: str,
        trace_id: str,
        graph_spec: dict[str, Any],
        node_states: dict[str, dict[str, Any]],
        completed_nodes: list[str],
        failed_nodes: list[str],
        context: dict[str, Any],
        usage: dict[str, Any] | None = None
    ) -> CheckpointState:
        """Create a new checkpoint."""

        checkpoint_id = str(uuid.uuid4())

        # Extract usage information
        usage = usage or {}

        state = CheckpointState(
            run_id=run_id,
            trace_id=trace_id,
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            graph_spec=graph_spec,
            node_states=node_states.copy(),
            completed_nodes=completed_nodes.copy(),
            failed_nodes=failed_nodes.copy(),
            context=context.copy(),
            total_tokens=usage.get("total_tokens", 0),
            total_cost=usage.get("total_cost", 0.0),
            total_children=usage.get("total_children", 0)
        )

        return state

    def save_checkpoint(self, state: CheckpointState) -> Path:
        """Save checkpoint to disk."""
        filename = f"{state.run_id}_{state.checkpoint_id}.json"
        file_path = self.checkpoint_dir / filename

        with open(file_path, 'w') as f:
            f.write(state.to_json())

        return file_path

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointState | None:
        """Load checkpoint by ID."""
        # Look for files containing the checkpoint ID
        for file_path in self.checkpoint_dir.glob(f"*_{checkpoint_id}.json"):
            try:
                with open(file_path) as f:
                    return CheckpointState.from_json(f.read())
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_id}: {e}")
                continue

        return None

    def list_checkpoints(self, run_id: str | None = None) -> list[CheckpointState]:
        """List all checkpoints, optionally filtered by run_id."""
        checkpoints = []

        pattern = f"{run_id}_*.json" if run_id else "*.json"

        for file_path in self.checkpoint_dir.glob(pattern):
            try:
                with open(file_path) as f:
                    checkpoint = CheckpointState.from_json(f.read())
                    checkpoints.append(checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint {file_path}: {e}")
                continue

        # Sort by timestamp
        checkpoints.sort(key=lambda c: c.timestamp)
        return checkpoints

    def get_latest_checkpoint(self, run_id: str) -> CheckpointState | None:
        """Get the most recent checkpoint for a run."""
        checkpoints = self.list_checkpoints(run_id)
        return checkpoints[-1] if checkpoints else None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint by ID."""
        for file_path in self.checkpoint_dir.glob(f"*_{checkpoint_id}.json"):
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"Error deleting checkpoint {checkpoint_id}: {e}")
                return False

        return False

    def cleanup_old_checkpoints(self, run_id: str, keep_count: int = 10) -> None:
        """Keep only the most recent N checkpoints for a run."""
        checkpoints = self.list_checkpoints(run_id)

        if len(checkpoints) <= keep_count:
            return

        # Delete oldest checkpoints
        to_delete = checkpoints[:-keep_count]
        for checkpoint in to_delete:
            self.delete_checkpoint(checkpoint.checkpoint_id)


class DeterministicContext:
    """Provides deterministic execution context for replay."""

    def __init__(self, rng_seed: int, injected_time: float) -> None:
        self.rng_seed = rng_seed
        self.injected_time = injected_time
        self._rng = self._create_rng()

    def _create_rng(self) -> Any:
        """Create a seeded random number generator."""
        import random
        rng = random.Random(self.rng_seed)
        return rng

    def random(self) -> float:
        """Get deterministic random number."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Get deterministic random integer."""
        return self._rng.randint(a, b)

    def now(self) -> float:
        """Get injected time (for deterministic replay)."""
        return self.injected_time

    def uuid4(self) -> str:
        """Generate deterministic UUID."""
        # Use the RNG to generate bytes for UUID
        bytes_data = bytes([self._rng.randint(0, 255) for _ in range(16)])
        return str(uuid.UUID(bytes=bytes_data, version=4))

    def advance_time(self, seconds: float) -> None:
        """Advance the injected time."""
        self.injected_time += seconds


def create_deterministic_context(
    checkpoint: CheckpointState | None = None
) -> DeterministicContext:
    """Create a deterministic context, optionally from a checkpoint."""

    if checkpoint:
        return DeterministicContext(
            rng_seed=checkpoint.rng_seed,
            injected_time=checkpoint.injected_time
        )
    else:
        # Create new deterministic context
        seed = int(time.time() * 1000000) % (2**32)
        return DeterministicContext(
            rng_seed=seed,
            injected_time=time.time()
        )

