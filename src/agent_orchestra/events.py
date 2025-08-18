"""Event system and JSONL logging.

Defines the event schema and provides JSONL logging with OpenTelemetry integration.
Event schema is a frozen contract.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles


class EventType(Enum):
    """Event types (frozen contract)."""

    # Run lifecycle
    RUN_START = "run.start"
    RUN_END = "run.end"
    RUN_ERROR = "run.error"

    # Node lifecycle
    NODE_SCHEDULE = "node.schedule"
    NODE_START = "node.start"
    NODE_OUTPUT = "node.output"
    NODE_ERROR = "node.error"
    NODE_COMPLETE = "node.complete"

    # Policy events
    POLICY_BLOCK = "policy.block"
    BUDGET_EXCEEDED = "budget.exceeded"
    HITL_REQUIRED = "hitl.required"
    HITL_RESOLVED = "hitl.resolved"

    # System events
    CHECKPOINT_SAVED = "checkpoint.saved"
    TOOL_USAGE = "tool.usage"
    ADAPTER_CALL = "adapter.call"
    ADAPTER_RESULT = "adapter.result"

    # Client/Server lifecycle events
    CLIENT_START = "client.start"
    CLIENT_STOP = "client.stop"
    SERVER_START = "server.start"
    SERVER_READY = "server.ready"
    SERVER_ERROR = "server.error"

    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Event data structure (frozen contract).
    
    All events must include trace/span information for observability.
    """

    type: EventType
    run_id: str
    trace_id: str
    span_id: str
    timestamp: float = field(default_factory=time.time)
    parent_span_id: str | None = None
    node_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate span_id if not provided.
        
        Ensures that every event has a unique span ID for tracing.
        """
        if not self.span_id:
            self.span_id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the event with enum values converted to strings.
        """
        data = asdict(self)
        data['type'] = self.type.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string.
        
        Returns:
            JSON string representation of the event.
        """
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary.
        
        Args:
            data: Dictionary containing event data with 'type' as string.
            
        Returns:
            Event instance created from dictionary data.
            
        Raises:
            ValueError: If event type is invalid.
        """
        data = data.copy()
        data['type'] = EventType(data['type'])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string.
        
        Args:
            json_str: JSON string containing event data.
            
        Returns:
            Event instance created from JSON data.
            
        Raises:
            json.JSONDecodeError: If JSON string is invalid.
            ValueError: If event type is invalid.
        """
        return cls.from_dict(json.loads(json_str))


class EventSink:
    """Base class for event output destinations."""

    async def write(self, event: Event) -> None:
        """Write an event to the sink.
        
        Args:
            event: Event to write to the sink.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Close the sink and flush any pending writes.
        
        Default implementation does nothing. Subclasses should override if needed.
        """
        pass


class JSONLSink(EventSink):
    """JSONL file sink for events."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize JSONL sink with file path.
        
        Args:
            file_path: Path to the JSONL file for writing events.
        """
        self.file_path = Path(file_path)
        self._file: aiofiles.threadpool.text.AsyncTextIOWrapper | None = None

    async def _ensure_open(self) -> None:
        """Ensure file is open for writing."""
        if self._file is None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = await aiofiles.open(self.file_path, 'a')

    async def write(self, event: Event) -> None:
        """Write event as JSONL."""
        await self._ensure_open()
        if self._file:
            await self._file.write(event.to_json() + '\n')
            await self._file.flush()

    async def close(self) -> None:
        """Close the file."""
        if self._file:
            await self._file.close()
            self._file = None


class EventBuffer(EventSink):
    """In-memory buffer for events (useful for testing)."""

    def __init__(self, max_size: int = 10000) -> None:
        self.events: list[Event] = []
        self.max_size = max_size

    async def write(self, event: Event) -> None:
        """Add event to buffer."""
        self.events.append(event)
        if len(self.events) > self.max_size:
            self.events.pop(0)

    def get_events(self, event_type: EventType | None = None) -> list[Event]:
        """Get events, optionally filtered by type."""
        if event_type is None:
            return self.events.copy()
        return [e for e in self.events if e.type == event_type]

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()


class EventEmitter:
    """Event emitter with multiple sinks."""

    def __init__(self, sinks: list[EventSink] | None = None) -> None:
        self.sinks = sinks or []

    def add_sink(self, sink: EventSink) -> None:
        """Add an event sink."""
        self.sinks.append(sink)

    async def emit(self, event: Event) -> None:
        """Emit event to all sinks."""
        for sink in self.sinks:
            try:
                await sink.write(event)
            except Exception as e:
                # Don't let sink errors break the main flow
                print(f"Error writing to event sink: {e}")

    async def close(self) -> None:
        """Close all sinks."""
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as e:
                print(f"Error closing event sink: {e}")


def create_span_id() -> str:
    """Generate a new span ID."""
    return str(uuid.uuid4())


def create_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())


async def read_events_from_jsonl(file_path: str | Path) -> AsyncIterator[Event]:
    """Read events from a JSONL file."""
    async with aiofiles.open(file_path) as f:
        async for line in f:
            line = line.strip()
            if line:
                try:
                    yield Event.from_json(line)
                except Exception as e:
                    print(f"Error parsing event line: {e}")
                    continue

