"""Tests for events module."""

import pytest
import json
import tempfile
from pathlib import Path

from agent_orchestra.events import (
    Event, EventType, EventSink, JSONLSink, EventBuffer, EventEmitter,
    create_span_id, create_trace_id
)


def test_event_creation():
    """Test Event creation and serialization."""
    event = Event(
        type=EventType.NODE_START,
        run_id="test-run",
        trace_id="test-trace",
        span_id="test-span",
        payload={"test": "data"}
    )
    
    assert event.type == EventType.NODE_START
    assert event.run_id == "test-run"
    assert event.trace_id == "test-trace"
    assert event.span_id == "test-span"
    assert event.payload["test"] == "data"


def test_event_json_serialization():
    """Test Event JSON serialization and deserialization."""
    original = Event(
        type=EventType.RUN_START,
        run_id="test",
        trace_id="trace",
        span_id="span",
        payload={"key": "value"}
    )
    
    # To JSON
    json_str = original.to_json()
    data = json.loads(json_str)
    assert data["type"] == "run.start"
    assert data["payload"]["key"] == "value"
    
    # From JSON
    restored = Event.from_json(json_str)
    assert restored.type == EventType.RUN_START
    assert restored.run_id == "test"
    assert restored.payload["key"] == "value"


def test_event_auto_span_id():
    """Test automatic span ID generation."""
    event = Event(
        type=EventType.NODE_START,
        run_id="test",
        trace_id="trace",
        span_id=""  # Empty span_id should be auto-generated
    )
    
    # Should have generated a span_id
    assert event.span_id
    assert len(event.span_id) > 0


@pytest.mark.asyncio
async def test_event_buffer():
    """Test EventBuffer functionality."""
    buffer = EventBuffer(max_size=3)
    
    # Add events
    event1 = Event(type=EventType.RUN_START, run_id="test", trace_id="trace", span_id="span1")
    event2 = Event(type=EventType.NODE_START, run_id="test", trace_id="trace", span_id="span2") 
    event3 = Event(type=EventType.NODE_COMPLETE, run_id="test", trace_id="trace", span_id="span3")
    event4 = Event(type=EventType.RUN_END, run_id="test", trace_id="trace", span_id="span4")
    
    await buffer.write(event1)
    await buffer.write(event2)
    await buffer.write(event3)
    await buffer.write(event4)  # Should evict event1
    
    # Check buffer contents
    events = buffer.get_events()
    assert len(events) == 3
    assert events[0].span_id == "span2"  # event1 was evicted
    
    # Test filtering
    node_events = buffer.get_events(EventType.NODE_START)
    assert len(node_events) == 1
    assert node_events[0].span_id == "span2"
    
    # Test clear
    buffer.clear()
    assert len(buffer.get_events()) == 0


@pytest.mark.asyncio
async def test_jsonl_sink():
    """Test JSONLSink functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_events.jsonl"
        sink = JSONLSink(file_path)
        
        # Write events
        event1 = Event(type=EventType.RUN_START, run_id="test", trace_id="trace", span_id="span1")
        event2 = Event(type=EventType.RUN_END, run_id="test", trace_id="trace", span_id="span2")
        
        await sink.write(event1)
        await sink.write(event2)
        await sink.close()
        
        # Read and verify
        assert file_path.exists()
        
        lines = file_path.read_text().strip().split('\n')
        assert len(lines) == 2
        
        # Parse first line
        data1 = json.loads(lines[0])
        assert data1["type"] == "run.start"
        assert data1["span_id"] == "span1"


@pytest.mark.asyncio
async def test_event_emitter():
    """Test EventEmitter with multiple sinks."""
    buffer1 = EventBuffer()
    buffer2 = EventBuffer()
    
    emitter = EventEmitter([buffer1, buffer2])
    
    event = Event(type=EventType.NODE_START, run_id="test", trace_id="trace", span_id="span")
    
    await emitter.emit(event)
    
    # Both buffers should have the event
    assert len(buffer1.get_events()) == 1
    assert len(buffer2.get_events()) == 1
    
    # Events should be identical
    assert buffer1.get_events()[0].span_id == buffer2.get_events()[0].span_id


@pytest.mark.asyncio
async def test_event_emitter_error_handling():
    """Test EventEmitter handles sink errors gracefully."""
    
    class FailingSink(EventSink):
        async def write(self, event: Event) -> None:
            raise Exception("Sink error")
    
    buffer = EventBuffer()
    failing_sink = FailingSink()
    
    emitter = EventEmitter([buffer, failing_sink])
    
    event = Event(type=EventType.NODE_START, run_id="test", trace_id="trace", span_id="span")
    
    # Should not raise despite failing sink
    await emitter.emit(event)
    
    # Buffer should still have received the event
    assert len(buffer.get_events()) == 1


def test_span_and_trace_id_generation():
    """Test ID generation functions."""
    span_id = create_span_id()
    trace_id = create_trace_id()
    
    assert span_id
    assert trace_id
    assert span_id != trace_id
    
    # Should generate unique IDs
    span_id2 = create_span_id()
    assert span_id != span_id2


def test_event_type_enum():
    """Test EventType enum values."""
    assert EventType.RUN_START.value == "run.start"
    assert EventType.NODE_ERROR.value == "node.error"
    assert EventType.HITL_REQUIRED.value == "hitl.required"
    assert EventType.CHECKPOINT_SAVED.value == "checkpoint.saved"