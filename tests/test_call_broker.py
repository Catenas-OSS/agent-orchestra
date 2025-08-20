"""
Tests for CallBroker system - rate limiting, retries, and streaming support.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock

from agent_orchestra.orchestrator.call_broker import (
    CallBroker,
    ModelLimits,
    SlidingWindowCounter,
    DEFAULT_MODEL_LIMITS,
    create_default_broker
)


class TestSlidingWindowCounter:
    """Test the sliding window counter for rate limiting."""
    
    @pytest.mark.asyncio
    async def test_empty_counter_allows_requests(self):
        """Empty counter should allow requests within limit."""
        counter = SlidingWindowCounter(window_seconds=60)
        
        assert await counter.can_make_request(10) == True
        assert await counter.get_wait_time(10) == 0.0
    
    @pytest.mark.asyncio
    async def test_record_and_check_requests(self):
        """Counter should track requests correctly."""
        counter = SlidingWindowCounter(window_seconds=60)
        
        # Record requests up to limit
        for _ in range(5):
            await counter.record_request()
        
        assert await counter.can_make_request(5) == False
        assert await counter.can_make_request(6) == True
    
    @pytest.mark.asyncio
    async def test_sliding_window_expiry(self):
        """Old requests should expire from the window."""
        counter = SlidingWindowCounter(window_seconds=1)  # 1 second window
        
        # Fill up the counter
        for _ in range(3):
            await counter.record_request()
        
        assert await counter.can_make_request(3) == False
        
        # Wait for window to slide
        await asyncio.sleep(1.1)
        
        assert await counter.can_make_request(3) == True
    
    @pytest.mark.asyncio
    async def test_wait_time_calculation(self):
        """Wait time should be calculated correctly."""
        counter = SlidingWindowCounter(window_seconds=2)
        
        # Record requests
        await counter.record_request()
        await asyncio.sleep(0.5)
        await counter.record_request()
        
        # Should need to wait for first request to expire
        wait_time = await counter.get_wait_time(2)
        assert 1.0 < wait_time < 2.0  # Should be ~1.5 seconds


class TestCallBroker:
    """Test the CallBroker rate limiting and retry system."""
    
    def test_model_limits_configuration(self):
        """Test broker initialization with custom limits."""
        custom_limits = {
            "test:model": ModelLimits(rpm=10, rpd=100, max_concurrency=2)
        }
        default_limits = ModelLimits(rpm=5, rpd=50, max_concurrency=1)
        
        broker = CallBroker(custom_limits, default_limits)
        
        # Check model limits retrieval
        assert broker._get_model_limits("test:model").rpm == 10
        assert broker._get_model_limits("unknown:model").rpm == 5
    
    @pytest.mark.asyncio
    async def test_basic_agent_call_routing(self):
        """Test that agent calls are routed through the broker."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        mock_func = AsyncMock(return_value="test_result")
        
        result = await broker.call_agent_regular("test:model", mock_func)
        
        assert result == "test_result"
        mock_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_streaming_agent_call_routing(self):
        """Test that streaming agent calls preserve chunks."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        # Mock streaming function
        async def mock_stream():
            yield {"chunk": 1, "content": "first"}
            yield {"chunk": 2, "content": "second"}
            yield {"output": "final_result"}
        
        chunks = []
        async for chunk in broker.call_agent_streaming("test:model", mock_stream):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0]["chunk"] == 1
        assert chunks[1]["chunk"] == 2
        assert chunks[2]["output"] == "final_result"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self):
        """Test that rate limits are enforced."""
        # Very restrictive limits for testing
        limits = {"test:model": ModelLimits(rpm=2, rpd=10, max_concurrency=1)}
        broker = CallBroker(limits)
        
        mock_func = AsyncMock(return_value="result")
        
        # First two calls should work
        start_time = time.time()
        await broker.call_agent_regular("test:model", mock_func)
        await broker.call_agent_regular("test:model", mock_func)
        
        # Third call should be delayed by rate limiting
        await broker.call_agent_regular("test:model", mock_func)
        end_time = time.time()
        
        # Should have taken some time due to rate limiting
        elapsed = end_time - start_time
        assert elapsed > 0.5  # Should have waited at least some time
        
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_429_retry_logic(self):
        """Test that 429 errors are retried with exponential backoff."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        call_count = 0
        
        async def mock_func_with_429():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "success_after_retries"
        
        start_time = time.time()
        result = await broker.call_agent_regular("test:model", mock_func_with_429)
        end_time = time.time()
        
        assert result == "success_after_retries"
        assert call_count == 3
        # Should have taken time for retries with backoff
        assert (end_time - start_time) > 1.0
    
    @pytest.mark.asyncio
    async def test_429_retry_exhaustion(self):
        """Test that retries are exhausted for persistent 429 errors."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        async def always_429():
            raise Exception("429 Too Many Requests - persistent")
        
        with pytest.raises(Exception) as exc_info:
            await broker.call_agent_regular("test:model", always_429)
        
        assert "429" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_non_429_error_passthrough(self):
        """Test that non-429 errors are not retried."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        call_count = 0
        
        async def mock_func_with_other_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Some other error")
        
        with pytest.raises(ValueError):
            await broker.call_agent_regular("test:model", mock_func_with_other_error)
        
        # Should not have retried
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_concurrency_limiting(self):
        """Test that max_concurrency is enforced."""
        # Limit to 1 concurrent request
        limits = {"test:model": ModelLimits(rpm=100, rpd=1000, max_concurrency=1)}
        broker = CallBroker(limits)
        
        call_times = []
        
        async def slow_mock_func():
            start_time = time.time()
            await asyncio.sleep(0.2)  # Simulate slow operation
            end_time = time.time()
            call_times.append((start_time, end_time))
            return "result"
        
        # Start multiple concurrent calls
        tasks = [
            asyncio.create_task(broker.call_agent_regular("test:model", slow_mock_func))
            for _ in range(3)
        ]
        
        await asyncio.gather(*tasks)
        
        # Calls should have been serialized due to concurrency limit
        assert len(call_times) == 3
        # Each call should start after the previous one finishes (with some tolerance)
        for i in range(1, len(call_times)):
            prev_end = call_times[i-1][1]
            curr_start = call_times[i][0]
            assert curr_start >= (prev_end - 0.05)  # Small tolerance for timing
    
    @pytest.mark.asyncio
    async def test_streaming_429_retry(self):
        """Test that streaming calls handle 429 retries correctly."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        attempt_count = 0
        
        async def mock_stream_with_429():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("429 Too Many Requests")
            
            yield {"chunk": 1, "content": "retry_success"}
            yield {"output": "final_result"}
        
        chunks = []
        async for chunk in broker.call_agent_streaming("test:model", mock_stream_with_429):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0]["content"] == "retry_success"
        assert chunks[1]["output"] == "final_result"
        assert attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_broker_stats(self):
        """Test broker statistics collection."""
        limits = {"test:model": ModelLimits(rpm=10, rpd=100, max_concurrency=3)}
        broker = CallBroker(limits)
        
        # Make some calls to generate stats
        mock_func = AsyncMock(return_value="result")
        await broker.call_agent_regular("test:model", mock_func)
        await broker.call_agent_regular("test:model", mock_func)
        
        stats = await broker.get_stats()
        
        assert "test:model" in stats
        model_stats = stats["test:model"]
        
        assert model_stats["rpm_limit"] == 10
        assert model_stats["rpd_limit"] == 100
        assert model_stats["concurrent_limit"] == 3
        assert model_stats["rpm_used"] == 2
        assert model_stats["rpd_used"] == 2
        assert model_stats["concurrent_used"] == 0  # No active concurrent calls
    
    @pytest.mark.asyncio
    async def test_broker_shutdown(self):
        """Test graceful broker shutdown."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        # Shutdown should complete without error
        await broker.shutdown()
        
        # Multiple shutdowns should be safe
        await broker.shutdown()


class TestCallBrokerIntegration:
    """Integration tests for CallBroker with real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_models_concurrent(self):
        """Test broker handling multiple models concurrently."""
        limits = {
            "openai:gpt-4": ModelLimits(rpm=5, rpd=50, max_concurrency=2),
            "anthropic:claude": ModelLimits(rpm=10, rpd=100, max_concurrency=3)
        }
        broker = CallBroker(limits)
        
        gpt_mock = AsyncMock(return_value="gpt_result")
        claude_mock = AsyncMock(return_value="claude_result")
        
        # Run calls concurrently for different models
        gpt_tasks = [
            asyncio.create_task(broker.call_agent_regular("openai:gpt-4", gpt_mock))
            for _ in range(3)
        ]
        claude_tasks = [
            asyncio.create_task(broker.call_agent_regular("anthropic:claude", claude_mock))
            for _ in range(3)
        ]
        
        gpt_results = await asyncio.gather(*gpt_tasks)
        claude_results = await asyncio.gather(*claude_tasks)
        
        assert all(result == "gpt_result" for result in gpt_results)
        assert all(result == "claude_result" for result in claude_results)
        
        assert gpt_mock.call_count == 3
        assert claude_mock.call_count == 3
    
    @pytest.mark.asyncio
    async def test_default_broker_creation(self):
        """Test creating broker with default configuration."""
        broker = create_default_broker()
        
        assert isinstance(broker, CallBroker)
        
        # Should have default models configured
        assert len(broker.model_limits) >= 3  # Should have multiple default models
        
        # Test that we can get stats (even if empty initially)
        stats = await broker.get_stats()
        assert isinstance(stats, dict)
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_mixed_regular_and_streaming_calls(self):
        """Test broker handling both regular and streaming calls."""
        broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=5))
        
        regular_mock = AsyncMock(return_value="regular_result")
        
        async def stream_mock():
            yield {"chunk": 1}
            yield {"chunk": 2}
            yield {"output": "stream_result"}
        
        # Execute both types of calls
        regular_result = await broker.call_agent_regular("test:model", regular_mock)
        
        stream_chunks = []
        async for chunk in broker.call_agent_streaming("test:model", stream_mock):
            stream_chunks.append(chunk)
        
        assert regular_result == "regular_result"
        assert len(stream_chunks) == 3
        assert stream_chunks[-1]["output"] == "stream_result"
        
        await broker.shutdown()


# Test fixtures and utilities
@pytest.fixture
async def test_broker():
    """Provide a test broker instance."""
    broker = CallBroker({}, ModelLimits(rpm=100, rpd=1000, max_concurrency=10))
    yield broker
    await broker.shutdown()


@pytest.fixture
def restrictive_broker():
    """Provide a broker with restrictive limits for testing rate limiting."""
    limits = {"test:model": ModelLimits(rpm=2, rpd=10, max_concurrency=1)}
    return CallBroker(limits)


# Performance and stress tests
class TestCallBrokerPerformance:
    """Performance and stress tests for CallBroker."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_regular_calls(self):
        """Test broker under high regular call load."""
        broker = CallBroker({}, ModelLimits(rpm=1000, rpd=10000, max_concurrency=50))
        
        mock_func = AsyncMock(return_value="result")
        
        # Execute many concurrent calls
        tasks = [
            asyncio.create_task(broker.call_agent_regular("test:model", mock_func))
            for _ in range(100)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 100
        assert all(result == "result" for result in results)
        assert mock_func.call_count == 100
        
        # Should complete reasonably quickly
        elapsed = end_time - start_time
        assert elapsed < 10.0  # Should complete within 10 seconds
        
        await broker.shutdown()
    
    @pytest.mark.asyncio
    async def test_sliding_window_performance(self):
        """Test sliding window counter performance with many requests."""
        counter = SlidingWindowCounter(window_seconds=60)
        
        # Rapidly record many requests
        start_time = time.time()
        for _ in range(1000):
            await counter.record_request()
        
        # Check that we can still query efficiently
        can_make = await counter.can_make_request(1500)
        wait_time = await counter.get_wait_time(1500)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        assert can_make == True  # Should allow more requests
        assert wait_time == 0.0  # No wait needed
        assert elapsed < 5.0  # Should complete quickly