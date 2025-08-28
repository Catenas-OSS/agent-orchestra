"""
CallBroker - Production-grade LLM rate limiting and request management.

This module provides:
- Per-model rate limiting (RPM, RPD, max_concurrency)
- 429-aware retries with jittered exponential backoff
- Chunk passthrough for streaming (AGENT_CHUNK stays verbatim)
- Agent pool management and reuse
- Sliding window request counters
"""

from __future__ import annotations
import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Optional
from collections import deque
import logging
from ..logging import get_system_logger

logger = logging.getLogger(__name__)


@dataclass
class ModelLimits:
    """Rate limits for a specific model."""
    rpm: int = 60  # requests per minute
    rpd: int = 1000  # requests per day  
    max_concurrency: int = 10  # concurrent requests


class SlidingWindowCounter:
    """Efficient sliding window counter for rate limiting."""
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.requests: deque[float] = deque()
        self._lock = asyncio.Lock()
    
    async def can_make_request(self, limit: int) -> bool:
        """Check if we can make a request within the limit."""
        async with self._lock:
            now = time.time()
            # Remove requests outside the window
            while self.requests and now - self.requests[0] > self.window_seconds:
                self.requests.popleft()
            
            return len(self.requests) < limit
    
    async def record_request(self) -> None:
        """Record a new request timestamp."""
        async with self._lock:
            self.requests.append(time.time())
    
    async def get_wait_time(self, limit: int) -> float:
        """Get time to wait before next request is allowed."""
        async with self._lock:
            if len(self.requests) < limit:
                return 0.0
            
            # Find the oldest request that would be in the next window
            now = time.time()
            oldest_to_keep = now - self.window_seconds
            
            # Find first request that's still in the window
            for req_time in self.requests:
                if req_time > oldest_to_keep:
                    # Wait until this request falls out of the window
                    return req_time - oldest_to_keep
            
            return 0.0


class CallBroker:
    """
    Production-grade LLM call broker with rate limiting, retries, and streaming support.
    
    Features:
    - Per-model rate limits (RPM, RPD, concurrency)
    - 429 retry with jittered exponential backoff
    - Chunk passthrough for streaming
    - Request queuing and throttling
    """
    
    def __init__(self, model_limits: Dict[str, ModelLimits], default_limits: Optional[ModelLimits] = None):
        self.model_limits = model_limits
        self.default_limits = default_limits or ModelLimits()
        self._system_logger = get_system_logger()
        
        # Per-model tracking
        self.rpm_counters: Dict[str, SlidingWindowCounter] = {}
        self.rpd_counters: Dict[str, SlidingWindowCounter] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Request queues for throttling
        self.request_queues: Dict[str, asyncio.Queue] = {} # type: ignore
        self._queue_processors: Dict[str, asyncio.Task] = {} # type: ignore
        
        # Global state
        self._shutdown = False
        
        logger.info(f"CallBroker initialized with {len(model_limits)} model configurations")
        self._system_logger.info("call_broker", f"CallBroker initialized with {len(model_limits)} model configurations")
        
        # Log model limits for debugging
        for model, limits in model_limits.items():
            self._system_logger.debug("call_broker", f"Model {model}: RPM={limits.rpm}, RPD={limits.rpd}, concurrency={limits.max_concurrency}")
    
    def _get_model_limits(self, model: str) -> ModelLimits:
        """Get rate limits for a model, falling back to defaults."""
        return self.model_limits.get(model, self.default_limits)
    
    def _get_or_create_counters(self, model: str) -> tuple[SlidingWindowCounter, SlidingWindowCounter, asyncio.Semaphore]:
        """Get or create rate limiting infrastructure for a model."""
        if model not in self.rpm_counters:
            self.rpm_counters[model] = SlidingWindowCounter(window_seconds=60)
            self.rpd_counters[model] = SlidingWindowCounter(window_seconds=24 * 60 * 60)
            
            limits = self._get_model_limits(model)
            self.semaphores[model] = asyncio.Semaphore(limits.max_concurrency)
        
        return (
            self.rpm_counters[model],
            self.rpd_counters[model], 
            self.semaphores[model]
        )
    
    async def _wait_for_rate_limit(self, model: str) -> None:
        """Wait until rate limits allow a new request."""
        limits = self._get_model_limits(model)
        rpm_counter, rpd_counter, _ = self._get_or_create_counters(model)
        
        while True:
            # Check RPM limit
            if not await rpm_counter.can_make_request(limits.rpm):
                wait_time = await rpm_counter.get_wait_time(limits.rpm)
                if wait_time > 0:
                    logger.debug(f"Rate limit hit for {model}, waiting {wait_time:.2f}s (RPM)")
                    await asyncio.sleep(wait_time)
                    continue
            
            # Check RPD limit
            if not await rpd_counter.can_make_request(limits.rpd):
                wait_time = await rpd_counter.get_wait_time(limits.rpd)
                if wait_time > 0:
                    logger.warning(f"Daily rate limit hit for {model}, waiting {wait_time:.2f}s (RPD)")
                    await asyncio.sleep(wait_time)
                    continue
            
            # Both limits OK
            break
    
    async def _execute_with_429_retry(
        self,
        model: str,
        func: Callable[[], Any],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> Any:
        """Execute function with 429 retry logic and exponential backoff."""
        
        for attempt in range(max_retries + 1):
            try:
                # Record the request
                rpm_counter, rpd_counter, _ = self._get_or_create_counters(model)
                await rpm_counter.record_request()
                await rpd_counter.record_request()
                
                # Execute the function
                return await func()
                
            except Exception as e:
                # Check if it's a 429 error
                is_429 = (
                    "429" in str(e) or 
                    "rate limit" in str(e).lower() or
                    "too many requests" in str(e).lower()
                )
                
                if not is_429 or attempt >= max_retries:
                    # Not a 429 or out of retries, re-raise
                    raise
                
                # Calculate jittered exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = delay * 0.1 * (random.random() - 0.5)  # ±5% jitter
                total_delay = delay + jitter
                
                logger.warning(f"429 error for {model} (attempt {attempt + 1}/{max_retries + 1}), "
                             f"retrying in {total_delay:.2f}s: {e}")
                
                await asyncio.sleep(total_delay)
        
        # Should never reach here
        raise RuntimeError(f"Retry logic exhausted for {model}")
    
    async def call_agent_streaming(
        self,
        model: str,
        agent_func: Callable[[], AsyncGenerator[Dict[str, Any], None]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute streaming agent call with rate limiting and 429 retries.
        
        Preserves chunk passthrough - all chunks are yielded verbatim.
        """
        
        # Wait for rate limits
        await self._wait_for_rate_limit(model)
        
        # Acquire concurrency semaphore
        _, _, semaphore = self._get_or_create_counters(model)
        
        async with semaphore:
            logger.debug(f"Executing streaming call for {model}")
            
            # Execute with 429 retry, but handle streaming specially
            # For streaming, we handle retries by restarting the entire stream
            
            # For streaming, we need to handle retries differently
            # If we get a 429, we need to restart the entire stream
            for attempt in range(4):  # max 3 retries
                try:
                    # Record the request attempt
                    rpm_counter, rpd_counter, _ = self._get_or_create_counters(model)
                    await rpm_counter.record_request()
                    await rpd_counter.record_request()
                    
                    # Execute streaming function
                    async for chunk in agent_func():
                        # Pass chunks through unchanged (preserves AGENT_CHUNK format)
                        yield chunk
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    # Check if it's a 429 error
                    is_429 = (
                        "429" in str(e) or 
                        "rate limit" in str(e).lower() or
                        "too many requests" in str(e).lower()
                    )
                    
                    if not is_429 or attempt >= 3:
                        # Not a 429 or out of retries, re-raise
                        logger.error(f"Streaming call failed for {model}: {e}")
                        raise
                    
                    # Calculate delay for retry
                    delay = min(1.0 * (2 ** attempt), 60.0)
                    jitter = delay * 0.1 * (random.random() - 0.5)
                    total_delay = delay + jitter
                    
                    logger.warning(f"429 error in streaming call for {model} "
                                 f"(attempt {attempt + 1}/4), retrying in {total_delay:.2f}s")
                    
                    await asyncio.sleep(total_delay)
    
    async def call_agent_regular(
        self,
        model: str,
        agent_func: Callable[[], Any]
    ) -> Any:
        """
        Execute regular (non-streaming) agent call with rate limiting and 429 retries.
        """
        
        # Wait for rate limits
        await self._wait_for_rate_limit(model)
        
        # Acquire concurrency semaphore  
        _, _, semaphore = self._get_or_create_counters(model)
        
        async with semaphore:
            logger.debug(f"Executing regular call for {model}")
            
            # Execute with 429 retry
            return await self._execute_with_429_retry(model, agent_func)
    
    async def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current broker statistics."""
        stats = {}
        
        for model in self.model_limits.keys():
            if model in self.rpm_counters:
                rpm_counter = self.rpm_counters[model]
                rpd_counter = self.rpd_counters[model]
                semaphore = self.semaphores[model]
                limits = self._get_model_limits(model)
                
                # Get current counts (approximate, no lock needed for stats)
                now = time.time()
                rpm_count = sum(1 for t in rpm_counter.requests if now - t <= 60)
                rpd_count = sum(1 for t in rpd_counter.requests if now - t <= 24 * 60 * 60)
                
                stats[model] = {
                    "rpm_used": rpm_count,
                    "rpm_limit": limits.rpm,
                    "rpd_used": rpd_count, 
                    "rpd_limit": limits.rpd,
                    "concurrent_used": limits.max_concurrency - semaphore._value,
                    "concurrent_limit": limits.max_concurrency
                }
        
        return stats # type: ignore
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the broker."""
        self._shutdown = True
        
        # Cancel any queue processors
        for task in self._queue_processors.values(): # type: ignore
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("CallBroker shutdown complete")


# Default configurations for common models
DEFAULT_MODEL_LIMITS = {
    "openai:gpt-4o-mini": ModelLimits(rpm=3, rpd=200, max_concurrency=2),
    "openai:gpt-4o": ModelLimits(rpm=10, rpd=500, max_concurrency=5),
    "openai:gpt-3.5-turbo": ModelLimits(rpm=60, rpd=1000, max_concurrency=10),
    "anthropic:claude-3-haiku": ModelLimits(rpm=50, rpd=1000, max_concurrency=5),
    "anthropic:claude-3-sonnet": ModelLimits(rpm=20, rpd=500, max_concurrency=3),
}


def create_default_broker() -> CallBroker:
    """Create a CallBroker with sensible defaults for common models."""
    return CallBroker(DEFAULT_MODEL_LIMITS, ModelLimits(rpm=30, rpd=500, max_concurrency=5))