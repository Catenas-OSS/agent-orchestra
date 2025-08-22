"""
Simple event bus for TUI communication.
"""

from __future__ import annotations
import asyncio
from typing import Callable, Dict, Set, Any, NamedTuple


class Subscription(NamedTuple):
    callback: Callable
    topics: frozenset[str]


class EventBus:
    """A simple asynchronous event bus for decoupling components."""
    
    def __init__(self):
        self.subscriptions: Set[Subscription] = set()
    
    def subscribe(self, callback: Callable, topics: Set[str]) -> Subscription:
        """Subscribe a callback to a set of topics."""
        subscription = Subscription(callback, frozenset(topics))
        self.subscriptions.add(subscription)
        return subscription
    
    def unsubscribe(self, subscription: Subscription) -> None:
        """Unsubscribe a callback."""
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)
    
    async def publish(self, topic: str, data: Any) -> None:
        """Publish an event to all subscribers of a topic."""
        for sub in self.subscriptions:
            if topic in sub.topics:
                # Schedule callback to run in the event loop
                asyncio.create_task(sub.callback(topic, data))
    
    async def close(self) -> None:
        """Clear all subscriptions."""
        self.subscriptions.clear()
