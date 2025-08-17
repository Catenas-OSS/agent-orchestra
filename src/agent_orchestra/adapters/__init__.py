"""Adapter system for Agent Orchestra.

Adapters provide a clean interface to different agent backends.
The only allowed imports of external agent libraries should be in adapter modules.
"""

from agent_orchestra.adapters.base import AdapterResult, BaseAdapter, UsageMetrics

__all__ = ["BaseAdapter", "AdapterResult", "UsageMetrics"]

