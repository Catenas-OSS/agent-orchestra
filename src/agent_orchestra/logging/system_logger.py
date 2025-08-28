"""
Centralized logging infrastructure for Agent Orchestra TUI.
Captures all system logs and routes them properly to avoid console interference.
"""

from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import threading
import contextlib
import io
import sys


class LogLevel(Enum):
    """Log levels for the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """A structured log entry."""
    timestamp: float
    level: LogLevel
    component: str  # orchestrator, executor, workflow, cli, etc.
    message: str
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted_time(self) -> str:
        """Get human-readable timestamp."""
        return time.strftime('%H:%M:%S', time.localtime(self.timestamp))
    
    def __str__(self) -> str:
        """Format log entry for display."""
        node_info = f"[{self.node_id}] " if self.node_id else ""
        return f"{self.formatted_time} [{self.level.value}] {self.component}: {node_info}{self.message}"


class SystemLogger:
    """
    Centralized logger that captures all system output and routes it appropriately.
    Designed to work with TUI without interfering with terminal control.
    """
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.log_buffer: deque[LogEntry] = deque(maxlen=max_entries)
        self.component_buffers: Dict[str, deque[LogEntry]] = {}
        self.lock = threading.Lock()
        
        # Original stdout/stderr for restoration
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tui_mode = False
        
        # Setup Python logging integration
        self._setup_python_logging()
    
    def _setup_python_logging(self):
        """Setup Python's logging module to route to our system."""
        # Create a custom handler that routes to our logger
        class SystemLogHandler(logging.Handler):
            def __init__(self, system_logger: SystemLogger):
                super().__init__()
                self.system_logger = system_logger
            
            def emit(self, record):
                try:
                    level_map = {
                        logging.DEBUG: LogLevel.DEBUG,
                        logging.INFO: LogLevel.INFO,
                        logging.WARNING: LogLevel.WARNING,
                        logging.ERROR: LogLevel.ERROR,
                        logging.CRITICAL: LogLevel.CRITICAL,
                    }
                    level = level_map.get(record.levelno, LogLevel.INFO)
                    component = record.name if record.name != "root" else "system"
                    
                    # Capture the full logging message
                    message = record.getMessage()
                    if hasattr(record, 'pathname'):
                        # Add file context for debugging
                        filename = record.pathname.split('/')[-1] if record.pathname else 'unknown'
                        message = f"[{filename}:{record.lineno}] {message}"
                    
                    self.system_logger.log(level, component, message)
                except Exception:
                    pass  # Don't crash on logging errors
        
        # Add our handler to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Only remove StreamHandlers to prevent duplicate output, keep our handler
        handlers_to_remove = []
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, SystemLogHandler):
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)
        
        # Add our custom handler if not already present
        has_our_handler = any(isinstance(h, SystemLogHandler) for h in root_logger.handlers)
        if not has_our_handler:
            system_handler = SystemLogHandler(self)
            root_logger.addHandler(system_handler)
        
        # Immediately suppress known noisy third-party loggers
        noisy_loggers = [
            'mcp_use',
            'mcp_use.telemetry', 
            'mcp_use.telemetry.telemetry',
            'mcp',
            'mcp_client',
            'mcp_server',
            'httpx',
            'urllib3'
        ]
        
        for logger_name in noisy_loggers:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(logging.CRITICAL)
            noisy_logger.disabled = True
    
    def enable_tui_mode(self):
        """Enable TUI mode - capture stdout/stderr to prevent terminal interference."""
        if self._tui_mode:
            return
        
        self._tui_mode = True
        
        # Also suppress all external library logging by setting high threshold
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL)  # Only show critical errors from external libs
        
        # Specifically suppress known noisy loggers
        noisy_loggers = [
            'mcp_use',
            'mcp_use.telemetry',
            'mcp_use.telemetry.telemetry',
            'mcp',
            'mcp_client',
            'mcp_server',
            'httpx',
            'urllib3',
            'asyncio'
        ]
        
        for logger_name in noisy_loggers:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(logging.CRITICAL)
            noisy_logger.disabled = True
        
        # Create more aggressive capture that handles all print output
        class TUILogCapture(io.StringIO):
            def __init__(self, logger: SystemLogger, component: str, level: LogLevel, original_stream):
                super().__init__()
                self.logger = logger
                self.component = component
                self.level = level
                self.original_stream = original_stream
            
            def write(self, s):
                if s and s.strip():  # Capture any non-empty content
                    # Log to our system
                    self.logger.log(self.level, self.component, s.strip())
                return len(s)
            
            def flush(self):
                pass
            
            def isatty(self):
                return False
            
            def fileno(self):
                # Some libraries check for fileno - return original if available
                try:
                    return self.original_stream.fileno()
                except:
                    raise io.UnsupportedOperation("fileno")
        
        # Replace stdout/stderr with more comprehensive capture
        sys.stdout = TUILogCapture(self, "stdout", LogLevel.INFO, self._original_stdout)
        sys.stderr = TUILogCapture(self, "stderr", LogLevel.ERROR, self._original_stderr)
        
        # Also capture print function directly by monkey patching builtins
        import builtins
        self._original_print = builtins.print
        
        def captured_print(*args, **kwargs):
            # Convert print arguments to string
            if args:
                message = " ".join(str(arg) for arg in args)
                if message.strip():
                    self.log(LogLevel.INFO, "print", message.strip())
        
        builtins.print = captured_print
    
    def disable_tui_mode(self):
        """Disable TUI mode - restore normal stdout/stderr."""
        if not self._tui_mode:
            return
        
        self._tui_mode = False
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
        # Restore original print function
        if hasattr(self, '_original_print'):
            import builtins
            builtins.print = self._original_print
    
    def log(self, level: LogLevel, component: str, message: str, node_id: Optional[str] = None, **metadata):
        """Add a log entry."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            component=component,
            message=message,
            node_id=node_id,
            metadata=metadata
        )
        
        with self.lock:
            # Add to main buffer
            self.log_buffer.append(entry)
            
            # Add to component-specific buffer
            if component not in self.component_buffers:
                self.component_buffers[component] = deque(maxlen=200)
            self.component_buffers[component].append(entry)
    
    def debug(self, component: str, message: str, node_id: Optional[str] = None, **metadata):
        """Log debug message."""
        self.log(LogLevel.DEBUG, component, message, node_id, **metadata)
    
    def info(self, component: str, message: str, node_id: Optional[str] = None, **metadata):
        """Log info message."""
        self.log(LogLevel.INFO, component, message, node_id, **metadata)
    
    def warning(self, component: str, message: str, node_id: Optional[str] = None, **metadata):
        """Log warning message."""
        self.log(LogLevel.WARNING, component, message, node_id, **metadata)
    
    def error(self, component: str, message: str, node_id: Optional[str] = None, **metadata):
        """Log error message."""
        self.log(LogLevel.ERROR, component, message, node_id, **metadata)
    
    def critical(self, component: str, message: str, node_id: Optional[str] = None, **metadata):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, component, message, node_id, **metadata)
    
    def get_recent_logs(self, count: int = 50, component: Optional[str] = None, 
                       level: Optional[LogLevel] = None) -> List[LogEntry]:
        """Get recent log entries with optional filtering."""
        with self.lock:
            if component and component in self.component_buffers:
                logs = list(self.component_buffers[component])
            else:
                logs = list(self.log_buffer)
        
        # Filter by level if specified
        if level:
            logs = [log for log in logs if log.level == level]
        
        # Return most recent entries
        return logs[-count:] if len(logs) > count else logs
    
    def get_system_logs_formatted(self, count: int = 20) -> List[str]:
        """Get formatted system logs for TUI display."""
        logs = self.get_recent_logs(count)
        formatted = []
        
        for log in logs:
            # Color code by level
            if log.level == LogLevel.ERROR:
                prefix = "🔴"
            elif log.level == LogLevel.WARNING:
                prefix = "🟡"
            elif log.level == LogLevel.INFO:
                prefix = "🔵"
            elif log.level == LogLevel.DEBUG:
                prefix = "🟦"
            else:
                prefix = "⚪"
            
            # Truncate long messages for display
            message = log.message
            if len(message) > 60:
                message = message[:57] + "..."
            
            formatted.append(f"{prefix} {log.formatted_time} [{log.component}] {message}")
        
        return formatted
    
    def get_component_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics by component."""
        stats = {}
        
        with self.lock:
            for component, buffer in self.component_buffers.items():
                level_counts = {level.value: 0 for level in LogLevel}
                for entry in buffer:
                    level_counts[entry.level.value] += 1
                
                stats[component] = {
                    "total": len(buffer),
                    **level_counts
                }
        
        return stats
    
    def clear_logs(self, component: Optional[str] = None):
        """Clear logs for a component or all logs."""
        with self.lock:
            if component and component in self.component_buffers:
                self.component_buffers[component].clear()
            else:
                self.log_buffer.clear()
                for buffer in self.component_buffers.values():
                    buffer.clear()
    
    @contextlib.contextmanager
    def capture_prints(self, component: str = "unknown"):
        """Context manager to capture print statements temporarily."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        class TempCapture(io.StringIO):
            def __init__(self, logger: SystemLogger, comp: str, level: LogLevel):
                super().__init__()
                self.logger = logger
                self.component = comp
                self.level = level
            
            def write(self, s):
                if s.strip():
                    self.logger.log(self.level, self.component, s.strip())
                return len(s)
        
        try:
            sys.stdout = TempCapture(self, component, LogLevel.INFO)
            sys.stderr = TempCapture(self, component, LogLevel.ERROR)
            yield self
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


# Global system logger instance
_global_logger: Optional[SystemLogger] = None


def get_system_logger() -> SystemLogger:
    """Get the global system logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SystemLogger()
    return _global_logger


def init_logging(tui_mode: bool = False) -> SystemLogger:
    """Initialize the logging system."""
    logger = get_system_logger()
    if tui_mode:
        logger.enable_tui_mode()
    return logger


def log_info(component: str, message: str, node_id: Optional[str] = None, **metadata):
    """Convenience function to log info message."""
    get_system_logger().info(component, message, node_id, **metadata)


def log_error(component: str, message: str, node_id: Optional[str] = None, **metadata):
    """Convenience function to log error message."""
    get_system_logger().error(component, message, node_id, **metadata)


def log_warning(component: str, message: str, node_id: Optional[str] = None, **metadata):
    """Convenience function to log warning message."""
    get_system_logger().warning(component, message, node_id, **metadata)


def log_debug(component: str, message: str, node_id: Optional[str] = None, **metadata):
    """Convenience function to log debug message."""
    get_system_logger().debug(component, message, node_id, **metadata)