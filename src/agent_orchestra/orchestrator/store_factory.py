"""
Store Factory for Agent Orchestra

Provides convenient creation of different store types with environment-based configuration.
Supports both JSONL and SQLite stores with sensible defaults.
"""

import os
from pathlib import Path
from typing import Union, Optional, Dict, Any

from .store import JsonlRunStore
from .store_sqlite import SQLiteRunStore
from ..logging import get_system_logger


def create_store(
    kind: str = "auto",
    path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Union[JsonlRunStore, SQLiteRunStore]:
    """
    Create a run store instance.
    
    Args:
        kind: Store type - "jsonl", "sqlite", or "auto" (default)
        path: Path to store location (auto-configured if None)
        **kwargs: Additional store-specific arguments
        
    Returns:
        Configured store instance
        
    Environment Variables:
        AO_STORE: Store type ("jsonl" or "sqlite")
        AO_STORE_PATH: Store path
        AO_DB_PATH: SQLite database path (legacy)
        AO_RUNS_DIR: JSONL runs directory (legacy)
    """
    # Determine store kind from environment or parameter
    if kind == "auto":
        kind = os.getenv("AO_STORE", "jsonl").lower()
    
    kind = kind.lower()
    
    # Configure path based on store type and environment
    if path is None:
        if kind == "sqlite":
            path = os.getenv("AO_STORE_PATH") or os.getenv("AO_DB_PATH", ".ao_runs/ao.sqlite3")
        elif kind == "jsonl":
            path = os.getenv("AO_STORE_PATH") or os.getenv("AO_RUNS_DIR", ".ao_runs")
        else:
            raise ValueError(f"Unknown store kind: {kind}. Use 'jsonl' or 'sqlite'")
    
    # Create store instance
    system_logger = get_system_logger()
    system_logger.info("store_factory", f"Creating {kind} store at: {path}")
    
    if kind == "sqlite":
        store = SQLiteRunStore(db_path=path, **kwargs)
        system_logger.info("store_factory", f"SQLite store created: {path}")
        return store
    elif kind == "jsonl":
        store = JsonlRunStore(root=path, **kwargs)
        system_logger.info("store_factory", f"JSONL store created: {path}")
        return store
    else:
        system_logger.error("store_factory", f"Unknown store kind: {kind}")
        raise ValueError(f"Unknown store kind: {kind}. Use 'jsonl' or 'sqlite'")


def create_development_store(**kwargs) -> Union[JsonlRunStore, SQLiteRunStore]:
    """
    Create a store optimized for development.
    
    Uses SQLite by default for better development experience (fast queries, UI potential),
    but can be overridden with environment variables.
    """
    kind = os.getenv("AO_STORE", "sqlite").lower()
    
    if kind == "sqlite":
        return create_store(
            kind="sqlite",
            path=".ao_runs/dev.sqlite3",
            **kwargs
        )
    else:
        return create_store(
            kind="jsonl", 
            path=".ao_runs",
            flush_every=1,  # Immediate flushing for development
            **kwargs
        )


def create_production_store(**kwargs) -> Union[JsonlRunStore, SQLiteRunStore]:
    """
    Create a store optimized for production.
    
    Uses sensible defaults for production workloads with proper performance tuning.
    """
    kind = os.getenv("AO_STORE", "sqlite").lower()
    
    if kind == "sqlite":
        return create_store(
            kind="sqlite",
            path=os.getenv("AO_STORE_PATH", "/data/ao_runs/production.sqlite3"),
            busy_timeout=5000,  # Higher timeout for production
            **kwargs
        )
    else:
        return create_store(
            kind="jsonl",
            path=os.getenv("AO_STORE_PATH", "/data/ao_runs"),
            flush_every=5,  # Batch flushing for performance
            max_chunk_size=50000,  # Larger chunks for production
            **kwargs
        )


def create_memory_store(**kwargs) -> SQLiteRunStore:
    """
    Create an in-memory SQLite store for testing.
    
    Data is lost when the process exits - only use for testing!
    """
    return SQLiteRunStore(db_path=":memory:", **kwargs)


def get_store_info(store: Union[JsonlRunStore, SQLiteRunStore]) -> Dict[str, Any]:
    """
    Get information about a store instance.
    
    Returns:
        Dictionary with store type, path, and configuration details
    """
    if isinstance(store, SQLiteRunStore):
        return {
            "type": "sqlite",
            "path": str(store.path),
            "exists": store.path.exists(),
            "size_bytes": store.path.stat().st_size if store.path.exists() else 0,
            "wal_mode": True,
            "concurrent_access": True
        }
    elif isinstance(store, JsonlRunStore):
        return {
            "type": "jsonl", 
            "path": str(store.root),
            "exists": store.root.exists(),
            "flush_every": store.flush_every,
            "max_chunk_size": store.max_chunk_size,
            "concurrent_access": "limited"
        }
    else:
        return {"type": "unknown", "path": "unknown"}


# Convenience functions for specific configurations

def create_test_store(test_name: str = "test") -> SQLiteRunStore:
    """Create an isolated store for testing."""
    return SQLiteRunStore(db_path=f":memory:_{test_name}")


def create_ci_store() -> Union[JsonlRunStore, SQLiteRunStore]:
    """Create a store optimized for CI/CD environments."""
    # Use JSONL in CI by default (simpler, no SQLite dependencies)
    return create_store(
        kind=os.getenv("AO_STORE", "jsonl"),
        path=os.getenv("AO_STORE_PATH", "./ci_runs"),
        flush_every=1
    )


def migrate_store(
    source_store: Union[JsonlRunStore, SQLiteRunStore],
    target_kind: str,
    target_path: Optional[Union[str, Path]] = None
) -> Union[JsonlRunStore, SQLiteRunStore]:
    """
    Migrate data from one store type to another.
    
    Note: This is a placeholder for future implementation.
    For now, use the migration script: scripts/migrate_jsonl_to_sqlite.py
    """
    raise NotImplementedError(
        "Store migration not yet implemented. "
        "Use scripts/migrate_jsonl_to_sqlite.py for JSONL to SQLite migration."
    )


# Configuration validation

def validate_store_config() -> Dict[str, Any]:
    """
    Validate current store configuration from environment variables.
    
    Returns:
        Dictionary with validation results and recommendations
    """
    config = {
        "store_type": os.getenv("AO_STORE", "auto"),
        "store_path": os.getenv("AO_STORE_PATH"),
        "legacy_db_path": os.getenv("AO_DB_PATH"),
        "legacy_runs_dir": os.getenv("AO_RUNS_DIR"),
        "issues": [],
        "recommendations": []
    }
    
    # Check for conflicting configuration
    if config["store_path"] and (config["legacy_db_path"] or config["legacy_runs_dir"]):
        config["issues"].append(
            "Both AO_STORE_PATH and legacy paths (AO_DB_PATH/AO_RUNS_DIR) are set"
        )
        config["recommendations"].append(
            "Use AO_STORE_PATH only, remove AO_DB_PATH and AO_RUNS_DIR"
        )
    
    # Check store type validity
    if config["store_type"] not in ["auto", "jsonl", "sqlite"]:
        config["issues"].append(f"Invalid AO_STORE value: {config['store_type']}")
        config["recommendations"].append("Set AO_STORE to 'jsonl' or 'sqlite'")
    
    # Check path accessibility
    if config["store_path"]:
        path = Path(config["store_path"])
        if config["store_type"] == "sqlite" or path.suffix == ".sqlite3":
            # SQLite path
            if not path.parent.exists():
                config["issues"].append(f"SQLite parent directory does not exist: {path.parent}")
                config["recommendations"].append(f"Create directory: mkdir -p {path.parent}")
        else:
            # JSONL directory
            if not path.exists():
                config["recommendations"].append(f"JSONL directory will be created: {path}")
    
    return config


def print_store_config():
    """Print current store configuration and validation results."""
    import logging
    logger = logging.getLogger(__name__)
    
    config = validate_store_config()
    
    print("🏪 Agent Orchestra Store Configuration")
    print("=" * 45)
    print(f"Store type: {config['store_type']}")
    print(f"Store path: {config['store_path'] or 'auto-configured'}")
    
    if config['legacy_db_path']:
        print(f"Legacy DB path: {config['legacy_db_path']}")
    if config['legacy_runs_dir']:
        print(f"Legacy runs dir: {config['legacy_runs_dir']}")
    
    if config['issues']:
        print("\n⚠️  Issues:")
        for issue in config['issues']:
            print(f"   • {issue}")
            logger.warning(f"Store config issue: {issue}")
    
    if config['recommendations']:
        print("\n💡 Recommendations:")
        for rec in config['recommendations']:
            print(f"   • {rec}")
            logger.info(f"Store config recommendation: {rec}")
    
    if not config['issues']:
        print("\n✅ Configuration looks good!")
        logger.info("Store configuration validated successfully")


def _run_examples():
    """Run store factory examples for testing and demonstration."""
    print("🏪 Store Factory Examples")
    print("=" * 30)
    
    # Show current configuration
    print_store_config()
    
    print("\nExample store creation:")
    
    # Create different types of stores
    examples = [
        ("Auto (environment)", lambda: create_store()),
        ("Development SQLite", lambda: create_development_store()),
        ("Production", lambda: create_production_store()), 
        ("Memory (testing)", lambda: create_memory_store()),
        ("Explicit JSONL", lambda: create_store("jsonl", ".test_runs")),
        ("Explicit SQLite", lambda: create_store("sqlite", ".test.db")),
    ]
    
    for name, factory in examples:
        try:
            store = factory()
            info = get_store_info(store)
            print(f"✅ {name}: {info['type']} at {info['path']}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print("\nEnvironment variables:")
    print("  AO_STORE=sqlite|jsonl")
    print("  AO_STORE_PATH=/path/to/store")
    print("  AO_DB_PATH=/path/to/db.sqlite3 (legacy)")
    print("  AO_RUNS_DIR=/path/to/runs (legacy)")


if __name__ == "__main__":
    _run_examples()