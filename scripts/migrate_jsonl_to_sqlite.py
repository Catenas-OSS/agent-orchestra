#!/usr/bin/env python3
"""
JSONL to SQLite Migration Script for Agent Orchestra

Migrates existing JSONL-based run data to SQLite format while preserving
all data integrity and enabling rollback if needed.

Usage:
    python scripts/migrate_jsonl_to_sqlite.py [--source .ao_runs] [--target .ao_runs/ao.sqlite3] [--dry-run]
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Set
import sqlite3

# Add the src directory to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_orchestra.orchestrator.store_sqlite import SQLiteRunStore
from agent_orchestra.orchestrator.store import SavedNode, SavedForeachItem


class JSONLMigrator:
    """Handles migration from JSONL to SQLite format."""
    
    def __init__(self, source_dir: Path, target_db: Path, dry_run: bool = False):
        self.source_dir = Path(source_dir)
        self.target_db = Path(target_db)
        self.dry_run = dry_run
        self.stats = {
            'runs_migrated': 0,
            'events_migrated': 0,
            'nodes_migrated': 0,
            'foreach_items_migrated': 0,
            'errors': []
        }
    
    def discover_runs(self) -> List[str]:
        """Discover all run directories in source."""
        if not self.source_dir.exists():
            print(f"❌ Source directory {self.source_dir} does not exist")
            return []
        
        run_dirs = []
        for item in self.source_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it looks like a run directory
                if any(item.glob('*.json')) or any(item.glob('*.jsonl')):
                    run_dirs.append(item.name)
        
        return sorted(run_dirs)
    
    def load_jsonl_run_data(self, run_id: str) -> Dict[str, Any]:
        """Load all data for a run from JSONL format."""
        run_dir = self.source_dir / run_id
        
        data = {
            'run_id': run_id,
            'meta': {},
            'status': {},
            'events': [],
            'nodes': {},
            'foreach_items': {},
            'gate_pruning': set()
        }
        
        try:
            # Load metadata
            meta_file = run_dir / "meta.json"
            if meta_file.exists():
                data['meta'] = json.loads(meta_file.read_text())
            
            # Load status
            status_file = run_dir / "status.json"
            if status_file.exists():
                data['status'] = json.loads(status_file.read_text())
            
            # Load events from JSONL
            events_file = run_dir / "events.jsonl"
            if events_file.exists():
                for line in events_file.read_text().strip().split('\\n'):
                    if line.strip():
                        data['events'].append(json.loads(line))
            
            # Load node results
            nodes_file = run_dir / "nodes.json"
            if nodes_file.exists():
                data['nodes'] = json.loads(nodes_file.read_text())
            
            # Load foreach items
            for foreach_file in run_dir.glob("foreach_*.json"):
                node_id = foreach_file.stem.replace("foreach_", "")
                data['foreach_items'][node_id] = json.loads(foreach_file.read_text())
            
            # Load gate pruning
            gate_file = run_dir / "gate_pruning.json"
            if gate_file.exists():
                pruned_list = json.loads(gate_file.read_text())
                data['gate_pruning'] = set(pruned_list)
                
        except Exception as e:
            self.stats['errors'].append(f"Error loading run {run_id}: {e}")
            return None
        
        return data
    
    def validate_migration_data(self, data: Dict[str, Any]) -> bool:
        """Validate that migration data is complete and consistent."""
        run_id = data['run_id']
        
        # Check required fields
        if not data.get('events'):
            print(f"⚠️  Run {run_id}: No events found")
        
        if not data.get('nodes'):
            print(f"⚠️  Run {run_id}: No node results found")
        
        # Validate event sequence
        events = data['events']
        if events:
            sequences = [e.get('event_seq', 0) for e in events]
            if len(sequences) != len(set(sequences)):
                print(f"⚠️  Run {run_id}: Duplicate event sequences found")
                return False
        
        return True
    
    async def migrate_run_to_sqlite(self, store: SQLiteRunStore, data: Dict[str, Any]) -> bool:
        """Migrate a single run to SQLite."""
        run_id = data['run_id']
        
        if not self.validate_migration_data(data):
            return False
        
        try:
            # Create mock run_spec from metadata
            class MockRunSpec:
                def __init__(self, goal: str):
                    self.goal = goal
            
            goal = data['meta'].get('goal', 'Migrated from JSONL')
            run_spec = MockRunSpec(goal)
            
            if not self.dry_run:
                # Start the run
                await store.start_run(run_id, run_spec)
                
                # Migrate events
                for event_data in data['events']:
                    # Create Event-like object
                    class MockEvent:
                        def __init__(self, event_dict):
                            self.__dict__.update(event_dict)
                    
                    event = MockEvent(event_data)
                    await store.append_event(run_id, event)
                
                # Migrate node results
                for node_id, node_data in data['nodes'].items():
                    saved_node = SavedNode(
                        node_id=node_id,
                        signature=node_data['signature'],
                        result=node_data['result']
                    )
                    await store.save_node_result(run_id, saved_node)
                
                # Migrate foreach items
                for node_id, items_data in data['foreach_items'].items():
                    for item_index_str, item_data in items_data.items():
                        item_index = int(item_index_str)
                        saved_item = SavedForeachItem(
                            node_id=node_id,
                            item_index=item_index,
                            signature=item_data['signature'],
                            result=item_data['result']
                        )
                        await store.save_foreach_item(run_id, saved_item)
                
                # Migrate gate pruning
                if data['gate_pruning']:
                    await store.save_gate_pruning(run_id, data['gate_pruning'])
                
                # Set final status
                status = data['status'].get('status', 'unknown')
                if status == 'complete':
                    await store.mark_run_complete(run_id)
                elif status == 'error':
                    error_msg = data['status'].get('error', 'Unknown error during migration')
                    await store.mark_run_error(run_id, error_msg)
                elif status == 'canceled':
                    await store.mark_run_canceled(run_id)
            
            # Update stats
            self.stats['runs_migrated'] += 1
            self.stats['events_migrated'] += len(data['events'])
            self.stats['nodes_migrated'] += len(data['nodes'])
            self.stats['foreach_items_migrated'] += sum(len(items) for items in data['foreach_items'].values())
            
            return True
            
        except Exception as e:
            error_msg = f"Error migrating run {run_id}: {e}"
            self.stats['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    async def migrate_all_runs(self) -> bool:
        """Migrate all discovered runs."""
        run_ids = self.discover_runs()
        
        if not run_ids:
            print("📭 No runs found to migrate")
            return True
        
        print(f"🔍 Discovered {len(run_ids)} runs to migrate")
        
        # Initialize SQLite store
        if not self.dry_run:
            store = SQLiteRunStore(self.target_db)
        else:
            store = None
        
        success_count = 0
        
        for i, run_id in enumerate(run_ids, 1):
            print(f"📦 [{i}/{len(run_ids)}] Migrating run: {run_id}")
            
            # Load JSONL data
            data = self.load_jsonl_run_data(run_id)
            if data is None:
                continue
            
            # Migrate to SQLite
            if self.dry_run:
                print(f"   📊 Would migrate: {len(data['events'])} events, {len(data['nodes'])} nodes")
                success_count += 1
            else:
                if await self.migrate_run_to_sqlite(store, data):
                    success_count += 1
                    print(f"   ✅ Migrated successfully")
                else:
                    print(f"   ❌ Migration failed")
        
        return success_count == len(run_ids)
    
    def create_backup(self) -> Path:
        """Create backup of source directory."""
        if self.dry_run:
            print("🔄 Would create backup (dry-run mode)")
            return None
        
        timestamp = int(time.time())
        backup_path = self.source_dir.parent / f"{self.source_dir.name}_backup_{timestamp}"
        
        print(f"💾 Creating backup: {backup_path}")
        shutil.copytree(self.source_dir, backup_path)
        
        return backup_path
    
    def verify_migration(self, store: SQLiteRunStore, original_runs: List[str]) -> bool:
        """Verify migration integrity."""
        if self.dry_run:
            print("✅ Would verify migration (dry-run mode)")
            return True
        
        print("🔍 Verifying migration integrity...")
        
        try:
            # Check all runs were migrated
            async def _verify():
                migrated_runs = await store.list_runs(limit=1000)
                migrated_run_ids = {run['run_id'] for run in migrated_runs}
                
                missing_runs = set(original_runs) - migrated_run_ids
                if missing_runs:
                    print(f"❌ Missing runs after migration: {missing_runs}")
                    return False
                
                # Spot check a few runs
                for run_id in original_runs[:3]:  # Check first 3 runs
                    stats = await store.get_run_statistics(run_id)
                    if not stats:
                        print(f"❌ No statistics for migrated run: {run_id}")
                        return False
                
                return True
            
            import asyncio
            return asyncio.run(_verify())
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def print_summary(self):
        """Print migration summary."""
        print("\\n" + "=" * 60)
        print("📊 MIGRATION SUMMARY")
        print("=" * 60)
        
        if self.dry_run:
            print("🔄 DRY RUN MODE - No changes made")
        
        print(f"✅ Runs migrated: {self.stats['runs_migrated']}")
        print(f"📝 Events migrated: {self.stats['events_migrated']}")
        print(f"🔧 Nodes migrated: {self.stats['nodes_migrated']}")
        print(f"🔄 Foreach items migrated: {self.stats['foreach_items_migrated']}")
        
        if self.stats['errors']:
            print(f"❌ Errors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if len(self.stats['errors']) > 5:
                print(f"   ... and {len(self.stats['errors']) - 5} more")
        else:
            print("✅ No errors")


async def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate JSONL run data to SQLite")
    parser.add_argument("--source", default=".ao_runs", help="Source JSONL directory")
    parser.add_argument("--target", default=".ao_runs/ao.sqlite3", help="Target SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup")
    parser.add_argument("--force", action="store_true", help="Overwrite existing SQLite database")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    target_db = Path(args.target)
    
    print("🚀 Agent Orchestra JSONL → SQLite Migration")
    print("=" * 50)
    print(f"📂 Source: {source_dir}")
    print(f"🎯 Target: {target_db}")
    print(f"🔄 Dry run: {args.dry_run}")
    print()
    
    # Check if target already exists
    if target_db.exists() and not args.force and not args.dry_run:
        print(f"❌ Target database {target_db} already exists")
        print("   Use --force to overwrite, or --dry-run to test migration")
        return 1
    
    # Initialize migrator
    migrator = JSONLMigrator(source_dir, target_db, args.dry_run)
    
    # Discover runs before any changes
    original_runs = migrator.discover_runs()
    
    # Create backup if requested
    backup_path = None
    if not args.no_backup and not args.dry_run:
        backup_path = migrator.create_backup()
    
    try:
        # Perform migration
        success = await migrator.migrate_all_runs()
        
        # Verify migration
        if success and not args.dry_run:
            store = SQLiteRunStore(target_db)
            success = migrator.verify_migration(store, original_runs)
        
        # Print summary
        migrator.print_summary()
        
        if success:
            print("\\n🎉 Migration completed successfully!")
            if backup_path:
                print(f"💾 Backup created at: {backup_path}")
            if not args.dry_run:
                print(f"🎯 SQLite database ready at: {target_db}")
            return 0
        else:
            print("\\n❌ Migration completed with errors")
            return 1
            
    except Exception as e:
        print(f"\\n💥 Migration failed: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    exit(exit_code)