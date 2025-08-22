"""
Utility for creating diff previews of filesystem changes.
Used by TOOL_END events to show what changed.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any
from pathlib import Path


def create_diff_preview(file_path: str, content_before: Optional[str] = None, content_after: Optional[str] = None, max_lines: int = 10) -> Optional[str]:
    """
    Create a diff preview for filesystem changes.
    
    Args:
        file_path: Path to the file that changed
        content_before: File content before change (None if file was created)
        content_after: File content after change (None if file was deleted)
        max_lines: Maximum lines to show in preview
        
    Returns:
        Diff preview string or None if no meaningful diff
    """
    
    if content_before is None and content_after is None:
        return None
    
    # File creation
    if content_before is None:
        lines = content_after.splitlines()[:max_lines] if content_after else []
        preview = "\n".join(f"+ {line}" for line in lines)
        if len(content_after.splitlines()) > max_lines:
            preview += f"\n... ({len(content_after.splitlines()) - max_lines} more lines)"
        return preview
    
    # File deletion  
    if content_after is None:
        lines = content_before.splitlines()[:max_lines]
        preview = "\n".join(f"- {line}" for line in lines)
        if len(content_before.splitlines()) > max_lines:
            preview += f"\n... ({len(content_before.splitlines()) - max_lines} more lines)"
        return preview
    
    # File modification - simple line-by-line diff
    before_lines = content_before.splitlines()
    after_lines = content_after.splitlines()
    
    # Find first and last differing lines
    first_diff = 0
    while first_diff < min(len(before_lines), len(after_lines)):
        if before_lines[first_diff] != after_lines[first_diff]:
            break
        first_diff += 1
    
    # If files are identical
    if first_diff == len(before_lines) == len(after_lines):
        return None
    
    # Show context around changes
    start = max(0, first_diff - 2)
    end = min(len(before_lines), len(after_lines), first_diff + max_lines - 2)
    
    diff_lines = []
    
    # Add context before changes
    for i in range(start, first_diff):
        diff_lines.append(f"  {before_lines[i]}")
    
    # Add changes
    changes_shown = 0
    for i in range(first_diff, min(end, len(before_lines), len(after_lines))):
        if changes_shown >= max_lines:
            break
        if i < len(before_lines) and i < len(after_lines):
            if before_lines[i] != after_lines[i]:
                diff_lines.append(f"- {before_lines[i]}")
                diff_lines.append(f"+ {after_lines[i]}")
                changes_shown += 2
            else:
                diff_lines.append(f"  {before_lines[i]}")
                changes_shown += 1
        elif i < len(before_lines):
            diff_lines.append(f"- {before_lines[i]}")
            changes_shown += 1
        elif i < len(after_lines):
            diff_lines.append(f"+ {after_lines[i]}")
            changes_shown += 1
    
    # Handle length differences
    if len(after_lines) > len(before_lines):
        for i in range(len(before_lines), min(len(after_lines), len(before_lines) + max_lines)):
            diff_lines.append(f"+ {after_lines[i]}")
    elif len(before_lines) > len(after_lines):
        for i in range(len(after_lines), min(len(before_lines), len(after_lines) + max_lines)):
            diff_lines.append(f"- {before_lines[i]}")
    
    return "\n".join(diff_lines) if diff_lines else None


def create_fs_change_summary(file_path: str, operation: str, bytes_before: Optional[int] = None, bytes_after: Optional[int] = None) -> Dict[str, Any]:
    """
    Create summary metadata for filesystem changes.
    
    Args:
        file_path: Path to the changed file
        operation: Type of operation (create, modify, delete)
        bytes_before: File size before change
        bytes_after: File size after change
        
    Returns:
        Summary dictionary for TOOL_END event
    """
    
    summary = {
        "path": file_path,
        "operation": operation,
    }
    
    if bytes_before is not None:
        summary["bytes_before"] = bytes_before
    if bytes_after is not None:
        summary["bytes_after"] = bytes_after
        
    # Calculate size change
    if bytes_before is not None and bytes_after is not None:
        summary["size_change"] = bytes_after - bytes_before
    elif bytes_before is None and bytes_after is not None:
        summary["size_change"] = bytes_after  # File created
    elif bytes_before is not None and bytes_after is None:
        summary["size_change"] = -bytes_before  # File deleted
        
    return summary