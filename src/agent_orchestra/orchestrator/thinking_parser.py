"""
Utility functions for parsing thinking content from agent streaming responses.
Handles XML tags and other structured thinking formats.
"""

import re
from typing import Optional, Tuple, List


def extract_thinking_content(text: str) -> Optional[str]:
    """
    Extract thinking content from various XML tag formats.
    
    Args:
        text: Raw text that may contain thinking tags
        
    Returns:
        The thinking content if found, None otherwise
    """
    if not text or not isinstance(text, str):
        return None
    
    # Common thinking tag patterns
    thinking_patterns = [
        r'<thinking>(.*?)</thinking>',
        r'<thinking>(.*?)</thinking>',
        r'<thought>(.*?)</thought>',
        r'<reasoning>(.*?)</reasoning>',
        r'<analysis>(.*?)</analysis>',
        r'<planning>(.*?)</planning>',
    ]
    
    for pattern in thinking_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            thinking_content = match.group(1).strip()
            if thinking_content:
                return thinking_content
    
    return None


def has_thinking_content(text: str) -> bool:
    """
    Check if text contains any thinking content markers.
    
    Args:
        text: Text to check
        
    Returns:
        True if thinking content is detected
    """
    return extract_thinking_content(text) is not None


def extract_all_thinking_segments(text: str) -> List[str]:
    """
    Extract all thinking segments from text that may contain multiple thinking blocks.
    
    Args:
        text: Text that may contain multiple thinking segments
        
    Returns:
        List of thinking content strings
    """
    if not text or not isinstance(text, str):
        return []
    
    segments = []
    thinking_patterns = [
        r'<thinking>(.*?)</thinking>',
        r'<thinking>(.*?)</thinking>',
        r'<thought>(.*?)</thought>',
        r'<reasoning>(.*?)</reasoning>',
        r'<analysis>(.*?)</analysis>',
        r'<planning>(.*?)</planning>',
    ]
    
    for pattern in thinking_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            content = match.strip()
            if content and content not in segments:
                segments.append(content)
    
    return segments


def is_partial_thinking_tag(text: str) -> bool:
    """
    Check if text contains a partial/incomplete thinking tag.
    This helps identify streaming chunks that are building up thinking content.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to contain partial thinking tags
    """
    if not text or not isinstance(text, str):
        return False
    
    # Check for opening tags without closing tags
    opening_patterns = [
        r'<thinking(?:\s|>)',
        r'<thinking(?:\s|>)',
        r'<thought(?:\s|>)',
        r'<reasoning(?:\s|>)',
        r'<analysis(?:\s|>)',
        r'<planning(?:\s|>)',
    ]
    
    closing_patterns = [
        r'</thinking>',
        r'</thinking>',
        r'</thought>',
        r'</reasoning>',
        r'</analysis>',
        r'</planning>',
    ]
    
    has_opening = any(re.search(pattern, text, re.IGNORECASE) for pattern in opening_patterns)
    has_closing = any(re.search(pattern, text, re.IGNORECASE) for pattern in closing_patterns)
    
    # If we have opening tag but no closing, it might be partial
    return has_opening and not has_closing


def clean_thinking_content(content: str) -> str:
    """
    Clean and format thinking content for display.
    
    Args:
        content: Raw thinking content
        
    Returns:
        Cleaned content suitable for display
    """
    if not content:
        return ""
    
    # Remove excessive whitespace while preserving structure
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1]:  # Preserve single empty lines between content
            cleaned_lines.append("")
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def format_thinking_for_display(content: str, max_length: int = 300) -> str:
    """
    Format thinking content for TUI display with appropriate truncation.
    
    Args:
        content: Thinking content to format
        max_length: Maximum length for display
        
    Returns:
        Formatted thinking content
    """
    if not content:
        return ""
    
    cleaned = clean_thinking_content(content)
    
    if len(cleaned) <= max_length:
        return cleaned
    
    # Truncate at word boundary if possible
    truncated = cleaned[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can truncate at a reasonable word boundary
        truncated = truncated[:last_space]
    
    return truncated + "..."