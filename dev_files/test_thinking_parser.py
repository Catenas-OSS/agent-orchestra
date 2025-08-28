#!/usr/bin/env python3
"""
Test script to verify thinking content parsing works correctly.
"""

import sys
sys.path.insert(0, 'src')

from agent_orchestra.orchestrator.thinking_parser import (
    extract_thinking_content,
    has_thinking_content,
    extract_all_thinking_segments,
    format_thinking_for_display
)

def test_thinking_parser():
    """Test thinking content extraction."""
    
    # Test cases with different thinking tag formats
    test_cases = [
        # Standard thinking tags
        {
            "input": "<thinking>This is my thought process about the problem</thinking>",
            "expected": "This is my thought process about the problem"
        },
        # Claude-style thinking tags
        {
            "input": "<thinking>Let me analyze this step by step:\n1. First I need to understand\n2. Then I should consider</thinking>",
            "expected": "Let me analyze this step by step:\n1. First I need to understand\n2. Then I should consider"
        },
        # Mixed content
        {
            "input": "Some regular text <thinking>My internal thoughts</thinking> and more text",
            "expected": "My internal thoughts"
        },
        # No thinking content
        {
            "input": "Just regular text without any thinking tags",
            "expected": None
        },
        # Multiple thinking blocks
        {
            "input": "<thinking>First thought</thinking> some text <thinking>Second thought</thinking>",
            "expected": "First thought"  # Should return the first one
        },
        # Different tag types
        {
            "input": "<reasoning>This is my reasoning process</reasoning>",
            "expected": "This is my reasoning process"
        },
        # Empty thinking tags
        {
            "input": "<thinking></thinking>",
            "expected": None
        }
    ]
    
    print("Testing thinking content extraction...")
    
    for i, case in enumerate(test_cases):
        result = extract_thinking_content(case["input"])
        expected = case["expected"]
        
        if result == expected:
            print(f"✅ Test {i+1}: PASSED")
        else:
            print(f"❌ Test {i+1}: FAILED")
            print(f"   Input: {case['input'][:50]}...")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
    
    print("\nTesting has_thinking_content...")
    
    # Test has_thinking_content function
    assert has_thinking_content("<thinking>test</thinking>") == True
    assert has_thinking_content("no thinking here") == False
    assert has_thinking_content("<reasoning>test</reasoning>") == True
    print("✅ has_thinking_content tests passed")
    
    print("\nTesting extract_all_thinking_segments...")
    
    # Test multiple segments
    multi_input = "<thinking>First</thinking> text <reasoning>Second</reasoning> more <thinking>Third</thinking>"
    segments = extract_all_thinking_segments(multi_input)
    expected_segments = ["First", "Second", "Third"]
    
    if segments == expected_segments:
        print("✅ Multiple segments test passed")
    else:
        print(f"❌ Multiple segments test failed: {segments}")
    
    print("\nTesting format_thinking_for_display...")
    
    # Test formatting
    long_text = "This is a very long thinking process that goes on and on " * 10
    formatted = format_thinking_for_display(long_text, 100)
    
    if len(formatted) <= 103:  # 100 + "..."
        print("✅ Formatting test passed")
    else:
        print(f"❌ Formatting test failed: length {len(formatted)}")
    
    print("\n🎉 All tests completed!")


def test_tui_integration():
    """Test how the thinking parser integrates with TUI components."""
    
    print("\nTesting TUI integration...")
    
    # Simulate different chunk types that might come from agents
    test_chunks = [
        {
            "text": "<thinking>I need to analyze this request carefully</thinking>Regular response text"
        },
        {
            "message": "Just a regular message without thinking"
        },
        {
            "content": "<reasoning>Let me think about the approach:\n1. First step\n2. Second step</reasoning>"
        },
        {
            "text": "thinking about this problem",  # keyword-based fallback
        },
        {
            "data": "<thinking>Nested thinking in data field</thinking>"
        }
    ]
    
    for i, chunk in enumerate(test_chunks):
        chunk_text = chunk.get("text", "") or chunk.get("message", "") or chunk.get("content", "") or chunk.get("data", "")
        
        thinking_content = extract_thinking_content(chunk_text)
        if thinking_content:
            formatted = format_thinking_for_display(thinking_content, 200)
            print(f"✅ Chunk {i+1}: Found thinking - {formatted[:50]}...")
        else:
            # Fallback to keyword detection
            thinking_indicators = ["thinking", "reasoning", "analysis", "plan"]
            if any(indicator in chunk_text.lower() for indicator in thinking_indicators):
                print(f"✅ Chunk {i+1}: Keyword-based detection - {chunk_text[:50]}...")
            else:
                print(f"⚪ Chunk {i+1}: No thinking detected")
    
    print("🎉 TUI integration test completed!")


if __name__ == "__main__":
    test_thinking_parser()
    test_tui_integration()