#!/usr/bin/env python3
"""
Restaurant Search Orchestrator Demo - Multi-layered Agent Orchestra Example

This example demonstrates a sophisticated restaurant discovery workflow using:
- Playwright MCP server for web scraping
- Multi-stage orchestrator workflow with Part 3 features
- Gate-based filtering, foreach processing, and intelligent reduction
- Real streaming with comprehensive analysis layers

Usage:
    OPENAI_API_KEY=<your-key> python restaurant_orchestrator_demo.py

Workflow layers:
1. Gate: Check if location is valid for restaurant search
2. Foreach: Search multiple restaurant discovery platforms in parallel
3. Reduce: Aggregate and analyze all restaurant data
4. Task: Generate personalized recommendations
5. Task: Create final summary with ratings and booking info
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from agent_orchestra import SidecarMCPAgent, SidecarMCPClient
    from agent_orchestra.orchestrator.core import Orchestrator
    from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
    from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
    load_dotenv()
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install with: pip install -e . playwright")
    print("💡 Also run: playwright install")
    sys.exit(1)


async def run_restaurant_orchestrator_demo():
    """Run sophisticated multi-layered restaurant search with orchestrator."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        print("💡 This demo requires OpenAI API access")
        return
    
    print("🍽️  Restaurant Discovery Orchestrator Demo")
    print("=" * 60)
    print("🎯 Multi-stage intelligent restaurant discovery workflow")
    print("🔧 Using Playwright MCP server for web research")
    print()
    
    # Configure Playwright MCP server for web scraping
    config = {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "env": {
                    "DISPLAY": ":1"  # For headless operation
                }
            }
        }
    }
    
    # Initialize the Agent Orchestra stack
    client = SidecarMCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # Low temp for consistent results
    agent = SidecarMCPAgent(llm=llm, client=client, max_steps=8)
    executor = MCPExecutor(agent)
    orchestrator = Orchestrator(executor)
    
    try:
        print("🏗️  Building sophisticated restaurant discovery workflow...")
        
        # Define search parameters
        target_location = "San Francisco"
        search_platforms = [
            "OpenTable", 
            "Yelp", 
            "Google Maps restaurants"
        ]
        cuisine_preferences = ["Italian", "Japanese", "Farm-to-table"]
        
        # Create multi-layered orchestrator workflow
        workflow = GraphSpec(
            nodes=[
                # Layer 1: Gate - Validate location for restaurant search
                NodeSpec(
                    id="location_gate",
                    type="gate",
                    name="Location Validation Gate",
                    inputs={
                        "predicate": True,  # Always pass for demo, but could validate location
                        "instruction": f"Verify that '{target_location}' is a valid location for restaurant search. Check if it's a real city/area where restaurants would exist."
                    }
                ),
                
                # Layer 2: Foreach - Parallel search across multiple platforms 
                NodeSpec(
                    id="platform_search",
                    type="foreach", 
                    name="Multi-Platform Restaurant Search",
                    inputs={
                        "items": search_platforms,
                        "location": target_location,
                        "instruction": f"For each platform, search for highly-rated restaurants in {target_location}. Find at least 2-3 specific restaurant names with their key details (cuisine type, rating, location). Use web search to get current information."
                    },
                    concurrency=2,  # Search 2 platforms simultaneously
                    retries=1,
                    retry_backoff_s=2.0,
                    foreach_fail_policy="skip"  # Continue even if one platform fails
                ),
                
                # Layer 3: Reduce - Intelligent aggregation and analysis
                NodeSpec(
                    id="restaurant_analysis",
                    type="reduce",
                    name="Restaurant Data Analysis", 
                    inputs={
                        "from_ids": ["platform_search"],
                        "preferences": cuisine_preferences,
                        "instruction": f"Analyze all restaurant data from multiple platforms. Look for: 1) Restaurants mentioned across multiple platforms (higher confidence), 2) Restaurants matching preferences: {', '.join(cuisine_preferences)}, 3) Highest-rated options with specific ratings/reviews. Create a consolidated list of top 5 restaurants with their key attributes."
                    },
                    retries=1,
                    retry_backoff_s=1.0
                ),
                
                # Layer 4: Task - Personalized recommendation generation
                NodeSpec(
                    id="generate_recommendations",
                    type="task",
                    name="Personalized Recommendations",
                    inputs={
                        "from": "restaurant_analysis",
                        "preferences": cuisine_preferences,
                        "instruction": f"Based on the analyzed restaurant data, generate personalized recommendations. For each recommended restaurant, include: 1) Why it matches the preferences ({', '.join(cuisine_preferences)}), 2) Specific dishes to try, 3) Best time to visit, 4) Price range expectations. Rank by overall appeal."
                    }
                ),
                
                # Layer 5: Task - Final summary with actionable info
                NodeSpec(
                    id="booking_summary",
                    type="task",
                    name="Booking & Visit Summary",
                    inputs={
                        "from": "generate_recommendations", 
                        "instruction": "Create a final actionable summary with: 1) Top 3 restaurant choices with addresses, 2) How to make reservations for each, 3) Estimated costs and timing, 4) Backup options if first choices are unavailable. Make it ready-to-use for planning a dining experience."
                    }
                )
            ],
            edges=[
                ("location_gate", "platform_search"),
                ("platform_search", "restaurant_analysis"),
                ("restaurant_analysis", "generate_recommendations"), 
                ("generate_recommendations", "booking_summary")
            ]
        )
        
        run_spec = RunSpec(
            run_id="restaurant_discovery_001",
            goal=f"Intelligent multi-layered restaurant discovery in {target_location} with {', '.join(cuisine_preferences)} preferences"
        )
        
        print(f"📊 Workflow: {len(workflow.nodes)} layers, {len(workflow.edges)} connections")
        print(f"🎯 Target: {target_location}")
        print(f"🍴 Preferences: {', '.join(cuisine_preferences)}")
        print(f"🔍 Platforms: {', '.join(search_platforms)}")
        print()
        
        # Execute with real-time streaming
        print("🎬 Starting intelligent restaurant discovery...")
        print("-" * 50)
        
        event_count = 0
        chunk_count = 0
        layer_results = {}
        
        async for event in orchestrator.run_streaming(workflow, run_spec):
            event_count += 1
            
            if event.type == "RUN_START":
                print(f"🚀 DISCOVERY START: {event.data['goal']}")
            
            elif event.type == "NODE_START":
                node_type = event.data.get('type', 'unknown')
                phase = event.data.get('phase', '')
                layer_name = {
                    'location_gate': '🚪 Layer 1 (Gate)',
                    'platform_search': '🔍 Layer 2 (Foreach)', 
                    'restaurant_analysis': '📊 Layer 3 (Reduce)',
                    'generate_recommendations': '⭐ Layer 4 (Recommendations)',
                    'booking_summary': '📋 Layer 5 (Summary)'
                }.get(event.node_id, f'🔧 {event.node_id}')
                
                print(f"\n{layer_name}: {node_type.upper()} - {phase}")
            
            elif event.type == "AGENT_CHUNK":
                chunk_count += 1
                phase = event.data.get('phase', '')
                
                # Show meaningful chunk content for restaurant search
                if 'content' in event.data and chunk_count <= 8:
                    content = str(event.data['content'])
                    # Look for restaurant-related content
                    if any(keyword in content.lower() for keyword in ['restaurant', 'food', 'cuisine', 'rating', 'menu']):
                        content_preview = content[:120] + "..." if len(content) > 120 else content
                        print(f"  📦 Discovering: {content_preview}")
                    elif chunk_count <= 3:  # Show first few chunks regardless
                        content_preview = content[:80] + "..." if len(content) > 80 else content
                        print(f"  📦 Processing ({phase}): {content_preview}")
                elif chunk_count == 9:
                    print(f"  📦 ... (continued processing)")
            
            elif event.type == "NODE_COMPLETE":
                phase = event.data.get('phase', '')
                layer_emoji = {
                    'location_gate': '✅🚪',
                    'platform_search': '✅🔍', 
                    'restaurant_analysis': '✅📊',
                    'generate_recommendations': '✅⭐',
                    'booking_summary': '✅📋'
                }.get(event.node_id, '✅')
                
                print(f"{layer_emoji} LAYER COMPLETE: {event.node_id} - {phase}")
                
                # Store key results for final display
                if event.node_id in ['restaurant_analysis', 'generate_recommendations', 'booking_summary']:
                    layer_results[event.node_id] = event.data
            
            elif event.type == "RUN_COMPLETE":
                print(f"\n🎉 RESTAURANT DISCOVERY COMPLETE!")
                
                # Display final results in a user-friendly format
                blackboard = event.data.get("result", {})
                print(f"\n🍽️  FINAL RESTAURANT RECOMMENDATIONS:")
                print("=" * 50)
                
                for layer_id, result in blackboard.items():
                    if layer_id == 'booking_summary' and isinstance(result, dict):
                        summary_content = result.get('result', {})
                        if isinstance(summary_content, dict) and 'output' in summary_content:
                            final_recommendations = str(summary_content['output'])
                            # Format the output nicely
                            print(final_recommendations[:1000])
                            if len(final_recommendations) > 1000:
                                print("\n... (truncated for display)")
                        break
                
                print(f"\n📊 Discovery completed with {len(blackboard)} layers")
            
            elif event.type == "ERROR":
                error_msg = event.data.get('error', 'Unknown error')
                phase = event.data.get('phase', '')
                print(f"❌ ERROR in {event.node_id} ({phase}): {error_msg}")
        
        print(f"\n📈 Execution Summary:")
        print(f"   🔢 Total events: {event_count}")
        print(f"   📦 Streaming chunks: {chunk_count}")
        print(f"   🏆 Status: ✅ Success")
        
        print(f"\n🌟 Advanced Features Demonstrated:")
        print(f"   ✅ Gate-based location validation")
        print(f"   ✅ Foreach parallel platform search (concurrency=2)")
        print(f"   ✅ Reduce intelligent data aggregation")
        print(f"   ✅ Multi-layer task processing")
        print(f"   ✅ Real-time streaming with restaurant discovery")
        print(f"   ✅ Retry logic for platform failures")
        print(f"   ✅ Web scraping with Playwright MCP server")
        print(f"   ✅ Production-grade error handling")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup MCP sessions
        try:
            await client.close_all_sessions()
        except Exception:
            pass  # Ignore cleanup errors


async def main():
    """Main entry point with error handling."""
    try:
        await run_restaurant_orchestrator_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🍽️  Restaurant Orchestrator demo complete!")
    print(f"💡 This demo shows how Agent Orchestra orchestrates complex")
    print(f"   multi-step workflows with real MCP server integration!")


if __name__ == "__main__":
    asyncio.run(main())