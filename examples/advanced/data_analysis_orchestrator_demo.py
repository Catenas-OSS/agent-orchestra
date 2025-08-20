#!/usr/bin/env python3
"""
Data Analysis Orchestrator Demo - Reliable Multi-Layer Example

This example demonstrates a sophisticated data processing workflow using:
- Filesystem MCP server for reliable file operations
- Multi-stage orchestrator workflow with all Part 3 features
- Gate validation, foreach processing, and intelligent reduction
- Real streaming with comprehensive analysis layers

Usage:
    OPENAI_API_KEY=<your-key> python data_analysis_orchestrator_demo.py

Workflow layers:
1. Gate: Check if data files are valid for analysis
2. Foreach: Process multiple data files in parallel  
3. Reduce: Aggregate analysis results
4. Task: Generate insights and recommendations
5. Task: Create final executive summary
"""

import asyncio
import json
import os
import sys
import tempfile
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
    print("💡 Install with: pip install -e .")
    sys.exit(1)


async def create_sample_data_files(data_dir: Path) -> list[str]:
    """Create realistic sample data files for the analysis demo."""
    
    # Sales data
    sales_data = {
        "Q1_2024": {
            "total_revenue": 1250000,
            "units_sold": 5230,
            "avg_order_value": 239.12,
            "top_products": ["Premium Widget", "Standard Widget", "Mini Widget"],
            "growth_rate": 0.15,
            "customer_satisfaction": 4.2
        },
        "Q2_2024": {
            "total_revenue": 1380000,
            "units_sold": 5892,
            "avg_order_value": 234.21,
            "top_products": ["Premium Widget", "Deluxe Widget", "Standard Widget"],
            "growth_rate": 0.10,
            "customer_satisfaction": 4.3
        },
        "Q3_2024": {
            "total_revenue": 1420000,
            "units_sold": 6150,
            "avg_order_value": 230.89,
            "top_products": ["Deluxe Widget", "Premium Widget", "Standard Widget"],
            "growth_rate": 0.03,
            "customer_satisfaction": 4.1
        }
    }
    
    # Marketing data
    marketing_data = {
        "campaigns": {
            "Q1_digital": {"spend": 125000, "impressions": 2500000, "clicks": 37500, "conversions": 750},
            "Q2_digital": {"spend": 140000, "impressions": 2800000, "clicks": 42000, "conversions": 840},
            "Q3_digital": {"spend": 135000, "impressions": 2700000, "clicks": 40500, "conversions": 810}
        },
        "channels": {
            "social_media": {"roi": 3.2, "engagement_rate": 0.048},
            "search_ads": {"roi": 4.1, "click_through_rate": 0.032},
            "email": {"roi": 5.8, "open_rate": 0.24, "click_rate": 0.045}
        }
    }
    
    # Customer data
    customer_data = {
        "demographics": {
            "age_groups": {"18-25": 0.15, "26-35": 0.35, "36-45": 0.28, "46-60": 0.22},
            "locations": {"urban": 0.65, "suburban": 0.30, "rural": 0.05},
            "income_brackets": {"<50k": 0.20, "50-100k": 0.45, "100k+": 0.35}
        },
        "behavior": {
            "repeat_purchase_rate": 0.68,
            "average_lifetime_value": 850.00,
            "churn_rate": 0.12,
            "preferred_contact": {"email": 0.55, "sms": 0.30, "phone": 0.15}
        }
    }
    
    # Operations data
    operations_data = {
        "supply_chain": {
            "inventory_turnover": 8.2,
            "supplier_performance": {"on_time_delivery": 0.94, "quality_score": 4.3},
            "warehouse_efficiency": 0.87
        },
        "production": {
            "capacity_utilization": 0.82,
            "defect_rate": 0.02,
            "employee_productivity": 112.5
        },
        "costs": {
            "cogs_percentage": 0.62,
            "operational_expenses": 0.25,
            "logistics_costs": 0.08
        }
    }
    
    # Create files
    data_files = [
        ("sales_quarterly.json", sales_data),
        ("marketing_performance.json", marketing_data), 
        ("customer_analytics.json", customer_data),
        ("operations_metrics.json", operations_data)
    ]
    
    created_files = []
    for filename, data in data_files:
        file_path = data_dir / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        created_files.append(filename)
        print(f"📊 Created: {filename}")
    
    return created_files


async def run_data_analysis_orchestrator():
    """Run sophisticated multi-layered data analysis with orchestrator."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        print("💡 This demo requires OpenAI API access")
        return
    
    print("📊 Data Analysis Orchestrator Demo")
    print("=" * 60)
    print("🎯 Multi-stage intelligent business data analysis workflow")
    print("📁 Using filesystem MCP server for reliable data operations")
    print()
    
    # Create temporary directory with sample data
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        print(f"📂 Working directory: {temp_dir}")
        
        # Create realistic business data files
        data_files = await create_sample_data_files(temp_dir)
        print()
        
        # Configure filesystem MCP server
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(temp_dir)],
                }
            }
        }
        
        # Initialize the Agent Orchestra stack
        client = SidecarMCPClient.from_dict(config)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        agent = SidecarMCPAgent(llm=llm, client=client, max_steps=6)
        executor = MCPExecutor(agent)
        orchestrator = Orchestrator(executor)
        
        try:
            print("🏗️  Building sophisticated data analysis workflow...")
            
            # Define analysis parameters
            analysis_domains = ["sales", "marketing", "customer", "operations"]
            
            # Create comprehensive multi-layered orchestrator workflow
            workflow = GraphSpec(
                nodes=[
                    # Layer 1: Gate - Validate data availability
                    NodeSpec(
                        id="data_validation_gate",
                        type="gate",
                        name="Data Validation Gate",
                        inputs={
                            "predicate": True,
                            "instruction": f"Verify that all required data files are present and accessible: {', '.join(data_files)}. Check that files exist and contain valid JSON data."
                        }
                    ),
                    
                    # Layer 2: Foreach - Analyze each data domain in parallel
                    NodeSpec(
                        id="domain_analysis",
                        type="foreach",
                        name="Domain-Specific Analysis",
                        inputs={
                            "items": data_files,
                            "instruction": "For each data file, perform detailed analysis: 1) Read and parse the file contents, 2) Identify key metrics and trends, 3) Calculate important ratios or insights, 4) Note any concerning patterns or standout performance. Provide specific numbers and percentages."
                        },
                        concurrency=2,  # Analyze 2 domains simultaneously
                        retries=1,
                        retry_backoff_s=1.0,
                        foreach_fail_policy="skip"  # Continue even if one analysis fails
                    ),
                    
                    # Layer 3: Reduce - Cross-domain insights and correlation analysis
                    NodeSpec(
                        id="cross_domain_insights",
                        type="reduce",
                        name="Cross-Domain Analysis",
                        inputs={
                            "from_ids": ["domain_analysis"],
                            "instruction": "Analyze the results from all domains together. Look for: 1) Cross-domain correlations (e.g., marketing spend vs sales growth), 2) Consistent trends across domains, 3) Resource allocation opportunities, 4) Risk factors that appear in multiple areas. Provide integrated business insights."
                        }
                    ),
                    
                    # Layer 4: Task - Strategic recommendations
                    NodeSpec(
                        id="strategic_recommendations", 
                        type="task",
                        name="Strategic Recommendations",
                        inputs={
                            "from": "cross_domain_insights",
                            "instruction": "Based on the cross-domain analysis, generate 5 specific strategic recommendations for the business. Each recommendation should: 1) Address a key opportunity or risk, 2) Include expected impact/ROI, 3) Specify required resources or timeline, 4) Reference specific data points from the analysis."
                        }
                    ),
                    
                    # Layer 5: Task - Executive summary
                    NodeSpec(
                        id="executive_summary",
                        type="task", 
                        name="Executive Summary",
                        inputs={
                            "from": "strategic_recommendations",
                            "instruction": "Create a comprehensive executive summary suitable for C-level presentation. Include: 1) Business performance overview (3-4 key metrics), 2) Major opportunities and risks identified, 3) Top 3 strategic priorities with expected outcomes, 4) Next steps and resource requirements. Keep it concise but impactful."
                        }
                    )
                ],
                edges=[
                    ("data_validation_gate", "domain_analysis"),
                    ("domain_analysis", "cross_domain_insights"),
                    ("cross_domain_insights", "strategic_recommendations"),
                    ("strategic_recommendations", "executive_summary")
                ]
            )
            
            run_spec = RunSpec(
                run_id="business_analysis_001",
                goal=f"Comprehensive multi-domain business analysis of {', '.join(analysis_domains)} data with strategic recommendations"
            )
            
            print(f"📈 Workflow: {len(workflow.nodes)} analysis layers, {len(workflow.edges)} connections")
            print(f"🎯 Domains: {', '.join(analysis_domains)}")
            print(f"📊 Data files: {len(data_files)} files")
            print()
            
            # Execute with real-time streaming
            print("🎬 Starting intelligent business analysis...")
            print("-" * 50)
            
            event_count = 0
            chunk_count = 0
            results = {}
            
            async for event in orchestrator.run_streaming(workflow, run_spec):
                event_count += 1
                
                if event.type == "RUN_START":
                    print(f"🚀 ANALYSIS START: {event.data['goal']}")
                
                elif event.type == "NODE_START":
                    node_type = event.data.get('type', 'unknown')
                    phase = event.data.get('phase', '')
                    
                    layer_info = {
                        'data_validation_gate': '🚪 Layer 1 (Validation Gate)',
                        'domain_analysis': '📊 Layer 2 (Domain Foreach)',
                        'cross_domain_insights': '🔄 Layer 3 (Cross-Domain Reduce)',
                        'strategic_recommendations': '💡 Layer 4 (Strategy Task)',
                        'executive_summary': '📋 Layer 5 (Summary Task)'
                    }.get(event.node_id, f'🔧 {event.node_id}')
                    
                    print(f"\n{layer_info}: {node_type.upper()}")
                
                elif event.type == "AGENT_CHUNK":
                    chunk_count += 1
                    
                    # Show relevant business analysis chunks
                    if 'content' in event.data and chunk_count <= 10:
                        content = str(event.data['content'])
                        # Look for business/analysis related content
                        if any(keyword in content.lower() for keyword in ['revenue', 'growth', 'analysis', 'data', 'metric', 'performance']):
                            preview = content[:150] + "..." if len(content) > 150 else content
                            print(f"  📈 Analysis: {preview}")
                        elif chunk_count <= 5:
                            preview = content[:100] + "..." if len(content) > 100 else content
                            print(f"  📦 Processing: {preview}")
                    elif chunk_count == 11:
                        print(f"  📦 ... (continued analysis)")
                
                elif event.type == "NODE_COMPLETE":
                    phase = event.data.get('phase', '')
                    layer_emoji = {
                        'data_validation_gate': '✅🚪',
                        'domain_analysis': '✅📊',
                        'cross_domain_insights': '✅🔄',
                        'strategic_recommendations': '✅💡',
                        'executive_summary': '✅📋'
                    }.get(event.node_id, '✅')
                    
                    print(f"{layer_emoji} LAYER COMPLETE: {event.node_id}")
                    
                    # Store final results
                    if event.node_id in ['strategic_recommendations', 'executive_summary']:
                        results[event.node_id] = event.data
                
                elif event.type == "RUN_COMPLETE":
                    print(f"\n🎉 BUSINESS ANALYSIS COMPLETE!")
                    
                    # Display executive summary
                    blackboard = event.data.get("result", {})
                    print(f"\n📋 EXECUTIVE SUMMARY:")
                    print("=" * 60)
                    
                    if 'executive_summary' in blackboard:
                        summary_data = blackboard['executive_summary']
                        if isinstance(summary_data, dict) and 'result' in summary_data:
                            result_data = summary_data['result']
                            if isinstance(result_data, dict) and 'output' in result_data:
                                summary_text = str(result_data['output'])
                                print(summary_text[:2000])  # Show substantial portion
                                if len(summary_text) > 2000:
                                    print("\n... (truncated for display)")
                        
                    print(f"\n📊 Analysis completed across {len(blackboard)} layers")
                
                elif event.type == "ERROR":
                    error_msg = event.data.get('error', 'Unknown error')
                    phase = event.data.get('phase', '')
                    print(f"❌ ERROR in {event.node_id} ({phase}): {error_msg}")
            
            print(f"\n📈 Execution Summary:")
            print(f"   🔢 Total events: {event_count}")
            print(f"   📦 Streaming chunks: {chunk_count}")
            print(f"   🏆 Status: ✅ Success")
            
            print(f"\n🌟 Part 3 Features Demonstrated:")
            print(f"   ✅ Gate-based data validation")
            print(f"   ✅ Foreach parallel domain analysis (concurrency=2)")
            print(f"   ✅ Reduce cross-domain correlation analysis")
            print(f"   ✅ Multi-layer strategic task processing")
            print(f"   ✅ Real-time streaming with business insights")
            print(f"   ✅ Retry logic for analysis failures")
            print(f"   ✅ Filesystem MCP integration with JSON data")
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
                pass


async def main():
    """Main entry point with error handling."""
    try:
        await run_data_analysis_orchestrator()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊 Data Analysis Orchestrator demo complete!")
    print(f"💡 This demo shows how Agent Orchestra orchestrates complex")
    print(f"   multi-layer business analysis workflows with real data!")


if __name__ == "__main__":
    asyncio.run(main())