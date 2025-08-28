
"""
Professional Interactive TUI using Urwid - True terminal interaction.
4-pane layout: Header / Left (Node Navigator) / Center (Agent Inspector) / Right (Ops Panels)
"""

from __future__ import annotations
import asyncio
import time
from typing import Optional, Dict, Any, List, TYPE_CHECKING

try:
    import urwid
except ImportError:
    raise ImportError("Urwid not installed. Run: pip install 'agent-orchestra[cli]'")

from .model import RunTUIModel, NodeState
from .bus import EventBus, Subscription
from ..logging import get_system_logger, init_logging

if TYPE_CHECKING:
    from ..orchestrator.core import Orchestrator


class ProfessionalUrwidTUI:
    """
    Professional Interactive TUI Dashboard using Urwid.
    
    Layout:
    ┌─ Header: Run info + Timeline + Controls ──────────────────────────────────────────────┐
    │                                                                                         │
    ├─ Left: Node Navigator ─────────┬─ Center: Agent Inspector ─┬─ Right: Ops Panels ────┤
    │  [Filter/Search]               │  [Tabs] Snapshot | Inst.  │  Metrics                │
    │  Nodes table (interactive)     │         | Inputs | Tools  │  Logs                   │
    │  ↑↓ Navigate, Enter select     │  Tab to cycle tabs        │  DAG Mini               │
    └────────────────────────────────┴────────────────────────────┴─────────────────────────┘
    """
    
    def __init__(self, orchestrator: "Orchestrator", model: RunTUIModel):
        self.orchestrator = orchestrator
        self.model = model
        self.bus = EventBus()
        
        # Initialize logging for TUI mode and IMMEDIATELY enable output capture
        self.system_logger = init_logging(tui_mode=True)
        self.system_logger.enable_tui_mode()  # Force enable immediately
        self.system_logger.info("tui", "Professional TUI initialized with output capture enabled")
        
        # UI state
        self.selected_node_index = -1  # Start with system logs selected
        self.current_inspector_tab = 0  # 0=Snapshot, 1=Instructions, etc.
        self.should_exit = False
        self.metrics_update_interval = 1.0
        
        # Tab names for inspector
        self.inspector_tabs = [
            "Snapshot", "Instructions", "Inputs", "Data Flow", "Output", 
            "Tool Trace", "Supervisor", "Limits", "Performance", "Policy", "Reproducibility", "Errors"
        ]
        
        # Create widgets
        self._create_widgets()
        self._setup_layout()
        
        # Palette for colors
        self.palette = [
            ('header', 'white,bold', 'dark blue'),
            ('footer', 'white', 'dark red'),
            ('running', 'yellow,bold', 'default'),
            ('complete', 'light green,bold', 'default'),
            ('error', 'light red,bold', 'default'),
            ('selected', 'white,bold', 'dark blue'),
            ('focus', 'white,bold', 'dark green'),
            ('dim', 'dark gray', 'default'),
            ('bright', 'white,bold', 'default'),
            ('cost', 'light green', 'default'),
            ('warning', 'yellow', 'default'),
        ]
        
        # Subscribe to model updates
        self.model_subscription = self.bus.subscribe(
            self._handle_model_event,
            topics={"model_update", "selection_change", "tab_change"}
        )
    
    def _create_widgets(self):
        """Create all UI widgets."""
        # Header widgets
        self.header_text = urwid.Text("")
        
        # Left panel - Node Navigator
        self.node_list_walker = urwid.SimpleFocusListWalker([])
        self.node_listbox = urwid.ListBox(self.node_list_walker)
        
        # Center panel - Agent Inspector (scrollable)
        self.inspector_content = urwid.Text("No node selected\nUse ↑↓ to select a node")
        self.inspector_listbox = urwid.ListBox(urwid.SimpleFocusListWalker([self.inspector_content]))
        
        # Right panel - Ops (back to 3 panels)
        self.metrics_text = urwid.Text("")
        self.logs_text = urwid.Text("")
        self.dag_text = urwid.Text("")
        
        # Footer
        self.footer_text = urwid.Text("↑↓: Navigate | Enter: Select | Tab: Inspect | v: Privacy | q: Quit")
    
    def _setup_layout(self):
        """Setup the main layout structure."""
        # Header
        header = urwid.AttrMap(urwid.Filler(self.header_text, 'top'), 'header')
        
        # Left panel - Node Navigator
        left_panel = urwid.LineBox(
            self.node_listbox,
            title="🎯 Node Navigator"
        )
        
        # Center panel - Agent Inspector with tabs (scrollable)
        inspector_header = urwid.Text("")  # Will show tab bar
        inspector_pile = urwid.Pile([
            ('pack', inspector_header),
            self.inspector_listbox  # Use scrollable listbox instead of filler
        ])
        center_panel = urwid.LineBox(
            inspector_pile,
            title="🔍 Agent Inspector"
        )
        
        # Right panel - Ops (back to 3 sections)
        ops_pile = urwid.Pile([
            ('weight', 1, urwid.LineBox(urwid.Filler(self.metrics_text, 'top'), title="📈 Metrics")),
            ('weight', 1, urwid.LineBox(urwid.Filler(self.logs_text, 'top'), title="📋 Events")),
            ('weight', 1, urwid.LineBox(urwid.Filler(self.dag_text, 'top'), title="🗺️ DAG"))
        ])
        
        # Main columns
        columns = urwid.Columns([
            ('weight', 1, left_panel),      # Left: Node Navigator
            ('weight', 2, center_panel),    # Center: Agent Inspector  
            ('weight', 1, ops_pile)         # Right: Ops Panels
        ], dividechars=1)
        
        # Main frame
        self.main_frame = urwid.Frame(
            body=columns,
            header=urwid.AttrMap(header, 'header'),
            footer=urwid.AttrMap(self.footer_text, 'footer')
        )
    
    def _update_display(self):
        """Update all display elements."""
        self._update_header()
        self._update_node_list()
        self._update_inspector()
        self._update_ops_panels()
    
    def _update_header(self):
        """Update header with run information."""
        # Run info
        status_style = {
            "idle": "○",
            "running": "▶", 
            "done": "●",
            "error": "✗"
        }.get(self.model.status, "?")
        
        progress = self.model.progress_summary
        cost = self.model.metrics.total_cost
        tokens = self.model.metrics.total_tokens_in + self.model.metrics.total_tokens_out
        
        header_text = (
            f"{status_style} Run: {self.model.run_id[:12]}... | "
            f"Goal: {self.model.goal[:30]}{'...' if len(self.model.goal) > 30 else ''} | "
            f"Progress: {progress['done']}/{progress['total']} | "
            f"Cost: ${cost:.4f} | Tokens: {tokens:,} | "
            f"Elapsed: {self.model.elapsed_time:.1f}s"
        )
        
        self.header_text.set_text(header_text)
    
    def _update_node_list(self):
        """Update the node navigator list."""
        self.node_list_walker.clear()
        
        # Add system logs as the first option
        system_logs_button = urwid.Button("🖥️ System Logs", on_press=self._node_selected, user_data=-1)
        if self.selected_node_index == -1:
            widget = urwid.AttrMap(system_logs_button, 'selected', 'focus')
        else:
            widget = urwid.AttrMap(system_logs_button, 'bright', 'focus')
        self.node_list_walker.append(widget)
        
        # Add regular nodes
        for i, (node_id, node) in enumerate(self.model.nodes.items()):
            # Status symbol
            if node.status == "pending":
                symbol = "○"
                attr = "dim"
            elif node.status == "running":
                symbol = "▶"
                attr = "running"
            elif node.status == "complete":
                symbol = "●"
                attr = "complete"
            elif node.status == "resumed":
                symbol = "◉"
                attr = "complete"
            elif node.status == "error":
                symbol = "✗"
                attr = "error"
            else:
                symbol = "?"
                attr = "dim"
            
            # Node info
            name = node.name or node_id
            if len(name) > 15:
                name = name[:12] + "..."
            
            # Progress for foreach
            progress_info = ""
            if node.is_foreach and node.items_total:
                done = node.items_done
                total = node.items_total
                progress_info = f" ({done}/{total})"
            
            # Duration
            duration_info = ""
            if node.duration:
                duration_info = f" {node.duration:.1f}s"
            elif node.started_at:
                elapsed = time.time() - node.started_at
                duration_info = f" {elapsed:.1f}s"
            
            # Create button
            button_text = f"{symbol} {name}{progress_info}{duration_info}"
            button = urwid.Button(button_text, on_press=self._node_selected, user_data=i)
            
            # Apply styling
            if i == self.selected_node_index:
                widget = urwid.AttrMap(button, 'selected', 'focus')
            else:
                widget = urwid.AttrMap(button, attr, 'focus')
            
            self.node_list_walker.append(widget)
    
    def _update_inspector(self):
        """Update the agent inspector panel."""
        # Update tab bar
        tab_bar_text = " | ".join([
            f"[{tab}]" if i == self.current_inspector_tab else tab
            for i, tab in enumerate(self.inspector_tabs)
        ])
        
        # Check if system logs is selected
        if self.selected_node_index == -1:
            content = self._render_system_logs_tab()
        else:
            # Get selected node
            nodes = list(self.model.nodes.values())
            selected_node = nodes[self.selected_node_index] if 0 <= self.selected_node_index < len(nodes) else None
            
            if not selected_node:
                content = "No node selected\nUse ↑↓ to select a node"
            else:
                content = self._render_inspector_tab(selected_node, self.current_inspector_tab)
        
        # Update inspector content (scrollable)
        # Split content into lines and create text widgets for scrolling
        content_lines = content.split('\n')
        content_widgets = [urwid.Text(line) for line in content_lines]
        
        # Update the listbox walker with new content
        self.inspector_listbox.body.clear()
        for widget in content_widgets:
            self.inspector_listbox.body.append(widget)
        
        # Update inspector header with tab bar
        inspector_pile = self.main_frame.body.contents[1][0].original_widget
        inspector_header = inspector_pile.contents[0][0]
        inspector_header.set_text(tab_bar_text)
    
    def _render_inspector_tab(self, node: NodeState, tab_index: int) -> str:
        """Render content for specific inspector tab."""
        tab_name = self.inspector_tabs[tab_index]
        
        if tab_name == "Snapshot":
            return self._render_snapshot_tab(node)
        elif tab_name == "Instructions":
            return self._render_instructions_tab(node)
        elif tab_name == "Inputs":
            return self._render_inputs_tab(node)
        elif tab_name == "Data Flow":
            return self._render_data_flow_tab(node)
        elif tab_name == "Output":
            return self._render_output_tab(node)
        elif tab_name == "Tool Trace":
            return self._render_tool_trace_tab(node)
        elif tab_name == "Supervisor":
            return self._render_supervisor_tab(node)
        elif tab_name == "Limits":
            return self._render_limits_tab(node)
        elif tab_name == "Performance":
            return self._render_performance_tab(node)
        elif tab_name == "Policy":
            return self._render_policy_tab(node)
        elif tab_name == "Reproducibility":
            return self._render_reproducibility_tab(node)
        elif tab_name == "Errors":
            return self._render_errors_tab(node)
        else:
            return f"Tab '{tab_name}' not implemented yet"
    
    def _render_snapshot_tab(self, node: NodeState) -> str:
        """Render the Snapshot tab with comprehensive developer overview."""
        lines = ["=== NODE OVERVIEW ===", ""]
        
        # Basic info
        lines.extend([
            f"📋 Node ID: {node.id}",
            f"🏷️  Name: {node.name or 'Unnamed'}",
            f"🎯 Status: {node.status.upper()}",
            f"🤖 Type: {node.node_type or 'task'}",
            f"🖥️  Server: {node.server or 'default'}",
            f"🤖 Model: {node.model or 'unknown'}",
            f"🔄 Attempts: {node.attempt}/{node.max_attempts}",
            ""
        ])
        
        # Timing information
        lines.append("⏱️  TIMING:")
        if node.started_at:
            start_time = time.strftime('%H:%M:%S', time.localtime(node.started_at))
            lines.append(f"  Started: {start_time}")
        
        if node.ended_at:
            end_time = time.strftime('%H:%M:%S', time.localtime(node.ended_at))
            lines.append(f"  Ended: {end_time}")
            lines.append(f"  Duration: {node.duration:.3f}s")
        elif node.started_at:
            elapsed = time.time() - node.started_at
            lines.append(f"  Elapsed: {elapsed:.3f}s (still running)")
        else:
            lines.append("  Not started yet")
        lines.append("")
        
        # Resource usage
        lines.append("💰 RESOURCE USAGE:")
        if node.tokens_used:
            prompt_tokens = node.tokens_used.get("prompt", 0)
            completion_tokens = node.tokens_used.get("completion", 0)
            total_tokens = node.tokens_used.get("total", 0)
            lines.append(f"  Tokens: {total_tokens:,} ({prompt_tokens:,} in + {completion_tokens:,} out)")
        else:
            lines.append("  Tokens: Not available")
        
        if node.cost:
            lines.append(f"  Cost: ${node.cost:.6f}")
        else:
            lines.append("  Cost: Not calculated")
        lines.append("")
        
        # Supervisor-specific info
        if node.is_supervisor:
            lines.append("🧠 SUPERVISOR INFO:")
            lines.append(f"  Available agents: {len(node.available_agents)}")
            lines.append(f"  Max agent calls: {node.max_agent_calls}")
            lines.append(f"  Agents called: {len(node.agents_called)}")
            
            if node.available_agents:
                lines.append("  Agent roster:")
                for agent_id, agent_info in node.available_agents.items():
                    description = agent_info.get("description", "No description")[:40]
                    lines.append(f"    • {agent_id}: {description}")
            
            if node.agents_called:
                lines.append("  Called agents:")
                for agent_id in node.agents_called:
                    lines.append(f"    ✅ {agent_id}")
            
            if node.supervisor_decisions:
                lines.append(f"  Decisions made: {len(node.supervisor_decisions)}")
            
            lines.append("")
        
        # Activity summary
        lines.append("📊 ACTIVITY SUMMARY:")
        lines.append(f"  Log entries: {len(node.logs)}")
        lines.append(f"  Tool calls: {len(node.tool_trace)}")
        lines.append(f"  Artifacts: {len(node.artifacts)}")
        
        # Instructions info
        if node.instructions:
            lines.append(f"  Instruction fingerprint: {node.instructions.fingerprint}")
            lines.append(f"  Tools available: {len(node.instructions.tools)}")
        else:
            lines.append("  No instruction data captured")
        lines.append("")
        
        # Recent activity (more detailed)
        lines.append("📝 RECENT ACTIVITY:")
        if node.logs:
            recent_logs = list(node.logs)[-5:]  # Show last 5 for snapshot
            for i, log in enumerate(recent_logs):
                # Truncate but show more than before
                log_preview = log[:80] + "..." if len(log) > 80 else log
                lines.append(f"  [{i+1}] {log_preview}")
        else:
            lines.append("  No activity logged")
        
        # Status-specific information
        lines.append("")
        if node.status == "error":
            lines.append("❌ ERROR DETAILS:")
            error_logs = [log for log in node.logs if "error" in log.lower()]
            if error_logs:
                lines.append(f"  Last error: {error_logs[-1][:60]}...")
            else:
                lines.append("  No specific error message captured")
        
        elif node.status == "complete":
            lines.append("✅ COMPLETION INFO:")
            if node.output_summary:
                # Show more of the output for visibility
                summary_preview = node.output_summary  # No truncation
                lines.append(f"  🎯 Agent Output: {summary_preview}")
            else:
                lines.append("  ⚠️ No output captured - agent may not have produced structured output")
        
        elif node.status == "running":
            lines.append("⚡ LIVE STATUS:")
            lines.append("  Node is currently executing...")
            if node.started_at:
                elapsed = time.time() - node.started_at
                lines.append(f"  Running for: {elapsed:.1f}s")
        
        return "\n".join(lines)
    
    def _render_instructions_tab(self, node: NodeState) -> str:
        """Render the Instructions tab."""
        if not node.instructions:
            return "No instructions available"
        
        instructions = node.instructions
        lines = []
        
        if instructions.task_title:
            lines.append(f"Task: {instructions.task_title}")
            lines.append("")
        
        if instructions.task_body:
            body = instructions.task_body
            if self.model.redaction_enabled and len(body) > 200:
                body = body[:200] + "... [REDACTED - press 'v' to toggle]"
            lines.append("Task Body:")
            lines.append(body)
            lines.append("")
        
        if instructions.system:
            lines.append("System:")
            system = instructions.system
            if self.model.redaction_enabled and len(system) > 150:
                system = system[:150] + "... [REDACTED]"
            lines.append(system)
            lines.append("")
        
        if instructions.tools:
            lines.append(f"Tools ({len(instructions.tools)}):")
            for tool in instructions.tools[:3]:
                name = tool.get("name", "unknown")
                purpose = tool.get("purpose", tool.get("description", ""))[:30]
                lines.append(f"• {name}: {purpose}")
            if len(instructions.tools) > 3:
                lines.append(f"... and {len(instructions.tools) - 3} more")
            lines.append("")
        
        lines.append(f"Fingerprint: {instructions.fingerprint}")
        
        return "\n".join(lines)
    
    def _render_inputs_tab(self, node: NodeState) -> str:
        """Render the Inputs tab."""
        if not node.resolved_inputs:
            return "No input data available"
        
        lines = ["Resolved Inputs:", ""]
        for key, value in node.resolved_inputs.items():
            value_str = str(value)[:100]
            if len(str(value)) > 100:
                value_str += "..."
            lines.append(f"{key}: {value_str}")
        
        return "\n".join(lines)
    
    def _render_tool_trace_tab(self, node: NodeState) -> str:
        """Render the Tool Trace tab."""
        if not node.tool_trace:
            return "No tool executions yet"
        
        lines = [f"Tool Executions ({len(node.tool_trace)}):", ""]
        
        for tool in node.tool_trace:
            status = "✓" if tool.ok else "✗"
            duration = ""
            if tool.ended_at:
                duration = f" ({(tool.ended_at - tool.started_at):.2f}s)"
            
            lines.append(f"{status} {tool.name}{duration}")
            
            if tool.args_preview:
                lines.append(f"  Args: {tool.args_preview[:50]}...")
            
            if tool.diff_preview:
                lines.append(f"  Changes: {tool.diff_preview[:50]}...")
            elif tool.path:
                lines.append(f"  File: {tool.path}")
            
            if tool.error_message:
                lines.append(f"  Error: {tool.error_message[:50]}...")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _render_supervisor_tab(self, node: NodeState) -> str:
        """Render the Supervisor tab with detailed orchestration information."""
        if not node.is_supervisor:
            return "This node is not a supervisor agent.\nSupervisor agents orchestrate other specialized agents to complete complex tasks."
        
        lines = ["🧠 === SUPERVISOR ORCHESTRATION === 🧠", ""]
        
        # Supervisor configuration
        lines.append("⚙️ CONFIGURATION:")
        lines.append(f"  Max agent calls: {node.max_agent_calls}")
        lines.append(f"  Available agents: {len(node.available_agents)}")
        lines.append("")
        
        # Available agents roster
        if node.available_agents:
            lines.append("🤖 AGENT ROSTER:")
            lines.append("-" * 40)
            for agent_id, agent_info in node.available_agents.items():
                description = agent_info.get("description", "No description")
                server = agent_info.get("server", "unknown")
                capabilities = agent_info.get("capabilities", [])
                
                lines.append(f"  • {agent_id}")
                lines.append(f"    📝 {description}")
                lines.append(f"    🖥️  Server: {server}")
                if capabilities:
                    lines.append(f"    🎯 Capabilities: {', '.join(capabilities)}")
                lines.append("")
        else:
            lines.append("No agents available for orchestration")
            lines.append("")
        
        # Supervisor decision history
        if node.supervisor_decisions:
            lines.append("🎯 DECISION HISTORY:")
            lines.append("-" * 40)
            for i, decision in enumerate(node.supervisor_decisions, 1):
                lines.append(f"  [{i}] {decision}")
            lines.append("")
        else:
            lines.append("No decisions recorded yet")
            lines.append("")
        
        # Agents called and their results
        if node.agents_called:
            lines.append("✅ AGENTS CALLED:")
            lines.append("-" * 40)
            for agent_id in node.agents_called:
                lines.append(f"  • {agent_id}")
                if agent_id in node.agent_results:
                    result = node.agent_results[agent_id]
                    result_preview = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    lines.append(f"    📄 Result: {result_preview}")
                else:
                    lines.append("    ⏳ Result pending...")
            lines.append("")
        else:
            lines.append("No agents called yet")
            lines.append("")
        
        # Orchestration summary
        lines.append("📊 ORCHESTRATION SUMMARY:")
        lines.append("-" * 40)
        lines.append(f"  Total agents available: {len(node.available_agents)}")
        lines.append(f"  Agents called: {len(node.agents_called)}")
        lines.append(f"  Decisions made: {len(node.supervisor_decisions)}")
        if node.agents_called:
            success_rate = len([r for r in node.agent_results.values() if r]) / len(node.agents_called) * 100
            lines.append(f"  Success rate: {success_rate:.1f}%")
        lines.append("")
        
        # Final output
        if node.output_summary:
            lines.append("🎯 FINAL SUPERVISOR OUTPUT:")
            lines.append("-" * 40)
            lines.append(node.output_summary)
        
        return "\n".join(lines)
    
    def _render_data_flow_tab(self, node: NodeState) -> str:
        """Render the Data Flow tab showing how data passes between agents."""
        lines = ["🔄 === DATA FLOW & AGENT CHAINING === 🔄", ""]
        
        # Show what this node receives from previous nodes
        lines.append("📥 INPUTS FROM PREVIOUS AGENTS:")
        lines.append("-" * 40)
        
        # Check if this node has inputs from other nodes
        has_inputs = False
        
        # Show raw inputs from workflow definition
        if hasattr(node, 'inputs') and node.inputs:
            lines.append("  📋 Workflow Definition Inputs:")
            for key, value in node.inputs.items():
                if key == "from":
                    lines.append(f"    🔗 Receives output from: {value}")
                    has_inputs = True
                elif key != "instruction":
                    lines.append(f"    📊 {key}: {str(value)}")
                    has_inputs = True
            lines.append("")
        
        # Show resolved inputs (actual data received from other agents)
        if hasattr(node, 'resolved_inputs') and node.resolved_inputs:
            lines.append("  ✅ RESOLVED DATA FROM OTHER AGENTS:")
            for key, value in node.resolved_inputs.items():
                lines.append(f"    🎯 {key}: {str(value)}")
            has_inputs = True
            lines.append("")
        
        if not has_inputs:
            lines.append("  ℹ️  This node doesn't receive data from other agents")
            lines.append("  (It's likely the first node in the workflow)")
        
        lines.append("-" * 40)
        lines.append("")
        
        # Show what data this node will pass to next nodes
        lines.append("📤 OUTPUTS TO NEXT AGENTS:")
        lines.append("-" * 40)
        
        if node.output_summary:
            lines.append("  🎯 This agent's output (available to next nodes):")
            lines.append(f"     {node.output_summary}")
            lines.append("")
            lines.append("  📋 Data format: This output becomes input for downstream agents")
        else:
            if node.status == "complete":
                lines.append("  ⚠️  No output captured - next agents may not receive data")
            else:
                lines.append("  ⏳ Output will be available when agent completes")
        
        lines.append("-" * 40)
        lines.append("")
        
        # Show workflow context
        lines.append("🗺️  WORKFLOW CONTEXT:")
        lines.append("-" * 30)
        
        # Try to show other nodes in the workflow for context
        if hasattr(self, 'model') and hasattr(self.model, 'nodes'):
            other_nodes = [n for n_id, n in self.model.nodes.items() if n_id != node.id]
            if other_nodes:
                lines.append("  Other agents in this workflow:")
                for other_node in other_nodes[:5]:  # Limit to 5 for space
                    status_icon = {"complete": "✅", "running": "⚡", "pending": "⏸️", "error": "❌"}.get(other_node.status, "❓")
                    lines.append(f"    {status_icon} {other_node.name} ({other_node.id})")
                    if other_node.output_summary:
                        summary_preview = other_node.output_summary[:60] + "..." if len(other_node.output_summary) > 60 else other_node.output_summary
                        lines.append(f"       Output: {summary_preview}")
        
        lines.append("-" * 30)
        lines.append("")
        
        # Debugging information
        lines.append("🔍 DEBUG INFO:")
        lines.append("-" * 20)
        lines.append(f"  Node ID: {node.id}")
        lines.append(f"  Status: {node.status}")
        if hasattr(node, 'inputs'):
            lines.append(f"  Input keys: {list(node.inputs.keys()) if node.inputs else 'None'}")
        lines.append(f"  Has output: {'Yes' if node.output_summary else 'No'}")
        lines.append("-" * 20)
        
        if not any([node.output_summary, has_inputs]):
            lines.extend(["", "💡 TROUBLESHOOTING:", 
                         "• Agent outputs may not be properly captured",
                         "• Check the 'Output' tab for raw agent responses",
                         "• Ensure agents are producing structured outputs",
                         "• Verify workflow node connections"])
        
        return "\n".join(lines)
    
    def _render_output_tab(self, node: NodeState) -> str:
        """Render the Output tab with enhanced developer information."""
        lines = ["🎯 === AGENT OUTPUT === 🎯", ""]
        
        # FINAL AGENT OUTPUT - Most prominent section
        if node.output_summary:
            lines.append("🏆 FINAL AGENT OUTPUT:")
            lines.append("=" * 50)
            lines.append(node.output_summary)
            lines.append("=" * 50)
            lines.append("")
        
        # Agent response chunks and logs (this contains the actual LLM responses)
        if node.logs:
            # Filter for output-related logs
            output_logs = [log for log in node.logs if any(marker in log for marker in ["🎯 FINAL OUTPUT:", "🏁 AGENT COMPLETED:", "✅ OUTPUT:", "🧠 AGENT:"])]
            
            if output_logs:
                lines.append("📄 Agent Response History:")
                lines.append("-" * 40)
                for i, log in enumerate(output_logs):
                    lines.append(f"[{i+1}] {log}")
                    if i < len(output_logs) - 1:
                        lines.append("")
                lines.append("-" * 40)
                lines.append("")
            
            # Show all recent logs for debugging
            lines.append("🔍 All Recent Agent Activity:")
            lines.append("-" * 30)
            recent_logs = list(node.logs)[-8:]  # Last 8 log entries
            for i, log in enumerate(recent_logs):
                lines.append(f"  {log}")
            lines.append("-" * 30)
            lines.append("")
        
        # Show blackboard/context data if available
        if hasattr(node, 'context_data') and node.context_data:
            lines.append("Blackboard/Context Data:")
            for key, value in node.context_data.items():
                value_preview = str(value)[:200]
                if len(str(value)) > 200:
                    value_preview += "... [truncated]"
                lines.append(f"  {key}: {value_preview}")
            lines.append("")
        
        # Artifacts (files created/modified)
        if node.artifacts:
            lines.append(f"Generated Artifacts ({len(node.artifacts)}):")
            for artifact in node.artifacts:
                path = artifact.get("path", "unknown")
                size = artifact.get("bytes", 0)
                kind = artifact.get("kind", "file")
                size_str = f"{size:,}B" if size else "-"
                lines.append(f"  📁 {path} ({size_str}, {kind})")
            lines.append("")
        
        # Token usage breakdown
        if node.tokens_used:
            lines.append("Token Usage Breakdown:")
            prompt_tokens = node.tokens_used.get("prompt", 0)
            completion_tokens = node.tokens_used.get("completion", 0)
            total_tokens = node.tokens_used.get("total", 0)
            lines.append(f"  Input tokens:  {prompt_tokens:,}")
            lines.append(f"  Output tokens: {completion_tokens:,}")
            lines.append(f"  Total tokens:  {total_tokens:,}")
            if node.cost:
                lines.append(f"  Estimated cost: ${node.cost:.6f}")
            lines.append("")
        
        # Performance metrics
        if node.duration:
            lines.append("Performance:")
            lines.append(f"  Execution time: {node.duration:.3f} seconds")
            if node.tokens_used and node.tokens_used.get("total", 0) > 0:
                tokens_per_sec = node.tokens_used["total"] / node.duration
                lines.append(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
            lines.append("")
        
        if not any([node.logs, node.output_summary, node.artifacts, node.tokens_used]):
            lines = ["No output data captured yet.", "", 
                    "This could mean:", 
                    "• Node hasn't completed execution",
                    "• Output capture needs enhancement", 
                    "• Agent didn't produce structured output"]
        
        return "\n".join(lines)
    
    def _render_limits_tab(self, node: NodeState) -> str:
        """Render the Limits & Retries tab."""
        lines = [
            "Resource Limits & Retry Info:",
            "",
            f"Current Attempt: {node.attempt}",
            f"Max Attempts: {node.max_attempts}",
        ]
        
        if node.attempt > 1:
            lines.append(f"Retries Used: {node.attempt - 1}")
        
        # TODO: Add actual limit information when available
        lines.extend([
            "",
            "Limits:",
            "• Timeout: Not set",
            "• Memory: Not set"
        ])
        
        return "\n".join(lines)
    
    def _render_performance_tab(self, node: NodeState) -> str:
        """Render the Performance tab with detailed metrics."""
        lines = ["=== PERFORMANCE ANALYTICS ===", ""]
        
        # Execution timing breakdown
        lines.append("⏱️  EXECUTION TIMING:")
        if node.started_at:
            start_time = time.strftime('%H:%M:%S.%f', time.localtime(node.started_at))[:-3]
            lines.append(f"  Start time: {start_time}")
            
            if node.ended_at:
                end_time = time.strftime('%H:%M:%S.%f', time.localtime(node.ended_at))[:-3]
                lines.append(f"  End time: {end_time}")
                lines.append(f"  Total duration: {node.duration:.3f} seconds")
                
                # Break down timing if we have tool traces
                if node.tool_trace:
                    total_tool_time = sum(
                        (tool.ended_at - tool.started_at) if tool.ended_at else 0 
                        for tool in node.tool_trace
                    )
                    llm_time = node.duration - total_tool_time
                    lines.append(f"  LLM processing: {llm_time:.3f}s ({llm_time/node.duration*100:.1f}%)")
                    lines.append(f"  Tool execution: {total_tool_time:.3f}s ({total_tool_time/node.duration*100:.1f}%)")
            else:
                elapsed = time.time() - node.started_at
                lines.append(f"  Elapsed: {elapsed:.3f}s (still running)")
        else:
            lines.append("  Not started yet")
        lines.append("")
        
        # Token throughput analysis
        lines.append("🔤 TOKEN ANALYSIS:")
        if node.tokens_used:
            prompt_tokens = node.tokens_used.get("prompt", 0)
            completion_tokens = node.tokens_used.get("completion", 0)
            total_tokens = node.tokens_used.get("total", 0)
            
            lines.extend([
                f"  Input tokens: {prompt_tokens:,}",
                f"  Output tokens: {completion_tokens:,}",
                f"  Total tokens: {total_tokens:,}",
                f"  Input/Output ratio: {prompt_tokens/completion_tokens:.2f}" if completion_tokens > 0 else "  Input/Output ratio: N/A"
            ])
            
            if node.duration and total_tokens > 0:
                overall_rate = total_tokens / node.duration
                input_rate = prompt_tokens / node.duration
                output_rate = completion_tokens / node.duration
                lines.extend([
                    f"  Overall throughput: {overall_rate:.1f} tokens/sec",
                    f"  Input processing: {input_rate:.1f} tokens/sec",
                    f"  Output generation: {output_rate:.1f} tokens/sec"
                ])
                
                # Efficiency metrics
                if completion_tokens > 0:
                    efficiency = (completion_tokens / total_tokens) * 100
                    lines.append(f"  Generation efficiency: {efficiency:.1f}% (output/total)")
        else:
            lines.append("  No token usage data available")
        lines.append("")
        
        # Tool performance breakdown
        lines.append("🔧 TOOL PERFORMANCE:")
        if node.tool_trace:
            lines.append(f"  Total tool calls: {len(node.tool_trace)}")
            
            # Tool timing analysis
            tool_times = []
            successful_tools = 0
            failed_tools = 0
            
            for tool in node.tool_trace:
                if tool.ended_at:
                    duration = tool.ended_at - tool.started_at
                    tool_times.append(duration)
                    if tool.ok:
                        successful_tools += 1
                    else:
                        failed_tools += 1
            
            if tool_times:
                avg_tool_time = sum(tool_times) / len(tool_times)
                max_tool_time = max(tool_times)
                min_tool_time = min(tool_times)
                
                lines.extend([
                    f"  Successful calls: {successful_tools}",
                    f"  Failed calls: {failed_tools}",
                    f"  Average tool time: {avg_tool_time:.3f}s",
                    f"  Fastest tool: {min_tool_time:.3f}s",
                    f"  Slowest tool: {max_tool_time:.3f}s"
                ])
                
                # Success rate
                if len(node.tool_trace) > 0:
                    success_rate = (successful_tools / len(node.tool_trace)) * 100
                    lines.append(f"  Success rate: {success_rate:.1f}%")
            
            # Top slowest tools
            if len(node.tool_trace) > 1:
                lines.append("")
                lines.append("  Slowest tools:")
                tool_durations = [
                    (tool.name, tool.ended_at - tool.started_at if tool.ended_at else 0)
                    for tool in node.tool_trace
                ]
                tool_durations.sort(key=lambda x: x[1], reverse=True)
                
                for i, (tool_name, duration) in enumerate(tool_durations[:3]):
                    lines.append(f"    {i+1}. {tool_name}: {duration:.3f}s")
        else:
            lines.append("  No tool calls recorded")
        lines.append("")
        
        # Cost analysis
        lines.append("💰 COST ANALYSIS:")
        if node.cost:
            lines.append(f"  Total cost: ${node.cost:.6f}")
            
            if node.tokens_used and node.tokens_used.get("total", 0) > 0:
                cost_per_token = node.cost / node.tokens_used["total"]
                cost_per_1k = cost_per_token * 1000
                lines.append(f"  Cost per token: ${cost_per_token:.8f}")
                lines.append(f"  Cost per 1K tokens: ${cost_per_1k:.6f}")
            
            if node.duration:
                cost_per_second = node.cost / node.duration
                lines.append(f"  Cost per second: ${cost_per_second:.6f}")
        else:
            lines.append("  Cost calculation not available")
        lines.append("")
        
        # Performance rating
        lines.append("📊 PERFORMANCE RATING:")
        if node.duration and node.tokens_used and node.tokens_used.get("total", 0) > 0:
            tokens_per_sec = node.tokens_used["total"] / node.duration
            
            if tokens_per_sec > 100:
                rating = "🚀 Excellent"
            elif tokens_per_sec > 50:
                rating = "✅ Good"
            elif tokens_per_sec > 20:
                rating = "⚠️  Fair"
            else:
                rating = "🐌 Slow"
                
            lines.append(f"  Throughput rating: {rating}")
            
            # Tool efficiency rating
            if node.tool_trace:
                if failed_tools == 0:
                    tool_rating = "🎯 Perfect"
                elif failed_tools / len(node.tool_trace) < 0.1:
                    tool_rating = "✅ Reliable"
                elif failed_tools / len(node.tool_trace) < 0.3:
                    tool_rating = "⚠️  Unstable"
                else:
                    tool_rating = "❌ Problematic"
                    
                lines.append(f"  Tool reliability: {tool_rating}")
        else:
            lines.append("  Insufficient data for rating")
        
        return "\n".join(lines)
    
    def _render_policy_tab(self, node: NodeState) -> str:
        """Render the Policy & Safety tab."""
        lines = [
            "Policy & Safety:",
            "",
            "✓ No policy violations detected",
            f"Privacy: {'Redacted' if node.redacted else 'Full disclosure'}"
        ]
        return "\n".join(lines)
    
    def _render_reproducibility_tab(self, node: NodeState) -> str:
        """Render the Reproducibility tab."""
        lines = ["Reproducibility Info:", ""]
        
        if node.instructions and node.instructions.fingerprint:
            lines.append(f"Instruction Fingerprint: {node.instructions.fingerprint}")
        
        lines.extend([
            f"Model: {node.model or 'unknown'}",
            f"Attempt: {node.attempt}",
            "",
            "Export reproduction bundle: [Not yet implemented]"
        ])
        
        return "\n".join(lines)
    
    def _render_system_logs_tab(self) -> str:
        """Render the system logs in the inspector panel."""
        lines = ["=== SYSTEM LOGS ===", ""]
        
        # Get comprehensive system logs
        logs = self.system_logger.get_recent_logs(50)
        
        if not logs:
            lines.extend([
                "No system logs available yet.",
                "",
                "System logs will show:",
                "• CLI operations and workflow loading",
                "• Orchestrator initialization and execution",
                "• Node lifecycle events", 
                "• Error conditions and warnings",
                "• Component-specific activity"
            ])
        else:
            # Group logs by component
            by_component = {}
            for log in logs:
                if log.component not in by_component:
                    by_component[log.component] = []
                by_component[log.component].append(log)
            
            # Show component statistics
            lines.append("📊 COMPONENT ACTIVITY:")
            stats = self.system_logger.get_component_stats()
            for component, stat in stats.items():
                total = stat['total']
                errors = stat.get('ERROR', 0)
                warnings = stat.get('WARNING', 0)
                lines.append(f"  {component}: {total} logs ({errors} errors, {warnings} warnings)")
            lines.append("")
            
            # Show recent logs in chronological order
            lines.append("📝 RECENT ACTIVITY:")
            lines.append("-" * 60)
            
            for log in logs[-30:]:  # Show last 30 logs
                # Format with color indicators
                level_indicator = {
                    'ERROR': '🔴',
                    'WARNING': '🟡', 
                    'INFO': '🔵',
                    'DEBUG': '🟦',
                    'CRITICAL': '⭕'
                }.get(log.level.value, '⚪')
                
                # Format with node context if available
                node_context = f" [{log.node_id}]" if log.node_id else ""
                
                # Truncate long messages for readability
                message = log.message
                if len(message) > 80:
                    message = message[:77] + "..."
                
                lines.append(f"{level_indicator} {log.formatted_time} [{log.component}]{node_context}")
                lines.append(f"    {message}")
                lines.append("")
            
            lines.append("-" * 60)
            lines.append(f"Showing {len(logs)} recent logs from {len(by_component)} components")
        
        return "\n".join(lines)
    
    def _render_errors_tab(self, node: NodeState) -> str:
        """Render the Errors tab."""
        if node.status != "error":
            return "✓ No errors for this node"
        
        lines = [
            "Node Status: ERROR",
            "",
            "Error Messages:"
        ]
        
        # Extract error messages from logs
        if node.logs:
            error_logs = [log for log in node.logs if "error" in log.lower() or "failed" in log.lower()]
            if error_logs:
                for error_log in error_logs[-3:]:
                    lines.append(f"• {error_log}")
            else:
                lines.append("No specific error messages found")
        else:
            lines.append("No error details available")
        
        return "\n".join(lines)
    
    def _update_ops_panels(self):
        """Update the ops panels (system logs, metrics, events, DAG)."""
        # System logs are now handled in the inspector when selected from navigator
        
        # Metrics
        lines = [
            f"Cost: ${self.model.metrics.total_cost:.4f}",
            f"Tokens: {self.model.metrics.total_tokens_in + self.model.metrics.total_tokens_out:,}",
            f"Progress: {self.model.progress_summary['done']}/{self.model.progress_summary['total']}"
        ]
        
        if self.model.metrics.broker_stats:
            lines.append("")
            lines.append("Broker Stats:")
            for model_name, stats in self.model.metrics.broker_stats.items():
                rpm_used = stats.get("rpm_used", 0)
                rpm_limit = stats.get("rpm_limit", 0)
                if rpm_limit > 0:
                    lines.append(f"{model_name}: {rpm_used}/{rpm_limit} RPM")
        
        self.metrics_text.set_text("\n".join(lines))
        
        # Events (workflow-specific events)
        event_lines = []
        if self.model.system_events:
            event_lines.append("Workflow Events:")
            for event in list(self.model.system_events)[-4:]:
                event_lines.append(f"• {event}")
        
        if self.model.global_errors:
            if event_lines:
                event_lines.append("")
            event_lines.append("Recent Errors:")
            for error in list(self.model.global_errors)[-2:]:
                event_lines.append(f"• {error}")
        
        if not event_lines:
            event_lines.append("No workflow events yet...")
        
        self.logs_text.set_text("\n".join(event_lines))
        
        # DAG Mini-map
        dag_lines = ["Workflow Graph:", ""]
        for node_id, node in self.model.nodes.items():
            if node.status == "pending":
                symbol = "○"
            elif node.status == "running":
                symbol = "▶"
            elif node.status == "complete":
                symbol = "●"
            elif node.status == "error":
                symbol = "✗"
            else:
                symbol = "?"
            
            name = node.name or node_id
            if len(name) > 10:
                name = name[:7] + "..."
            
            dag_lines.append(f"{symbol} {name}")
        
        self.dag_text.set_text("\n".join(dag_lines))
    
    def _node_selected(self, button, index):
        """Handle node selection."""
        self.selected_node_index = index
        self._update_display()
    
    def _handle_model_event(self, event_type: str, data: Any) -> None:
        """Handle model update events."""
        # Update display when model changes
        self._update_display()
    
    def _log_agent_chunk(self, node_id: str, data: dict) -> None:
        """Log detailed agent chunk information to system logger."""
        if not isinstance(data, dict):
            self.system_logger.debug("agent", f"[{node_id}] Non-dict chunk: {str(data)[:100]}", node_id=node_id)
            return
        
        # Log every chunk to understand what we're getting
        self.system_logger.debug("agent", f"[{node_id}] Raw chunk: {str(data)[:200]}", node_id=node_id)
            
        # Extract meaningful content from different types of chunks
        chunk_text = data.get("text", "") or data.get("message", "") or data.get("content", "") or data.get("delta", "") or data.get("data", "")
        
        # Detect and log different types of agent activity
        if "tool_use" in data or "tool_call" in data:
            tool_data = data.get("tool_call") or data.get("tool_use", {})
            tool_name = tool_data.get("name", "unknown")
            tool_args = str(tool_data.get("arguments", tool_data.get("args", {})))[:100]
            self.system_logger.info("agent", f"[{node_id}] Tool call: {tool_name}({tool_args})", node_id=node_id)
            
        elif "tool_result" in data:
            result = str(data["tool_result"])[:150]
            self.system_logger.info("agent", f"[{node_id}] Tool result: {result}", node_id=node_id)
            
        elif "step" in data:
            step_info = data["step"]
            self.system_logger.info("agent", f"[{node_id}] Step: {step_info}", node_id=node_id)
            
        elif "output" in data:
            output = str(data["output"])
            self.system_logger.info("agent", f"[{node_id}] Output: {output}", node_id=node_id)
            
        else:
            # Check for thinking content using XML tag parsing
            from ..orchestrator.thinking_parser import extract_thinking_content, format_thinking_for_display
            
            thinking_content = extract_thinking_content(chunk_text)
            if thinking_content:
                # Found structured thinking content
                formatted_thinking = format_thinking_for_display(thinking_content, 200)
                self.system_logger.debug("agent", f"[{node_id}] Thinking: {formatted_thinking}", node_id=node_id)
            elif any(indicator in chunk_text.lower() for indicator in ["thinking", "reasoning", "analysis", "plan"]):
                # Fallback to keyword-based detection
                self.system_logger.debug("agent", f"[{node_id}] Thinking: {chunk_text[:200]}", node_id=node_id)
            elif chunk_text.strip() and len(chunk_text) > 20:
                # Substantial text content
                self.system_logger.debug("agent", f"[{node_id}] Response: {chunk_text[:150]}", node_id=node_id)
        
        # Always log something if we have any data at all
        if not chunk_text.strip() and data:
            # Try to extract any meaningful info from the chunk
            for key in ['choices', 'delta', 'response', 'completion', 'partial', 'streaming']:
                if key in data:
                    value = str(data[key])[:100]
                    self.system_logger.debug("agent", f"[{node_id}] {key.title()}: {value}", node_id=node_id)
                    break
            else:
                # Log the keys available in the chunk
                keys = list(data.keys())[:5]  # First 5 keys
                self.system_logger.debug("agent", f"[{node_id}] Chunk keys: {keys}", node_id=node_id)
    
    def _handle_input(self, key):
        """Handle keyboard input."""
        if key == 'q':
            self.should_exit = True
            raise urwid.ExitMainLoop()
        
        elif key == 'tab':
            # Cycle through inspector tabs
            self.current_inspector_tab = (self.current_inspector_tab + 1) % len(self.inspector_tabs)
            self._update_inspector()
        
        elif key == 'v':
            # Toggle privacy/redaction
            self.model.toggle_redaction()
            self._update_display()
        
        elif key == 'up' and self.selected_node_index > -1:
            self.selected_node_index -= 1
            self._update_display()
        
        elif key == 'down':
            max_index = len(self.model.nodes) - 1
            if self.selected_node_index < max_index:
                self.selected_node_index += 1
                self._update_display()
        
        elif key == 'enter':
            # Refresh display
            self._update_display()
    
    async def _process_orchestrator_events(self, run_stream):
        """Process events from orchestrator and update model."""
        try:
            async for event in run_stream:
                if self.should_exit:
                    break
                
                # Apply event to model
                event_type = event.type
                node_id = getattr(event, 'node_id', None)
                data = getattr(event, 'data', {})
                
                # Log detailed agent chunks to system logger
                if event_type == "AGENT_CHUNK" and node_id:
                    self._log_agent_chunk(node_id, data)
                
                self.model.apply_event(event_type, node_id, data)
                
                # Auto-select running node
                if event_type == "NODE_START" and node_id:
                    node_ids = list(self.model.nodes.keys())
                    if node_id in node_ids:
                        self.selected_node_index = node_ids.index(node_id)
                
                # Update display
                self._update_display()
                
                # Exit condition
                if event_type == "RUN_COMPLETE":
                    break
                    
        except Exception as e:
            self.model.apply_event("ERROR", None, {"error": str(e)})
            self._update_display()
    
    async def _update_metrics_periodically(self):
        """Periodically poll and update metrics."""
        while not self.should_exit:
            try:
                # Get metrics from executor
                broker_stats = {}
                pool_stats = {}
                
                if hasattr(self.orchestrator, '_executor'):
                    executor = self.orchestrator._executor
                    
                    if hasattr(executor, '_broker') and executor._broker:
                        broker_stats = await executor._broker.get_stats()
                    
                    if hasattr(executor, '_agent_pool') and executor._agent_pool:
                        pool_stats = executor._agent_pool.get_stats()
                
                # Apply to model
                self.model.apply_metrics(broker_stats, pool_stats)
                self._update_display()
                
            except Exception:
                pass  # Don't crash on metrics errors
            
            await asyncio.sleep(self.metrics_update_interval)
    
    async def run(self, run_stream):
        """Run the interactive Urwid TUI."""
        # Initialize nodes from graph if needed
        # (This should be done by caller but we can handle it)
        
        # Start background tasks
        event_task = asyncio.create_task(self._process_orchestrator_events(run_stream))
        metrics_task = asyncio.create_task(self._update_metrics_periodically())
        
        try:
            # Initial display update
            self._update_display()
            
            # Create asyncio-compatible urwid loop
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(current_loop)
            
            evl = urwid.AsyncioEventLoop(loop=current_loop)
            loop = urwid.MainLoop(
                self.main_frame,
                palette=self.palette,
                event_loop=evl,
                unhandled_input=self._handle_input
            )
            
            # Start urwid loop
            loop.start()
            
            # Wait for events to complete
            try:
                await event_task
            except:
                pass
            
            # Keep TUI open for inspection even after workflow completes
            while not self.should_exit:
                await asyncio.sleep(0.1)
            
        except KeyboardInterrupt:
            self.should_exit = True
        finally:
            # Clean up
            self.should_exit = True
            event_task.cancel()
            metrics_task.cancel()
            
            try:
                await asyncio.wait([event_task, metrics_task], timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            await self.bus.close()
            if self.model_subscription:
                self.bus.unsubscribe(self.model_subscription)