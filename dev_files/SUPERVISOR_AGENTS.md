# Smart Agent Orchestration with Supervisor Agents 🧠

Agent Orchestra now supports **supervisor agents** that can dynamically choose and orchestrate specialized agents based on task requirements. This creates truly intelligent workflows that adapt to different scenarios.

## 🌟 Key Features

- **Dynamic Agent Selection**: Supervisor agents analyze tasks and choose the most appropriate specialized agents
- **Multi-Agent Coordination**: One supervisor can orchestrate multiple specialized agents in sequence
- **Hierarchical Workflows**: Multiple supervisor agents can work together for complex projects
- **Streaming Support**: Real-time visibility into agent decision-making and execution
- **Flexible Architecture**: Easy to define new agent types and capabilities

## 📋 How It Works

### 1. Define Specialized Agents

```python
available_agents = {
    "ui_designer": {
        "description": "Creates beautiful, modern UI designs and layouts",
        "capabilities": ["responsive design", "color schemes", "typography"],
        "server": "claude"
    },
    "frontend_developer": {
        "description": "Implements HTML, CSS, and JavaScript for web interfaces", 
        "capabilities": ["HTML5", "CSS3", "JavaScript", "responsive frameworks"],
        "server": "claude"
    },
    "content_writer": {
        "description": "Creates engaging, SEO-optimized content",
        "capabilities": ["copywriting", "SEO", "brand voice"],
        "server": "claude"
    }
}
```

### 2. Create Supervisor Nodes

```python
NodeSpec(
    id="website_supervisor",
    type="supervisor",  # New node type!
    name="Website Building Supervisor",
    inputs={
        "requirements": "requirements_node",
        "project_type": "portfolio website"
    },
    available_agents=available_agents,
    max_agent_calls=4,  # Can call up to 4 agents
    server_name="claude"
)
```

### 3. Supervisor Decision Format

The supervisor agent uses a structured format to make decisions:

```
CALL_AGENT: ui_designer
TASK: Create modern portfolio design
INPUT: Target audience: potential clients, Style: clean and professional
---

CALL_AGENT: content_writer  
TASK: Write engaging portfolio copy
INPUT: Brand voice: creative but professional, Sections: about, portfolio, contact
---

FINAL_RESULT: Created complete portfolio website with modern design and engaging content
```

## 🎯 Use Cases

### 1. Website Building
- Supervisor analyzes requirements
- Calls UI designer, developer, content writer as needed
- Coordinates their outputs into final website

### 2. Content Creation
- Supervisor analyzes content goals
- Calls researchers, writers, editors, SEO specialists
- Produces optimized, high-quality content

### 3. Software Development
- Supervisor analyzes project requirements
- Calls architects, developers, testers, DevOps engineers
- Delivers complete software solutions

### 4. Business Analysis
- Supervisor analyzes business problems
- Calls data analysts, market researchers, strategists
- Provides comprehensive business recommendations

## 🔧 Technical Implementation

### New Node Type: "supervisor"

- Added to `NodeSpec.type` enum in `types.py`
- Implemented `_run_supervisor_node()` in `core.py`  
- Supports both streaming and non-streaming execution
- Integrates with existing agent pool and broker systems

### Agent Selection Logic

1. Supervisor receives task and available agents
2. Analyzes requirements using LLM reasoning
3. Outputs structured agent calls in specified format
4. Orchestrator parses and executes each agent call
5. Combines results into final output

### Streaming Integration

- Supervisor reasoning is streamed in real-time
- Individual agent executions are also streamed
- Complete visibility into multi-agent coordination
- Enhanced TUI shows supervisor decisions and agent outputs

## 🚀 Example Usage

```python
# Run the test
python test_supervisor_workflow.py

# Or use in your own workflows
from website_builder_workflow import create_website_workflow, simple_run
from agent_orchestra.orchestrator.core import Orchestrator

orchestrator = Orchestrator(executor=your_executor)
workflow = create_website_workflow()

async for event in orchestrator.run_streaming(workflow, simple_run):
    if event.type == "AGENT_CHUNK":
        print(f"Agent output: {event.data}")
```

## 🎨 Benefits for Website Building

The supervisor agent approach is perfect for creating lovable websites because:

1. **Intelligent Specialization**: Each agent focuses on what they do best
2. **Adaptive Workflows**: Different project types get different agent combinations  
3. **Quality Coordination**: Supervisor ensures agents work together effectively
4. **Scalable Architecture**: Easy to add new specialist agents
5. **User-Friendly**: Complex multi-agent coordination feels simple and natural

## 🔮 Future Enhancements

- **Agent Learning**: Supervisors learn from past decisions
- **Performance Optimization**: Automatic load balancing across agents
- **Visual Orchestration**: GUI for designing supervisor workflows
- **Agent Marketplace**: Share and discover specialized agents
- **Multi-Modal Agents**: Support for image, audio, and video processing agents

---

*The supervisor agent system transforms Agent Orchestra from a simple workflow engine into an intelligent multi-agent coordination platform. Perfect for building the next generation of AI-powered applications!* 🌟