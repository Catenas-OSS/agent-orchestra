# Agentic Orchestra: Production-Ready Multi-Agentic Orchestration Platform

**Agentic Orchestra** is a production-grade, open-source framework for building sophisticated multi-agent workflows with enterprise-level features. Built on top of the Model Context Protocol (MCP), it provides advanced orchestration, rate limiting, agent pooling, and comprehensive observability for real-world AI applications.

## 🚀 **Production-Ready Features**

Agentic Orchestra has been battle-tested and includes all the polish improvements needed for real-world deployment:

- **🏊 Profile-Based Agent Pooling** - Intelligent agent reuse with race-safe creation and no duplicate initialization
- **⚡ Global Rate Limiting** - Per-model RPM/RPD limits with 429-aware retries and jittered exponential backoff  
- **🔀 Multi-Server Routing** - Single MCP client with dynamic server-name routing per workflow node
- **🛡️ Security & Safety** - Path validation, directory traversal prevention, and secure parameter handling
- **🎯 Advanced Orchestration** - DAG workflows with concurrent `foreach`, intelligent `reduce`, and conditional `gate` nodes
- **📊 Comprehensive Telemetry** - Event-driven architecture with structured logging and performance metrics
- **🧹 Clean Async Management** - Proper resource lifecycle with graceful startup/shutdown

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│   MCPExecutor    │───▶│   AgentPool     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         │              │   CallBroker     │    │ SidecarMCPAgent │
         │              │  (Rate Limiting) │    │ (with Telemetry)│
         │              └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       │                       ▼
┌─────────────────┐              │              ┌─────────────────┐
│   GraphSpec     │              │              │ SidecarMCPClient│
│   (Workflow)    │              │              │ (Multi-Server)  │
└─────────────────┘              │              └─────────────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   Broker Stats   │    │   MCP Servers   │
                        │   (Monitoring)   │    │  (fs, web, etc) │
                        └──────────────────┘    └─────────────────┘
```

## 🎯 **Key Components**

### **Orchestrator**
The central workflow engine that executes DAG-based workflows with support for:
- **Task Nodes** - Single agent operations
- **Foreach Nodes** - Concurrent batch processing with configurable concurrency
- **Reduce Nodes** - Intelligent aggregation of multiple results
- **Gate Nodes** - Conditional workflow control

### **AgentPool (Profile-Based)**
Production-grade agent management with:
- **Profile Keys** - `(server_name, model_key, policy_id)` for precise agent categorization
- **Race-Safe Creation** - Double-checked locking prevents duplicate agent initialization
- **Agent Reuse** - Automatic sharing of agents across workflow nodes with same profile
- **Resource Limits** - Configurable max agents per run with automatic cleanup

### **CallBroker (Rate Limiting)**
Global rate limiting system with:
- **Per-Model Limits** - Separate RPM, RPD, and concurrency limits per model
- **429-Aware Retries** - Automatic retry with jittered exponential backoff
- **Sliding Window Counters** - Precise rate tracking with time-based windows
- **Request Queuing** - Fair scheduling across multiple agents

### **MCPExecutor (Multi-Server)**
Enhanced executor with:
- **Server-Name Routing** - Dynamic routing to different MCP servers per node
- **Parameter Filtering** - Clean parameter handling for backward compatibility
- **Output Capture** - Enhanced result processing with fallback to text
- **Streaming Support** - Real-time chunk processing with telemetry

### **SidecarMCPAgent (Enhanced)**
Drop-in replacement for `mcp-use` MCPAgent with:
- **Telemetry Integration** - Comprehensive event emission and performance tracking
- **Parameter Safety** - Secure handling of server_name and other routing parameters
- **Enhanced Error Handling** - Better error reporting and recovery
- **Full API Compatibility** - 100% compatible with existing `mcp-use` code

## 📦 **Installation**

### Prerequisites
- Python 3.11+
- Node.js 18+ (for MCP servers)
- OpenAI API key (or other LLM provider)

### Install Agentic Orchestra
```bash
pip install agentic-orchestra
```

### Install MCP Servers
```bash
# Filesystem server
npm install -g @modelcontextprotocol/server-filesystem

# Web browser server  
npm install -g @modelcontextprotocol/server-playwright

# Or use npx to run without global install
npx @modelcontextprotocol/server-filesystem --help
```

## 🚀 **Quick Start**

### Simple Agent Usage (Drop-in Replacement)
```python
import asyncio
from agent_orchestra import SidecarMCPClient, SidecarMCPAgent
from langchain_openai import ChatOpenAI

async def simple_example():
    # Configure MCP client
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", 
                        "--stdio", "--root", "/tmp"]
            }
        }
    }
    
    client = SidecarMCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = SidecarMCPAgent(llm=llm, client=client)
    
    result = await agent.run("List the files in the current directory")
    print(result)
    
    await client.close_all_sessions()

asyncio.run(simple_example())
```

### Production Workflow with All Features
```python
import asyncio
from agent_orchestra import SidecarMCPClient, SidecarMCPAgent
from agent_orchestra.orchestrator.core import Orchestrator
from agent_orchestra.orchestrator.types import GraphSpec, NodeSpec, RunSpec
from agent_orchestra.orchestrator.executors_mcp import MCPExecutor
from agent_orchestra.orchestrator.broker_config import create_development_broker
from agent_orchestra.orchestrator.agent_pool import AgentPool, create_default_agent_factory
from langchain_openai import ChatOpenAI

async def production_workflow():
    # Multi-server MCP configuration
    config = {
        "mcpServers": {
            "fs_business": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", 
                        "--stdio", "--root", "/business/data"]
            },
            "fs_reports": {
                "command": "npx", 
                "args": ["-y", "@modelcontextprotocol/server-filesystem",
                        "--stdio", "--root", "/reports/output"]
            },
            "web": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-playwright", "--stdio"]
            }
        }
    }
    
    # Create production components
    client = SidecarMCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # Profile-based agent pool
    agent_factory = create_default_agent_factory(client, llm)
    agent_pool = AgentPool(agent_factory, max_agents_per_run=10)
    
    # Global rate limiting
    broker = create_development_broker()
    
    # Production-ready executor
    executor = MCPExecutor(
        agent=None,  # No template agent needed
        default_server="fs_business",
        broker=broker,
        agent_pool=agent_pool,
        model_key="openai:gpt-4o-mini"
    )
    
    orchestrator = Orchestrator(executor)
    
    # Define multi-server workflow
    workflow = GraphSpec(
        nodes=[
            # Concurrent data processing
            NodeSpec(
                id="read_sales_data",
                type="foreach",
                server_name="fs_business",  # Route to business filesystem
                inputs={
                    "items": ["sales.json", "marketing.json", "operations.json"],
                    "instruction": "Read and summarize each business file"
                },
                concurrency=3
            ),
            
            # Cross-department analysis  
            NodeSpec(
                id="analyze_trends",
                type="reduce",
                inputs={
                    "from_ids": ["read_sales_data"],
                    "instruction": "Analyze trends across all departments"
                }
            ),
            
            # Web research for market context
            NodeSpec(
                id="market_research",
                type="task",
                server_name="web",  # Route to web browser
                inputs={
                    "from": "analyze_trends",
                    "instruction": "Research current market trends for context"
                }
            ),
            
            # Save final report
            NodeSpec(
                id="save_report",
                type="task", 
                server_name="fs_reports",  # Route to reports filesystem
                inputs={
                    "from": "market_research",
                    "instruction": "Create executive summary and save as report.pdf"
                }
            )
        ],
        edges=[
            ("read_sales_data", "analyze_trends"),
            ("analyze_trends", "market_research"),
            ("market_research", "save_report")
        ]
    )
    
    run_spec = RunSpec(
        run_id="business_analysis_001",
        goal="Multi-department business analysis with market research"
    )
    
    # Execute with full observability
    print("🚀 Starting production workflow...")
    async for event in orchestrator.run_streaming(workflow, run_spec):
        if event.type == "NODE_START":
            print(f"🔄 Starting {event.node_id}")
        elif event.type == "NODE_COMPLETE":
            print(f"✅ Completed {event.node_id}")
        elif event.type == "AGENT_CHUNK":
            print(f"   🧠 Agent progress: {event.data.get('content', '')[:50]}...")
        elif event.type == "RUN_COMPLETE":
            print(f"🎉 Workflow completed successfully!")
    
    # Get production metrics
    broker_stats = await broker.get_stats()
    pool_stats = await agent_pool.get_pool_stats()
    
    print(f"\n📊 Production Metrics:")
    print(f"   🏊 Agent profiles created: {len(pool_stats['profiles'])}")
    for profile_key, profile_info in pool_stats['profiles'].items():
        server = profile_info['server_name'] or 'default'
        usage = profile_info['usage_count']
        print(f"      {server} server: {usage} uses")
    
    for model, stats in broker_stats.items():
        if stats['rpm_used'] > 0:
            print(f"   📈 {model}: {stats['rpm_used']}/{stats['rpm_limit']} RPM used")
    
    # Clean shutdown
    await broker.shutdown()
    await agent_pool.shutdown()
    await client.close_all_sessions()

# Run with proper error handling
if __name__ == "__main__":
    asyncio.run(production_workflow())
```

## 📊 **Examples**

The `examples/` directory contains production-ready demonstrations:

### **Production Examples**
- **`polished_part4_demo.py`** - Complete production workflow with all features
- **`polished_simple_demo.py`** - Simple demo without complex MCP setup  
- **`polished_verification_demo.py`** - Verification of all polish improvements
- **`part4_complete_demo.py`** - CallBroker + AgentPool integration

### **Core Feature Examples**
- **`basic_orchestration.py`** - Simple DAG workflow
- **`foreach_example.py`** - Concurrent batch processing
- **`reduce_example.py`** - Data aggregation patterns
- **`gate_example.py`** - Conditional workflow control

### **Integration Examples**
- **`multi_server_example.py`** - Multiple MCP servers in one workflow
- **`rate_limiting_example.py`** - CallBroker rate limiting demonstration
- **`agent_pooling_example.py`** - AgentPool management patterns

## 🧪 **Testing**

Agentic Orchestra includes comprehensive test coverage:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_polish_improvements.py -v  # Production features
python -m pytest tests/test_orchestration.py -v       # Core orchestration
python -m pytest tests/test_agent_pool.py -v          # Agent management
```

**Test Coverage:**
- **Polish Improvements** - All 10 production-ready improvements
- **Race Conditions** - Concurrent agent creation safety
- **Path Validation** - Security and directory traversal prevention
- **Rate Limiting** - Global rate limiting across multiple agents
- **Multi-Server** - Server routing and profile management

## 🔧 **Configuration**

### **CallBroker Configuration**
```python
from agent_orchestra.orchestrator.call_broker import CallBroker, ModelLimits

# Custom rate limits
limits = {
    "openai:gpt-4": ModelLimits(rpm=60, rpd=1000, max_concurrency=10),
    "openai:gpt-4o-mini": ModelLimits(rpm=200, rpd=5000, max_concurrency=20),
    "anthropic:claude-3": ModelLimits(rpm=50, rpd=800, max_concurrency=5)
}

broker = CallBroker(limits)
```

### **AgentPool Configuration**  
```python
from agent_orchestra.orchestrator.agent_pool import AgentPool, AgentSpec

# Profile-based agent management
async def custom_factory(spec: AgentSpec):
    # Custom agent creation logic based on spec
    return SidecarMCPAgent(...)

pool = AgentPool(custom_factory, max_agents_per_run=15)

# Get agent for specific profile
spec = AgentSpec(
    server_name="fs_business",
    model_key="openai:gpt-4",
    policy_id="standard"
)
agent = await pool.get(spec, "run_123")
```

### **Multi-Server MCP Configuration**
```python
from agent_orchestra.orchestrator.fs_utils import create_multi_server_config

configs = {
    "fs_sales": {"root": "/data/sales"},
    "fs_reports": {"root": "/data/reports"},
    "playwright": {"type": "playwright"},
    "custom_server": {
        "command": "python",
        "args": ["-m", "my_custom_server", "--stdio"]
    }
}

mcp_config = create_multi_server_config(configs)
```

## 🛡️ **Security Features**

### **Path Validation**
```python
from agent_orchestra.orchestrator.fs_utils import fs_args

# Safe path handling with directory traversal prevention
root = Path("/safe/root")
try:
    safe_args = fs_args(root, "../../etc/passwd")  # Raises ValueError
except ValueError as e:
    print(f"Security violation prevented: {e}")
```

### **Parameter Filtering**
Agentic Orchestra automatically filters potentially unsafe parameters before passing them to underlying MCP agents, ensuring backward compatibility while maintaining security.

## 📈 **Performance & Monitoring**

### **Built-in Metrics**
```python
# Get real-time broker statistics
broker_stats = await broker.get_stats()
print(f"RPM usage: {broker_stats['openai:gpt-4']['rpm_used']}")

# Get agent pool statistics  
pool_stats = await agent_pool.get_pool_stats()
print(f"Active agents: {pool_stats['total_agents']}")
print(f"Profile usage: {pool_stats['profiles']}")
```

### **Event-Driven Telemetry**
```python
# Access structured events during execution
async for event in orchestrator.run_streaming(workflow, run_spec):
    if event.type == "AGENT_CHUNK":
        # Log or emit to external monitoring
        telemetry_system.emit({
            "timestamp": event.timestamp,
            "node_id": event.node_id, 
            "content": event.data
        })
```

## 🤝 **Migration from mcp-use**

Agentic Orchestra is designed as a drop-in replacement. To migrate:

1. **Replace imports:**
   ```python
   # Old
   from mcp_use import MCPClient, MCPAgent
   
   # New  
   from agent_orchestra import SidecarMCPClient as MCPClient
   from agent_orchestra import SidecarMCPAgent as MCPAgent
   ```

2. **Optional: Add production features:**
   ```python
   # Add rate limiting
   from agent_orchestra.orchestrator.broker_config import create_development_broker
   broker = create_development_broker()
   
   # Add agent pooling
   from agent_orchestra.orchestrator.agent_pool import AgentPool
   pool = AgentPool(agent_factory)
   ```

3. **Optional: Use orchestration:**
   ```python
   # Define workflows instead of sequential calls
   from agent_orchestra.orchestrator import Orchestrator, GraphSpec, NodeSpec
   ```

## 📚 **Documentation**

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and component overview
- **[Production Deployment](docs/DEPLOYMENT.md)** - Best practices for production use
- **[API Reference](docs/API.md)** - Comprehensive API documentation
- **[Migration Guide](docs/MIGRATION.md)** - Detailed migration from mcp-use
- **[Performance Tuning](docs/PERFORMANCE.md)** - Optimization strategies

## 🎯 **Production Readiness Checklist**

Agentic Orchestra has been thoroughly tested and includes all features needed for production deployment:

- ✅ **Race-safe agent creation** with double-checked locking
- ✅ **Global rate limiting** with 429-aware retries
- ✅ **Profile-based agent pooling** with automatic cleanup
- ✅ **Multi-server routing** with parameter filtering
- ✅ **Security validations** preventing directory traversal
- ✅ **Comprehensive error handling** with graceful degradation
- ✅ **Resource lifecycle management** with proper async cleanup
- ✅ **Production monitoring** with structured events and metrics
- ✅ **Backward compatibility** with existing mcp-use code
- ✅ **Comprehensive test coverage** including race conditions

## 🛠️ **Development**

### **Setup Development Environment**
```bash
git clone https://github.com/your-org/agentic-orchestra
cd agentic-orchestra
pip install -e .
pip install -r requirements-dev.txt
```

### **Run Tests**
```bash
python -m pytest tests/ -v --cov=agent_orchestra
```

### **Code Quality**
```bash
ruff check .                    # Linting
ruff format .                   # Formatting  
mypy src/agent_orchestra/       # Type checking
```

## 📄 **License**

Agentic Orchestra is licensed under the [MIT License](LICENSE).

## 🙏 **Contributing**

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Key Areas for Contribution:**
- Additional MCP server integrations
- Enhanced telemetry and monitoring features
- Performance optimizations
- Documentation improvements
- Example workflows and use cases

## 🌟 **Roadmap**

**Upcoming Features:**
- OpenTelemetry integration for distributed tracing
- Human-in-the-loop (HITL) workflow nodes
- Advanced policy enforcement with RBAC
- Workflow versioning and rollback
- Distributed execution across multiple nodes
- Enhanced security with request signing

---

**Agentic Orchestra: Production-Ready Multi-Agentic Orchestration** 🎼

*Built for enterprises, loved by developers.*