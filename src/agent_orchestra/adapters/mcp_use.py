"""MCP-USE adapter implementation.

This is the ONLY module that should import from mcp-use.
All other modules should use the adapter interface.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from agent_orchestra.adapters.base import AdapterResult, BaseAdapter, UsageMetrics
from agent_orchestra.events import Event, EventEmitter, EventType
from agent_orchestra.llm_factory import LLMFactory, LLMFactoryError
from agent_orchestra.tools_loader import ToolsLoader

# Only mcp-use imports allowed in this file
try:
    from mcp_use import MCPAgent, MCPClient
    MCP_USE_AVAILABLE = True
except ImportError:
    MCP_USE_AVAILABLE = False
    MCPAgent = None
    MCPClient = None


class MCPUseAdapter(BaseAdapter):
    """Adapter for MCP-USE agents."""

    def __init__(self, event_emitter: EventEmitter | None = None) -> None:
        if not MCP_USE_AVAILABLE:
            raise ImportError(
                "mcp-use is not installed. Install with: pip install mcp-use"
            )

        self._event_emitter = event_emitter
        self._tools_loader = ToolsLoader()

    async def call(
        self,
        agent: dict[str, Any],  # Always a resolved spec now
        tool: str,
        *,
        inputs: dict[str, Any],
        meta: dict[str, Any]
    ) -> AdapterResult:
        """Execute a tool call through MCP-USE."""
        start_time = time.time()
        trace_id = meta.get("trace_id", "")
        span_id = meta.get("span_id", "")
        run_id = meta.get("run_id", "")

        try:
            # Get pre-built MCP client from meta
            mcp_client = meta.get("mcp_client")
            if not mcp_client:
                raise ValueError("MCPClient not provided in meta - orchestrator should pass it")

            # Get or create agent from resolved spec
            mcp_agent = await self._get_agent_from_spec(agent, mcp_client, meta)

            # Format the input for the agent
            prompt = self._format_tool_call(tool, inputs)
            
            # Emit AGENT_START event
            if self._event_emitter:
                await self._event_emitter.emit(Event(
                    type=EventType.AGENT_START,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={"agent_id": agent.get("id", "unknown"), "tool": tool, "prompt_length": len(prompt)}
                ))

            # Execute the agent with tool call monitoring
            server_name = meta.get("server_name")
            stream_mode = meta.get("stream", False)
            
            if stream_mode:
                # Return streaming generator wrapped in AdapterResult
                # Note: Streaming mode returns a generator, not a final result
                stream_gen = self._run_with_tool_monitoring(
                    mcp_agent, prompt, server_name, run_id, trace_id, span_id, stream=True
                )
                
                # For streaming, return a special AdapterResult with the generator
                return AdapterResult(
                    output=stream_gen,  # Async generator
                    usage=UsageMetrics(latency_ms=0),  # Will be calculated later
                    metadata={
                        "agent_id": agent.get("id"),
                        "tool": tool,
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "streaming": True
                    }
                )
            else:
                # Execute normally
                result = await self._run_with_tool_monitoring(
                    mcp_agent, prompt, server_name, run_id, trace_id, span_id, stream=False
                )

                # Calculate usage metrics
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)

                usage = UsageMetrics(
                    tokens=self._estimate_tokens(prompt, str(result)),
                    cost=0.0,  # TODO: Calculate actual cost based on provider
                    latency_ms=latency_ms
                )
                
                # Emit AGENT_COMPLETE event
                if self._event_emitter:
                    await self._event_emitter.emit(Event(
                        type=EventType.AGENT_COMPLETE,
                        run_id=run_id,
                        trace_id=trace_id,
                        span_id=span_id,
                        payload={
                            "agent_id": agent.get("id", "unknown"), 
                            "tool": tool, 
                            "tokens": usage.tokens,
                            "latency_ms": latency_ms,
                            "result_length": len(str(result))
                        }
                    ))

                return AdapterResult(
                    output=result,
                    usage=usage,
                    metadata={
                        "agent_id": agent.get("id"),
                        "tool": tool,
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "streaming": False
                    }
                )

        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Emit AGENT_ERROR event
            if self._event_emitter:
                await self._event_emitter.emit(Event(
                    type=EventType.AGENT_ERROR,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={
                        "agent_id": agent.get("id", "unknown"), 
                        "tool": tool, 
                        "error": str(e)
                    }
                ))

            return AdapterResult(
                output=None,
                usage=UsageMetrics(latency_ms=latency_ms),
                errors=[str(e)],
                metadata={
                    "agent_id": agent.get("id"),
                    "tool": tool,
                    "trace_id": trace_id,
                    "span_id": span_id
                }
            )

    async def get_available_tools(self, agent: dict[str, Any]) -> list[str]:
        """Get available tools for an MCP agent."""
        try:
            # Return tools from agent spec allowed_tools, or query client
            allowed_tools = agent.get("allowed_tools", [])
            if allowed_tools:
                return allowed_tools
            
            # TODO: Query MCP client for available tools if none specified
            # This is a placeholder - actual implementation depends on mcp-use API
            return ["web_search", "browser", "analysis", "file_operations"]
        except Exception:
            return []

    async def health_check(self) -> bool:
        """Check if MCP-USE is available and working."""
        try:
            # Simple test to ensure mcp-use is working
            test_config = {
                "mcpServers": {
                    "test": {
                        "command": "echo",
                        "args": ["test"]
                    }
                }
            }
            client = MCPClient.from_dict(test_config)
            return True
        except Exception:
            return False


    async def _get_agent_from_spec(
        self, 
        agent_spec: dict[str, Any], 
        mcp_client: MCPClient, 
        meta: dict[str, Any]
    ) -> MCPAgent:
        """Get or create MCP agent from resolved spec and pre-built client."""
        # Cache key: (agent_id, model_id, api_key_hash, tools_hash, server_manager_flag, disallowed/allowed_hash)
        agent_id = agent_spec.get("id", "unknown")
        model = agent_spec.get("model", "")
        api_key = agent_spec.get("api_key", "")
        instructions = agent_spec.get("instructions", "")
        use_server_manager = agent_spec.get("use_server_manager", True)
        disallowed_tools = agent_spec.get("disallowed_tools", [])
        allowed_tools = agent_spec.get("allowed_tools", [])
        
        # Get tools config digest from meta (orchestrator calculates this)
        tools_digest = meta.get("tools_digest", "no_tools")
        
        # Create comprehensive cache key for agent caching
        # Include tool access control in cache key (allow-list precedence)
        if allowed_tools is not None:
            tool_control_hash = f"allowed_{hash(tuple(sorted(allowed_tools)))}"
        elif disallowed_tools:
            tool_control_hash = f"disallowed_{hash(tuple(sorted(disallowed_tools)))}"
        else:
            tool_control_hash = "no_tool_control"
            
        cache_components = [
            agent_id or "inline",
            model,
            str(hash(api_key)) if api_key else "no_key",
            tools_digest,
            str(use_server_manager),
            tool_control_hash,
            str(hash(instructions)) if instructions else "no_instructions"
        ]
        agent_cache_key = "_".join(cache_components)
        
        # Check agent cache in meta (orchestrator manages per-run cache)
        agent_cache = meta.get("agent_cache", {})
        if agent_cache_key in agent_cache:
            # Reuse cached agent for this run
            cached_agent = agent_cache[agent_cache_key]
            
            # Emit AGENT_INIT for cached agent reuse
            if self._event_emitter:
                await self._event_emitter.emit(Event(
                    type=EventType.AGENT_INIT,
                    run_id=meta.get("run_id", ""),
                    trace_id=meta.get("trace_id", ""),
                    span_id=meta.get("span_id", ""),
                    payload={
                        "id": agent_id,
                        "model": model,
                        "cached": True,
                        "cache_key": agent_cache_key[:50] + "..." if len(agent_cache_key) > 50 else agent_cache_key
                    }
                ))
                
            return cached_agent
        
        # Create LLM cache key (separate from agent cache)
        llm_cache_key = f"{model}_{hash(api_key)}_{hash(instructions)}"
        
        # Get or create LLM from cache
        llm_factory = meta.get("llm_factory")
        if not llm_factory:
            raise ValueError("LLM factory not provided in meta")
            
        # Check LLM cache in meta (orchestrator manages this) 
        llm_cache = meta.get("llm_cache", {})
        if llm_cache_key not in llm_cache:
            # Create raw LLM from spec using factory (no prompt wrapping!)
            llm_config = agent_spec.get("llm", {})
            
            # Apply configuration precedence: ENV > agent spec > defaults
            if api_key:
                llm_config["api_key"] = api_key
                
            # Create raw LLM - mcp_use needs bind_tools() support
            llm = llm_factory.get_llm(model, llm_config)
            
            # Do NOT wrap with prompt | llm - breaks bind_tools()
            # Instructions will be passed to MCPAgent constructor instead
                    
            llm_cache[llm_cache_key] = llm
        else:
            llm = llm_cache[llm_cache_key]
        
        # Always create fresh MCPAgent (no conversation state sharing)
        max_steps = agent_spec.get("max_steps", 8)
        disallowed_tools = agent_spec.get("disallowed_tools")
        allowed_tools = agent_spec.get("allowed_tools")
        use_server_manager = agent_spec.get("use_server_manager", True)
        
        # Apply tool access control with allow-list precedence
        # If allowed_tools is specified, it takes precedence over disallowed_tools
        if allowed_tools is not None:
            # Use allowed_tools only (ignore disallowed_tools)
            final_disallowed_tools = None
            final_allowed_tools = allowed_tools
        else:
            # Use disallowed_tools if no allowed_tools specified
            final_disallowed_tools = disallowed_tools
            final_allowed_tools = None
        
        # Prepare instruction parameters for MCPAgent
        mcp_agent_kwargs = {
            "llm": llm,
            "client": mcp_client,
            "max_steps": max_steps,
            "use_server_manager": use_server_manager
        }
        
        # Add tool filtering parameters (check what mcp-use supports)
        if final_allowed_tools is not None:
            # Check if MCPAgent supports allowed_tools parameter
            import inspect
            mcp_agent_params = inspect.signature(MCPAgent.__init__).parameters
            if "allowed_tools" in mcp_agent_params:
                mcp_agent_kwargs["allowed_tools"] = final_allowed_tools
            else:
                # Fallback: use disallowed_tools by inverting allowed_tools
                # This would require knowing all available tools to compute the difference
                # For now, emit a warning and use disallowed_tools=None
                print(f"⚠️  WARNING: MCPAgent doesn't support allowed_tools parameter. Tool filtering may be incomplete.")
                mcp_agent_kwargs["disallowed_tools"] = None
        elif final_disallowed_tools is not None:
            mcp_agent_kwargs["disallowed_tools"] = final_disallowed_tools
        
        # Pass instructions via MCPAgent parameters (not prompt wrapper)
        if instructions:
            # Check if instructions contain template variables
            if "{" in instructions and "}" in instructions:
                mcp_agent_kwargs["system_prompt_template"] = instructions
            else:
                mcp_agent_kwargs["system_prompt"] = instructions
        
        # Add any additional instructions from agent spec
        additional_instructions = agent_spec.get("additional_instructions")
        if additional_instructions:
            mcp_agent_kwargs["additional_instructions"] = additional_instructions
        
        # Create the MCPAgent
        agent = MCPAgent(**mcp_agent_kwargs)
        
        # Cache the agent in orchestrator's agent cache
        agent_cache[agent_cache_key] = agent
        
        # Emit AGENT_INIT event
        if self._event_emitter:
            # Prepare tool control info for event payload
            tool_control_info = {}
            if final_allowed_tools is not None:
                tool_control_info = {
                    "tool_control_type": "allowed",
                    "tool_control_count": len(final_allowed_tools)
                }
            elif final_disallowed_tools:
                tool_control_info = {
                    "tool_control_type": "disallowed", 
                    "tool_control_count": len(final_disallowed_tools)
                }
            else:
                tool_control_info = {
                    "tool_control_type": "none",
                    "tool_control_count": 0
                }
                
            await self._event_emitter.emit(Event(
                type=EventType.AGENT_INIT,
                run_id=meta.get("run_id", ""),
                trace_id=meta.get("trace_id", ""),
                span_id=meta.get("span_id", ""),
                payload={
                    "id": agent_id,
                    "model": model,
                    "max_steps": max_steps,
                    "use_server_manager": use_server_manager,
                    "has_instructions": bool(instructions),
                    "cached": False,
                    "cache_key": agent_cache_key[:50] + "..." if len(agent_cache_key) > 50 else agent_cache_key,
                    **tool_control_info
                }
            ))
        
        return agent

    async def _run_with_tool_monitoring(
        self, 
        agent: MCPAgent, 
        prompt: str, 
        server_name: str | None,
        run_id: str,
        trace_id: str, 
        span_id: str,
        *,
        stream: bool = False
    ) -> Any:
        """Run agent with tool call/result event monitoring (debug level)."""
        # TODO: Implement tool call monitoring when debug mode is enabled
        # For now, this is a placeholder that just calls the agent
        # In a full implementation, we would need to hook into the agent's 
        # tool execution loop to emit TOOL_CALL and TOOL_RESULT events
        
        if stream:
            # Use streaming interface if requested
            return self.astream(agent, prompt, server_name=server_name)
        else:
            # Use regular run interface
            if server_name is not None:
                return await agent.run(prompt, server_name=server_name)
            else:
                return await agent.run(prompt)


    def _format_tool_call(self, tool: str, inputs: dict[str, Any]) -> str:
        """Format tool call as a prompt for the agent."""
        if not inputs:
            return f"Use the {tool} tool"

        # Convert inputs to a natural language prompt
        input_parts = []
        for key, value in inputs.items():
            input_parts.append(f"{key}: {value}")

        return f"Use the {tool} tool with the following inputs: {', '.join(input_parts)}"

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Rough token estimation (replace with actual tokenizer if needed)."""
        # Very rough estimate: ~4 characters per token
        total_chars = len(prompt) + len(response)
        return total_chars // 4

    async def build_client(
        self,
        tools_config_dict_or_path: dict[str, Any] | str | Path,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None
    ) -> MCPClient:
        """Build MCPClient from config dict or path.
        
        Args:
            tools_config_dict_or_path: Either a config dictionary or path to config file
            run_id: Run ID for event emission
            trace_id: Trace ID for event emission
            span_id: Span ID for event emission
            
        Returns:
            MCPClient instance
            
        Raises:
            ToolsLoaderError: If loading/validation fails
        """
        # Load and validate config
        if isinstance(tools_config_dict_or_path, (str, Path)):
            config = self._tools_loader.load(tools_config_dict_or_path)
        else:
            # Validate dict config
            self._tools_loader._validate(tools_config_dict_or_path)
            self._tools_loader._validate_edge_cases(tools_config_dict_or_path)
            config = tools_config_dict_or_path

        # Emit client.start event
        if self._event_emitter and run_id and trace_id and span_id:
            server_count = len(config.get("mcpServers", {}))
            await self._event_emitter.emit(Event(
                type=EventType.CLIENT_START,
                run_id=run_id,
                trace_id=trace_id,
                span_id=span_id,
                payload={"server_count": server_count}
            ))

            # Emit server.start events
            for server_name, server_config in config.get("mcpServers", {}).items():
                transport = "http-sse" if "url" in server_config else "stdio"
                await self._event_emitter.emit(Event(
                    type=EventType.SERVER_START,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={"name": server_name, "transport": transport}
                ))

        # Create client
        client = MCPClient.from_dict(config)

        # Optional readiness check
        if self._event_emitter and run_id and trace_id and span_id:
            await self._check_server_readiness(client, config, run_id, trace_id, span_id)

        return client

    def build_agent(
        self,
        llm: Any,
        client: MCPClient,
        max_steps: int,
        *,
        disallowed_tools: list[str] | None = None,
        use_server_manager: bool = False
    ) -> MCPAgent:
        """Build MCPAgent with specified configuration.
        
        Args:
            llm: Language model instance
            client: MCPClient instance
            max_steps: Maximum number of steps for agent execution
            disallowed_tools: Optional list of tools to disallow
            use_server_manager: Whether to use server manager
            
        Returns:
            MCPAgent instance
        """
        return MCPAgent(
            llm=llm,
            client=client,
            max_steps=max_steps,
            disallowed_tools=disallowed_tools,
            use_server_manager=use_server_manager
        )

    async def run(
        self,
        agent: MCPAgent,
        prompt: str,
        *,
        server_name: str | None = None
    ) -> Any:
        """Run agent with prompt.
        
        Args:
            agent: MCPAgent instance
            prompt: Input prompt for the agent
            server_name: Optional server name to pass through
            
        Returns:
            Agent execution result
        """
        if server_name is not None:
            return await agent.run(prompt, server_name=server_name)
        else:
            return await agent.run(prompt)
            
    async def astream(
        self,
        agent: MCPAgent,
        prompt: str,
        *,
        server_name: str | None = None
    ) -> Any:
        """Stream agent execution with prompt.
        
        Args:
            agent: MCPAgent instance
            prompt: Input prompt for the agent
            server_name: Optional server name to pass through
            
        Returns:
            Async generator of execution steps/events
        """
        # Check if MCPAgent supports streaming
        import inspect
        if hasattr(agent, 'astream') and callable(agent.astream):
            # Use built-in streaming support
            if server_name is not None:
                async for chunk in agent.astream(prompt, server_name=server_name):
                    yield chunk
            else:
                async for chunk in agent.astream(prompt):
                    yield chunk
        else:
            # Fallback: run normally and yield final result
            # This provides streaming interface even if underlying agent doesn't support it
            result = await self.run(agent, prompt, server_name=server_name)
            yield result

    async def close_all(
        self,
        client: MCPClient,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None
    ) -> None:
        """Close all client sessions.
        
        Args:
            client: MCPClient instance to close
            run_id: Run ID for event emission
            trace_id: Trace ID for event emission
            span_id: Span ID for event emission
        """
        try:
            await client.close_all_sessions()
        finally:
            # Always emit client.stop event
            if self._event_emitter and run_id and trace_id and span_id:
                await self._event_emitter.emit(Event(
                    type=EventType.CLIENT_STOP,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={}
                ))

    async def _check_server_readiness(
        self,
        client: MCPClient,
        config: dict[str, Any],
        run_id: str,
        trace_id: str,
        span_id: str
    ) -> None:
        """Check server readiness with optional timeout.
        
        Args:
            client: MCPClient instance
            config: Server configuration
            run_id: Run ID for events
            trace_id: Trace ID for events
            span_id: Span ID for events
        """
        for server_name, server_config in config.get("mcpServers", {}).items():
            timeout_s = server_config.get("timeout_s")
            if timeout_s is None:
                continue  # Skip readiness check if no timeout specified

            start_time = time.time()
            try:
                # Simple readiness check - try to list tools with timeout
                await asyncio.wait_for(
                    self._ping_server_readiness(client, server_name),
                    timeout=timeout_s
                )

                # Emit server.ready
                elapsed_ms = int((time.time() - start_time) * 1000)
                await self._event_emitter.emit(Event(
                    type=EventType.SERVER_READY,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={"name": server_name, "ms": elapsed_ms}
                ))

            except TimeoutError:
                await self._event_emitter.emit(Event(
                    type=EventType.SERVER_ERROR,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={"name": server_name, "reason": f"Timeout after {timeout_s}s"}
                ))
            except Exception as e:
                await self._event_emitter.emit(Event(
                    type=EventType.SERVER_ERROR,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={"name": server_name, "reason": str(e)}
                ))

    async def _ping_server_readiness(self, client: MCPClient, server_name: str) -> None:
        """Ping server for readiness (light check).
        
        Args:
            client: MCPClient instance
            server_name: Name of server to ping
            
        Raises:
            Exception: If server is not ready or unreachable
        """
        try:
            # Check if we can get basic server information
            # Try different lightweight operations based on what mcp-use supports
            
            # Method 1: Try to list available tools (most common)
            if hasattr(client, 'list_tools'):
                tools = await client.list_tools(server_name=server_name)
                # If we get a response, server is ready
                return
                
            # Method 2: Try to get server info/capabilities
            if hasattr(client, 'get_server_info'):
                info = await client.get_server_info(server_name)
                return
                
            # Method 3: Try to list resources (if supported)
            if hasattr(client, 'list_resources'):
                resources = await client.list_resources(server_name=server_name)
                return
                
            # Method 4: Generic ping if available
            if hasattr(client, 'ping'):
                await client.ping(server_name)
                return
                
            # If no specific methods available, assume ready
            # (The actual connection will be tested when first used)
            
        except Exception as e:
            # Re-raise to indicate server is not ready
            raise Exception(f"Server readiness check failed: {str(e)}")

    async def build_agent_from_spec(
        self,
        agent_spec: dict[str, Any],
        client: MCPClient,
        llm_factory: LLMFactory | None = None,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        span_id: str | None = None
    ) -> MCPAgent:
        """Build MCPAgent from agent specification.
        
        Args:
            agent_spec: Agent specification dict
            client: MCPClient instance
            llm_factory: LLM factory instance (creates one if None)
            run_id: Run ID for event emission
            trace_id: Trace ID for event emission
            span_id: Span ID for event emission
            
        Returns:
            MCPAgent instance configured from spec
            
        Raises:
            LLMFactoryError: If LLM creation fails
        """
        agent_id = agent_spec.get("id", "unknown")
        
        try:
            # Create LLM factory if not provided
            if llm_factory is None:
                llm_factory = LLMFactory()
            
            # Create LLM from spec
            model_id = agent_spec["model"]
            llm_config = agent_spec.get("llm", {})
            
            # Pass through API key if provided in agent spec
            if "api_key" in agent_spec:
                llm_config["api_key"] = agent_spec["api_key"]
                
            llm = llm_factory.get_llm(model_id, llm_config)
            
            # Wrap with system prompt if instructions provided
            instructions = agent_spec.get("instructions")
            if instructions:
                try:
                    # Import LangChain here to avoid hard dependency
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.runnables import RunnablePassthrough
                    
                    # Create system prompt wrapper
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", instructions),
                        ("human", "{input}")
                    ])
                    
                    # Chain prompt with LLM
                    llm = prompt | llm
                    
                except ImportError:
                    # If LangChain core not available, just use LLM without system prompt
                    pass
            
            # Get agent configuration
            max_steps = agent_spec.get("max_steps", 8)
            disallowed_tools = agent_spec.get("disallowed_tools")
            use_server_manager = agent_spec.get("use_server_manager", True)
            
            # Create MCPAgent
            agent = MCPAgent(
                llm=llm,
                client=client,
                max_steps=max_steps,
                disallowed_tools=disallowed_tools,
                use_server_manager=use_server_manager
            )
            
            # Emit AGENT_INIT event
            if self._event_emitter and run_id and trace_id and span_id:
                await self._event_emitter.emit(Event(
                    type=EventType.AGENT_INIT,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={
                        "id": agent_id,
                        "model": model_id,
                        "max_steps": max_steps,
                        "use_server_manager": use_server_manager,
                        "disallowed_tools_count": len(disallowed_tools) if disallowed_tools else 0,
                        "has_instructions": bool(instructions)
                    }
                ))
            
            return agent
            
        except Exception as e:
            # Emit AGENT_ERROR event
            if self._event_emitter and run_id and trace_id and span_id:
                await self._event_emitter.emit(Event(
                    type=EventType.AGENT_ERROR,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    payload={
                        "id": agent_id,
                        "reason": str(e)
                    }
                ))
            
            # Re-raise the original exception
            raise

