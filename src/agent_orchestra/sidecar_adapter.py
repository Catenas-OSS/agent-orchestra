"""SidecarLangChainAdapter - Extended LangChain adapter with metadata attachment.

This module provides an enhanced version of LangChainAdapter that adds metadata
attachment to tools, safety analysis, and extended functionality for resources
and prompts while maintaining complete API compatibility.

Typical usage:
    # Create adapter
    adapter = SidecarLangChainAdapter()
    
    # Create tools with metadata attachment
    tools = adapter.create_tools_sync(
        client=mcp_client,
        allowed_tools=["search", "resource_read"]
    )
    
    # Filter tools by safety requirements
    safe_tools = adapter.filter_tools_by_safety(
        tools,
        {"system_command": False, "file_system_access": False}
    )
"""

from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import re

from mcp_use.adapters import LangChainAdapter

logger = logging.getLogger(__name__)


class SidecarLangChainAdapter(LangChainAdapter):  # pyright: ignore[reportInheritanceFromUntyped]
    """
    Extended LangChainAdapter that attaches Sidecar metadata to tools.
    
    This class extends the LangChainAdapter from mcp-use to attach additional
    metadata to the tools it creates, enabling safety analysis, tracking, and
    extended functionality for resources and prompts.
    
    Attributes:
        _sidecar_metadata_cache: Cache for storing schema hashes and metadata
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SidecarLangChainAdapter."""
        super().__init__(*args, **kwargs)
        self._sidecar_metadata_cache: Dict[str, Any] = {}
    
    def _convert_tool(self, tool_info: Dict[str, Any], server_name: str) -> Any:
        """
        Override _convert_tool to attach Sidecar metadata to tools.
        
        This method extends the parent _convert_tool method to attach additional
        metadata to the created tools, including schema hash for versioning,
        server origin tracking, and safety flags analysis.
        
        Args:
            tool_info: Tool information dictionary with name, description, and schema
            server_name: Name of the MCP server this tool belongs to
            
        Returns:
            Any: LangChain BaseTool instance with attached Sidecar metadata
        """
        # Call parent conversion first
        langchain_tool = super()._convert_tool(tool_info, server_name) # type: ignore
        
        # Generate schema hash for versioning/caching
        schema_str = json.dumps(tool_info.get('inputSchema', {}), sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
        
        # Attach Sidecar metadata while preserving existing MCP metadata
        sidecar_metadata = { # pyright: ignore[reportUnknownVariableType]
            '_sidecar_schema_hash': schema_hash,
            '_sidecar_server_origin': server_name,
            '_sidecar_tool_name': tool_info.get('name', 'unknown'),
            '_sidecar_safety_flags': self._analyze_safety_flags(tool_info)
        }
        
        # Add to existing metadata rather than replacing
        for key, value in sidecar_metadata.items(): # type: ignore
            setattr(langchain_tool, key, value) # pyright: ignore[reportUnknownArgumentType]
        
        logger.debug(f"Attached Sidecar metadata to tool {tool_info.get('name')}: {sidecar_metadata}")
        
        return langchain_tool # type: ignore
    
    def _analyze_safety_flags(self, tool_info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Analyze tool for potential safety characteristics and risks.
        
        This method examines a tool's name and description to identify potential
        safety concerns such as destructive operations, file system access,
        network access, or system command execution.
        
        Args:
            tool_info: Tool information dictionary containing name and description
            
        Returns:
            Dict[str, bool]: Dictionary of safety flags where True indicates
                the tool has the specified characteristic or risk
        """
        tool_name = tool_info.get('name', '').lower()
        description = tool_info.get('description', '').lower()
        
        safety_flags = {
            'potentially_destructive': False,
            'file_system_access': False,
            'network_access': False,
            'system_command': False
        }
        
        # Analyze based on common patterns
        destructive_keywords = ['delete', 'remove', 'destroy', 'kill', 'terminate', 'drop']
        if any(keyword in tool_name or keyword in description for keyword in destructive_keywords):
            safety_flags['potentially_destructive'] = True
        
        fs_keywords = ['file', 'directory', 'folder', 'path', 'write', 'read', 'create']
        if any(keyword in tool_name or keyword in description for keyword in fs_keywords):
            safety_flags['file_system_access'] = True
        
        network_keywords = ['http', 'url', 'fetch', 'download', 'upload', 'request', 'api']
        if any(keyword in tool_name or keyword in description for keyword in network_keywords):
            safety_flags['network_access'] = True
        
        system_keywords = ['exec', 'command', 'shell', 'bash', 'system', 'process']
        if any(keyword in tool_name or keyword in description for keyword in system_keywords):
            safety_flags['system_command'] = True
        
        return safety_flags
    
    def create_tools_sync(
        self,
        client: Any,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None
    ) -> List[Any]:  # pyright: ignore[reportUnknownReturnType]
        """
        Create LangChain tools with Sidecar metadata synchronously.
        
        This method provides a synchronous interface for creating LangChain tools
        with Sidecar metadata attached. It preserves the exact public API signature
        while adding Sidecar metadata through the _convert_tool hook.
        
        Args:
            client: MCP client instance with configured servers and sessions
            allowed_tools: Optional list of allowed tool names to filter the tools
            disallowed_tools: Optional list of disallowed tool names to exclude
            
        Returns:
            List[Any]: List of LangChain BaseTool instances with Sidecar metadata attached
            
        Note:
            This method logs metadata information about the created tools
        """
        # Use parent implementation which will call our _convert_tool override
        tools = super().create_tools(client, allowed_tools, disallowed_tools) # type: ignore
        
        # Log metadata summary
        logger.info(f"Created {len(tools)} tools with Sidecar metadata") # type: ignore
        for tool in tools: # type: ignore
            if hasattr(tool, '_sidecar_server_origin'): # type: ignore
                logger.debug(f"Tool {tool.name} from server {tool._sidecar_server_origin} " # type: ignore
                           f"with hash {getattr(tool, '_sidecar_schema_hash', 'unknown')}") # type: ignore
        
        return tools # type: ignore
    
    def get_tool_metadata(self, tool: Any) -> Dict[str, Any]:  # pyright: ignore[reportUnknownReturnType]
        """
        Extract Sidecar metadata from a tool.
        
        This method extracts both Sidecar and MCP metadata from a tool that was
        created by this adapter. It can be used for inspection, debugging, or
        advanced tool handling.
        
        Args:
            tool: LangChain tool with Sidecar metadata, created by this adapter
            
        Returns:
            Dict[str, Any]: Dictionary containing all Sidecar and MCP metadata
                attached to the tool
        """
        metadata = {}
        
        sidecar_attrs = [
            '_sidecar_schema_hash',
            '_sidecar_server_origin', 
            '_sidecar_tool_name',
            '_sidecar_safety_flags'
        ]
        
        for attr in sidecar_attrs:
            if hasattr(tool, attr):
                metadata[attr] = getattr(tool, attr)
        
        # Also include existing MCP metadata for completeness
        mcp_attrs = ['_mcp_server', '_mcp_original_schema']
        for attr in mcp_attrs:
            if hasattr(tool, attr):
                metadata[attr] = getattr(tool, attr)
        
        return metadata # type: ignore
    
    def filter_tools_by_safety(
        self,
        tools: List[Any],
        safety_requirements: Dict[str, bool]
    ) -> List[Any]:
        """
        Filter tools based on safety requirements.
        
        This method filters a list of tools based on their safety flags, removing
        tools that violate the specified safety requirements. It can be used to
        ensure that only safe tools are provided to an agent.
        
        Args:
            tools: List of tools with Sidecar metadata, created by this adapter
            safety_requirements: Dictionary where keys are safety flags (like
                'system_command', 'file_system_access') and values are booleans
                indicating whether the flag must be False to include the tool
            
        Returns:
            List[Any]: Filtered list of tools that meet all safety requirements
            
        Note:
            Tools without safety metadata will be included by default
        """
        filtered_tools = []
        
        for tool in tools:
            safety_flags = getattr(tool, '_sidecar_safety_flags', {})
            
            # Check if tool violates any safety requirements
            violates_safety = False
            for flag, should_be_false in safety_requirements.items():
                if should_be_false and safety_flags.get(flag, False):
                    violates_safety = True
                    break
            
            if not violates_safety:
                filtered_tools.append(tool) # type: ignore
            else:
                logger.debug(f"Filtered out tool {tool.name} due to safety requirements")
        
        return filtered_tools # type: ignore
    
    def fix_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON Schema 'type': ['string', 'null'] to 'anyOf' format.
        
        This helper method fixes a common issue with JSON Schema validation where
        the 'type' field is an array. It converts such schemas to use 'anyOf' which
        is more widely supported by JSON Schema validators.
        
        Args:
            schema: The JSON schema to fix, with potential array 'type' fields
            
        Returns:
            Dict[str, Any]: The fixed JSON schema using 'anyOf' instead of array types
        """
        if isinstance(schema, dict): # type: ignore
            if "type" in schema and isinstance(schema["type"], list):
                schema["anyOf"] = [{"type": t} for t in schema["type"]] # type: ignore
                del schema["type"]  # Remove 'type' and standardize to 'anyOf'
            for key, value in schema.items():
                schema[key] = self.fix_schema(value)  # Apply recursively
        return schema
    
    def _parse_mcp_tool_result(self, tool_result: Any) -> str:
        """Parse the content of a CallToolResult into a string.
        
        This helper method converts the complex structure of an MCP tool result
        into a single string representation. It handles different result types
        including text, image, resource, and blob data.
        
        Args:
            tool_result: The result object from calling an MCP tool
            
        Returns:
            str: A string representation of the tool result content
            
        Note:
            This method has enhanced error handling compared to the base implementation
        """
        # Enhanced version with better error handling and content parsing
        try:
            if hasattr(tool_result, 'isError') and tool_result.isError:
                return f"Tool execution failed: {tool_result.content}"
            
            if not hasattr(tool_result, 'content') or not tool_result.content:
                return "Tool execution returned no content"
            
            decoded_result = ""
            for item in tool_result.content:
                if hasattr(item, 'type'):
                    if item.type == "text":
                        decoded_result += getattr(item, 'text', str(item))
                    elif item.type == "image":
                        decoded_result += getattr(item, 'data', str(item))
                    elif item.type == "resource":
                        resource = getattr(item, 'resource', item)
                        if hasattr(resource, 'text'):
                            decoded_result += resource.text
                        elif hasattr(resource, 'blob'):
                            decoded_result += (
                                resource.blob.decode() if isinstance(resource.blob, bytes) 
                                else str(resource.blob)
                            )
                        else:
                            decoded_result += str(resource)
                    else:
                        decoded_result += str(item)
                else:
                    decoded_result += str(item)
            
            return decoded_result
            
        except Exception as e:
            return f"Error parsing tool result: {str(e)}"
    
    def _convert_resource(self, mcp_resource: Any, connector: Any) -> Any:
        """Convert an MCP resource to LangChain's tool format.
        
        This method converts an MCP resource into a LangChain tool that returns
        the resource content when called. It adds Sidecar metadata to the tool
        for safety analysis and tracking.
        
        Args:
            mcp_resource: The MCP resource object with uri, name, and description
            connector: The MCP connector to use for accessing the resource
            
        Returns:
            Any: A LangChain-compatible tool with Sidecar metadata
        """
        def _sanitize(name: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]+", "_", name).lower().strip("_")
        
        # This would be a full implementation - simplified for now
        # The actual implementation would create a LangChain tool class
        # that calls connector.read_resource(mcp_resource.uri)
        
        class ResourceTool:
            def __init__(self) -> None:
                """Initialize the ResourceTool with metadata."""
                self.name = _sanitize(getattr(mcp_resource, 'name', f"resource_{getattr(mcp_resource, 'uri', 'unknown')}"))
                self.description = getattr(mcp_resource, 'description', f"Return the content of the resource located at URI {getattr(mcp_resource, 'uri', 'unknown')}")
                self._mcp_server = getattr(connector, 'public_identifier', 'unknown')
                self._mcp_original_schema: Dict[str, Any] = {}
                
                # Add Sidecar metadata
                self._sidecar_server_origin = self._mcp_server
                self._sidecar_tool_name = self.name
                self._sidecar_schema_hash = hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()[:16]
                self._sidecar_safety_flags = {
                    'potentially_destructive': False,
                    'file_system_access': True,  # Resources typically access files
                    'network_access': 'http' in str(getattr(mcp_resource, 'uri', '')).lower(),
                    'system_command': False
                }
        
        return ResourceTool()
    
    def _convert_prompt(self, mcp_prompt: Any, connector: Any) -> Any:
        """Convert an MCP prompt to LangChain's tool format.
        
        This method converts an MCP prompt into a LangChain tool that executes
        the prompt when called. It adds Sidecar metadata to the tool for safety
        analysis and tracking.
        
        Args:
            mcp_prompt: The MCP prompt object with name and description
            connector: The MCP connector to use for executing the prompt
            
        Returns:
            Any: A LangChain-compatible tool with Sidecar metadata
        """
        class PromptTool:
            def __init__(self) -> None:
                """Initialize the ResourceTool with metadata."""
                self.name = getattr(mcp_prompt, 'name', 'unknown_prompt')
                self.description = getattr(mcp_prompt, 'description', f"Execute prompt {self.name}")
                self._mcp_server = getattr(connector, 'public_identifier', 'unknown')
                self._mcp_original_schema: Dict[str, Any] = {}
                
                # Add Sidecar metadata
                self._sidecar_server_origin = self._mcp_server
                self._sidecar_tool_name = self.name
                self._sidecar_schema_hash = hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()[:16]
                self._sidecar_safety_flags = {
                    'potentially_destructive': False,
                    'file_system_access': False,
                    'network_access': False,
                    'system_command': False
                }
        
        return PromptTool()
    
    async def create_tools(
        self,
        client: Any,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None
    ) -> List[Any]:  # pyright: ignore[reportUnknownReturnType]
        """
        Create LangChain tools with full resource and prompt support asynchronously.
        
        This async method extends the parent create_tools to also include resources
        and prompts as tools, in addition to the standard MCP tools. It adds Sidecar
        metadata to all tools for safety analysis and tracking.
        
        Args:
            client: MCP client instance with configured servers and sessions
            allowed_tools: Optional list of allowed tool names to filter the tools
            disallowed_tools: Optional list of disallowed tool names to exclude
            
        Returns:
            List[Any]: List of LangChain BaseTool instances with Sidecar metadata
                attached, including resources and prompts as tools
            
        Raises:
            Exception: If the client has no sessions or if tools cannot be created
        """
        # Get standard tools from parent
        import inspect
        
        parent_method = super().create_tools
        if inspect.iscoroutinefunction(parent_method):
            tools = await parent_method(client, allowed_tools, disallowed_tools) # type: ignore
        else:
            tools = parent_method(client, allowed_tools, disallowed_tools) # type: ignore
        
        # Add resource tools
        try:
            if hasattr(client, 'sessions'):
                for server_name, session in client.sessions.items():
                    if hasattr(session, 'list_resources'):
                        try:
                            resources = await session.list_resources()
                            for resource in resources:
                                if allowed_tools is None or resource.name in allowed_tools:
                                    if disallowed_tools is None or resource.name not in disallowed_tools:
                                        resource_tool = self._convert_resource(resource, session)
                                        tools.append(resource_tool) # type: ignore
                        except Exception as e:
                            logger.warning(f"Failed to get resources from {server_name}: {e}")
            
            # Add prompt tools
            if hasattr(client, 'sessions'):
                for server_name, session in client.sessions.items():
                    if hasattr(session, 'list_prompts'):
                        try:
                            prompts = await session.list_prompts()
                            for prompt in prompts:
                                if allowed_tools is None or prompt.name in allowed_tools:
                                    if disallowed_tools is None or prompt.name not in disallowed_tools:
                                        prompt_tool = self._convert_prompt(prompt, session)
                                        tools.append(prompt_tool) # type: ignore
                        except Exception as e:
                            logger.warning(f"Failed to get prompts from {server_name}: {e}")
                            
        except Exception as e:
            logger.warning(f"Failed to add resources/prompts: {e}")
        
        return tools # pyright: ignore[reportUnknownVariableType]
    
    async def _create_tools_from_connectors(self, connectors: List[Any]) -> List[Any]:  # pyright: ignore[reportUnknownReturnType]
        """Create tools from connectors with support for resources and prompts.
        
        This helper method creates tools from a list of connectors, delegating to the
        parent method and then adding Sidecar metadata to all created tools.
        
        Args:
            connectors: List of MCP connectors to create tools from
            
        Returns:
            List[Any]: List of LangChain BaseTool instances with Sidecar metadata
            
        Raises:
            Exception: If tools cannot be created from the connectors
        """
        # This would extend the parent method to include resources and prompts
        # For now, delegate to parent and add metadata
        try:
            tools = await super()._create_tools_from_connectors(connectors) # type: ignore
            
            # Add metadata to each tool
            for tool in tools: # pyright: ignore[reportUnknownVariableType]
                if not hasattr(tool, '_sidecar_schema_hash'): # type: ignore
                    # Add basic sidecar metadata if not already present
                    schema_str = json.dumps(getattr(tool, 'args_schema', {}), sort_keys=True, default=str) # type: ignore
                    tool._sidecar_schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]
                    tool._sidecar_server_origin = getattr(tool, '_mcp_server', 'unknown') # type: ignore
                    tool._sidecar_tool_name = getattr(tool, 'name', 'unknown') # type: ignore
                    tool._sidecar_safety_flags = self._analyze_safety_flags({
                        'name': tool.name, # type: ignore
                        'description': getattr(tool, 'description', '') # type: ignore
                    })
            
            return tools # type: ignore
            
        except Exception as e:
            logger.error(f"Error creating tools from connectors: {e}")
            return []