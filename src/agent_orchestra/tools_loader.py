"""Tools configuration loader and validator.

Handles loading, validation, and canonicalization of MCP tools configurations.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    jsonschema = None


class ToolsLoaderError(Exception):
    """Error loading or validating tools configuration.
    
    Raised when tools configuration files cannot be loaded,
    parsed, or validated against the schema.
    """
    pass


class ToolsLoader:
    """Loads and validates MCP tools configurations.
    
    Handles loading tools configuration files in YAML or JSON format,
    validates them against the schema, and provides canonicalization
    for consistent hashing and comparison.
    """

    def __init__(self, schema_path: str | Path | None = None):
        """Initialize tools loader.
        
        Args:
            schema_path: Path to tools.schema.json, defaults to bundled schema
        """
        if schema_path is None:
            # Default to schema in package
            package_dir = Path(__file__).parent.parent.parent
            schema_path = package_dir / "schemas" / "tools.schema.json"

        self.schema_path = Path(schema_path)
        self._schema: dict[str, Any] | None = None

    def _load_schema(self) -> dict[str, Any]:
        """Load the JSON schema from file.
        
        Returns:
            The loaded schema dictionary
            
        Raises:
            ToolsLoaderError: If schema file not found or invalid
        """
        if not self.schema_path.exists():
            raise ToolsLoaderError(f"Schema file not found: {self.schema_path}")

        try:
            with open(self.schema_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ToolsLoaderError(f"Failed to load schema: {e}")

    @property
    def schema(self) -> dict[str, Any]:
        """Load and cache the JSON schema.
        
        Returns:
            The tools configuration JSON schema
            
        Raises:
            ToolsLoaderError: If schema file not found or invalid
        """
        if self._schema is None:
            self._schema = self._load_schema()
        return self._schema

    def load(self, config_path: str | Path) -> dict[str, Any]:
        """Load tools configuration from file.
        
        Args:
            config_path: Path to .yaml/.yml/.json config file
            
        Returns:
            Dict shaped for MCPClient.from_dict()
            
        Raises:
            ToolsLoaderError: If loading or validation fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ToolsLoaderError(f"Tools config file not found: {config_path}")

        # Load file based on extension
        config: Any
        try:
            if config_path.suffix.lower() in {'.yaml', '.yml'}:
                if not YAML_AVAILABLE or yaml is None:
                    raise ToolsLoaderError(
                        "PyYAML not installed. Install with: pip install PyYAML"
                    )
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path) as f:
                    config = json.load(f)
            else:
                raise ToolsLoaderError(
                    f"Unsupported file format: {config_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )
        except Exception as e:
            # Handle both YAML and JSON parsing errors, plus file I/O errors
            if YAML_AVAILABLE and yaml is not None:
                if hasattr(yaml, 'YAMLError') and isinstance(e, yaml.YAMLError):
                    raise ToolsLoaderError(f"Failed to parse {config_path}: {e}")
            if isinstance(e, (json.JSONDecodeError, OSError)):
                raise ToolsLoaderError(f"Failed to parse {config_path}: {e}")
            # Re-raise ToolsLoaderError as-is
            if isinstance(e, ToolsLoaderError):
                raise
            # Catch-all for other exceptions
            raise ToolsLoaderError(f"Failed to parse {config_path}: {e}")

        if not isinstance(config, dict):
            raise ToolsLoaderError(f"Config must be an object, got {type(config).__name__}")

        # At this point, config is guaranteed to be a dict
        config_dict: dict[str, Any] = config

        # Expand environment variables and paths
        config_dict = self._expand_variables(config_dict)

        # Validate against schema
        self._validate(config_dict)

        # Additional validation checks
        self._validate_edge_cases(config_dict)

        return config_dict

    def validate(self, config_path: str | Path) -> None:
        """Validate tools configuration file.
        
        Loads and validates the configuration file without returning
        the loaded configuration. Useful for validation-only checks.
        
        Args:
            config_path: Path to config file
            
        Raises:
            ToolsLoaderError: If validation fails
        """
        self.load(config_path)  # Will raise if invalid

    def canonicalize(self, config: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize config for consistent hashing.
        
        Transforms configuration into a canonical form with sorted keys
        and normalized string values for consistent hashing and comparison.
        
        Args:
            config: Tools configuration dict
            
        Returns:
            Canonical dict with sorted keys and normalized values
        """
        def _canonicalize_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                # Sort keys and recursively canonicalize values
                return {k: _canonicalize_recursive(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [_canonicalize_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Normalize line endings and strip trailing whitespace
                return obj.replace('\r\n', '\n').replace('\r', '\n').rstrip()
            else:
                return obj

        result = _canonicalize_recursive(config)
        # Ensure we return a dict[str, Any]
        if not isinstance(result, dict):
            raise ToolsLoaderError("Canonicalization failed to produce a dictionary")
        return result

    def _expand_variables(self, config: dict[str, Any]) -> dict[str, Any]:
        """Expand environment variables and home directory in config.
        
        Recursively processes the configuration to expand environment
        variables (${VAR}) and home directory paths (~) in string values.
        
        Args:
            config: Raw configuration dict
            
        Returns:
            Configuration with expanded variables
        """
        def _expand_string(value: str) -> str:
            # Expand environment variables ${VAR} and home directory ~
            expanded = os.path.expandvars(value)
            expanded = os.path.expanduser(expanded)
            return expanded

        def _expand_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _expand_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_expand_recursive(item) for item in obj]
            elif isinstance(obj, str):
                return _expand_string(obj)
            else:
                return obj

        result = _expand_recursive(config)
        # Ensure we return a dict[str, Any]
        if not isinstance(result, dict):
            raise ToolsLoaderError("Variable expansion failed to produce a dictionary")
        return result

    def _validate(self, config: dict[str, Any]) -> None:
        """Validate config against JSON schema.
        
        Validates the configuration against the tools schema if
        jsonschema is available. Skips validation if not installed.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ToolsLoaderError: If validation fails
        """
        if not JSONSCHEMA_AVAILABLE or jsonschema is None:
            # Skip schema validation if jsonschema not available
            return

        try:
            jsonschema.validate(config, self.schema)
        except jsonschema.ValidationError as e:
            raise ToolsLoaderError(f"Schema validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            raise ToolsLoaderError(f"Invalid schema: {e.message}")

    def _validate_edge_cases(self, config: dict[str, Any]) -> None:
        """Additional validation checks beyond schema.
        
        Performs domain-specific validation that cannot be expressed
        in JSON schema, such as server name uniqueness and transport validation.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ToolsLoaderError: If validation fails
        """
        # Check mcpServers exists and is not empty
        mcp_servers = config.get("mcpServers", {})
        if not mcp_servers:
            raise ToolsLoaderError(
                "mcpServers must be present and contain at least one server"
            )

        # Check for duplicate server names (case insensitive)
        server_names = list(mcp_servers.keys())
        server_names_lower = [name.lower() for name in server_names]
        if len(server_names_lower) != len(set(server_names_lower)):
            raise ToolsLoaderError("Duplicate server names found (case insensitive)")

        # Validate each server configuration
        for name, server_config in mcp_servers.items():
            self._validate_server(name, server_config)

    def _validate_server(self, name: str, config: dict[str, Any]) -> None:
        """Validate individual server configuration.
        
        Checks that each server has valid transport configuration
        (either stdio or HTTP/SSE), validates commands exist on PATH,
        and validates URL formats.
        
        Args:
            name: Server name
            config: Server configuration
            
        Raises:
            ToolsLoaderError: If validation fails
        """
        # Check for exactly one transport type
        has_url = "url" in config
        has_command = "command" in config

        if has_url and has_command:
            raise ToolsLoaderError(
                f"Server '{name}' cannot have both 'url' and 'command'"
            )

        if not has_url and not has_command:
            raise ToolsLoaderError(
                f"Server '{name}' must have either 'url' or 'command'"
            )

        # Validate stdio transport
        if has_command:
            command = config["command"]

            # Check if command exists on PATH
            if not shutil.which(command):
                # Provide helpful suggestions
                suggestions = []
                if command in {"npx", "npm"}:
                    suggestions.append("Install Node.js: https://nodejs.org/")
                elif command in {"uvx", "uv"}:
                    suggestions.append("Install uv: pip install uv")
                elif command.startswith("python"):
                    suggestions.append("Ensure Python is in PATH")

                suggestion_text = ""
                if suggestions:
                    suggestion_text = f" Suggestions: {'; '.join(suggestions)}"

                raise ToolsLoaderError(
                    f"Server '{name}': command '{command}' not found on PATH.{suggestion_text}"
                )

        # Validate HTTP/SSE transport
        if has_url:
            url = config["url"]
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError("Invalid URL format")
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError("URL must use http or https scheme")
            except ValueError as e:
                raise ToolsLoaderError(
                    f"Server '{name}': invalid URL '{url}': {e}"
                )


def load_tools_config(config_path: str | Path) -> dict[str, Any]:
    """Convenience function to load tools configuration.
    
    Creates a ToolsLoader instance and loads the specified configuration file.
    
    Args:
        config_path: Path to tools config file
        
    Returns:
        Configuration dict shaped for MCPClient.from_dict()
        
    Raises:
        ToolsLoaderError: If loading fails
    """
    loader = ToolsLoader()
    return loader.load(config_path)


def validate_tools_config(config_path: str | Path) -> None:
    """Convenience function to validate tools configuration.
    
    Creates a ToolsLoader instance and validates the specified configuration file.
    
    Args:
        config_path: Path to tools config file
        
    Raises:
        ToolsLoaderError: If validation fails
    """
    loader = ToolsLoader()
    loader.validate(config_path)
