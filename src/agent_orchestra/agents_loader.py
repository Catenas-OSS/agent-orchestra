"""Agents configuration loader and validator.

Handles loading, validation, and canonicalization of agent specifications.
"""

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class AgentsLoaderError(Exception):
    """Error loading or validating agents configuration."""
    pass


class AgentsLoaderResult:
    """Result from loading agents."""

    def __init__(
        self,
        agents_list: list[dict[str, Any]],
        canonical_json: dict[str, Any],
        by_id: dict[str, dict[str, Any]]
    ):
        self.agents_list = agents_list
        self.canonical_json = canonical_json
        self.by_id = by_id


class AgentsLoader:
    """Loads and validates agent specifications."""

    def __init__(self, schema_path: str | Path | None = None):
        """Initialize agents loader.
        
        Args:
            schema_path: Path to agents.schema.json, defaults to bundled schema
        """
        if schema_path is None:
            # Default to schema in package
            package_dir = Path(__file__).parent.parent.parent
            schema_path = package_dir / "schemas" / "agents.schema.json"

        self.schema_path = Path(schema_path)
        self._schema: dict[str, Any] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        """Load and cache the JSON schema."""
        if self._schema is None:
            if not self.schema_path.exists():
                raise AgentsLoaderError(f"Schema file not found: {self.schema_path}")

            try:
                with open(self.schema_path) as f:
                    self._schema = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise AgentsLoaderError(f"Failed to load schema: {e}")

        return self._schema

    def load(self, agents_path: str | Path) -> AgentsLoaderResult:
        """Load agents from directory or file.
        
        Args:
            agents_path: Path to directory containing .yaml/.yml/.json files, or single file
            
        Returns:
            AgentsLoaderResult with agents_list, canonical_json, by_id
            
        Raises:
            AgentsLoaderError: If loading or validation fails
        """
        agents_path = Path(agents_path)

        if not agents_path.exists():
            raise AgentsLoaderError(f"Agents path not found: {agents_path}")

        # Collect all agent files
        agent_files = []
        if agents_path.is_file():
            agent_files = [agents_path]
        elif agents_path.is_dir():
            # Find all .yaml/.yml/.json files
            for pattern in ["*.yaml", "*.yml", "*.json"]:
                agent_files.extend(agents_path.glob(pattern))

            if not agent_files:
                raise AgentsLoaderError(f"No agent files found in directory: {agents_path}")
        else:
            raise AgentsLoaderError(f"Invalid path: {agents_path}")

        # Load and merge all agents
        all_agents = []
        seen_ids = {}

        for file_path in sorted(agent_files):
            agents_from_file = self._load_file(file_path)

            for agent in agents_from_file:
                agent_id = agent.get("id")
                if not agent_id:
                    raise AgentsLoaderError(f"Agent missing 'id' in file: {file_path}")

                # Check for duplicate IDs
                if agent_id in seen_ids:
                    raise AgentsLoaderError(
                        f"Duplicate agent ID '{agent_id}' found in {file_path} "
                        f"(previously defined in {seen_ids[agent_id]})"
                    )

                seen_ids[agent_id] = file_path
                all_agents.append(agent)

        if not all_agents:
            raise AgentsLoaderError("No agents found in any files")

        # Expand environment variables
        all_agents = self._expand_variables(all_agents)

        # Validate against schema
        self._validate(all_agents)

        # Additional validation checks
        self._validate_edge_cases(all_agents)

        # Create return objects
        canonical_json = self._canonicalize({"agents": all_agents})
        by_id = {agent["id"]: agent for agent in all_agents}

        return AgentsLoaderResult(all_agents, canonical_json, by_id)

    def validate(self, agents_path: str | Path) -> None:
        """Validate agents configuration.
        
        Args:
            agents_path: Path to directory or file
            
        Raises:
            AgentsLoaderError: If validation fails
        """
        self.load(agents_path)  # Will raise if invalid

    def _load_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Load agents from a single file.
        
        Args:
            file_path: Path to agent file
            
        Returns:
            List of agent specifications
        """
        try:
            if file_path.suffix.lower() in {'.yaml', '.yml'}:
                if not YAML_AVAILABLE:
                    raise AgentsLoaderError(
                        "PyYAML not installed. Install with: pip install PyYAML"
                    )
                with open(file_path) as f:
                    content = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path) as f:
                    content = json.load(f)
            else:
                raise AgentsLoaderError(
                    f"Unsupported file format: {file_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )
        except (yaml.YAMLError, json.JSONDecodeError, OSError) as e:
            raise AgentsLoaderError(f"Failed to parse {file_path}: {e}")

        # Handle both single agent and list of agents
        if isinstance(content, dict):
            # Single agent in file
            return [content]
        elif isinstance(content, list):
            # List of agents
            return content
        else:
            raise AgentsLoaderError(
                f"File {file_path} must contain either an agent object or list of agents, "
                f"got {type(content).__name__}"
            )

    def _expand_variables(self, agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expand environment variables and home directory in agent configs.
        
        Args:
            agents: List of agent specifications
            
        Returns:
            Agents with expanded variables
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

        return _expand_recursive(agents)

    def _validate(self, agents: list[dict[str, Any]]) -> None:
        """Validate agents against JSON schema.
        
        Args:
            agents: List of agent specifications
            
        Raises:
            AgentsLoaderError: If validation fails
        """
        if not JSONSCHEMA_AVAILABLE:
            # Skip schema validation if jsonschema not available
            return

        try:
            jsonschema.validate(agents, self.schema)
        except jsonschema.ValidationError as e:
            raise AgentsLoaderError(f"Schema validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            raise AgentsLoaderError(f"Invalid schema: {e.message}")

    def _validate_edge_cases(self, agents: list[dict[str, Any]]) -> None:
        """Additional validation checks beyond schema.
        
        Args:
            agents: List of agent specifications
            
        Raises:
            AgentsLoaderError: If validation fails
        """
        for agent in agents:
            agent_id = agent.get("id", "unknown")

            # Validate ID pattern
            if not agent_id or "@v" not in agent_id:
                raise AgentsLoaderError(
                    f"Agent '{agent_id}' has invalid ID format. "
                    "Must be 'name@vN' (e.g., 'researcher@v1')"
                )

            # Validate model has provider prefix
            model = agent.get("model", "")
            if ":" not in model:
                raise AgentsLoaderError(
                    f"Agent '{agent_id}' model '{model}' missing provider prefix. "
                    "Use format 'provider:model' (e.g., 'openai:gpt-4o-mini')"
                )

            # Validate no overlap between allowed and disallowed tools
            allowed_tools = set(agent.get("allowed_tools", []))
            disallowed_tools = set(agent.get("disallowed_tools", []))

            overlap = allowed_tools & disallowed_tools
            if overlap:
                raise AgentsLoaderError(
                    f"Agent '{agent_id}' has overlapping allowed and disallowed tools: "
                    f"{', '.join(sorted(overlap))}"
                )

    def _canonicalize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize agents data for consistent hashing.
        
        Args:
            data: Agents data to canonicalize
            
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

        return _canonicalize_recursive(data)


def load_agents_config(agents_path: str | Path) -> AgentsLoaderResult:
    """Convenience function to load agents configuration.
    
    Args:
        agents_path: Path to agents directory or file
        
    Returns:
        AgentsLoaderResult with loaded and validated agents
        
    Raises:
        AgentsLoaderError: If loading fails
    """
    loader = AgentsLoader()
    return loader.load(agents_path)


def validate_agents_config(agents_path: str | Path) -> None:
    """Convenience function to validate agents configuration.
    
    Args:
        agents_path: Path to agents directory or file
        
    Raises:
        AgentsLoaderError: If validation fails
    """
    loader = AgentsLoader()
    loader.validate(agents_path)
