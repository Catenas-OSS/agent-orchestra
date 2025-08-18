"""LLM factory for provider-agnostic model instantiation.

Handles dynamic provider resolution and LangChain model creation.
"""

import os
from typing import Any


class LLMFactoryError(Exception):
    """Error creating LLM instance."""
    pass


class LLMFactory:
    """Factory for creating LangChain LLM instances."""

    def get_llm(self, model_id: str, config: dict[str, Any] | None = None) -> Any:
        """Create LangChain LLM instance from model ID and config.
        
        Args:
            model_id: Model identifier with provider prefix (e.g., 'openai:gpt-4o-mini')
            config: Optional provider-specific configuration overrides
            
        Returns:
            LangChain ChatModel instance
            
        Raises:
            LLMFactoryError: If provider package missing, API key missing, or model invalid
        """
        if config is None:
            config = {}

        # Parse provider prefix
        if ":" not in model_id:
            raise LLMFactoryError(
                f"Model ID '{model_id}' missing provider prefix. "
                "Use format 'provider:model' (e.g., 'openai:gpt-4o-mini')"
            )

        provider, model_name = model_id.split(":", 1)

        if provider == "openai":
            return self._create_openai_llm(model_name, config)
        elif provider == "anthropic":
            return self._create_anthropic_llm(model_name, config)
        elif provider == "groq":
            return self._create_groq_llm(model_name, config)
        else:
            raise LLMFactoryError(
                f"Unsupported provider '{provider}'. "
                "Supported providers: openai, anthropic, groq"
            )

    def _create_openai_llm(self, model_name: str, config: dict[str, Any]) -> Any:
        """Create OpenAI LLM instance.
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o-mini')
            config: Configuration overrides
            
        Returns:
            ChatOpenAI instance
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise LLMFactoryError(
                "OpenAI provider not available. Install with: "
                "pip install 'agent-orchestra[openai]' or pip install langchain-openai"
            )

        # Check for API key
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMFactoryError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide 'api_key' in agent config"
            )

        # Create model with config overrides
        model_config = {
            "model": model_name,
            "api_key": api_key,
            **config
        }

        try:
            return ChatOpenAI(**model_config)
        except Exception as e:
            raise LLMFactoryError(f"Failed to create OpenAI model '{model_name}': {e}")

    def _create_anthropic_llm(self, model_name: str, config: dict[str, Any]) -> Any:
        """Create Anthropic LLM instance.
        
        Args:
            model_name: Anthropic model name (e.g., 'claude-3-5-sonnet')
            config: Configuration overrides
            
        Returns:
            ChatAnthropic instance
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise LLMFactoryError(
                "Anthropic provider not available. Install with: "
                "pip install 'agent-orchestra[anthropic]' or pip install langchain-anthropic"
            )

        # Check for API key
        api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMFactoryError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or provide 'api_key' in agent config"
            )

        # Create model with config overrides
        model_config = {
            "model": model_name,
            "api_key": api_key,
            **config
        }

        try:
            return ChatAnthropic(**model_config)
        except Exception as e:
            raise LLMFactoryError(f"Failed to create Anthropic model '{model_name}': {e}")

    def _create_groq_llm(self, model_name: str, config: dict[str, Any]) -> Any:
        """Create Groq LLM instance.
        
        Args:
            model_name: Groq model name (e.g., 'llama3-70b')
            config: Configuration overrides
            
        Returns:
            ChatGroq instance
        """
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise LLMFactoryError(
                "Groq provider not available. Install with: "
                "pip install 'agent-orchestra[groq]' or pip install langchain-groq"
            )

        # Check for API key
        api_key = config.get("api_key") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMFactoryError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or provide 'api_key' in agent config"
            )

        # Create model with config overrides
        model_config = {
            "model": model_name,
            "api_key": api_key,
            **config
        }

        try:
            return ChatGroq(**model_config)
        except Exception as e:
            raise LLMFactoryError(f"Failed to create Groq model '{model_name}': {e}")


def get_llm(model_id: str, config: dict[str, Any] | None = None) -> Any:
    """Convenience function to create LLM instance.
    
    Args:
        model_id: Model identifier with provider prefix
        config: Optional configuration overrides
        
    Returns:
        LangChain ChatModel instance
        
    Raises:
        LLMFactoryError: If creation fails
    """
    factory = LLMFactory()
    return factory.get_llm(model_id, config)
