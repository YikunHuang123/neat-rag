from typing import Any
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from neat_rag.config import settings
from neat_rag.exceptions import LLMProviderError
from neat_rag.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_LLM_PROVIDERS = ["openai", "gemini", "anthropic", "ollama"]


def get_llm() -> Any:
    """
    Return a pydantic-ai model object with API key explicitly injected from
    settings, so .env-sourced keys work without being exported to os.environ.

    Supported providers (LLM_PROVIDER setting):
      "openai"    — OpenAI API
      "gemini"    — Google Gemini
      "anthropic" — Anthropic
      "ollama"    — Local Ollama (OpenAI-compatible)
    """
    provider = settings.LLM_PROVIDER.lower()
    model = settings.LLM_MODEL

    if provider == "openai":
        logger.info("LLM provider: OpenAI", model=model)
        return OpenAIModel(model_name=model, provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY))

    elif provider == "gemini":
        logger.info("LLM provider: Google Gemini", model=model)
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        return GoogleModel(model_name=model, provider=GoogleProvider(api_key=settings.GEMINI_API_KEY))

    elif provider == "anthropic":
        logger.info("LLM provider: Anthropic", model=model)
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        return AnthropicModel(model_name=model, provider=AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY))

    elif provider == "ollama":
        logger.info("LLM provider: Ollama (local)", model=model, base_url=settings.OLLAMA_BASE_URL)
        ollama_provider = OpenAIProvider(
            base_url=f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1",
            api_key=settings.OLLAMA_API_KEY,
        )
        return OpenAIModel(model_name=model, provider=ollama_provider)

    else:
        raise LLMProviderError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Supported options: {SUPPORTED_LLM_PROVIDERS}"
        )
