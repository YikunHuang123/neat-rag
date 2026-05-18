from typing import Any, Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from neat_rag.config import settings
from neat_rag.exceptions import LLMProviderError
from neat_rag.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_LLM_PROVIDERS = ["openai", "gemini", "anthropic", "ollama", "deepseek"]


def get_llm() -> Any:
    """
    Return a pydantic-ai model object with API key explicitly injected from
    settings, so .env-sourced keys work without being exported to os.environ.

    Supported providers (LLM_PROVIDER setting):
      "openai"    — OpenAI API
      "gemini"    — Google Gemini
      "anthropic" — Anthropic
      "ollama"    — Local Ollama (OpenAI-compatible)
      "deepseek"  — DeepSeek API (OpenAI-compatible)
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

    elif provider == "deepseek":
        logger.info("LLM provider: DeepSeek", model=model, base_url=settings.DEEPSEEK_BASE_URL)
        deepseek_provider = OpenAIProvider(
            base_url=f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/v1" if not settings.DEEPSEEK_BASE_URL.endswith("/v1") else settings.DEEPSEEK_BASE_URL,
            api_key=settings.DEEPSEEK_API_KEY,
        )
        return OpenAIModel(model_name=model, provider=deepseek_provider)

    else:
        raise LLMProviderError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Supported options: {SUPPORTED_LLM_PROVIDERS}"
        )


def get_llm_with_override(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
) -> Any:
    """
    Create an LLM instance with explicitly provided credentials.
    Falls back to settings values when api_key / base_url are empty.
    Used for per-request model overrides from the UI.
    """
    p = provider.lower().strip()
    logger.info("LLM override active", provider=p, model=model)

    if p == "openai":
        key = api_key or settings.OPENAI_API_KEY
        return OpenAIModel(model_name=model, provider=OpenAIProvider(api_key=key))

    elif p == "gemini":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        key = api_key or settings.GEMINI_API_KEY
        return GoogleModel(model_name=model, provider=GoogleProvider(api_key=key))

    elif p == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        key = api_key or settings.ANTHROPIC_API_KEY
        return AnthropicModel(model_name=model, provider=AnthropicProvider(api_key=key))

    elif p == "ollama":
        url = base_url or settings.OLLAMA_BASE_URL
        return OpenAIModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{url.rstrip('/')}/v1",
                api_key="ollama",
            ),
        )

    elif p == "deepseek":
        key = api_key or settings.DEEPSEEK_API_KEY
        url = base_url or settings.DEEPSEEK_BASE_URL
        base = f"{url.rstrip('/')}/v1" if not url.rstrip("/").endswith("/v1") else url
        return OpenAIModel(model_name=model, provider=OpenAIProvider(base_url=base, api_key=key))

    elif p == "custom":
        if not base_url:
            raise LLMProviderError("Custom provider requires a Base URL.")
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return OpenAIModel(
            model_name=model,
            provider=OpenAIProvider(base_url=base, api_key=api_key or "custom"),
        )

    else:
        raise LLMProviderError(
            f"Unsupported override provider '{p}'. "
            f"Supported: {SUPPORTED_LLM_PROVIDERS + ['custom']}"
        )


def get_langchain_llm(provider: Optional[str] = None, model: Optional[str] = None) -> Any:
    """
    Return a LangChain-compatible ChatModel instance (BaseChatModel).
    Used for evaluation (ragas) and other LangChain-based utilities.

    Supported providers: "openai", "gemini", "deepseek", "ollama".
    """
    p = (provider or settings.EVAL_LLM_PROVIDER).lower()
    m = model or settings.EVAL_LLM_MODEL

    if p == "openai":
        from langchain_openai import ChatOpenAI
        logger.info("LangChain LLM: OpenAI", model=m)
        return ChatOpenAI(model=m, api_key=settings.OPENAI_API_KEY)

    elif p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("LangChain LLM: Gemini", model=m)
        # Note: Gemini OpenAI-compatible base_url can also be used via ChatOpenAI
        # but ChatGoogleGenerativeAI is the native LangChain way.
        return ChatGoogleGenerativeAI(model=m, google_api_key=settings.GEMINI_API_KEY)

    elif p == "deepseek":
        from langchain_openai import ChatOpenAI
        logger.info("LangChain LLM: DeepSeek", model=m)
        return ChatOpenAI(
            model=m,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL.rstrip("/") + "/v1" if not settings.DEEPSEEK_BASE_URL.endswith("/v1") else settings.DEEPSEEK_BASE_URL,
        )

    elif p == "ollama":
        from langchain_openai import ChatOpenAI
        logger.info("LangChain LLM: Ollama", model=m)
        return ChatOpenAI(
            model=m,
            api_key=settings.OLLAMA_API_KEY,
            base_url=f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1",
        )

    else:
        raise LLMProviderError(f"Unsupported LangChain provider '{p}'")
