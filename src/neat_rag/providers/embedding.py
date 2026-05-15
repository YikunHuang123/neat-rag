from typing import List, Optional, Callable
import openai

from neat_rag.config import settings, SUPPORTED_EMBEDDING_PROVIDERS
from neat_rag.exceptions import EmbeddingProviderError
from neat_rag.logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 100  # OpenAI allows up to 2048; 100 keeps retries cheap


class OpenAIEmbedder:
    """
    Embeds text using any OpenAI-compatible embeddings endpoint.
    Works with OpenAI, Gemini (via /v1beta/openai/), and local models.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, dimensions: Optional[int] = None):
        self.model = model
        self.dimensions = dimensions
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts. Handles internal batching to stay under API limits.
        Returns vectors in the same order as input.
        """
        if not texts:
            return []

        extra_kwargs = {}
        if self.dimensions is not None:
            extra_kwargs["dimensions"] = self.dimensions

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    **extra_kwargs,
                )
                all_vectors.extend(item.embedding for item in response.data)
                logger.debug("Embedded batch", model=self.model, batch_start=i, batch_size=len(batch))
            except Exception as e:
                logger.error("Embedding batch failed", model=self.model, batch_start=i, error=str(e))
                raise EmbeddingProviderError(f"Embedding failed: {e}") from e

        return all_vectors

    async def embed_one(self, text: str) -> List[float]:
        """Convenience wrapper for a single text."""
        return (await self.embed([text]))[0]


# ---------------------------------------------------------------------------
# Provider registry
#
# To add a new provider: add ONE entry here. get_embedder() and get_langchain_embedder() require no changes.
#
# Each entry has two zero-argument factory callables (lazy — settings are read at call time, not at module import time):
#   "embedder"  → OpenAIEmbedder       used by the retrieval pipeline
#   "langchain" → LangChain Embeddings  used by SemanticChunker
# ---------------------------------------------------------------------------

def _lc_openai_compat(api_key: str, base_url: Optional[str], check_len: bool):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=settings.CHUNKING_MODEL,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=check_len,
    )


def _lc_gemini():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    model = settings.CHUNKING_MODEL
    if not model.startswith("models/"):
        model = f"models/{model}"
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.GEMINI_API_KEY)


def _embedder_custom() -> OpenAIEmbedder:
    if not settings.EMBEDDING_BASE_URL:
        raise EmbeddingProviderError("EMBEDDING_PROVIDER=custom requires EMBEDDING_BASE_URL to be set.")
    return OpenAIEmbedder(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY or "none",
        base_url=settings.EMBEDDING_BASE_URL,
    )


_REGISTRY: dict[str, dict[str, Callable]] = {
    "openai": {
        "embedder":  lambda: OpenAIEmbedder(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY),
        "langchain": lambda: _lc_openai_compat(settings.OPENAI_API_KEY, None, True),
    },
    "gemini": {
        "embedder":  lambda: OpenAIEmbedder(model=settings.EMBEDDING_MODEL, api_key=settings.GEMINI_API_KEY, base_url=settings.GEMINI_BASE_URL, dimensions=settings.EMBEDDING_DIM),
        "langchain": _lc_gemini,
    },
    "ollama": {
        "embedder":  lambda: OpenAIEmbedder(model=settings.EMBEDDING_MODEL, api_key=settings.OLLAMA_API_KEY, base_url=f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1"),
        "langchain": lambda: _lc_openai_compat(settings.OLLAMA_API_KEY, f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1", False),
    },
    "custom": {
        "embedder":  _embedder_custom,
        "langchain": lambda: _lc_openai_compat(settings.CHUNKING_API_KEY, settings.CHUNKING_BASE_URL, False),
    },
}

# ---------------------------------------------------------------------------

def get_embedder(provider: Optional[str] = None) -> OpenAIEmbedder:
    """Return an OpenAIEmbedder for the given provider (defaults to EMBEDDING_PROVIDER)."""
    key = (provider or settings.EMBEDDING_PROVIDER).lower()
    if key not in _REGISTRY:
        raise EmbeddingProviderError(
            f"Unsupported EMBEDDING_PROVIDER '{key}'. Supported options are: {SUPPORTED_EMBEDDING_PROVIDERS}"
        )
    return _REGISTRY[key]["embedder"]()


def get_langchain_embedder():
    """Return a LangChain Embeddings object for SemanticChunker (uses CHUNKING_PROVIDER)."""
    key = settings.CHUNKING_PROVIDER.lower()
    if key not in _REGISTRY:
        raise EmbeddingProviderError(
            f"Unsupported CHUNKING_PROVIDER '{key}'. Supported options are: {SUPPORTED_EMBEDDING_PROVIDERS}"
        )
    return _REGISTRY[key]["langchain"]()
