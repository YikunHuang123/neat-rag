from typing import List, Optional
import openai

from neat_rag.config import settings
from neat_rag.exceptions import EmbeddingProviderError
from neat_rag.logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 100  # OpenAI allows up to 2048; 100 keeps retries cheap


class OpenAIEmbedder:
    """
    Embeds text using any OpenAI-compatible embeddings endpoint.
    Works with OpenAI, Gemini (via /v1beta/openai/), and local models.
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts. Handles internal batching to stay under API limits.
        Returns vectors in the same order as input.
        """
        if not texts:
            return []

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                response = await self._client.embeddings.create(
                    model=self.model,
                    input=batch,
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


def get_embedder(provider: Optional[str] = None) -> OpenAIEmbedder:
    """
    Return an embedder for the given provider name.
    Supported: "openai" (default), "gemini".
    Reads model name and API keys from settings.
    """
    provider = (provider or "openai").lower()

    if provider == "gemini":
        return OpenAIEmbedder(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    return OpenAIEmbedder(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
