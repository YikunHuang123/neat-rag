"""
Reranker providers for Phase 5 advanced retrieval.

Supported providers (RERANKER_PROVIDER setting):
  "bge"    — BAAI/bge-reranker-v2-m3 via sentence-transformers (local)
  "cohere" — Cohere Rerank API (cloud)
"""
import asyncio
from typing import Any

from neat_rag.config import settings
from neat_rag.exceptions import RerankerProviderError
from neat_rag.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_RERANKER_PROVIDERS = ["bge", "cohere"]


class CrossEncoderReranker:
    """
    Local cross-encoder reranker using sentence-transformers CrossEncoder.
    The model is downloaded and loaded lazily on first use (~500 MB for bge-reranker-v2-m3).
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            logger.info("Loading cross-encoder reranker", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder reranker ready", model=self._model_name)
        except ImportError as e:
            raise RerankerProviderError(
                "sentence-transformers is required for RERANKER_PROVIDER='bge'. "
                "Install it with: pip install sentence-transformers"
            ) from e

    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Score (query, passage) pairs. Returns scores aligned with `passages` order."""
        self._load()
        pairs = [(query, p) for p in passages]
        # CrossEncoder.predict() is synchronous — offload to a thread pool.
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self._model.predict, pairs)
        return [float(s) for s in scores]


class CohereReranker:
    """Reranker backed by the Cohere Rerank API."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self._model_name = model_name
        self._api_key = api_key

    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Score passages via the Cohere API. Returns scores aligned with `passages` order."""
        try:
            import cohere  # type: ignore[import]
        except ImportError as e:
            raise RerankerProviderError(
                "cohere package is required for RERANKER_PROVIDER='cohere'. "
                "Install it with: pip install cohere"
            ) from e

        client = cohere.AsyncClientV2(api_key=self._api_key)
        response = await client.rerank(
            model=self._model_name,
            query=query,
            documents=passages,
        )
        scores = [0.0] * len(passages)
        for result in response.results:
            scores[result.index] = float(result.relevance_score)
        return scores


def get_reranker() -> CrossEncoderReranker | CohereReranker:
    """Return a reranker instance based on RERANKER_PROVIDER setting."""
    provider = settings.RERANKER_PROVIDER.lower()
    model = settings.RERANKER_MODEL

    if provider == "bge":
        return CrossEncoderReranker(model_name=model)

    if provider == "cohere":
        if not settings.COHERE_API_KEY:
            raise RerankerProviderError(
                "RERANKER_PROVIDER='cohere' requires COHERE_API_KEY to be set."
            )
        return CohereReranker(model_name=model, api_key=settings.COHERE_API_KEY)

    raise RerankerProviderError(
        f"Unsupported RERANKER_PROVIDER '{provider}'. "
        f"Supported options: {SUPPORTED_RERANKER_PROVIDERS}"
    )
