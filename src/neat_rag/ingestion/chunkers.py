from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from neat_rag.exceptions import ChunkingError
from neat_rag.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RawChunk:
    """
    Intermediate chunk produced by chunkers before document_id / UUID are assigned.
    The pipeline converts these into Chunk domain models.
    """
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecursiveChunker:
    """
    Splits text using LangChain's RecursiveCharacterTextSplitter.
    Uses add_start_index=True so start_char is accurate (fixes original bug).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[RawChunk]:
        if not content.strip():
            return []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,  # puts 'start_index' in each chunk's metadata
            )
            docs = splitter.create_documents([content], metadatas=[metadata or {}])

            result = []
            for doc in docs:
                text = doc.page_content.strip()
                if len(text) < self.min_chunk_size:
                    continue
                start = doc.metadata.get("start_index", 0)
                # Strip 'start_index' from metadata — it's promoted to a first-class field
                chunk_meta = {k: v for k, v in doc.metadata.items() if k != "start_index"}
                chunk_meta["chunk_method"] = "recursive"
                result.append(RawChunk(
                    content=text,
                    start_char=start,
                    end_char=start + len(text),
                    metadata=chunk_meta,
                ))
            return result
        except Exception as e:
            logger.error("Recursive chunking failed", error=str(e))
            raise ChunkingError(f"Recursive chunking failed: {e}") from e


class SemanticChunker:
    """
    Splits text using LangChain's SemanticChunker (embedding-based breakpoints).
    Falls back to RecursiveChunker on any failure.
    """

    def __init__(
        self,
        breakpoint_threshold_type: str = "percentile",
        min_chunk_size: int = 100,
    ):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.min_chunk_size = min_chunk_size

    def chunk(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[RawChunk]:
        if not content.strip():
            return []
        try:
            from langchain_experimental.text_splitter import SemanticChunker as _LCSemanticChunker
            from langchain_openai import OpenAIEmbeddings
            from neat_rag.config import settings

            embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
            splitter = _LCSemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
            )
            docs = splitter.create_documents([content], metadatas=[metadata or {}])

            result = []
            cursor = 0
            for doc in docs:
                text = doc.page_content.strip()
                if len(text) < self.min_chunk_size:
                    cursor += len(doc.page_content)
                    continue
                # SemanticChunker doesn't expose start_index; scan from cursor
                start = content.find(text, cursor)
                if start == -1:
                    start = cursor
                end = start + len(text)
                cursor = end
                chunk_meta = {**doc.metadata, "chunk_method": "semantic"}
                result.append(RawChunk(
                    content=text,
                    start_char=start,
                    end_char=end,
                    metadata=chunk_meta,
                ))
            return result

        except Exception as e:
            logger.warning("Semantic chunking failed, falling back to recursive", error=str(e))
            fallback = RecursiveChunker(min_chunk_size=self.min_chunk_size)
            chunks = fallback.chunk(content, metadata)
            for c in chunks:
                c.metadata["chunk_method"] = "fallback"
            return chunks


class TokenBasedChunker:
    """
    Splits text by token count using tiktoken.
    Ensures no chunk exceeds max_tokens, preventing embedding model context overflow.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        encoding_name: str = "cl100k_base",
        min_chunk_size: int = 50,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding_name = encoding_name
        self.min_chunk_size = min_chunk_size

    def chunk(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[RawChunk]:
        if not content.strip():
            return []
        try:
            import tiktoken
            enc = tiktoken.get_encoding(self.encoding_name)
            tokens = enc.encode(content)

            result = []
            step = max(1, self.max_tokens - self.overlap_tokens)
            char_cursor = 0

            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + self.max_tokens]
                text = enc.decode(chunk_tokens).strip()
                if len(text) < self.min_chunk_size:
                    continue
                start = content.find(text, char_cursor)
                if start == -1:
                    start = char_cursor
                end = start + len(text)
                char_cursor = start  # overlap means next chunk may start before end
                chunk_meta = {
                    **(metadata or {}),
                    "chunk_method": "token",
                    "token_count": len(chunk_tokens),
                }
                result.append(RawChunk(
                    content=text,
                    start_char=start,
                    end_char=end,
                    metadata=chunk_meta,
                ))
            return result
        except Exception as e:
            logger.error("Token-based chunking failed", error=str(e))
            raise ChunkingError(f"Token-based chunking failed: {e}") from e


def get_chunker(
    strategy: str = "recursive",
    **kwargs,
) -> "RecursiveChunker | SemanticChunker | TokenBasedChunker":
    """
    Return a chunker by strategy name.
    Strategies: "recursive" (default), "semantic", "token".
    Extra kwargs are forwarded to the chunker constructor.
    """
    strategy = strategy.lower()
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    if strategy == "token":
        return TokenBasedChunker(**kwargs)
    raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose: recursive, semantic, token")
