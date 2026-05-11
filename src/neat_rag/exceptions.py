class NeatRagError(Exception):
    """Base exception for all Neat-RAG project errors."""
    pass

# --- Database Errors ---
class DatabaseError(NeatRagError):
    """Base exception for database-related errors."""
    pass

class RecordNotFoundError(DatabaseError):
    """Raised when a requested database record is not found (e.g., Document, Session)."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Raised when the application fails to connect to the database."""
    pass

# --- Provider Errors (LLM, Embedding, Reranker) ---
class ProviderError(NeatRagError):
    """Base exception for external provider errors."""
    pass

class LLMProviderError(ProviderError):
    """Raised when an LLM provider API call fails."""
    pass

class EmbeddingProviderError(ProviderError):
    """Raised when an embedding provider API call fails."""
    pass

class RerankerProviderError(ProviderError):
    """Raised when a reranker provider API call fails."""
    pass

# --- Ingestion & Processing Errors ---
class IngestionError(NeatRagError):
    """Base exception for document ingestion and processing errors."""
    pass

class UnsupportedFileTypeError(IngestionError):
    """Raised when attempting to ingest a file with an unsupported extension."""
    pass

class ExtractionError(IngestionError):
    """Raised when text/data extraction from a document fails (e.g., malformed PDF)."""
    pass

class ChunkingError(IngestionError):
    """Raised when document chunking fails."""
    pass

# --- Retrieval Errors ---
class RetrievalError(NeatRagError):
    """Base exception for retrieval and search pipeline errors."""
    pass

# --- Agent & Tool Errors ---
class AgentError(NeatRagError):
    """Base exception for Agent orchestrator errors."""
    pass

class ToolExecutionError(AgentError):
    """Raised when a specific tool executed by the Agent fails."""
    pass
