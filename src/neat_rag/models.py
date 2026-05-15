from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator

# --- Enums ---

class SearchType(str, Enum):
    """Available search strategies."""
    VECTOR = "vector"
    HYBRID = "hybrid"

class JobStatus(str, Enum):
    """Status of an asynchronous ingestion job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MessageRole(str, Enum):
    """Role of the message sender."""
    USER = "user"
    AGENT = "agent"

class FeedbackRating(int, Enum):
    """User feedback rating (e.g., 1 for thumbs up, -1 for thumbs down)."""
    UP = 1
    DOWN = -1

# --- Domain Entities (Database representations) ---

class Document(BaseModel):
    """Represents a fully ingested document."""
    id: str
    title: str
    source: str # e.g., filename or URL
    mime_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_count: int = 0

class Chunk(BaseModel):
    """Represents a text chunk from a document."""
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
class Session(BaseModel):
    """Represents a chat session."""
    id: str
    user_id: Optional[str] = None
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Message(BaseModel):
    """Represents a message within a chat session."""
    id: str
    session_id: str
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Job(BaseModel):
    """Represents an async ingestion job."""
    id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0 # 0.0 to 1.0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Feedback(BaseModel):
    """Represents user feedback on an agent's message."""
    id: str
    message_id: str
    user_id: Optional[str] = None
    rating: FeedbackRating
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ApiKey(BaseModel):
    """Represents an API key used for authenticating requests."""
    id: str
    hashed_key: str
    owner: str
    scopes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None

# --- Internal Retrieval & Processing Models ---

class SearchHit(BaseModel):
    """Represents a single retrieved chunk during search/rerank."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    document_title: str
    document_source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is within expected bounds (though some rerankers output unbounded logits, we normalize later)."""
        return v

class SearchResult(BaseModel):
    """Represents the complete result of a retrieval operation."""
    hits: List[SearchHit] = Field(default_factory=list)
    total_hits: int = 0
    search_type: SearchType
    query_time_ms: float

class Citation(BaseModel):
    """Represents a citation linking an agent's response to a specific source chunk."""
    citation_number: int # e.g., [1]
    chunk_id: str
    document_id: str
    document_title: str
    document_source: str
    content_snippet: str
