from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from neat_rag.models import FeedbackRating, JobStatus, MessageRole, SearchType


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# --- Chat ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str
    search_type: SearchType = SearchType.HYBRID
    # Optional per-request LLM override (set from the UI model picker)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None


class CitationResponse(BaseModel):
    citation_number: int
    document_title: str
    document_source: str
    content_snippet: str


class ChatResponse(BaseModel):
    session_id: str
    content: str
    message_id: Optional[str] = None
    citations: List[CitationResponse] = Field(default_factory=list)


# --- Documents ---

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    message: str = "File uploaded. Ingestion started."


class DocumentResponse(BaseModel):
    id: str
    title: str
    source: str
    mime_type: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunk_count: int


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int


class UpdateDocumentRequest(BaseModel):
    metadata: Dict[str, Any]


# --- Jobs ---

class JobResponse(BaseModel):
    id: str
    filename: str
    status: JobStatus
    progress: float
    error: Optional[str]
    created_at: datetime
    updated_at: datetime


class JobListResponse(BaseModel):
    items: List[JobResponse]


# --- Sessions ---

class SessionCreateRequest(BaseModel):
    # user_id: Optional[str] = None
    title: str = "New Chat"


class SessionUpdateRequest(BaseModel):
    title: str


class SessionResponse(BaseModel):
    id: str
    user_id: Optional[str]
    title: str
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    items: List[SessionResponse]
    total: int


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: MessageRole
    content: str
    created_at: datetime
    metadata: Dict[str, Any]


class MessageListResponse(BaseModel):
    items: List[MessageResponse]


# --- Feedback ---

class FeedbackRequest(BaseModel):
    message_id: str
    rating: FeedbackRating
    user_id: Optional[str] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    message_id: str
    user_id: Optional[str]
    rating: FeedbackRating
    comment: Optional[str]
    created_at: datetime


# --- Health ---

class HealthLive(BaseModel):
    status: str = "ok"


class HealthReady(BaseModel):
    status: str
    db: bool
    timestamp: datetime
