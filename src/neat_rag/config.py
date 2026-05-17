import os
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Force load .env and override any existing environment variables.
load_dotenv(dotenv_path=".env", override=True)

SUPPORTED_EMBEDDING_PROVIDERS = ["openai", "gemini", "ollama", "custom"]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Configuration
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8058
    API_URL: str = "http://localhost:8058"
    CORS_ALLOWED_ORIGINS: List[str] = ["*"]

    # Streamlit Configuration
    UI_HOST: str = "0.0.0.0"
    UI_PORT: int = 8501

    # Database Configuration
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rag"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_SYNC(self) -> str:
        # Used for alembic migrations
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # --- Provider API Keys & Base URLs ---
    OPENAI_API_KEY: str = ""

    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    ANTHROPIC_API_KEY: str = ""

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_API_KEY: str = "ollama"  # Ollama ignores this, but SDK requires a non-empty string

    # --- LLM Configuration ---
    # Provider options: "openai", "gemini", "ollama", "anthropic", "deepseek"
    # Recommended models:
    #   openai    : gpt-4o, gpt-4o-mini
    #   gemini    : gemini-2.5-pro, gemini-2.5-flash
    #   anthropic : claude-3-7-sonnet-latest, claude-3-5-haiku-latest
    #   deepseek  : deepseek-chat, deepseek-reasoner
    #   ollama    : qwen2.5, llama3
    #
    # [WARNING] Llama 3.1 (8B) Note:
    # This model has known "Tool Calling" stability issues in multi-language contexts (e.g., Chinese).
    # It may output gibberish/random characters when generating JSON parameters for tools.
    # For local OLLAMA usage, QWEN 2.5 is STRONGLY RECOMMENDED over Llama 3.1 for RAG tasks.
    LLM_PROVIDER: str = "deepseek"
    LLM_MODEL: str = "deepseek-chat"

    # Set to True when using a vision-capable LLM (e.g. gpt-4o, llava, qwen2-vl).
    # When enabled, relevant images are included alongside the question at inference time.
    # Non-multimodal LLMs still receive the VLM description + OCR text as fallback.
    LLM_MULTIMODAL: bool = True

    # --- Evaluation Configuration ---
    # Provider for ragas/evaluation (recommends high-reasoning models)
    EVAL_LLM_PROVIDER: str = "deepseek"
    EVAL_LLM_MODEL: str = "deepseek-chat"

    # --- Embedding Configuration (Retrieval) ---
    # Options: "openai", "gemini", "ollama", "custom"
    EMBEDDING_PROVIDER: str = "gemini"
    # Recommended combinations:
    #   openai : text-embedding-3-small (1536 dim)
    #   gemini : gemini-embedding-001 (1536 dim via MRL, same MTEB as 3072)
    #   ollama : bge-m3 (1024 dim, for multilanguage), nomic-embed-text (768 dim, lite, only for English)
    # Note: gemini-embedding-001 natively outputs 3072 dims but supports Matryoshka
    # truncation to 1536 (no quality loss) or 768 (0.26% loss) — keeps within
    # pgvector HNSW's 2000-dim limit. EMBEDDING_DIM controls output dimensionality.
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 1536

    # Used only when EMBEDDING_PROVIDER="custom"
    EMBEDDING_BASE_URL: str = "http://host.docker.internal:11434/v1"
    EMBEDDING_API_KEY: str = ""    

    # --- Chunking Configuration (Ingestion) ---
    # Which provider to use for semantic chunking.
    CHUNKING_PROVIDER: str = "gemini"
    CHUNKING_MODEL: str = "gemini-embedding-001"
    
    # Custom/Local settings for chunking:
    CHUNKING_BASE_URL: str = "http://host.docker.internal:11434/v1"
    CHUNKING_API_KEY: str = ""

    # --- Reranker Configuration ---
    # Options for RERANKER_PROVIDER: 
    #   "bge"    - Local CrossEncoder models (requires sentence-transformers)
    #   "cohere" - Cohere Rerank API (requires COHERE_API_KEY)
    #
    # Recommended Models for "bge" (Local):
    #   - BAAI/bge-reranker-v2-m3        : SOTA quality, Multilingual (Large: ~2GB)
    #   - cross-encoder/ms-marco-MiniLM-L-6-v2 : Fast, English-only (Small: ~90MB)
    #   - jinaai/jina-reranker-v1-tiny-en : Long-context (8k), English (Tiny: ~60MB)
    #
    # Recommended Models for "cohere" (Cloud):
    #   - rerank-v3.5              : Newest, highest quality reasoning (Multilingual)
    #   - rerank-multilingual-v3.0 : Standard multilingual model
    #   - rerank-english-v3.0      : Optimized for English
    #
    RERANKER_PROVIDER: str = "cohere"
    RERANKER_MODEL: str = "rerank-multilingual-v3.0"
    COHERE_API_KEY: str = ""

    # --- Advanced Retrieval Settings ---
    DEFAULT_SEARCH_TYPE: str = "hybrid" # Options: "vector", "hybrid"
    ENABLE_RERANK: bool = True
    ENABLE_HYDE: bool = False
    ENABLE_MULTI_QUERY: bool = False
    MULTI_QUERY_COUNT: int = 3
    
    # Reranking logic: retrieve CANDIDATE_K -> rerank -> return TOP_K
    RERANK_TOP_K: int = 5
    RETRIEVE_CANDIDATE_K: int = 25

    # --- Agent / Prompt Configuration ---
    # Injected into the system prompt template via {domain}
    DOMAIN_DESCRIPTION: str = "a knowledge base of indexed documents"

    # --- Security & Rate Limiting ---
    # Set ENABLE_AUTH=true in production and issue keys via the /admin/keys endpoint
    ENABLE_AUTH: bool = False
    # When True, all authenticated users share a single dataset
    # When False (default), each API key owner sees only their own documents/sessions.
    DATABASE_SHARED: bool = True
    # slowapi limit strings — see https://limits.readthedocs.io/en/stable/string-notation.html
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_CHAT: str = "10/minute"

    # --- Image Support ---
    IMAGE_STORAGE_PATH: str = "./data/images"
    # Languages used for OCR (pytesseract).  Separate multiple codes with '+'.
    # Must match installed tesseract language packs (tesseract-ocr-<code>).
    # Common codes: eng chi_sim chi_tra jpn kor ara fra deu spa rus ita por hin
    # Set to "eng" if only English packs are installed locally.
    OCR_LANGUAGES: str = "eng+deu+chi_sim+chi_tra+jpn+kor+ara+fra+spa+rus+ita+por+hin"

    # --- Infrastructure ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Vector Store Backend ---
    # Options: "pgvector" (default, uses PostgreSQL + pgvector extension)
    #          "qdrant"   (uses Qdrant dedicated vector database)
    VECTOR_STORE_BACKEND: str = "qdrant"

    # --- Qdrant Configuration (only used when VECTOR_STORE_BACKEND="qdrant") ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "chunks"

    @field_validator("EMBEDDING_PROVIDER", "CHUNKING_PROVIDER")
    @classmethod
    def validate_providers(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(f"Provider '{v}' is not supported. Must be one of {SUPPORTED_EMBEDDING_PROVIDERS}")
        return v

settings = Settings()
