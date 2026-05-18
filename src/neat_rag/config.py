from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# pydantic-settings reads .env via env_file, but load_dotenv with override=True
# ensures .env values win over any pre-existing shell environment variables.
load_dotenv(dotenv_path=".env", override=True)

SUPPORTED_EMBEDDING_PROVIDERS = ["openai", "gemini", "ollama", "custom"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App / API ---
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8058
    API_URL: str = "http://localhost:8058"
    CORS_ALLOWED_ORIGINS: list[str] = ["*"]

    # --- Streamlit UI ---
    UI_HOST: str = "0.0.0.0"
    UI_PORT: int = 8501

    # --- PostgreSQL ---
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rag"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Vector Store Backend ---
    # "pgvector" — integrated PostgreSQL + pgvector extension
    # "qdrant"   — dedicated Qdrant instance
    VECTOR_STORE_BACKEND: str = "qdrant"

    # Qdrant (only used when VECTOR_STORE_BACKEND="qdrant")
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "chunks"

    # --- Provider API Keys & Base URLs ---
    OPENAI_API_KEY: str = ""

    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    ANTHROPIC_API_KEY: str = ""

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # host.docker.internal resolves to the host machine from inside a Docker container.
    # Change to http://localhost:11434 for local (non-Docker) development.
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_API_KEY: str = "ollama"  # Ollama ignores this; SDK requires a non-empty value

    # --- LLM ---
    # Provider options: "openai" | "gemini" | "anthropic" | "deepseek" | "ollama"
    # Recommended models:
    #   openai    : gpt-4o, gpt-4o-mini
    #   gemini    : gemini-2.5-pro, gemini-2.5-flash
    #   anthropic : claude-3-7-sonnet-latest, claude-3-5-haiku-latest
    #   deepseek  : deepseek-chat, deepseek-reasoner
    #   ollama    : qwen2.5 (recommended), llama3.2
    #
    # WARNING — Llama 3.1 (8B) has known tool-call instability in multilingual contexts:
    # it can emit random characters when serialising JSON tool arguments.
    # Use Qwen 2.5 for local RAG tasks instead.
    #
    # LLM_PROVIDER: str = "deepseek"
    # LLM_MODEL: str = "deepseek-chat"
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "qwen2.5:7b"

    # When True, vision-capable LLMs (e.g. gpt-4o, llava, qwen2-vl) receive images
    # directly at inference time. Non-multimodal LLMs fall back to VLM description + OCR text.
    LLM_MULTIMODAL: bool = True

    # --- Evaluation ---
    # Separate provider/model for RAGAS scoring; high-reasoning models give more reliable scores.
    # EVAL_LLM_PROVIDER: str = "deepseek"
    # EVAL_LLM_MODEL: str = "deepseek-chat"
    EVAL_LLM_PROVIDER: str = "ollama"
    EVAL_LLM_MODEL: str = "qwen2.5:7b"

    # --- Embeddings ---
    # Provider options: "openai" | "gemini" | "ollama" | "custom"
    # Recommended combinations:
    #   openai : text-embedding-3-small (1536 dim)
    #   gemini : gemini-embedding-001 — natively 3072 dim, but supports Matryoshka
    #            truncation to 1536 (no quality loss) or 768 (−0.26%) to stay within
    #            pgvector HNSW's 2000-dim limit. EMBEDDING_DIM controls output size.
    #   ollama : bge-m3 (1024 dim, multilingual), nomic-embed-text (768 dim, English only)
    #
    # EMBEDDING_PROVIDER: str = "gemini"
    # EMBEDDING_MODEL: str = "gemini-embedding-001"
    # EMBEDDING_DIM: int = 1536
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIM: int = 1024

    # Only used when EMBEDDING_PROVIDER="custom"
    EMBEDDING_BASE_URL: str = "http://host.docker.internal:11434/v1"
    EMBEDDING_API_KEY: str = ""

    # --- Chunking ---
    # Provider/model used for semantic chunking during ingestion (same options as embeddings).
    #
    # CHUNKING_PROVIDER: str = "gemini"
    # CHUNKING_MODEL: str = "gemini-embedding-001"
    CHUNKING_PROVIDER: str = "ollama"
    CHUNKING_MODEL: str = "bge-m3"

    # Only used when CHUNKING_PROVIDER="custom"
    CHUNKING_BASE_URL: str = "http://host.docker.internal:11434/v1"
    CHUNKING_API_KEY: str = ""

    # --- Reranker ---
    # Provider options:
    #   "bge"    — local CrossEncoder via sentence-transformers (no API key needed)
    #   "cohere" — Cohere Rerank API (requires COHERE_API_KEY)
    #
    # Recommended BGE models (local) - bge:
    #   BAAI/bge-reranker-v2-m3            — SOTA multilingual quality (~2 GB)
    #   cross-encoder/ms-marco-MiniLM-L-6-v2 — fast, English-only (~90 MB)
    #   jinaai/jina-reranker-v1-tiny-en    — 8k context, English, tiny (~60 MB)
    #
    # Recommended Cohere models (cloud) - cohere:
    #   rerank-v3.5              — highest quality, multilingual
    #   rerank-multilingual-v3.0 — standard multilingual
    #   rerank-english-v3.0      — optimised for English
    #
    # RERANKER_PROVIDER: str = "cohere"
    # RERANKER_MODEL: str = "rerank-multilingual-v3.0"
    # COHERE_API_KEY: str = ""
    RERANKER_PROVIDER: str = "bge"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    COHERE_API_KEY: str = ""    # only used when RERANKER_PROVIDER="cohere"

    # --- Retrieval ---
    DEFAULT_SEARCH_TYPE: str = "hybrid"  # "vector" | "hybrid"
    ENABLE_RERANK: bool = True
    ENABLE_HYDE: bool = False
    ENABLE_MULTI_QUERY: bool = False
    MULTI_QUERY_COUNT: int = 3

    # Two-stage retrieval: fetch RETRIEVE_CANDIDATE_K chunks, rerank, return RERANK_TOP_K.
    RERANK_TOP_K: int = 5
    RETRIEVE_CANDIDATE_K: int = 25

    # --- Agent ---
    # Injected into the system prompt as {domain} to scope the agent's persona.
    DOMAIN_DESCRIPTION: str = "a knowledge base of indexed documents"

    # --- Security & Rate Limiting ---
    # Static operator key for /admin/* endpoints — set a strong value in .env for production.
    # Grants admin access regardless of DB state (bootstrap + permanent operator key).
    # DB-backed API keys with scopes=["admin"] grant the same access and can be revoked.
    ADMIN_BOOTSTRAP_KEY: str = "admin"

    # --- Hugging Face ---
    # Optional: Provide a token to enable higher rate limits and access to private models.
    # The SDKs (transformers, sentence-transformers) will also pick this up from the environment.
    HF_TOKEN: str = ""

    # When True, all API keys share a single dataset (user_id is never propagated to the DB).
    # When False, each key owner sees only their own documents and sessions.
    DATABASE_SHARED: bool = True

    # slowapi limit strings — see https://limits.readthedocs.io/en/stable/string-notation.html
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_CHAT: str = "10/minute"

    # --- Image Support ---
    IMAGE_STORAGE_PATH: str = "./data/images"

    # Tesseract language codes to enable for OCR. Separate multiple with '+'.
    # Must match installed tesseract-ocr-<code> packages.
    # Use "eng" if only the English pack is available locally.
    OCR_LANGUAGES: str = "eng+deu+chi_sim+chi_tra+jpn+kor+ara+fra+spa+rus+ita+por+hin"

    # --- Computed properties ---
    @property
    def DATABASE_URL(self) -> str:
        """DSN for asyncpg.create_pool — uses the standard postgresql:// scheme."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """DSN for Alembic (psycopg2/sync driver)."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @field_validator("EMBEDDING_PROVIDER", "CHUNKING_PROVIDER")
    @classmethod
    def validate_providers(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Provider '{v}' is not supported. Must be one of {SUPPORTED_EMBEDDING_PROVIDERS}"
            )
        return v


settings = Settings()
