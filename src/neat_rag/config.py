from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_API_KEY: str = "ollama"  # Ollama ignores this, but SDK requires a non-empty string

    # --- LLM Configuration ---
    # Provider options: "openai", "gemini", "ollama", "anthropic"
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"

    # --- Embedding Configuration (Retrieval) ---
    # Options: "openai", "gemini", "ollama", "custom"
    EMBEDDING_PROVIDER: str = "gemini"
    
    # Recommended combinations:
    #   openai : text-embedding-3-small (1536 dim)
    #   gemini : gemini-embedding-001 (3072 dim)
    #   ollama : nomic-embed-text (768 dim)
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 3072

    # Used only when EMBEDDING_PROVIDER="custom"
    EMBEDDING_BASE_URL: str = ""   
    EMBEDDING_API_KEY: str = ""    

    # --- Chunking Configuration (Ingestion) ---
    # Which provider to use for semantic chunking. If empty, follows EMBEDDING_PROVIDER.
    CHUNKING_PROVIDER: str = "gemini"
    CHUNKING_MODEL: str = "gemini-embedding-001"
    
    # Custom/Local settings for chunking:
    CHUNKING_BASE_URL: str = ""
    CHUNKING_API_KEY: str = ""

    # --- Reranker Configuration ---
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # --- Infrastructure ---
    REDIS_URL: str = "redis://localhost:6379/0"

    @field_validator("EMBEDDING_PROVIDER", "CHUNKING_PROVIDER")
    @classmethod
    def validate_providers(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(f"Provider '{v}' is not supported. Must be one of {SUPPORTED_EMBEDDING_PROVIDERS}")
        return v

settings = Settings()
