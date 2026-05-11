import logging
import sys
import structlog
from neat_rag.config import settings

def setup_logging():
    """
    Configure structlog and the standard logging library.
    Logs will be output as JSON in production, and as human-readable text in development.
    """
    # 1. Setup standard logging level based on config
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # 2. Define shared processors for both structlog and standard logging
    shared_processors = [
        structlog.contextvars.merge_contextvars, # Allows injecting context (e.g., request_id, session_id)
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # 3. Configure structlog
    structlog.configure(
        processors=shared_processors + [
            # If development, use pretty console output. If prod, use JSON.
            structlog.dev.ConsoleRenderer(colors=True) 
            if settings.APP_ENV == "development" 
            else structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """
    Return a structlog bound logger.
    Usage:
        from neat_rag.logger import get_logger
        logger = get_logger(__name__)
        logger.info("message", key="value")
    """
    return structlog.get_logger(name)
