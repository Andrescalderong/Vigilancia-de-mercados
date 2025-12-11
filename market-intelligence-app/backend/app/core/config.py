"""
Market Intelligence Platform - Configuration
============================================
Configuración centralizada para la plataforma de inteligencia de mercado.
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Configuración principal de la aplicación."""
    
    # Application
    APP_NAME: str = "Market Intelligence AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: str = "sqlite:///./market_intelligence.db"
    
    # Vector Store (ChromaDB)
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "market_intelligence"
    
    # LLM Configuration
    LLM_PROVIDER: str = "anthropic"  # Options: anthropic, openai, local
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Sentence Transformers model
    EMBEDDING_DIMENSION: int = 384
    
    # External APIs
    FINNHUB_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    SEC_EDGAR_USER_AGENT: str = "MarketIntelligence/1.0 (contact@example.com)"
    
    # RAG Configuration
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.7
    
    # Verification Settings
    VERIFICATION_MIN_SOURCES: int = 2
    VERIFICATION_CONFIDENCE_THRESHOLD: float = 0.85
    ENABLE_TRIPLE_VERIFICATION: bool = True
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hour
    ENABLE_CACHE: bool = True
    REDIS_URL: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Obtener configuración con caché."""
    return Settings()


# Configuración de logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "market_intelligence": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


# Constantes de la aplicación
class Constants:
    """Constantes globales de la aplicación."""
    
    # Tipos de fuentes de datos
    SOURCE_TYPES = {
        "PRIMARY": "primary",
        "SECONDARY": "secondary",
        "ALTERNATIVE": "alternative",
        "REGULATORY": "regulatory",
    }
    
    # Niveles de confianza
    CONFIDENCE_LEVELS = {
        "HIGH": (0.90, 1.0),
        "MEDIUM": (0.75, 0.90),
        "LOW": (0.50, 0.75),
        "UNCERTAIN": (0.0, 0.50),
    }
    
    # Estados de verificación
    VERIFICATION_STATUS = {
        "PENDING": "pending",
        "VERIFYING": "verifying",
        "CROSS_CHECKING": "cross_checking",
        "VERIFIED": "verified",
        "FAILED": "failed",
    }
    
    # Tipos de señales de mercado
    SIGNAL_TYPES = {
        "PATENT": "patent",
        "FUNDING": "funding",
        "REGULATORY": "regulatory",
        "M_AND_A": "m&a",
        "EXECUTIVE": "executive",
        "EARNINGS": "earnings",
        "NEWS": "news",
    }
    
    # Sectores de mercado
    MARKET_SECTORS = [
        "Technology",
        "Healthcare",
        "Finance",
        "Energy",
        "Consumer",
        "Industrial",
        "Materials",
        "Utilities",
        "Real Estate",
        "Communications",
    ]
