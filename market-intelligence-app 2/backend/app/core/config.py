"""
Configuration Module - Extended
================================
Configuración centralizada con soporte para Google APIs, Finnhub y Cloud deployment.
"""

import os
from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de la aplicación con todas las integraciones."""
    
    # ============================================
    # APPLICATION
    # ============================================
    app_name: str = "Market Intelligence Platform"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # ============================================
    # LLM PROVIDERS
    # ============================================
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", description="Claude model")
    anthropic_max_tokens: int = Field(default=4096, description="Max tokens for Claude")
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")

    google_api_key: Optional[str] = Field(default=None, description="Google API key (Gemini)")
    
    # ============================================
    # GOOGLE SEARCH / CLOUD
    # ============================================
    google_search_engine_id: Optional[str] = Field(default=None, description="Google Custom Search Engine ID")
    google_cloud_project: Optional[str] = Field(default=None, description="Google Cloud Project ID")
    google_search_enabled: bool = Field(default=True, description="Enable Google Search")
    google_search_results_per_query: int = Field(default=10, description="Results per search")
    vertex_ai_enabled: bool = Field(default=False, description="Enable Vertex AI")
    vertex_ai_location: str = Field(default="us-central1", description="Vertex AI location")
    
    # ============================================
    # FINANCIAL DATA APIS
    # ============================================
    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API key")
    finnhub_base_url: str = Field(default="https://finnhub.io/api/v1", description="Finnhub base URL")
    finnhub_rate_limit: int = Field(default=60, description="Requests per minute (free tier)")
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    yahoo_finance_enabled: bool = Field(default=True, description="Enable Yahoo Finance")
    
    # ============================================
    # NEWS & MEDIA APIS
    # ============================================
    news_api_key: Optional[str] = Field(default=None, description="NewsAPI.org key")
    newsdata_api_key: Optional[str] = Field(default=None, description="NewsData.io key")
    
    # ============================================
    # COMPLIANCE & DUE DILIGENCE
    # ============================================
    opensanctions_enabled: bool = Field(default=True, description="Enable OpenSanctions")
    opensanctions_api_url: str = Field(default="https://api.opensanctions.org", description="OpenSanctions API")
    ofac_enabled: bool = Field(default=True, description="Enable OFAC checks")
    
    # ============================================
    # RAG CONFIGURATION
    # ============================================
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    embedding_dimension: int = Field(default=384, description="Embedding dimension")
    chroma_persist_dir: str = Field(default="./data/chroma_db", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="market_intelligence", description="Default collection")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Chunk overlap")
    rag_top_k: int = Field(default=5, description="Top K documents to retrieve")
    rag_similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    rag_rerank_enabled: bool = Field(default=True, description="Enable reranking")
    
    # ============================================
    # VERIFICATION SYSTEM
    # ============================================
    verification_enabled: bool = Field(default=True, description="Enable triple verification")
    verification_min_sources: int = Field(default=2, description="Minimum sources for verification")
    verification_confidence_threshold: float = Field(default=0.85, description="Confidence threshold")
    verification_timeout: int = Field(default=30, description="Verification timeout in seconds")
    
    # ============================================
    # CACHING
    # ============================================
    redis_url: Optional[str] = Field(default="redis://localhost:6379", description="Redis URL")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_default: int = Field(default=3600, description="Default cache TTL in seconds")
    cache_ttl_market_data: int = Field(default=300, description="Market data cache TTL (5 min)")
    cache_ttl_news: int = Field(default=900, description="News cache TTL (15 min)")
    
    # ============================================
    # DATABASE
    # ============================================
    database_url: Optional[str] = Field(default=None, description="PostgreSQL URL for persistence")
    
    # ============================================
    # SECURITY
    # ============================================
    secret_key: str = Field(default="change-me-in-production", description="Secret key")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    allowed_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    
    # ============================================
    # CLOUD DEPLOYMENT
    # ============================================
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for storage")
    gcp_project_id: Optional[str] = Field(default=None, description="GCP project ID")
    gcp_region: str = Field(default="us-central1", description="GCP region")
    gcs_bucket: Optional[str] = Field(default=None, description="GCS bucket for storage")
    
    # ============================================
    # MONITORING
    # ============================================
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    health_check_interval: int = Field(default=30, description="Health check interval")
    
    # ============================================
    # FEATURE FLAGS
    # ============================================
    feature_due_diligence: bool = Field(default=True, description="Enable Due Diligence module")
    feature_competitive_intel: bool = Field(default=True, description="Enable Competitive Intelligence")
    feature_predictive: bool = Field(default=True, description="Enable Predictive Intelligence")
    feature_weak_signals: bool = Field(default=True, description="Enable Weak Signal Detection")


    # ============================================================
    # LLM HELPERS (CORREGIDOS)
    # ============================================================

    @property
    def has_llm_provider(self) -> bool:
        """Indica si existe al menos un proveedor LLM disponible."""
        return bool(
            self.anthropic_api_key or
            self.openai_api_key or
            self.google_api_key
        )

    def get_llm_config(self) -> dict:
        """
        Devuelve el proveedor LLM activo en este orden de prioridad:
        1. Anthropic
        2. OpenAI
        3. Google Gemini
        """
        if self.anthropic_api_key:
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
                "max_tokens": self.anthropic_max_tokens,
            }

        if self.openai_api_key:
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "max_tokens": 4096,
            }

        if self.google_api_key:
            return {
                "provider": "google",
                "api_key": self.google_api_key,
                "model": "gemini-1.5-flash",  # puedes personalizar aquí
                "max_tokens": 8192,
            }

        return {"provider": None}


    # ============================================================
    # OTHER HELPERS
    # ============================================================

    @property
    def has_google_search(self) -> bool:
        return bool(self.google_api_key and self.google_search_engine_id)

    @property
    def has_finnhub(self) -> bool:
        return bool(self.finnhub_api_key)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def get_enabled_features(self) -> List[str]:
        features = []
        if self.feature_due_diligence:
            features.append("due_diligence")
        if self.feature_competitive_intel:
            features.append("competitive_intelligence")
        if self.feature_predictive:
            features.append("predictive_intelligence")
        if self.feature_weak_signals:
            features.append("weak_signals")
        return features


class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"
    case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_api_status() -> dict:
    settings = get_settings()
    return {
        "llm": {
            "anthropic": bool(settings.anthropic_api_key),
            "openai": bool(settings.openai_api_key),
            "google": bool(settings.google_api_key),
            "primary": settings.get_llm_config()["provider"],
        },
        "google": {
            "search": settings.has_google_search,
            "vertex_ai": settings.vertex_ai_enabled,
        },
        "financial": {
            "finnhub": settings.has_finnhub,
            "alpha_vantage": bool(settings.alpha_vantage_api_key),
            "yahoo_finance": settings.yahoo_finance_enabled,
        },
        "news": {
            "newsapi": bool(settings.news_api_key),
            "newsdata": bool(settings.newsdata_api_key),
        },
        "compliance": {
            "opensanctions": settings.opensanctions_enabled,
            "ofac": settings.ofac_enabled,
        },
        "features": settings.get_enabled_features(),
    }
