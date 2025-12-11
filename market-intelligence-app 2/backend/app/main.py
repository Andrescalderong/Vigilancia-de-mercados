"""
Market Intelligence Platform - API Principal
=============================================
FastAPI backend con todos los módulos:
- Inteligencia de Mercado (RAG + Verificación)
- Due Diligence / AML
- Inteligencia Competitiva
- Inteligencia Predictiva
- Integración Google + Finnhub
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Core
from .core.config import get_settings, get_api_status
from .core.rag_engine import get_rag_engine

# Servicios
from .services.google_service import get_google_service
from .services.due_diligence import get_due_diligence_service, EntityType, RiskLevel
from .services.competitive_intelligence import get_competitive_intelligence_service, SignalType
from .services.predictive_intelligence import get_predictive_intelligence_service, TimeHorizon

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("market_intelligence.api")


# ============================================
# MODELOS PYDANTIC
# ============================================

# --- General ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_verification: bool = Field(default=True)
    language: str = Field(default="es")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    sources: List[str] = Field(default=["rag", "google", "finnhub"])
    num_results: int = Field(default=10, ge=1, le=50)


class IngestRequest(BaseModel):
    content: str = Field(..., min_length=10)
    source_name: str = Field(..., min_length=1)
    source_type: str = Field(default="secondary")
    source_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# --- Due Diligence ---
class DueDiligenceRequest(BaseModel):
    entity_name: str = Field(..., min_length=2, max_length=200)
    entity_type: str = Field(default="company")  # company, person
    include_sanctions: bool = Field(default=True)
    include_pep: bool = Field(default=True)
    include_adverse_media: bool = Field(default=True)
    include_corporate_structure: bool = Field(default=True)


class SanctionsCheckRequest(BaseModel):
    name: str = Field(..., min_length=2)
    entity_type: str = Field(default="any")


class AMLScreeningRequest(BaseModel):
    entities: List[str] = Field(..., min_items=1, max_items=50)
    check_type: str = Field(default="full")  # full, sanctions, pep


# --- Competitive Intelligence ---
class CompetitorAnalysisRequest(BaseModel):
    company_name: str = Field(..., min_length=2)
    competitors: List[str] = Field(default=[])
    market: Optional[str] = None
    include_signals: bool = Field(default=True)


class MarketMonitorRequest(BaseModel):
    market_name: str = Field(..., min_length=2)
    key_players: List[str] = Field(default=[])
    signal_types: List[str] = Field(default=[])


class CompetitorCompareRequest(BaseModel):
    competitors: List[str] = Field(..., min_items=2, max_items=10)


# --- Predictive Intelligence ---
class PredictionRequest(BaseModel):
    subject: str = Field(..., min_length=2)
    time_horizon: str = Field(default="medium_term")  # short_term, medium_term, long_term
    include_scenarios: bool = Field(default=True)


class MarketForecastRequest(BaseModel):
    market: str = Field(..., min_length=2)
    time_horizon: str = Field(default="medium_term")
    include_weak_signals: bool = Field(default=True)


class WeakSignalRequest(BaseModel):
    domain: str = Field(..., min_length=2)
    categories: List[str] = Field(default=[])


# ============================================
# LIFECYCLE
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación."""
    logger.info("🚀 Starting Market Intelligence Platform...")
    
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Features enabled: {settings.get_enabled_features()}")
    
    # Inicializar RAG Engine
    try:
        rag = get_rag_engine()
        logger.info("✅ RAG Engine initialized")
    except Exception as e:
        logger.error(f"❌ RAG Engine error: {str(e)}")
    
    # Verificar APIs
    api_status = get_api_status()
    logger.info(f"API Status: {api_status}")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down...")
    try:
        google = get_google_service()
        await google.close_all()
        
        dd = get_due_diligence_service()
        await dd.close_all()
        
        ci = get_competitive_intelligence_service()
        await ci.close_all()
    except:
        pass
    
    logger.info("👋 Shutdown complete")


# ============================================
# APP INITIALIZATION
# ============================================

settings = get_settings()

app = FastAPI(
    title="Market Intelligence Platform",
    description="""
    Plataforma de Inteligencia de Mercado con IA.
    
    ## Módulos
    
    - **Intelligence**: Búsqueda y análisis RAG con verificación triple
    - **Due Diligence**: Verificación AML, sanciones, PEP, medios adversos
    - **Competitive**: Monitoreo de competidores y señales de mercado
    - **Predictive**: Análisis predictivo, tendencias y escenarios
    
    ## APIs Integradas
    
    - Google Custom Search
    - Finnhub (datos financieros)
    - OpenSanctions
    - Yahoo Finance
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    """Health check del sistema."""
    settings = get_settings()
    
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment
    }
    
    # Check RAG
    try:
        rag = get_rag_engine()
        health["rag_engine"] = "operational"
    except:
        health["rag_engine"] = "error"
        health["status"] = "degraded"
    
    # Check APIs
    health["apis"] = get_api_status()
    
    return health


@app.get("/api/v1/status")
async def api_status():
    """Estado detallado de APIs y servicios."""
    return {
        "apis": get_api_status(),
        "features": get_settings().get_enabled_features(),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================
# INTELLIGENCE ENDPOINTS (RAG + Verificación)
# ============================================

@app.post("/api/v1/query")
async def intelligence_query(request: QueryRequest):
    """
    Consulta de inteligencia con RAG y verificación.
    
    Procesa la consulta a través del motor RAG y aplica
    verificación triple si está habilitada.
    """
    try:
        rag = get_rag_engine()
        result = rag.query(
            query=request.query,
            top_k=request.top_k
        )
        
        response = {
            "query": request.query,
            "answer": result.answer,
            "sources": [
                {
                    "content": doc.page_content[:300],
                    "source": doc.metadata.get("source", "unknown"),
                    "type": doc.metadata.get("source_type", "unknown"),
                    "relevance": score
                }
                for doc, score in zip(result.source_documents, result.relevance_scores)
            ],
            "confidence": result.confidence_score,
            "verified": result.verification_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search")
async def multi_source_search(request: SearchRequest):
    """
    Búsqueda multi-fuente (RAG + Google + APIs financieras).
    """
    results = {
        "query": request.query,
        "sources": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # RAG Search
    if "rag" in request.sources:
        try:
            rag = get_rag_engine()
            rag_result = rag.retrieve(request.query, top_k=request.num_results)
            results["sources"]["rag"] = {
                "count": len(rag_result.documents),
                "results": [
                    {
                        "content": doc.page_content[:200],
                        "source": doc.metadata.get("source"),
                        "score": score
                    }
                    for doc, score in zip(rag_result.documents, rag_result.relevance_scores)
                ]
            }
        except Exception as e:
            results["sources"]["rag"] = {"error": str(e)}
    
    # Google Search
    if "google" in request.sources:
        try:
            google = get_google_service()
            google_result = await google.search.search(
                request.query, 
                num_results=min(request.num_results, 10)
            )
            results["sources"]["google"] = {
                "total": google_result.total_results,
                "results": [
                    {
                        "title": r.title,
                        "link": r.link,
                        "snippet": r.snippet,
                        "relevance": r.relevance_score
                    }
                    for r in google_result.results
                ]
            }
        except Exception as e:
            results["sources"]["google"] = {"error": str(e)}
    
    return results


@app.post("/api/v1/ingest")
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Indexa un documento en el sistema RAG.
    """
    try:
        rag = get_rag_engine()
        
        metadata = request.metadata or {}
        metadata.update({
            "source": request.source_name,
            "source_type": request.source_type,
            "source_url": request.source_url,
            "ingested_at": datetime.utcnow().isoformat()
        })
        
        doc_id = rag.ingest_document(
            content=request.content,
            metadata=metadata
        )
        
        return {
            "status": "success",
            "document_id": doc_id,
            "source": request.source_name,
            "chunks_created": len(request.content) // 1000 + 1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# DUE DILIGENCE / AML ENDPOINTS
# ============================================

@app.post("/api/v1/due-diligence/check")
async def due_diligence_check(request: DueDiligenceRequest):
    """
    Ejecuta verificación completa de Due Diligence.
    
    Incluye: sanciones, PEP, medios adversos, estructura corporativa.
    """
    try:
        dd_service = get_due_diligence_service()
        
        entity_type = EntityType.COMPANY if request.entity_type == "company" else EntityType.PERSON
        
        report = await dd_service.run_full_check(
            entity_name=request.entity_name,
            entity_type=entity_type
        )
        
        return {
            "entity": report.entity_name,
            "entity_type": report.entity_type.value,
            "risk_assessment": {
                "overall_risk": report.overall_risk.value,
                "risk_score": report.risk_score,
                "confidence": report.confidence_score
            },
            "risk_indicators": [
                {
                    "category": ind.category,
                    "description": ind.description,
                    "severity": ind.severity.value,
                    "source": ind.source
                }
                for ind in report.risk_indicators
            ],
            "sanctions_check": report.sanctions_check,
            "pep_check": report.pep_check,
            "adverse_media_count": len(report.adverse_media),
            "recommendations": report.recommendations,
            "timestamp": report.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Due Diligence error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/due-diligence/sanctions")
async def sanctions_check(request: SanctionsCheckRequest):
    """
    Verificación rápida de sanciones.
    """
    try:
        dd_service = get_due_diligence_service()
        result = await dd_service.sanctions_checker.check_entity(
            request.name,
            request.entity_type
        )
        return result
        
    except Exception as e:
        logger.error(f"Sanctions check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/due-diligence/batch-screening")
async def batch_aml_screening(request: AMLScreeningRequest, background_tasks: BackgroundTasks):
    """
    Screening AML por lotes para múltiples entidades.
    """
    try:
        dd_service = get_due_diligence_service()
        
        results = []
        for entity in request.entities:
            check = await dd_service.sanctions_checker.check_entity(entity)
            results.append({
                "entity": entity,
                "status": check.get("overall_status", "unknown"),
                "jurisdiction_risk": check.get("jurisdiction_risk", {})
            })
        
        return {
            "total_screened": len(results),
            "results": results,
            "summary": {
                "clear": sum(1 for r in results if r["status"] == "clear"),
                "matches": sum(1 for r in results if r["status"] == "match"),
                "potential_matches": sum(1 for r in results if r["status"] == "potential_match"),
                "jurisdiction_risk": sum(1 for r in results if r["status"] == "jurisdiction_risk")
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch screening error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# COMPETITIVE INTELLIGENCE ENDPOINTS
# ============================================

@app.post("/api/v1/competitive/analyze")
async def analyze_competitor(request: CompetitorAnalysisRequest):
    """
    Análisis completo de competidor(es).
    """
    try:
        ci_service = get_competitive_intelligence_service()
        
        result = await ci_service.get_competitive_overview(
            company_name=request.company_name,
            competitors=request.competitors if request.competitors else None,
            market=request.market
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Competitor analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/competitive/compare")
async def compare_competitors(request: CompetitorCompareRequest):
    """
    Comparación directa de competidores.
    """
    try:
        ci_service = get_competitive_intelligence_service()
        
        result = await ci_service.competitor_tracker.compare_competitors(
            request.competitors
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Competitor comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/competitive/market")
async def monitor_market(request: MarketMonitorRequest):
    """
    Monitoreo de mercado con detección de señales.
    """
    try:
        ci_service = get_competitive_intelligence_service()
        
        landscape = await ci_service.market_monitor.analyze_market(
            market_name=request.market_name,
            key_players=request.key_players if request.key_players else None
        )
        
        return {
            "market": landscape.market_name,
            "players_analyzed": len(landscape.key_players),
            "signals_detected": len(landscape.recent_signals),
            "signal_distribution": landscape.competitive_dynamics.get("signal_distribution", {}),
            "recent_signals": [
                {
                    "type": s.signal_type.value,
                    "competitor": s.competitor,
                    "title": s.title,
                    "strength": s.strength.value,
                    "impact": s.impact_assessment
                }
                for s in landscape.recent_signals[:10]
            ],
            "timestamp": landscape.analyzed_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Market monitor error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/competitive/signals")
async def get_competitive_signals(
    competitors: str = Query(..., description="Comma-separated competitor names"),
    signal_types: Optional[str] = Query(None, description="Comma-separated signal types"),
    min_strength: str = Query("weak", description="Minimum signal strength")
):
    """
    Obtiene alertas de señales competitivas.
    """
    try:
        ci_service = get_competitive_intelligence_service()
        
        competitor_list = [c.strip() for c in competitors.split(",")]
        
        # Parse signal types
        type_filter = None
        if signal_types:
            type_filter = [
                SignalType(t.strip()) 
                for t in signal_types.split(",")
                if t.strip() in [st.value for st in SignalType]
            ]
        
        from .services.competitive_intelligence import SignalStrength
        strength_map = {
            "weak": SignalStrength.WEAK,
            "moderate": SignalStrength.MODERATE,
            "strong": SignalStrength.STRONG,
            "confirmed": SignalStrength.CONFIRMED
        }
        
        signals = await ci_service.market_monitor.get_alerts(
            competitors=competitor_list,
            signal_types=type_filter,
            min_strength=strength_map.get(min_strength, SignalStrength.WEAK)
        )
        
        return {
            "competitors_monitored": competitor_list,
            "signals_found": len(signals),
            "signals": [
                {
                    "type": s.signal_type.value,
                    "competitor": s.competitor,
                    "title": s.title,
                    "description": s.description[:200],
                    "strength": s.strength.value,
                    "confidence": s.confidence,
                    "recommended_actions": s.recommended_actions
                }
                for s in signals[:20]
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Signals error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# PREDICTIVE INTELLIGENCE ENDPOINTS
# ============================================

@app.post("/api/v1/predictive/forecast")
async def market_forecast(request: MarketForecastRequest):
    """
    Genera pronóstico completo de mercado.
    """
    try:
        pi_service = get_predictive_intelligence_service()
        
        horizon_map = {
            "short_term": TimeHorizon.SHORT_TERM,
            "medium_term": TimeHorizon.MEDIUM_TERM,
            "long_term": TimeHorizon.LONG_TERM,
            "strategic": TimeHorizon.STRATEGIC
        }
        
        forecast = await pi_service.generate_market_forecast(
            market=request.market,
            time_horizon=horizon_map.get(request.time_horizon, TimeHorizon.MEDIUM_TERM),
            include_scenarios=True
        )
        
        return {
            "market": forecast.market_name,
            "horizon": forecast.forecast_horizon.value,
            "current_state": forecast.current_state,
            "trends": [
                {
                    "name": t.name,
                    "direction": t.direction.value,
                    "strength": t.strength,
                    "momentum": t.momentum,
                    "confidence": t.confidence
                }
                for t in forecast.trends
            ],
            "weak_signals": [
                {
                    "signal": s.signal,
                    "category": s.category,
                    "potential_impact": s.potential_impact,
                    "probability": s.probability,
                    "time_to_materialize": s.time_to_materialize
                }
                for s in forecast.weak_signals[:5]
            ],
            "predictions": [
                {
                    "prediction": p.prediction,
                    "probability": p.probability,
                    "confidence": p.confidence.value,
                    "expected_range": p.expected_date_range
                }
                for p in forecast.predictions
            ],
            "scenarios": forecast.scenarios,
            "key_uncertainties": forecast.key_uncertainties,
            "recommended_preparations": forecast.recommended_preparations,
            "confidence_score": forecast.confidence_score,
            "timestamp": forecast.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predictive/predict")
async def quick_prediction(request: PredictionRequest):
    """
    Predicción rápida para un tema específico.
    """
    try:
        pi_service = get_predictive_intelligence_service()
        
        result = await pi_service.get_quick_prediction(request.subject)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predictive/weak-signals")
async def detect_weak_signals(request: WeakSignalRequest):
    """
    Detecta señales débiles en un dominio.
    """
    try:
        pi_service = get_predictive_intelligence_service()
        
        signals = pi_service.signal_detector.detect_signals(
            domain=request.domain,
            categories=request.categories if request.categories else None
        )
        
        return {
            "domain": request.domain,
            "signals_detected": len(signals),
            "signals": [
                {
                    "signal": s.signal,
                    "category": s.category,
                    "potential_impact": s.potential_impact,
                    "probability": s.probability,
                    "time_to_materialize": s.time_to_materialize,
                    "sources": s.sources
                }
                for s in signals
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Weak signals error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# GOOGLE SEARCH ENDPOINTS
# ============================================

@app.get("/api/v1/google/search")
async def google_search(
    q: str = Query(..., min_length=2, description="Search query"),
    num: int = Query(10, ge=1, le=10, description="Number of results"),
    lang: str = Query("es", description="Language code"),
    country: str = Query("co", description="Country code")
):
    """
    Búsqueda directa en Google.
    """
    try:
        google = get_google_service()
        result = await google.search.search(
            query=q,
            num_results=num,
            language=lang,
            country=country
        )
        
        return {
            "query": result.query,
            "total_results": result.total_results,
            "search_time": result.search_time,
            "results": [
                {
                    "title": r.title,
                    "link": r.link,
                    "snippet": r.snippet,
                    "relevance": r.relevance_score
                }
                for r in result.results
            ]
        }
        
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/google/news")
async def google_news_search(
    q: str = Query(..., min_length=2, description="Search query"),
    days: int = Query(7, ge=1, le=30, description="Days back")
):
    """
    Búsqueda de noticias en Google.
    """
    try:
        google = get_google_service()
        result = await google.search.search_news(
            query=q,
            days_back=days
        )
        
        return {
            "query": result.query,
            "period": f"Last {days} days",
            "total_results": result.total_results,
            "results": [
                {
                    "title": r.title,
                    "link": r.link,
                    "snippet": r.snippet
                }
                for r in result.results
            ]
        }
        
    except Exception as e:
        logger.error(f"Google news error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STATS & ADMIN ENDPOINTS
# ============================================

@app.get("/api/v1/stats")
async def system_stats():
    """Estadísticas del sistema."""
    try:
        rag = get_rag_engine()
        
        return {
            "rag": {
                "documents_indexed": rag.vector_store.collection.count() if hasattr(rag.vector_store, 'collection') else 0,
                "embedding_model": rag.embedding_engine.model_name
            },
            "apis": get_api_status(),
            "features": get_settings().get_enabled_features(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if get_settings().debug else "Contact support",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================
# ROOT
# ============================================

@app.get("/")
async def root():
    """Información de la API."""
    return {
        "name": "Market Intelligence Platform",
        "version": "2.0.0",
        "modules": [
            "Intelligence (RAG + Verification)",
            "Due Diligence / AML",
            "Competitive Intelligence",
            "Predictive Intelligence"
        ],
        "docs": "/docs",
        "health": "/health"
    }
