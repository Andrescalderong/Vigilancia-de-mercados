"""
Market Intelligence API - FastAPI Application
==============================================
API REST para la plataforma de inteligencia de mercado.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import asyncio

from .core.config import get_settings, Constants
from .core.rag_engine import get_rag_engine, RAGEngine
from .agents.intelligence_agents import get_orchestrator, AgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market_intelligence.api")

# Initialize FastAPI
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Market Intelligence AI Platform
    
    Plataforma de inteligencia de mercado impulsada por IA con:
    - **RAG (Retrieval-Augmented Generation)** para respuestas verificables
    - **Sistema Multi-Agente** para análisis especializado
    - **Triple Verificación** para máxima confiabilidad
    
    ### Características principales:
    - 🔍 Búsqueda semántica en múltiples fuentes
    - 📊 Análisis de mercado y competencia
    - ✅ Verificación automática de información
    - 🎯 Detección de señales de mercado
    - 📈 Inteligencia predictiva
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class QueryRequest(BaseModel):
    """Solicitud de consulta de inteligencia."""
    query: str = Field(..., min_length=3, max_length=2000, description="Pregunta o consulta de inteligencia")
    include_verification: bool = Field(True, description="Incluir triple verificación")
    source_types: Optional[List[str]] = Field(None, description="Filtrar por tipos de fuente")
    max_results: int = Field(5, ge=1, le=20, description="Número máximo de resultados")

class QueryResponse(BaseModel):
    """Respuesta a una consulta de inteligencia."""
    task_id: str
    query: str
    answer: str
    confidence_score: float
    verification_status: str
    sources: List[Dict[str, Any]]
    execution_time_ms: float
    timestamp: datetime

class DocumentIngestRequest(BaseModel):
    """Solicitud para ingestar un documento."""
    content: str = Field(..., min_length=10, description="Contenido del documento")
    source_name: str = Field(..., min_length=1, description="Nombre de la fuente")
    source_type: str = Field("primary", description="Tipo de fuente")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")

class DocumentIngestResponse(BaseModel):
    """Respuesta de ingesta de documento."""
    success: bool
    chunks_indexed: int
    source_name: str
    message: str

class MarketSearchRequest(BaseModel):
    """Solicitud de búsqueda de mercado."""
    query: str = Field(..., description="Término de búsqueda")
    sectors: Optional[List[str]] = Field(None, description="Sectores a filtrar")
    top_k: int = Field(10, ge=1, le=50, description="Número de resultados")

class CompanySearchRequest(BaseModel):
    """Solicitud de búsqueda de empresa."""
    company_name: str = Field(..., description="Nombre de la empresa")
    include_financials: bool = Field(True, description="Incluir datos financieros")
    include_news: bool = Field(True, description="Incluir noticias recientes")

class VerificationRequest(BaseModel):
    """Solicitud de verificación de claim."""
    claim: str = Field(..., description="Afirmación a verificar")
    context: Optional[Dict[str, Any]] = Field(None, description="Contexto adicional")

class HealthResponse(BaseModel):
    """Respuesta de health check."""
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]


# ============== Dependencies ==============

def get_rag() -> RAGEngine:
    """Dependency para obtener RAG Engine."""
    return get_rag_engine()

def get_agent_orchestrator() -> AgentOrchestrator:
    """Dependency para obtener Agent Orchestrator."""
    return get_orchestrator()


# ============== API Endpoints ==============

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(rag: RAGEngine = Depends(get_rag)):
    """
    Health check del sistema.
    Verifica el estado de todos los componentes.
    """
    components = {
        "api": "healthy",
        "rag_engine": "healthy" if rag else "unhealthy",
        "vector_store": "healthy",  # TODO: Add actual check
        "llm": "configured" if settings.ANTHROPIC_API_KEY or settings.OPENAI_API_KEY else "not_configured"
    }
    
    overall_status = "healthy" if all(v in ["healthy", "configured"] for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        components=components
    )


# ============== Intelligence Endpoints ==============

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Intelligence"])
async def query_intelligence(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    Consulta de Inteligencia de Mercado
    
    Ejecuta el pipeline completo de RAG con verificación:
    1. Búsqueda semántica en fuentes indexadas
    2. Análisis con LLM
    3. Verificación triple (opcional)
    4. Síntesis de resultados
    
    **Ejemplo de consulta:**
    - "¿Cuál es el tamaño del mercado de IA en 2025?"
    - "Analiza la competencia en el sector de inteligencia de mercado"
    - "¿Qué tendencias emergentes hay en RAG para finanzas?"
    """
    try:
        # Ejecutar pipeline multi-agente
        result = await orchestrator.process_query(
            query=request.query,
            include_verification=request.include_verification,
            parallel=True
        )
        
        # Extraer respuesta principal
        synthesis_data = result.get("results", {}).get("synthesis", {}).get("data", {})
        analysis_data = result.get("results", {}).get("analysis", {}).get("data", {})
        
        answer = synthesis_data.get("synthesis", {}).get("executive_summary", "") or \
                 analysis_data.get("analysis", "No se pudo generar análisis.")
        
        # Agregar fuentes
        all_sources = []
        for agent_key, agent_result in result.get("results", {}).items():
            if isinstance(agent_result, dict) and "sources" in agent_result:
                for source in agent_result["sources"]:
                    if source not in [s["name"] for s in all_sources]:
                        all_sources.append({
                            "name": source,
                            "type": "verified",
                            "agent": agent_key
                        })
        
        # Determinar estado de verificación
        verification_result = result.get("results", {}).get("verification", {})
        verification_status = "verified"
        if verification_result:
            v_data = verification_result.get("data", {})
            verification_status = v_data.get("final_status", "pending")
        
        return QueryResponse(
            task_id=result.get("task_id", "unknown"),
            query=request.query,
            answer=answer,
            confidence_score=result.get("overall_confidence", 0.0),
            verification_status=verification_status,
            sources=all_sources[:10],  # Limitar a 10 fuentes
            execution_time_ms=result.get("total_execution_time_ms", 0),
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search", tags=["Intelligence"])
async def search_markets(
    request: MarketSearchRequest,
    rag: RAGEngine = Depends(get_rag)
):
    """
    Búsqueda de Mercados
    
    Búsqueda semántica rápida en la base de conocimiento.
    Ideal para exploración inicial de mercados.
    """
    try:
        result = rag.retrieve(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "results": [
                {
                    "content": doc.page_content[:300],
                    "source": doc.metadata.get("source", "unknown"),
                    "source_type": doc.metadata.get("source_type", "primary"),
                    "relevance_score": score
                }
                for doc, score in zip(result.documents, result.relevance_scores)
            ],
            "total_results": len(result.documents),
            "retrieval_time_ms": result.retrieval_time_ms
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/verify", tags=["Intelligence"])
async def verify_claim(
    request: VerificationRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator)
):
    """
    Verificación de Información
    
    Verifica una afirmación usando el sistema de triple verificación:
    1. Verificación primaria (fuentes oficiales)
    2. Verificación secundaria (cross-reference)
    3. Verificación terciaria (consenso)
    """
    try:
        from .agents.intelligence_agents import VerificationAgent, AgentTask
        import uuid
        
        agent = orchestrator.agents.get(
            list(orchestrator.agents.keys())[2]  # Verification agent
        )
        
        task = AgentTask(
            task_id=str(uuid.uuid4())[:8],
            query=request.claim,
            context=request.context or {}
        )
        
        result = await agent.execute(task)
        
        return {
            "claim": request.claim,
            "verification_result": result.data,
            "overall_confidence": result.confidence,
            "execution_time_ms": result.execution_time_ms,
            "recommendation": result.data.get("recommendation", "")
        }
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Data Ingestion Endpoints ==============

@app.post("/api/v1/ingest", response_model=DocumentIngestResponse, tags=["Data"])
async def ingest_document(
    request: DocumentIngestRequest,
    rag: RAGEngine = Depends(get_rag)
):
    """
    Ingestar Documento
    
    Añade un nuevo documento a la base de conocimiento.
    El documento será chunkeado y embedido automáticamente.
    
    **Tipos de fuente válidos:**
    - primary: Fuentes oficiales (SEC, reguladores)
    - secondary: Reportes de analistas, investigación
    - alternative: Datos alternativos (patentes, job postings)
    - regulatory: Documentos regulatorios
    """
    try:
        chunks_count = rag.ingest_document(
            content=request.content,
            source_name=request.source_name,
            source_type=request.source_type,
            metadata=request.metadata
        )
        
        return DocumentIngestResponse(
            success=True,
            chunks_indexed=chunks_count,
            source_name=request.source_name,
            message=f"Documento '{request.source_name}' indexado exitosamente con {chunks_count} chunks."
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest/batch", tags=["Data"])
async def ingest_batch(
    documents: List[DocumentIngestRequest],
    background_tasks: BackgroundTasks,
    rag: RAGEngine = Depends(get_rag)
):
    """
    Ingestar Múltiples Documentos
    
    Ingesta un batch de documentos en segundo plano.
    Útil para cargas masivas de datos.
    """
    def process_batch(docs: List[DocumentIngestRequest]):
        results = []
        for doc in docs:
            try:
                chunks = rag.ingest_document(
                    content=doc.content,
                    source_name=doc.source_name,
                    source_type=doc.source_type,
                    metadata=doc.metadata
                )
                results.append({"source": doc.source_name, "chunks": chunks, "success": True})
            except Exception as e:
                results.append({"source": doc.source_name, "error": str(e), "success": False})
        logger.info(f"Batch ingestion completed: {len(results)} documents processed")
    
    background_tasks.add_task(process_batch, documents)
    
    return {
        "status": "processing",
        "documents_queued": len(documents),
        "message": "Batch ingestion started in background"
    }


# ============== Analytics Endpoints ==============

@app.get("/api/v1/stats", tags=["Analytics"])
async def get_system_stats(rag: RAGEngine = Depends(get_rag)):
    """
    Estadísticas del Sistema
    
    Retorna métricas y estadísticas del sistema.
    """
    try:
        doc_count = rag.vector_store.collection.count()
        
        return {
            "documents_indexed": doc_count,
            "vector_store_status": "active",
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "rag_config": {
                "chunk_size": settings.RAG_CHUNK_SIZE,
                "chunk_overlap": settings.RAG_CHUNK_OVERLAP,
                "top_k": settings.RAG_TOP_K,
                "similarity_threshold": settings.RAG_SIMILARITY_THRESHOLD
            },
            "verification_config": {
                "min_sources": settings.VERIFICATION_MIN_SOURCES,
                "confidence_threshold": settings.VERIFICATION_CONFIDENCE_THRESHOLD,
                "triple_verification_enabled": settings.ENABLE_TRIPLE_VERIFICATION
            }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Error Handlers ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500
        }
    )


# ============== Startup/Shutdown Events ==============

@app.on_event("startup")
async def startup_event():
    """Inicialización al arranque."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Inicializar componentes
    try:
        rag = get_rag_engine()
        logger.info("RAG Engine initialized")
        
        orchestrator = get_orchestrator()
        logger.info("Agent Orchestrator initialized")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al apagar."""
    logger.info("Shutting down Market Intelligence Platform")


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )
