"""
Intelligence Agents - Multi-Agent System
=========================================
Sistema multi-agente para inteligencia de mercado.
Cada agente se especializa en un tipo de análisis.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json

from ..core.rag_engine import RAGEngine, get_rag_engine, GenerationResult
from ..core.config import get_settings, Constants

logger = logging.getLogger("market_intelligence.agents")


class AgentRole(Enum):
    """Roles de los agentes."""
    SEARCH = "search"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    PREDICTION = "prediction"
    MONITORING = "monitoring"


@dataclass
class AgentTask:
    """Tarea para un agente."""
    task_id: str
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None


@dataclass
class AgentResult:
    """Resultado de un agente."""
    agent_role: AgentRole
    task_id: str
    success: bool
    data: Dict[str, Any]
    confidence: float
    execution_time_ms: float
    sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BaseAgent(ABC):
    """Clase base para todos los agentes."""
    
    def __init__(self, role: AgentRole):
        self.role = role
        self.settings = get_settings()
        self.rag_engine = get_rag_engine()
        self.is_active = True
        logger.info(f"Agent initialized: {role.value}")
    
    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Ejecuta una tarea."""
        pass
    
    def _create_result(
        self,
        task: AgentTask,
        success: bool,
        data: Dict[str, Any],
        confidence: float,
        execution_time: float,
        sources: List[str] = None,
        errors: List[str] = None
    ) -> AgentResult:
        """Crea un resultado estandarizado."""
        return AgentResult(
            agent_role=self.role,
            task_id=task.task_id,
            success=success,
            data=data,
            confidence=confidence,
            execution_time_ms=execution_time,
            sources=sources or [],
            errors=errors or []
        )


class SearchAgent(BaseAgent):
    """
    Agente de Búsqueda
    ------------------
    Especializado en encontrar información relevante
    de múltiples fuentes internas y externas.
    """
    
    def __init__(self):
        super().__init__(AgentRole.SEARCH)
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Ejecuta búsqueda de información."""
        import time
        start_time = time.time()
        
        try:
            # Búsqueda en RAG
            rag_result = self.rag_engine.retrieve(task.query)
            
            # Extraer información relevante
            search_results = []
            for i, doc in enumerate(rag_result.documents):
                search_results.append({
                    "rank": i + 1,
                    "content": doc.page_content[:500],
                    "source": doc.metadata.get("source", "unknown"),
                    "source_type": doc.metadata.get("source_type", "primary"),
                    "relevance_score": rag_result.relevance_scores[i]
                })
            
            execution_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                task=task,
                success=True,
                data={
                    "query": task.query,
                    "results": search_results,
                    "total_found": len(search_results),
                    "retrieval_time_ms": rag_result.retrieval_time_ms
                },
                confidence=sum(rag_result.relevance_scores) / len(rag_result.relevance_scores) if rag_result.relevance_scores else 0,
                execution_time=execution_time,
                sources=[s.name for s in rag_result.sources]
            )
            
        except Exception as e:
            logger.error(f"SearchAgent error: {e}")
            return self._create_result(
                task=task,
                success=False,
                data={},
                confidence=0.0,
                execution_time=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )


class AnalysisAgent(BaseAgent):
    """
    Agente de Análisis
    ------------------
    Analiza datos de mercado, empresas y tendencias.
    Genera insights estructurados.
    """
    
    def __init__(self):
        super().__init__(AgentRole.ANALYSIS)
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Ejecuta análisis de mercado."""
        import time
        start_time = time.time()
        
        try:
            # Consulta RAG con análisis
            analysis_prompt = f"""Analiza la siguiente consulta de inteligencia de mercado:
            
{task.query}

Proporciona:
1. RESUMEN EJECUTIVO (3 puntos clave)
2. ANÁLISIS DE MERCADO (tamaño, crecimiento, drivers)
3. ANÁLISIS COMPETITIVO (principales jugadores, posicionamiento)
4. TENDENCIAS IDENTIFICADAS
5. OPORTUNIDADES Y RIESGOS
6. RECOMENDACIONES ACCIONABLES"""
            
            rag_result = self.rag_engine.query(analysis_prompt)
            
            # Estructurar análisis
            analysis_data = {
                "query": task.query,
                "analysis": rag_result.answer,
                "confidence": rag_result.confidence_score,
                "verification_status": rag_result.verification_status,
                "sources_count": len(rag_result.sources),
                "key_metrics": self._extract_metrics(rag_result.answer),
            }
            
            execution_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                task=task,
                success=True,
                data=analysis_data,
                confidence=rag_result.confidence_score,
                execution_time=execution_time,
                sources=[s.name for s in rag_result.sources]
            )
            
        except Exception as e:
            logger.error(f"AnalysisAgent error: {e}")
            return self._create_result(
                task=task,
                success=False,
                data={},
                confidence=0.0,
                execution_time=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
    
    def _extract_metrics(self, text: str) -> Dict[str, Any]:
        """Extrae métricas numéricas del análisis."""
        import re
        
        metrics = {}
        
        # Buscar valores monetarios
        money_pattern = r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?'
        money_matches = re.findall(money_pattern, text, re.IGNORECASE)
        if money_matches:
            metrics["monetary_values"] = money_matches[:5]
        
        # Buscar porcentajes
        percentage_pattern = r'[\d.]+%'
        percentage_matches = re.findall(percentage_pattern, text)
        if percentage_matches:
            metrics["percentages"] = percentage_matches[:5]
        
        # Buscar años
        year_pattern = r'\b20[2-3]\d\b'
        year_matches = re.findall(year_pattern, text)
        if year_matches:
            metrics["years_mentioned"] = list(set(year_matches))
        
        return metrics


class VerificationAgent(BaseAgent):
    """
    Agente de Verificación
    ----------------------
    Verifica la veracidad y consistencia de la información.
    Implementa triple verificación.
    """
    
    def __init__(self):
        super().__init__(AgentRole.VERIFICATION)
        self.verification_levels = ["primary", "secondary", "tertiary"]
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Ejecuta verificación de información."""
        import time
        start_time = time.time()
        
        try:
            claim = task.query
            context = task.context
            
            verification_results = []
            overall_confidence = 0.0
            
            # Nivel 1: Verificación primaria (fuentes oficiales)
            primary_check = await self._verify_primary(claim)
            verification_results.append({
                "level": "primary",
                "status": primary_check["status"],
                "confidence": primary_check["confidence"],
                "sources": primary_check["sources"]
            })
            
            # Nivel 2: Verificación secundaria (cross-reference)
            secondary_check = await self._verify_secondary(claim)
            verification_results.append({
                "level": "secondary",
                "status": secondary_check["status"],
                "confidence": secondary_check["confidence"],
                "sources": secondary_check["sources"]
            })
            
            # Nivel 3: Verificación terciaria (consenso)
            tertiary_check = await self._verify_tertiary(claim, verification_results)
            verification_results.append({
                "level": "tertiary",
                "status": tertiary_check["status"],
                "confidence": tertiary_check["confidence"],
                "discrepancies": tertiary_check.get("discrepancies", [])
            })
            
            # Calcular confianza global
            confidences = [v["confidence"] for v in verification_results]
            overall_confidence = sum(confidences) / len(confidences)
            
            # Determinar estado final
            if all(v["status"] == "verified" for v in verification_results):
                final_status = "verified"
            elif any(v["status"] == "failed" for v in verification_results):
                final_status = "failed"
            else:
                final_status = "partial"
            
            execution_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                task=task,
                success=final_status != "failed",
                data={
                    "claim": claim,
                    "final_status": final_status,
                    "verification_levels": verification_results,
                    "overall_confidence": overall_confidence,
                    "recommendation": self._get_recommendation(final_status, overall_confidence)
                },
                confidence=overall_confidence,
                execution_time=execution_time,
                sources=primary_check["sources"] + secondary_check["sources"]
            )
            
        except Exception as e:
            logger.error(f"VerificationAgent error: {e}")
            return self._create_result(
                task=task,
                success=False,
                data={},
                confidence=0.0,
                execution_time=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
    
    async def _verify_primary(self, claim: str) -> Dict[str, Any]:
        """Verificación contra fuentes primarias."""
        # Buscar en fuentes primarias (regulatory, official)
        result = self.rag_engine.retrieve(
            claim, 
            source_types=["primary", "regulatory"]
        )
        
        if result.documents:
            avg_score = sum(result.relevance_scores) / len(result.relevance_scores)
            return {
                "status": "verified" if avg_score > 0.8 else "partial",
                "confidence": avg_score,
                "sources": [s.name for s in result.sources]
            }
        
        return {
            "status": "unverified",
            "confidence": 0.0,
            "sources": []
        }
    
    async def _verify_secondary(self, claim: str) -> Dict[str, Any]:
        """Verificación contra fuentes secundarias."""
        result = self.rag_engine.retrieve(
            claim,
            source_types=["secondary", "alternative"]
        )
        
        if result.documents:
            avg_score = sum(result.relevance_scores) / len(result.relevance_scores)
            return {
                "status": "verified" if avg_score > 0.75 else "partial",
                "confidence": avg_score,
                "sources": [s.name for s in result.sources]
            }
        
        return {
            "status": "unverified",
            "confidence": 0.0,
            "sources": []
        }
    
    async def _verify_tertiary(
        self, 
        claim: str, 
        previous_results: List[Dict]
    ) -> Dict[str, Any]:
        """Verificación de consenso entre fuentes."""
        # Verificar consistencia entre niveles
        primary_confidence = previous_results[0]["confidence"]
        secondary_confidence = previous_results[1]["confidence"]
        
        # Detectar discrepancias
        discrepancies = []
        confidence_diff = abs(primary_confidence - secondary_confidence)
        
        if confidence_diff > 0.3:
            discrepancies.append({
                "type": "confidence_divergence",
                "description": f"Diferencia de confianza significativa entre fuentes: {confidence_diff:.2f}",
                "severity": "high" if confidence_diff > 0.5 else "medium"
            })
        
        # Calcular consenso
        consensus_confidence = (primary_confidence + secondary_confidence) / 2
        
        if primary_confidence > 0.8 and secondary_confidence > 0.8:
            status = "verified"
        elif primary_confidence > 0.6 or secondary_confidence > 0.6:
            status = "partial"
        else:
            status = "unverified"
        
        return {
            "status": status,
            "confidence": consensus_confidence,
            "discrepancies": discrepancies
        }
    
    def _get_recommendation(self, status: str, confidence: float) -> str:
        """Genera recomendación basada en verificación."""
        if status == "verified" and confidence > 0.9:
            return "Alta confiabilidad. Información verificada por múltiples fuentes."
        elif status == "verified" and confidence > 0.8:
            return "Confiable. Verificado con algunas limitaciones menores."
        elif status == "partial":
            return "Precaución. Verificación parcial. Se recomienda validación adicional."
        else:
            return "No verificado. Tratar como no confirmado. Requiere fuentes adicionales."


class SynthesisAgent(BaseAgent):
    """
    Agente de Síntesis
    ------------------
    Sintetiza información de múltiples agentes
    y genera reportes consolidados.
    """
    
    def __init__(self):
        super().__init__(AgentRole.SYNTHESIS)
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Sintetiza resultados de múltiples fuentes."""
        import time
        start_time = time.time()
        
        try:
            # Obtener resultados previos del contexto
            agent_results = task.context.get("agent_results", [])
            
            if not agent_results:
                # Ejecutar búsqueda y análisis si no hay resultados previos
                search_agent = SearchAgent()
                analysis_agent = AnalysisAgent()
                
                search_task = AgentTask(
                    task_id=f"{task.task_id}_search",
                    query=task.query
                )
                analysis_task = AgentTask(
                    task_id=f"{task.task_id}_analysis",
                    query=task.query
                )
                
                search_result = await search_agent.execute(search_task)
                analysis_result = await analysis_agent.execute(analysis_task)
                
                agent_results = [search_result, analysis_result]
            
            # Sintetizar resultados
            synthesis = self._synthesize_results(task.query, agent_results)
            
            execution_time = (time.time() - start_time) * 1000
            
            return self._create_result(
                task=task,
                success=True,
                data={
                    "query": task.query,
                    "synthesis": synthesis,
                    "agents_used": [r.agent_role.value for r in agent_results],
                    "aggregate_confidence": self._calculate_aggregate_confidence(agent_results)
                },
                confidence=self._calculate_aggregate_confidence(agent_results),
                execution_time=execution_time,
                sources=self._aggregate_sources(agent_results)
            )
            
        except Exception as e:
            logger.error(f"SynthesisAgent error: {e}")
            return self._create_result(
                task=task,
                success=False,
                data={},
                confidence=0.0,
                execution_time=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
    
    def _synthesize_results(
        self, 
        query: str, 
        results: List[AgentResult]
    ) -> Dict[str, Any]:
        """Sintetiza resultados de múltiples agentes."""
        synthesis = {
            "executive_summary": "",
            "key_findings": [],
            "data_points": [],
            "recommendations": [],
            "confidence_assessment": "",
            "gaps_identified": []
        }
        
        for result in results:
            if result.success:
                if result.agent_role == AgentRole.SEARCH:
                    # Extraer hallazgos de búsqueda
                    if "results" in result.data:
                        for r in result.data["results"][:3]:
                            synthesis["data_points"].append({
                                "content": r.get("content", "")[:200],
                                "source": r.get("source"),
                                "relevance": r.get("relevance_score")
                            })
                
                elif result.agent_role == AgentRole.ANALYSIS:
                    # Extraer análisis
                    if "analysis" in result.data:
                        synthesis["executive_summary"] = result.data["analysis"][:500]
                    if "key_metrics" in result.data:
                        synthesis["data_points"].extend([
                            {"metric": k, "values": v} 
                            for k, v in result.data["key_metrics"].items()
                        ])
                
                elif result.agent_role == AgentRole.VERIFICATION:
                    # Extraer estado de verificación
                    synthesis["confidence_assessment"] = result.data.get(
                        "recommendation", 
                        "Verificación no disponible"
                    )
        
        return synthesis
    
    def _calculate_aggregate_confidence(self, results: List[AgentResult]) -> float:
        """Calcula confianza agregada."""
        if not results:
            return 0.0
        
        successful = [r for r in results if r.success]
        if not successful:
            return 0.0
        
        return sum(r.confidence for r in successful) / len(successful)
    
    def _aggregate_sources(self, results: List[AgentResult]) -> List[str]:
        """Agrega fuentes de todos los resultados."""
        sources = set()
        for result in results:
            sources.update(result.sources)
        return list(sources)


class AgentOrchestrator:
    """
    Orquestador de Agentes
    ----------------------
    Coordina la ejecución de múltiples agentes
    para resolver consultas complejas.
    """
    
    def __init__(self):
        self.agents = {
            AgentRole.SEARCH: SearchAgent(),
            AgentRole.ANALYSIS: AnalysisAgent(),
            AgentRole.VERIFICATION: VerificationAgent(),
            AgentRole.SYNTHESIS: SynthesisAgent(),
        }
        self.settings = get_settings()
        logger.info("Agent Orchestrator initialized with agents: " + 
                   ", ".join(a.value for a in self.agents.keys()))
    
    async def process_query(
        self, 
        query: str,
        include_verification: bool = True,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa una consulta usando múltiples agentes.
        
        Args:
            query: Consulta del usuario
            include_verification: Incluir verificación
            parallel: Ejecutar agentes en paralelo
        
        Returns:
            Resultado consolidado de todos los agentes
        """
        import time
        import uuid
        
        start_time = time.time()
        task_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Processing query [{task_id}]: {query[:50]}...")
        
        # Crear tareas
        tasks = []
        
        # 1. Búsqueda
        search_task = AgentTask(task_id=f"{task_id}_search", query=query)
        tasks.append((AgentRole.SEARCH, search_task))
        
        # 2. Análisis
        analysis_task = AgentTask(task_id=f"{task_id}_analysis", query=query)
        tasks.append((AgentRole.ANALYSIS, analysis_task))
        
        # 3. Verificación (opcional)
        if include_verification:
            verify_task = AgentTask(task_id=f"{task_id}_verify", query=query)
            tasks.append((AgentRole.VERIFICATION, verify_task))
        
        # Ejecutar agentes
        results = {}
        
        if parallel:
            # Ejecución paralela
            async_tasks = [
                self.agents[role].execute(task) 
                for role, task in tasks
            ]
            agent_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            for (role, _), result in zip(tasks, agent_results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {role.value} failed: {result}")
                    results[role.value] = {"success": False, "error": str(result)}
                else:
                    results[role.value] = {
                        "success": result.success,
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time_ms": result.execution_time_ms,
                        "sources": result.sources
                    }
        else:
            # Ejecución secuencial
            for role, task in tasks:
                try:
                    result = await self.agents[role].execute(task)
                    results[role.value] = {
                        "success": result.success,
                        "data": result.data,
                        "confidence": result.confidence,
                        "execution_time_ms": result.execution_time_ms,
                        "sources": result.sources
                    }
                except Exception as e:
                    logger.error(f"Agent {role.value} failed: {e}")
                    results[role.value] = {"success": False, "error": str(e)}
        
        # 4. Síntesis final
        synthesis_task = AgentTask(
            task_id=f"{task_id}_synthesis",
            query=query,
            context={"agent_results": [
                AgentResult(
                    agent_role=AgentRole[k.upper()],
                    task_id=f"{task_id}_{k}",
                    success=v.get("success", False),
                    data=v.get("data", {}),
                    confidence=v.get("confidence", 0),
                    execution_time_ms=v.get("execution_time_ms", 0),
                    sources=v.get("sources", [])
                )
                for k, v in results.items() if "success" in v
            ]}
        )
        
        synthesis_result = await self.agents[AgentRole.SYNTHESIS].execute(synthesis_task)
        results["synthesis"] = {
            "success": synthesis_result.success,
            "data": synthesis_result.data,
            "confidence": synthesis_result.confidence,
            "execution_time_ms": synthesis_result.execution_time_ms,
            "sources": synthesis_result.sources
        }
        
        total_time = (time.time() - start_time) * 1000
        
        # Construir respuesta final
        return {
            "task_id": task_id,
            "query": query,
            "results": results,
            "overall_confidence": synthesis_result.confidence,
            "total_execution_time_ms": total_time,
            "agents_executed": list(results.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }


# Singleton instance
_orchestrator: Optional[AgentOrchestrator] = None

def get_orchestrator() -> AgentOrchestrator:
    """Obtiene la instancia singleton del orquestador."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
