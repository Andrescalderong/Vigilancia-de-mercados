"""
Predictive Intelligence Service
================================
Servicio de inteligencia predictiva para mercados y competencia.
Utiliza análisis de tendencias, patrones y señales débiles.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import math
import statistics

from ..core.config import get_settings
from ..core.rag_engine import get_rag_engine

logger = logging.getLogger("market_intelligence.services.predictive")


class TrendDirection(Enum):
    """Dirección de una tendencia."""
    STRONG_UP = "strong_up"
    UP = "up"
    STABLE = "stable"
    DOWN = "down"
    STRONG_DOWN = "strong_down"
    UNCERTAIN = "uncertain"


class PredictionConfidence(Enum):
    """Nivel de confianza de una predicción."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


class TimeHorizon(Enum):
    """Horizonte temporal de predicción."""
    SHORT_TERM = "short_term"      # 0-3 meses
    MEDIUM_TERM = "medium_term"    # 3-12 meses
    LONG_TERM = "long_term"        # 1-3 años
    STRATEGIC = "strategic"        # 3+ años


@dataclass
class TrendAnalysis:
    """Análisis de tendencia."""
    name: str
    direction: TrendDirection
    strength: float  # 0-1
    momentum: float  # -1 to 1
    data_points: int
    timeframe: str
    supporting_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class WeakSignal:
    """Señal débil detectada."""
    signal: str
    category: str
    potential_impact: str
    probability: float
    time_to_materialize: str
    sources: List[str] = field(default_factory=list)
    related_trends: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Prediction:
    """Predicción de mercado."""
    subject: str
    prediction: str
    confidence: PredictionConfidence
    probability: float  # 0-100
    time_horizon: TimeHorizon
    expected_date_range: str
    supporting_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    scenarios: Dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketForecast:
    """Pronóstico de mercado completo."""
    market_name: str
    current_state: Dict[str, Any]
    trends: List[TrendAnalysis]
    weak_signals: List[WeakSignal]
    predictions: List[Prediction]
    scenarios: Dict[str, Dict[str, Any]]
    key_uncertainties: List[str]
    recommended_preparations: List[str]
    forecast_horizon: TimeHorizon
    generated_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0


class TrendAnalyzer:
    """
    Analizador de tendencias.
    Detecta y analiza tendencias en datos de mercado.
    """
    
    def __init__(self):
        self.rag_engine = get_rag_engine()
    
    def analyze_trend(
        self, 
        topic: str, 
        data_points: List[Dict[str, Any]] = None
    ) -> TrendAnalysis:
        """
        Analiza una tendencia específica.
        
        Args:
            topic: Tema o métrica a analizar
            data_points: Puntos de datos históricos (opcional)
        
        Returns:
            Análisis de la tendencia
        """
        # Buscar información de tendencia en RAG
        query = f"{topic} trend growth trajectory forecast"
        result = self.rag_engine.retrieve(query, top_k=5)
        
        # Extraer evidencia
        evidence = [doc.page_content[:200] for doc in result.documents]
        
        # Analizar dirección y fuerza
        direction, strength = self._determine_direction(evidence)
        
        # Calcular momentum
        momentum = self._calculate_momentum(evidence)
        
        return TrendAnalysis(
            name=topic,
            direction=direction,
            strength=strength,
            momentum=momentum,
            data_points=len(result.documents),
            timeframe="Based on available data",
            supporting_evidence=evidence,
            confidence=sum(result.relevance_scores) / len(result.relevance_scores) if result.relevance_scores else 0
        )
    
    def _determine_direction(
        self, 
        evidence: List[str]
    ) -> Tuple[TrendDirection, float]:
        """Determina dirección y fuerza de la tendencia."""
        positive_indicators = [
            "growth", "increase", "rising", "expanding", "surge",
            "acceleration", "momentum", "boom", "uptick", "gain"
        ]
        negative_indicators = [
            "decline", "decrease", "falling", "shrinking", "drop",
            "slowdown", "contraction", "downturn", "loss", "reduction"
        ]
        
        positive_count = 0
        negative_count = 0
        
        for text in evidence:
            text_lower = text.lower()
            positive_count += sum(1 for ind in positive_indicators if ind in text_lower)
            negative_count += sum(1 for ind in negative_indicators if ind in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return TrendDirection.UNCERTAIN, 0.0
        
        score = (positive_count - negative_count) / total
        strength = abs(score)
        
        if score > 0.5:
            return TrendDirection.STRONG_UP, strength
        elif score > 0.1:
            return TrendDirection.UP, strength
        elif score < -0.5:
            return TrendDirection.STRONG_DOWN, strength
        elif score < -0.1:
            return TrendDirection.DOWN, strength
        return TrendDirection.STABLE, strength
    
    def _calculate_momentum(self, evidence: List[str]) -> float:
        """Calcula el momentum de la tendencia."""
        # Palabras que indican aceleración/desaceleración
        acceleration = ["accelerating", "faster", "rapid", "exponential", "surge"]
        deceleration = ["slowing", "plateauing", "maturing", "stabilizing"]
        
        accel_count = 0
        decel_count = 0
        
        for text in evidence:
            text_lower = text.lower()
            accel_count += sum(1 for a in acceleration if a in text_lower)
            decel_count += sum(1 for d in deceleration if d in text_lower)
        
        total = accel_count + decel_count
        if total == 0:
            return 0.0
        
        return (accel_count - decel_count) / total


class WeakSignalDetector:
    """
    Detector de señales débiles.
    Identifica indicadores tempranos de cambios futuros.
    """
    
    # Categorías de señales débiles
    SIGNAL_CATEGORIES = {
        "technology": [
            "emerging technology", "breakthrough", "prototype",
            "research paper", "patent", "beta", "pilot program"
        ],
        "regulatory": [
            "proposed regulation", "draft legislation", "policy review",
            "regulatory consultation", "compliance", "standard"
        ],
        "market_shift": [
            "new entrant", "pivot", "disruption", "convergence",
            "vertical integration", "platform shift"
        ],
        "consumer_behavior": [
            "changing preference", "adoption curve", "demographic shift",
            "usage pattern", "sentiment", "review trend"
        ],
        "economic": [
            "interest rate", "inflation", "employment", "GDP",
            "commodity price", "currency", "tariff"
        ],
        "talent": [
            "hiring trend", "skill shortage", "talent migration",
            "salary trend", "remote work", "expertise"
        ]
    }
    
    def __init__(self):
        self.rag_engine = get_rag_engine()
    
    def detect_signals(
        self, 
        domain: str,
        categories: List[str] = None
    ) -> List[WeakSignal]:
        """
        Detecta señales débiles en un dominio.
        
        Args:
            domain: Dominio de búsqueda
            categories: Categorías específicas a buscar
        
        Returns:
            Lista de señales débiles detectadas
        """
        signals = []
        search_categories = categories or list(self.SIGNAL_CATEGORIES.keys())
        
        for category in search_categories:
            if category not in self.SIGNAL_CATEGORIES:
                continue
                
            keywords = self.SIGNAL_CATEGORIES[category]
            
            for keyword in keywords[:3]:  # Limitar búsquedas
                query = f"{domain} {keyword} early indicator emerging"
                result = self.rag_engine.retrieve(query, top_k=2)
                
                for doc, score in zip(result.documents, result.relevance_scores):
                    if score > 0.7:  # Solo señales relevantes
                        signal = WeakSignal(
                            signal=f"{keyword.title()} en {domain}",
                            category=category,
                            potential_impact=self._assess_impact(category, score),
                            probability=score * 0.7,  # Conservador
                            time_to_materialize=self._estimate_timing(category),
                            sources=[doc.metadata.get("source", "RAG")],
                            related_trends=[]
                        )
                        signals.append(signal)
        
        # Deduplicar y ordenar por probabilidad
        unique_signals = self._deduplicate_signals(signals)
        unique_signals.sort(key=lambda x: x.probability, reverse=True)
        
        return unique_signals[:10]  # Top 10 señales
    
    def _assess_impact(self, category: str, relevance: float) -> str:
        """Evalúa el impacto potencial de una señal."""
        high_impact_categories = ["technology", "regulatory", "market_shift"]
        
        if category in high_impact_categories and relevance > 0.8:
            return "Alto - Potencial de transformación significativa"
        elif relevance > 0.75:
            return "Medio - Impacto notable en el mercado"
        return "Bajo - Impacto incremental"
    
    def _estimate_timing(self, category: str) -> str:
        """Estima timing de materialización."""
        timing_map = {
            "technology": "12-36 meses",
            "regulatory": "6-24 meses",
            "market_shift": "6-18 meses",
            "consumer_behavior": "3-12 meses",
            "economic": "3-6 meses",
            "talent": "6-12 meses"
        }
        return timing_map.get(category, "12-24 meses")
    
    def _deduplicate_signals(
        self, 
        signals: List[WeakSignal]
    ) -> List[WeakSignal]:
        """Elimina señales duplicadas."""
        seen = set()
        unique = []
        
        for signal in signals:
            key = (signal.category, signal.signal[:30])
            if key not in seen:
                seen.add(key)
                unique.append(signal)
        
        return unique


class ScenarioPlanner:
    """
    Planificador de escenarios.
    Genera escenarios futuros basados en incertidumbres clave.
    """
    
    def __init__(self):
        self.rag_engine = get_rag_engine()
    
    def generate_scenarios(
        self, 
        subject: str,
        key_uncertainties: List[str],
        time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    ) -> Dict[str, Dict[str, Any]]:
        """
        Genera escenarios basados en incertidumbres.
        
        Args:
            subject: Tema del escenario
            key_uncertainties: Principales incertidumbres
            time_horizon: Horizonte temporal
        
        Returns:
            Diccionario de escenarios
        """
        scenarios = {}
        
        # Escenario Optimista
        scenarios["optimistic"] = {
            "name": "Escenario Optimista",
            "probability": 0.25,
            "description": f"Crecimiento acelerado para {subject}",
            "key_assumptions": [
                f"Resolución favorable de: {key_uncertainties[0] if key_uncertainties else 'incertidumbres clave'}",
                "Adopción de mercado superior a expectativas",
                "Condiciones económicas favorables"
            ],
            "implications": [
                "Oportunidad de expansión agresiva",
                "Necesidad de escalar capacidad",
                "Potencial de liderazgo de mercado"
            ],
            "recommended_preparations": [
                "Preparar planes de escalamiento",
                "Asegurar recursos para crecimiento",
                "Desarrollar estrategia de liderazgo"
            ]
        }
        
        # Escenario Base
        scenarios["base"] = {
            "name": "Escenario Base",
            "probability": 0.50,
            "description": f"Desarrollo esperado para {subject}",
            "key_assumptions": [
                "Tendencias actuales continúan",
                "Incertidumbres se resuelven gradualmente",
                "Competencia mantiene patrones actuales"
            ],
            "implications": [
                "Crecimiento moderado sostenible",
                "Necesidad de diferenciación",
                "Optimización de operaciones"
            ],
            "recommended_preparations": [
                "Ejecutar estrategia actual",
                "Monitorear señales de cambio",
                "Mantener flexibilidad"
            ]
        }
        
        # Escenario Pesimista
        scenarios["pessimistic"] = {
            "name": "Escenario Pesimista",
            "probability": 0.20,
            "description": f"Desafíos significativos para {subject}",
            "key_assumptions": [
                f"Resolución desfavorable de: {key_uncertainties[0] if key_uncertainties else 'incertidumbres'}",
                "Competencia intensificada",
                "Condiciones de mercado adversas"
            ],
            "implications": [
                "Necesidad de ajustar expectativas",
                "Posible reestructuración",
                "Foco en eficiencia"
            ],
            "recommended_preparations": [
                "Desarrollar planes de contingencia",
                "Fortalecer posición financiera",
                "Identificar opciones de pivot"
            ]
        }
        
        # Escenario Disruptivo
        scenarios["disruptive"] = {
            "name": "Escenario Disruptivo",
            "probability": 0.05,
            "description": f"Cambio radical en {subject}",
            "key_assumptions": [
                "Disrupción tecnológica mayor",
                "Cambio regulatorio fundamental",
                "Evento de cisne negro"
            ],
            "implications": [
                "Transformación del mercado",
                "Obsolescencia de modelos actuales",
                "Nuevas oportunidades emergentes"
            ],
            "recommended_preparations": [
                "Monitorear señales de disrupción",
                "Mantener optionality",
                "Desarrollar capacidades de adaptación"
            ]
        }
        
        return scenarios


class PredictionEngine:
    """
    Motor de predicciones.
    Genera predicciones basadas en tendencias, señales y escenarios.
    """
    
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.signal_detector = WeakSignalDetector()
        self.scenario_planner = ScenarioPlanner()
        self.rag_engine = get_rag_engine()
    
    def generate_prediction(
        self, 
        subject: str,
        time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    ) -> Prediction:
        """
        Genera una predicción para un tema.
        
        Args:
            subject: Tema de la predicción
            time_horizon: Horizonte temporal
        
        Returns:
            Predicción generada
        """
        # Analizar tendencia
        trend = self.trend_analyzer.analyze_trend(subject)
        
        # Generar predicción base
        prediction_text = self._generate_prediction_text(subject, trend, time_horizon)
        
        # Calcular probabilidad y confianza
        probability = self._calculate_probability(trend)
        confidence = self._determine_confidence(trend, time_horizon)
        
        # Identificar factores
        supporting, risks = self._identify_factors(trend)
        
        return Prediction(
            subject=subject,
            prediction=prediction_text,
            confidence=confidence,
            probability=probability,
            time_horizon=time_horizon,
            expected_date_range=self._get_date_range(time_horizon),
            supporting_factors=supporting,
            risk_factors=risks,
            scenarios={"trend_direction": trend.direction.value}
        )
    
    def _generate_prediction_text(
        self, 
        subject: str, 
        trend: TrendAnalysis,
        horizon: TimeHorizon
    ) -> str:
        """Genera texto de la predicción."""
        direction_text = {
            TrendDirection.STRONG_UP: "crecimiento significativo",
            TrendDirection.UP: "crecimiento moderado",
            TrendDirection.STABLE: "estabilidad relativa",
            TrendDirection.DOWN: "contracción moderada",
            TrendDirection.STRONG_DOWN: "declive significativo",
            TrendDirection.UNCERTAIN: "evolución incierta"
        }
        
        horizon_text = {
            TimeHorizon.SHORT_TERM: "los próximos 3 meses",
            TimeHorizon.MEDIUM_TERM: "los próximos 6-12 meses",
            TimeHorizon.LONG_TERM: "los próximos 1-3 años",
            TimeHorizon.STRATEGIC: "el horizonte estratégico (3+ años)"
        }
        
        return f"Se proyecta {direction_text.get(trend.direction, 'evolución')} para {subject} durante {horizon_text.get(horizon, 'el período analizado')}. Fuerza de la tendencia: {trend.strength:.0%}. Momentum: {'positivo' if trend.momentum > 0 else 'negativo' if trend.momentum < 0 else 'neutral'}."
    
    def _calculate_probability(self, trend: TrendAnalysis) -> float:
        """Calcula probabilidad de la predicción."""
        base = 50.0
        
        # Ajustar por confianza
        base += (trend.confidence - 0.5) * 30
        
        # Ajustar por fuerza de tendencia
        base += trend.strength * 15
        
        # Ajustar por momentum
        base += trend.momentum * 10
        
        return max(10, min(90, base))
    
    def _determine_confidence(
        self, 
        trend: TrendAnalysis,
        horizon: TimeHorizon
    ) -> PredictionConfidence:
        """Determina nivel de confianza."""
        # Menos confianza a mayor horizonte
        horizon_penalty = {
            TimeHorizon.SHORT_TERM: 0,
            TimeHorizon.MEDIUM_TERM: 0.1,
            TimeHorizon.LONG_TERM: 0.2,
            TimeHorizon.STRATEGIC: 0.3
        }
        
        adjusted_confidence = trend.confidence - horizon_penalty.get(horizon, 0.2)
        
        if adjusted_confidence >= 0.75:
            return PredictionConfidence.HIGH
        elif adjusted_confidence >= 0.55:
            return PredictionConfidence.MEDIUM
        elif adjusted_confidence >= 0.35:
            return PredictionConfidence.LOW
        return PredictionConfidence.SPECULATIVE
    
    def _identify_factors(
        self, 
        trend: TrendAnalysis
    ) -> Tuple[List[str], List[str]]:
        """Identifica factores de soporte y riesgo."""
        supporting = []
        risks = []
        
        if trend.direction in [TrendDirection.STRONG_UP, TrendDirection.UP]:
            supporting.append("Tendencia de crecimiento confirmada")
            if trend.momentum > 0:
                supporting.append("Momentum positivo")
        else:
            risks.append("Tendencia no favorable")
        
        if trend.confidence > 0.7:
            supporting.append("Alta consistencia en datos")
        else:
            risks.append("Datos limitados o inconsistentes")
        
        if trend.data_points > 3:
            supporting.append("Múltiples fuentes de evidencia")
        else:
            risks.append("Evidencia limitada")
        
        return supporting, risks
    
    def _get_date_range(self, horizon: TimeHorizon) -> str:
        """Obtiene rango de fechas para el horizonte."""
        now = datetime.utcnow()
        ranges = {
            TimeHorizon.SHORT_TERM: (now, now + timedelta(days=90)),
            TimeHorizon.MEDIUM_TERM: (now + timedelta(days=90), now + timedelta(days=365)),
            TimeHorizon.LONG_TERM: (now + timedelta(days=365), now + timedelta(days=1095)),
            TimeHorizon.STRATEGIC: (now + timedelta(days=1095), now + timedelta(days=1825))
        }
        
        start, end = ranges.get(horizon, (now, now + timedelta(days=365)))
        return f"{start.strftime('%Y-%m')} a {end.strftime('%Y-%m')}"


class PredictiveIntelligenceService:
    """
    Servicio principal de Inteligencia Predictiva.
    """
    
    def __init__(self):
        self.prediction_engine = PredictionEngine()
        self.trend_analyzer = TrendAnalyzer()
        self.signal_detector = WeakSignalDetector()
        self.scenario_planner = ScenarioPlanner()
    
    async def generate_market_forecast(
        self, 
        market: str,
        time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM,
        include_scenarios: bool = True
    ) -> MarketForecast:
        """
        Genera pronóstico completo de mercado.
        
        Args:
            market: Mercado a pronosticar
            time_horizon: Horizonte temporal
            include_scenarios: Incluir planificación de escenarios
        
        Returns:
            Pronóstico completo del mercado
        """
        logger.info(f"Generating market forecast for: {market}")
        
        # Analizar tendencias principales
        main_trend = self.trend_analyzer.analyze_trend(market)
        related_trends = [
            self.trend_analyzer.analyze_trend(f"{market} technology"),
            self.trend_analyzer.analyze_trend(f"{market} adoption"),
            self.trend_analyzer.analyze_trend(f"{market} competition")
        ]
        
        # Detectar señales débiles
        weak_signals = self.signal_detector.detect_signals(market)
        
        # Generar predicción principal
        main_prediction = self.prediction_engine.generate_prediction(
            market, time_horizon
        )
        
        # Identificar incertidumbres
        key_uncertainties = [
            f"Ritmo de adopción tecnológica en {market}",
            "Evolución del entorno regulatorio",
            "Respuesta competitiva del mercado",
            "Condiciones macroeconómicas"
        ]
        
        # Generar escenarios
        scenarios = {}
        if include_scenarios:
            scenarios = self.scenario_planner.generate_scenarios(
                market, key_uncertainties, time_horizon
            )
        
        # Generar recomendaciones
        preparations = self._generate_preparations(
            main_prediction, weak_signals
        )
        
        return MarketForecast(
            market_name=market,
            current_state={
                "trend_direction": main_trend.direction.value,
                "trend_strength": main_trend.strength,
                "momentum": main_trend.momentum
            },
            trends=[main_trend] + related_trends,
            weak_signals=weak_signals,
            predictions=[main_prediction],
            scenarios=scenarios,
            key_uncertainties=key_uncertainties,
            recommended_preparations=preparations,
            forecast_horizon=time_horizon,
            confidence_score=main_prediction.probability / 100
        )
    
    def _generate_preparations(
        self, 
        prediction: Prediction,
        signals: List[WeakSignal]
    ) -> List[str]:
        """Genera recomendaciones de preparación."""
        preparations = []
        
        # Basado en confianza
        if prediction.confidence == PredictionConfidence.HIGH:
            preparations.append("Proceder con estrategia basada en predicción principal")
        else:
            preparations.append("Mantener flexibilidad para ajustar estrategia")
        
        # Basado en señales
        if signals:
            tech_signals = [s for s in signals if s.category == "technology"]
            if tech_signals:
                preparations.append("Monitorear desarrollo tecnológico emergente")
            
            reg_signals = [s for s in signals if s.category == "regulatory"]
            if reg_signals:
                preparations.append("Prepararse para cambios regulatorios potenciales")
        
        # Generales
        preparations.extend([
            "Establecer puntos de revisión periódica",
            "Desarrollar indicadores de alerta temprana",
            "Mantener reservas para oportunidades"
        ])
        
        return preparations
    
    async def get_quick_prediction(
        self, 
        topic: str
    ) -> Dict[str, Any]:
        """
        Genera predicción rápida para un tema.
        """
        prediction = self.prediction_engine.generate_prediction(topic)
        
        return {
            "topic": topic,
            "prediction": prediction.prediction,
            "probability": f"{prediction.probability:.0f}%",
            "confidence": prediction.confidence.value,
            "time_horizon": prediction.time_horizon.value,
            "expected_range": prediction.expected_date_range,
            "supporting_factors": prediction.supporting_factors,
            "risk_factors": prediction.risk_factors,
            "generated_at": prediction.generated_at.isoformat()
        }


# Singleton
_pi_service: Optional[PredictiveIntelligenceService] = None

def get_predictive_intelligence_service() -> PredictiveIntelligenceService:
    """Obtiene instancia singleton del servicio."""
    global _pi_service
    if _pi_service is None:
        _pi_service = PredictiveIntelligenceService()
    return _pi_service
