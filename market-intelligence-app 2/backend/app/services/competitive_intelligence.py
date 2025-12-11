"""
Competitive Intelligence Service
=================================
Servicio de monitoreo y análisis de competencia.
Detecta movimientos competitivos, cambios de mercado y amenazas.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings
from ..core.rag_engine import get_rag_engine

logger = logging.getLogger("market_intelligence.services.competitive")


class SignalType(Enum):
    """Tipos de señales competitivas."""
    PRODUCT_LAUNCH = "product_launch"
    PRICING_CHANGE = "pricing_change"
    MARKET_ENTRY = "market_entry"
    MARKET_EXIT = "market_exit"
    ACQUISITION = "acquisition"
    PARTNERSHIP = "partnership"
    EXECUTIVE_MOVE = "executive_move"
    FUNDING_ROUND = "funding_round"
    PATENT_FILING = "patent_filing"
    REGULATORY_CHANGE = "regulatory_change"
    EARNINGS_REPORT = "earnings_report"
    STRATEGIC_SHIFT = "strategic_shift"


class SignalStrength(Enum):
    """Fuerza de la señal detectada."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONFIRMED = "confirmed"


@dataclass
class CompetitiveSignal:
    """Señal competitiva detectada."""
    signal_type: SignalType
    competitor: str
    title: str
    description: str
    strength: SignalStrength
    impact_assessment: str
    source: str
    source_url: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    related_entities: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class CompetitorProfile:
    """Perfil completo de un competidor."""
    name: str
    industry: str
    market_position: str
    estimated_revenue: Optional[str] = None
    employee_count: Optional[str] = None
    headquarters: Optional[str] = None
    founded: Optional[str] = None
    key_products: List[str] = field(default_factory=list)
    key_executives: List[Dict[str, str]] = field(default_factory=list)
    recent_signals: List[CompetitiveSignal] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass 
class MarketLandscape:
    """Panorama del mercado competitivo."""
    market_name: str
    market_size: str
    growth_rate: str
    key_players: List[CompetitorProfile]
    market_trends: List[str]
    entry_barriers: List[str]
    regulatory_environment: str
    technology_trends: List[str]
    recent_signals: List[CompetitiveSignal]
    competitive_dynamics: Dict[str, Any]
    analyzed_at: datetime = field(default_factory=datetime.utcnow)


class SignalDetector:
    """
    Detector de señales competitivas.
    Analiza múltiples fuentes para detectar movimientos de competidores.
    """
    
    # Patrones para detección de señales
    SIGNAL_PATTERNS = {
        SignalType.PRODUCT_LAUNCH: [
            r"(launch|release|introduce|unveil|announce).*?(product|service|platform|solution)",
            r"new (product|offering|service|feature)",
            r"(beta|preview|early access)"
        ],
        SignalType.PRICING_CHANGE: [
            r"(price|pricing) (change|increase|decrease|adjustment)",
            r"new (pricing|subscription) (model|plan|tier)",
            r"(free tier|freemium|discount)"
        ],
        SignalType.ACQUISITION: [
            r"(acquire|acquisition|purchase|buy).*?(company|startup|firm)",
            r"(merger|merge) with",
            r"deal (worth|valued)"
        ],
        SignalType.PARTNERSHIP: [
            r"(partner|partnership|alliance|collaboration) with",
            r"(strategic|joint venture)",
            r"(integrate|integration) with"
        ],
        SignalType.FUNDING_ROUND: [
            r"(raise|secure|close).*?\$[\d.]+[BMK]",
            r"(series [A-Z]|seed|pre-seed) (round|funding)",
            r"(valuation|valued at)"
        ],
        SignalType.EXECUTIVE_MOVE: [
            r"(appoint|hire|join).*?(CEO|CTO|CFO|COO|VP|Director)",
            r"(leave|depart|resign|step down)",
            r"(new|former) (CEO|executive|leadership)"
        ],
        SignalType.PATENT_FILING: [
            r"(patent|IP) (filing|application|granted)",
            r"(intellectual property|trademark)"
        ],
        SignalType.MARKET_ENTRY: [
            r"(enter|expand|launch) (in|into).*?(market|region|country)",
            r"(international|global) expansion",
            r"open.*?(office|headquarters|presence)"
        ]
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.rag_engine = get_rag_engine()
    
    def detect_signals(self, text: str, competitor: str) -> List[CompetitiveSignal]:
        """
        Detecta señales competitivas en un texto.
        
        Args:
            text: Texto a analizar
            competitor: Nombre del competidor
        
        Returns:
            Lista de señales detectadas
        """
        signals = []
        text_lower = text.lower()
        
        for signal_type, patterns in self.SIGNAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Evaluar fuerza de la señal
                    strength = self._assess_signal_strength(text, pattern, matches)
                    
                    # Crear señal
                    signal = CompetitiveSignal(
                        signal_type=signal_type,
                        competitor=competitor,
                        title=f"{signal_type.value.replace('_', ' ').title()} detectado",
                        description=text[:300],
                        strength=strength,
                        impact_assessment=self._assess_impact(signal_type),
                        source="RAG Analysis",
                        confidence=self._calculate_confidence(matches, text)
                    )
                    
                    # Agregar acciones recomendadas
                    signal.recommended_actions = self._get_recommended_actions(signal_type)
                    
                    signals.append(signal)
                    break  # Un signal type por texto
        
        return signals
    
    def _assess_signal_strength(
        self, 
        text: str, 
        pattern: str, 
        matches: List
    ) -> SignalStrength:
        """Evalúa la fuerza de una señal."""
        # Indicadores de señal fuerte
        strong_indicators = [
            "confirmed", "official", "announced", "reported",
            "according to", "sources say", "press release"
        ]
        
        text_lower = text.lower()
        strong_count = sum(1 for ind in strong_indicators if ind in text_lower)
        
        if strong_count >= 2:
            return SignalStrength.CONFIRMED
        elif strong_count == 1:
            return SignalStrength.STRONG
        elif len(matches) > 1:
            return SignalStrength.MODERATE
        return SignalStrength.WEAK
    
    def _assess_impact(self, signal_type: SignalType) -> str:
        """Evalúa el impacto potencial de un tipo de señal."""
        high_impact = [
            SignalType.ACQUISITION, SignalType.MARKET_ENTRY,
            SignalType.STRATEGIC_SHIFT, SignalType.FUNDING_ROUND
        ]
        medium_impact = [
            SignalType.PRODUCT_LAUNCH, SignalType.PARTNERSHIP,
            SignalType.EXECUTIVE_MOVE, SignalType.PRICING_CHANGE
        ]
        
        if signal_type in high_impact:
            return "Alto impacto potencial - Requiere atención inmediata"
        elif signal_type in medium_impact:
            return "Impacto medio - Monitorear evolución"
        return "Impacto bajo - Mantener en radar"
    
    def _calculate_confidence(self, matches: List, text: str) -> float:
        """Calcula confianza de la detección."""
        base_confidence = 0.5
        
        # Más matches = más confianza
        base_confidence += min(0.2, len(matches) * 0.05)
        
        # Texto más largo = más contexto = más confianza
        if len(text) > 500:
            base_confidence += 0.1
        
        # Fuentes citadas
        if any(s in text.lower() for s in ["according to", "reported by", "source:"]):
            base_confidence += 0.15
        
        return min(0.95, base_confidence)
    
    def _get_recommended_actions(self, signal_type: SignalType) -> List[str]:
        """Genera acciones recomendadas para un tipo de señal."""
        actions = {
            SignalType.PRODUCT_LAUNCH: [
                "Analizar características del nuevo producto",
                "Comparar con nuestra oferta actual",
                "Evaluar impacto en posicionamiento"
            ],
            SignalType.PRICING_CHANGE: [
                "Analizar nueva estructura de precios",
                "Evaluar impacto en competitividad",
                "Considerar ajustes de pricing propios"
            ],
            SignalType.ACQUISITION: [
                "Evaluar sinergias del deal",
                "Identificar amenazas competitivas",
                "Analizar posibles gaps de mercado"
            ],
            SignalType.FUNDING_ROUND: [
                "Estimar runway y planes de expansión",
                "Anticipar movimientos de mercado",
                "Evaluar amenaza competitiva"
            ],
            SignalType.EXECUTIVE_MOVE: [
                "Investigar background del ejecutivo",
                "Anticipar cambios estratégicos",
                "Evaluar oportunidades de reclutamiento"
            ],
            SignalType.MARKET_ENTRY: [
                "Mapear estrategia de entrada",
                "Evaluar amenaza territorial",
                "Preparar respuesta competitiva"
            ]
        }
        return actions.get(signal_type, ["Monitorear evolución"])


class CompetitorTracker:
    """
    Rastreador de competidores.
    Mantiene perfiles actualizados de competidores clave.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_detector = SignalDetector()
        self.rag_engine = get_rag_engine()
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Cache de perfiles
        self._profiles_cache: Dict[str, CompetitorProfile] = {}
    
    async def get_competitor_profile(
        self, 
        competitor_name: str,
        refresh: bool = False
    ) -> CompetitorProfile:
        """
        Obtiene o actualiza el perfil de un competidor.
        
        Args:
            competitor_name: Nombre del competidor
            refresh: Forzar actualización
        
        Returns:
            Perfil del competidor
        """
        cache_key = competitor_name.lower()
        
        if not refresh and cache_key in self._profiles_cache:
            cached = self._profiles_cache[cache_key]
            # Check if cache is fresh (< 24 hours)
            if datetime.utcnow() - cached.last_updated < timedelta(hours=24):
                return cached
        
        # Construir perfil
        profile = CompetitorProfile(
            name=competitor_name,
            industry="Technology",  # TODO: Detectar automáticamente
            market_position="Competitor"
        )
        
        # Buscar información en RAG
        queries = [
            f"{competitor_name} company overview",
            f"{competitor_name} products services",
            f"{competitor_name} executives leadership",
            f"{competitor_name} recent news announcements"
        ]
        
        for query in queries:
            result = self.rag_engine.retrieve(query, top_k=3)
            
            for doc in result.documents:
                # Detectar señales en el contenido
                signals = self.signal_detector.detect_signals(
                    doc.page_content, 
                    competitor_name
                )
                profile.recent_signals.extend(signals)
        
        # Limitar señales recientes
        profile.recent_signals = sorted(
            profile.recent_signals,
            key=lambda x: x.detected_at,
            reverse=True
        )[:10]
        
        # Actualizar cache
        self._profiles_cache[cache_key] = profile
        
        return profile
    
    async def compare_competitors(
        self, 
        competitors: List[str]
    ) -> Dict[str, Any]:
        """
        Compara múltiples competidores.
        
        Args:
            competitors: Lista de nombres de competidores
        
        Returns:
            Análisis comparativo
        """
        profiles = []
        for competitor in competitors:
            profile = await self.get_competitor_profile(competitor)
            profiles.append(profile)
        
        return {
            "competitors_analyzed": len(profiles),
            "profiles": [
                {
                    "name": p.name,
                    "market_position": p.market_position,
                    "recent_signals_count": len(p.recent_signals),
                    "key_signals": [
                        {
                            "type": s.signal_type.value,
                            "title": s.title,
                            "strength": s.strength.value
                        }
                        for s in p.recent_signals[:3]
                    ]
                }
                for p in profiles
            ],
            "total_signals_detected": sum(len(p.recent_signals) for p in profiles),
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    async def close(self):
        await self.client.aclose()


class MarketMonitor:
    """
    Monitor de mercado.
    Rastrea cambios y tendencias en mercados específicos.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.competitor_tracker = CompetitorTracker()
        self.signal_detector = SignalDetector()
        self.rag_engine = get_rag_engine()
    
    async def analyze_market(
        self, 
        market_name: str,
        key_players: List[str] = None
    ) -> MarketLandscape:
        """
        Analiza un mercado completo.
        
        Args:
            market_name: Nombre del mercado
            key_players: Lista de jugadores clave (opcional)
        
        Returns:
            Panorama del mercado
        """
        # Buscar información del mercado
        market_query = f"{market_name} market size growth trends 2025"
        market_result = self.rag_engine.query(market_query)
        
        # Obtener perfiles de competidores
        competitor_profiles = []
        if key_players:
            for player in key_players[:5]:  # Limitar a 5
                profile = await self.competitor_tracker.get_competitor_profile(player)
                competitor_profiles.append(profile)
        
        # Recopilar todas las señales
        all_signals = []
        for profile in competitor_profiles:
            all_signals.extend(profile.recent_signals)
        
        # Ordenar por fecha
        all_signals.sort(key=lambda x: x.detected_at, reverse=True)
        
        return MarketLandscape(
            market_name=market_name,
            market_size="TBD",  # Extraer del análisis RAG
            growth_rate="TBD",
            key_players=competitor_profiles,
            market_trends=[],
            entry_barriers=[],
            regulatory_environment="",
            technology_trends=[],
            recent_signals=all_signals[:20],
            competitive_dynamics={
                "total_players_analyzed": len(competitor_profiles),
                "total_signals_detected": len(all_signals),
                "signal_distribution": self._count_signals_by_type(all_signals)
            }
        )
    
    def _count_signals_by_type(
        self, 
        signals: List[CompetitiveSignal]
    ) -> Dict[str, int]:
        """Cuenta señales por tipo."""
        counts = {}
        for signal in signals:
            type_name = signal.signal_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    async def get_alerts(
        self, 
        competitors: List[str],
        signal_types: List[SignalType] = None,
        min_strength: SignalStrength = SignalStrength.WEAK
    ) -> List[CompetitiveSignal]:
        """
        Obtiene alertas de competidores filtradas.
        
        Args:
            competitors: Lista de competidores a monitorear
            signal_types: Tipos de señales a incluir
            min_strength: Fuerza mínima de señal
        
        Returns:
            Lista de señales que cumplen criterios
        """
        all_signals = []
        
        for competitor in competitors:
            profile = await self.competitor_tracker.get_competitor_profile(competitor)
            all_signals.extend(profile.recent_signals)
        
        # Filtrar por tipo
        if signal_types:
            all_signals = [s for s in all_signals if s.signal_type in signal_types]
        
        # Filtrar por fuerza
        strength_order = [SignalStrength.WEAK, SignalStrength.MODERATE, 
                         SignalStrength.STRONG, SignalStrength.CONFIRMED]
        min_idx = strength_order.index(min_strength)
        all_signals = [
            s for s in all_signals 
            if strength_order.index(s.strength) >= min_idx
        ]
        
        # Ordenar por fecha
        all_signals.sort(key=lambda x: x.detected_at, reverse=True)
        
        return all_signals
    
    async def close(self):
        await self.competitor_tracker.close()


class CompetitiveIntelligenceService:
    """
    Servicio principal de Inteligencia Competitiva.
    """
    
    def __init__(self):
        self.market_monitor = MarketMonitor()
        self.competitor_tracker = CompetitorTracker()
    
    async def get_competitive_overview(
        self, 
        company_name: str,
        competitors: List[str] = None,
        market: str = None
    ) -> Dict[str, Any]:
        """
        Genera overview competitivo completo.
        """
        result = {
            "subject": company_name,
            "generated_at": datetime.utcnow().isoformat(),
            "competitive_landscape": {},
            "key_threats": [],
            "opportunities": [],
            "recommended_actions": []
        }
        
        # Analizar competidores
        if competitors:
            comparison = await self.competitor_tracker.compare_competitors(competitors)
            result["competitive_landscape"] = comparison
            
            # Identificar amenazas
            for profile in comparison.get("profiles", []):
                for signal in profile.get("key_signals", []):
                    if signal.get("strength") in ["strong", "confirmed"]:
                        result["key_threats"].append({
                            "competitor": profile["name"],
                            "signal": signal
                        })
        
        # Analizar mercado
        if market:
            landscape = await self.market_monitor.analyze_market(
                market, 
                competitors or []
            )
            result["market_analysis"] = {
                "market_name": landscape.market_name,
                "players_analyzed": len(landscape.key_players),
                "signals_detected": len(landscape.recent_signals)
            }
        
        return result
    
    async def close_all(self):
        await self.market_monitor.close()


# Singleton
_ci_service: Optional[CompetitiveIntelligenceService] = None

def get_competitive_intelligence_service() -> CompetitiveIntelligenceService:
    """Obtiene instancia singleton del servicio."""
    global _ci_service
    if _ci_service is None:
        _ci_service = CompetitiveIntelligenceService()
    return _ci_service
