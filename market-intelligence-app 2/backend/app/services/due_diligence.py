"""
Due Diligence & AML Service
============================
Servicio especializado para Due Diligence y Anti-Money Laundering.
Integra verificación de empresas, personas y señales de riesgo.
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

logger = logging.getLogger("market_intelligence.services.due_diligence")


class RiskLevel(Enum):
    """Niveles de riesgo para Due Diligence."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Tipos de entidades para verificación."""
    COMPANY = "company"
    PERSON = "person"
    TRANSACTION = "transaction"
    JURISDICTION = "jurisdiction"


@dataclass
class RiskIndicator:
    """Indicador de riesgo identificado."""
    category: str
    description: str
    severity: RiskLevel
    source: str
    evidence: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0


@dataclass
class DueDiligenceReport:
    """Reporte completo de Due Diligence."""
    entity_name: str
    entity_type: EntityType
    overall_risk: RiskLevel
    risk_score: float  # 0-100
    risk_indicators: List[RiskIndicator]
    sanctions_check: Dict[str, Any]
    pep_check: Dict[str, Any]  # Politically Exposed Persons
    adverse_media: List[Dict[str, Any]]
    corporate_structure: Dict[str, Any]
    financial_indicators: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0


class SanctionsChecker:
    """
    Verificador de Listas de Sanciones.
    Integra múltiples bases de datos de sanciones.
    """
    
    # Listas de sanciones conocidas
    SANCTIONS_LISTS = [
        "OFAC SDN",  # US Treasury
        "EU Sanctions",
        "UN Sanctions",
        "UK HMT",
        "INTERPOL",
    ]
    
    # Países de alto riesgo (FATF)
    HIGH_RISK_JURISDICTIONS = [
        "North Korea", "Iran", "Myanmar", "Syria",
        "Afghanistan", "Yemen", "Libya", "South Sudan",
        "Democratic Republic of Congo", "Central African Republic"
    ]
    
    # Países bajo monitoreo aumentado
    INCREASED_MONITORING = [
        "Albania", "Barbados", "Burkina Faso", "Cameroon",
        "Cayman Islands", "Croatia", "Democratic Republic of Congo",
        "Gibraltar", "Haiti", "Jamaica", "Jordan", "Mali",
        "Mozambique", "Nigeria", "Panama", "Philippines",
        "Senegal", "South Africa", "South Sudan", "Syria",
        "Tanzania", "Turkey", "Uganda", "United Arab Emirates",
        "Vietnam", "Yemen"
    ]
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def check_entity(self, name: str, entity_type: str = "any") -> Dict[str, Any]:
        """
        Verifica una entidad contra listas de sanciones.
        
        Args:
            name: Nombre de la entidad a verificar
            entity_type: Tipo de entidad (person, company, any)
        
        Returns:
            Resultado de la verificación de sanciones
        """
        results = {
            "entity": name,
            "checked_at": datetime.utcnow().isoformat(),
            "lists_checked": self.SANCTIONS_LISTS,
            "matches": [],
            "potential_matches": [],
            "jurisdiction_risk": self._check_jurisdiction_risk(name),
            "overall_status": "clear"
        }
        
        # Verificar patrones de nombres sospechosos
        name_analysis = self._analyze_name(name)
        if name_analysis["flags"]:
            results["name_analysis"] = name_analysis
        
        # Simular verificación (en producción, conectar a APIs reales)
        # OpenSanctions API, World-Check, Dow Jones, etc.
        
        # Determinar estado general
        if results["matches"]:
            results["overall_status"] = "match"
        elif results["potential_matches"]:
            results["overall_status"] = "potential_match"
        elif results["jurisdiction_risk"]["risk_level"] in ["high", "critical"]:
            results["overall_status"] = "jurisdiction_risk"
        
        return results
    
    def _check_jurisdiction_risk(self, name: str) -> Dict[str, Any]:
        """Verifica riesgo jurisdiccional."""
        name_lower = name.lower()
        
        for country in self.HIGH_RISK_JURISDICTIONS:
            if country.lower() in name_lower:
                return {
                    "risk_level": "critical",
                    "jurisdiction": country,
                    "reason": "FATF High-Risk Jurisdiction"
                }
        
        for country in self.INCREASED_MONITORING:
            if country.lower() in name_lower:
                return {
                    "risk_level": "medium",
                    "jurisdiction": country,
                    "reason": "FATF Increased Monitoring"
                }
        
        return {"risk_level": "low", "jurisdiction": None, "reason": None}
    
    def _analyze_name(self, name: str) -> Dict[str, Any]:
        """Analiza el nombre para detectar patrones sospechosos."""
        flags = []
        
        # Detectar shell company patterns
        shell_patterns = [
            r"\b(holding|offshore|trust|foundation|limited|ltd|llc|corp)\b",
            r"\b(international|global|worldwide|universal)\b",
            r"\b(investments?|ventures?|capital|assets?)\b"
        ]
        
        name_lower = name.lower()
        for pattern in shell_patterns:
            if re.search(pattern, name_lower):
                flags.append({
                    "type": "shell_company_pattern",
                    "pattern": pattern,
                    "severity": "low"
                })
        
        # Detectar uso excesivo de abreviaturas
        if len(re.findall(r'\b[A-Z]{2,}\b', name)) > 2:
            flags.append({
                "type": "excessive_abbreviations",
                "severity": "low"
            })
        
        return {"flags": flags, "analyzed_at": datetime.utcnow().isoformat()}
    
    async def close(self):
        await self.client.aclose()


class PEPChecker:
    """
    Verificador de Personas Políticamente Expuestas (PEP).
    """
    
    PEP_CATEGORIES = [
        "Head of State/Government",
        "Senior Government Official",
        "Senior Military Official",
        "Senior Judicial Official",
        "Senior Political Party Official",
        "Senior Executive of State-Owned Enterprise",
        "International Organization Official",
        "Family Member of PEP",
        "Close Associate of PEP"
    ]
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def check_person(self, name: str, country: str = None) -> Dict[str, Any]:
        """
        Verifica si una persona es PEP.
        
        Args:
            name: Nombre completo de la persona
            country: País de residencia/nacionalidad
        
        Returns:
            Resultado de la verificación PEP
        """
        results = {
            "person": name,
            "country": country,
            "checked_at": datetime.utcnow().isoformat(),
            "is_pep": False,
            "pep_category": None,
            "related_peps": [],
            "positions_held": [],
            "risk_level": "low"
        }
        
        # En producción, integrar con:
        # - World-Check
        # - Dow Jones Risk & Compliance
        # - Refinitiv
        # - OpenSanctions
        
        return results
    
    async def close(self):
        await self.client.aclose()


class AdverseMediaScanner:
    """
    Scanner de medios adversos.
    Busca noticias negativas y menciones problemáticas.
    """
    
    ADVERSE_KEYWORDS = {
        "criminal": ["fraud", "money laundering", "corruption", "bribery", 
                    "embezzlement", "tax evasion", "criminal", "indictment",
                    "prosecution", "conviction", "arrested", "charged"],
        "financial": ["bankruptcy", "default", "insolvency", "debt crisis",
                     "financial crime", "securities fraud", "insider trading"],
        "regulatory": ["sanctions", "violation", "fine", "penalty", "banned",
                      "suspended", "debarred", "blacklisted"],
        "reputational": ["scandal", "controversy", "allegations", "lawsuit",
                        "investigation", "probe", "misconduct"]
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rag_engine = get_rag_engine()
    
    async def scan_entity(self, name: str, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Escanea medios adversos para una entidad.
        
        Args:
            name: Nombre de la entidad
            lookback_days: Días hacia atrás para buscar
        
        Returns:
            Resultados del escaneo de medios adversos
        """
        results = {
            "entity": name,
            "scan_period": f"Last {lookback_days} days",
            "scanned_at": datetime.utcnow().isoformat(),
            "adverse_mentions": [],
            "categories_found": {},
            "severity_summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "overall_risk": "low"
        }
        
        # Buscar en RAG interno
        for category, keywords in self.ADVERSE_KEYWORDS.items():
            for keyword in keywords:
                query = f"{name} {keyword}"
                rag_result = self.rag_engine.retrieve(query, top_k=3)
                
                for doc, score in zip(rag_result.documents, rag_result.relevance_scores):
                    if score > 0.75:  # Alta relevancia
                        mention = {
                            "category": category,
                            "keyword": keyword,
                            "snippet": doc.page_content[:300],
                            "source": doc.metadata.get("source", "unknown"),
                            "relevance_score": score,
                            "severity": self._assess_severity(keyword, score)
                        }
                        results["adverse_mentions"].append(mention)
                        
                        # Actualizar conteo de severidad
                        results["severity_summary"][mention["severity"]] += 1
                        
                        # Actualizar categorías encontradas
                        if category not in results["categories_found"]:
                            results["categories_found"][category] = 0
                        results["categories_found"][category] += 1
        
        # Calcular riesgo general
        results["overall_risk"] = self._calculate_overall_risk(results["severity_summary"])
        
        return results
    
    def _assess_severity(self, keyword: str, relevance: float) -> str:
        """Evalúa la severidad de una mención."""
        high_severity_keywords = ["fraud", "money laundering", "sanctions", 
                                  "criminal", "conviction", "indictment"]
        
        if keyword in high_severity_keywords and relevance > 0.85:
            return "critical"
        elif keyword in high_severity_keywords:
            return "high"
        elif relevance > 0.85:
            return "medium"
        return "low"
    
    def _calculate_overall_risk(self, severity_summary: Dict[str, int]) -> str:
        """Calcula el riesgo general basado en las menciones."""
        if severity_summary["critical"] > 0:
            return "critical"
        elif severity_summary["high"] >= 3:
            return "critical"
        elif severity_summary["high"] > 0:
            return "high"
        elif severity_summary["medium"] >= 5:
            return "high"
        elif severity_summary["medium"] > 0:
            return "medium"
        return "low"
    
    async def close(self):
        await self.client.aclose()


class CorporateStructureAnalyzer:
    """
    Analizador de estructura corporativa.
    Identifica beneficial ownership y estructuras complejas.
    """
    
    # Indicadores de estructuras de alto riesgo
    HIGH_RISK_STRUCTURES = [
        "nominee_shareholder",
        "bearer_shares",
        "multiple_jurisdictions",
        "circular_ownership",
        "layered_entities",
        "trust_structures",
        "foundation_ownership"
    ]
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def analyze_structure(self, company_name: str) -> Dict[str, Any]:
        """
        Analiza la estructura corporativa de una empresa.
        
        Args:
            company_name: Nombre de la empresa
        
        Returns:
            Análisis de estructura corporativa
        """
        results = {
            "company": company_name,
            "analyzed_at": datetime.utcnow().isoformat(),
            "ownership_structure": {
                "beneficial_owners": [],
                "intermediate_entities": [],
                "ultimate_parent": None,
                "ownership_chain_length": 0
            },
            "jurisdictions_involved": [],
            "structure_risk_indicators": [],
            "complexity_score": 0,  # 0-100
            "transparency_score": 0,  # 0-100
            "overall_risk": "unknown"
        }
        
        # En producción, integrar con:
        # - OpenCorporates
        # - Dun & Bradstreet
        # - Bureau van Dijk (Orbis)
        # - Companies House API (UK)
        # - SEC EDGAR
        
        return results
    
    async def close(self):
        await self.client.aclose()


class DueDiligenceService:
    """
    Servicio principal de Due Diligence.
    Orquesta todos los componentes de verificación.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.sanctions_checker = SanctionsChecker()
        self.pep_checker = PEPChecker()
        self.adverse_media = AdverseMediaScanner()
        self.structure_analyzer = CorporateStructureAnalyzer()
        self.rag_engine = get_rag_engine()
    
    async def run_full_check(
        self, 
        entity_name: str,
        entity_type: EntityType = EntityType.COMPANY,
        include_related: bool = True
    ) -> DueDiligenceReport:
        """
        Ejecuta verificación completa de Due Diligence.
        
        Args:
            entity_name: Nombre de la entidad
            entity_type: Tipo de entidad
            include_related: Incluir entidades relacionadas
        
        Returns:
            Reporte completo de Due Diligence
        """
        logger.info(f"Running due diligence check for: {entity_name}")
        
        # Ejecutar todas las verificaciones en paralelo
        tasks = [
            self.sanctions_checker.check_entity(entity_name),
            self.adverse_media.scan_entity(entity_name),
        ]
        
        if entity_type == EntityType.PERSON:
            tasks.append(self.pep_checker.check_person(entity_name))
        elif entity_type == EntityType.COMPANY:
            tasks.append(self.structure_analyzer.analyze_structure(entity_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        sanctions_result = results[0] if not isinstance(results[0], Exception) else {}
        adverse_result = results[1] if not isinstance(results[1], Exception) else {}
        additional_result = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}
        
        # Construir indicadores de riesgo
        risk_indicators = self._build_risk_indicators(
            sanctions_result, adverse_result, additional_result
        )
        
        # Calcular score de riesgo
        risk_score, overall_risk = self._calculate_risk_score(risk_indicators)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            overall_risk, risk_indicators
        )
        
        return DueDiligenceReport(
            entity_name=entity_name,
            entity_type=entity_type,
            overall_risk=overall_risk,
            risk_score=risk_score,
            risk_indicators=risk_indicators,
            sanctions_check=sanctions_result,
            pep_check=additional_result if entity_type == EntityType.PERSON else {},
            adverse_media=adverse_result.get("adverse_mentions", []),
            corporate_structure=additional_result if entity_type == EntityType.COMPANY else {},
            financial_indicators={},
            recommendations=recommendations,
            confidence_score=self._calculate_confidence(results)
        )
    
    def _build_risk_indicators(
        self, 
        sanctions: Dict, 
        adverse: Dict, 
        additional: Dict
    ) -> List[RiskIndicator]:
        """Construye lista de indicadores de riesgo."""
        indicators = []
        
        # Indicadores de sanciones
        if sanctions.get("overall_status") == "match":
            indicators.append(RiskIndicator(
                category="sanctions",
                description="Match encontrado en lista de sanciones",
                severity=RiskLevel.CRITICAL,
                source="Sanctions Database",
                confidence=0.95
            ))
        elif sanctions.get("jurisdiction_risk", {}).get("risk_level") == "critical":
            indicators.append(RiskIndicator(
                category="jurisdiction",
                description=f"Jurisdicción de alto riesgo: {sanctions['jurisdiction_risk'].get('jurisdiction')}",
                severity=RiskLevel.HIGH,
                source="FATF",
                confidence=0.90
            ))
        
        # Indicadores de medios adversos
        adverse_risk = adverse.get("overall_risk", "low")
        if adverse_risk in ["critical", "high"]:
            indicators.append(RiskIndicator(
                category="adverse_media",
                description=f"Menciones adversas detectadas: {len(adverse.get('adverse_mentions', []))}",
                severity=RiskLevel.HIGH if adverse_risk == "critical" else RiskLevel.MEDIUM,
                source="Media Scan",
                evidence=str(adverse.get("categories_found", {})),
                confidence=0.80
            ))
        
        return indicators
    
    def _calculate_risk_score(
        self, 
        indicators: List[RiskIndicator]
    ) -> tuple[float, RiskLevel]:
        """Calcula score de riesgo agregado."""
        if not indicators:
            return 15.0, RiskLevel.LOW
        
        # Pesos por severidad
        weights = {
            RiskLevel.CRITICAL: 40,
            RiskLevel.HIGH: 25,
            RiskLevel.MEDIUM: 15,
            RiskLevel.LOW: 5,
            RiskLevel.UNKNOWN: 10
        }
        
        total_score = sum(weights.get(ind.severity, 0) for ind in indicators)
        total_score = min(100, total_score)  # Cap at 100
        
        # Determinar nivel general
        if total_score >= 75:
            return total_score, RiskLevel.CRITICAL
        elif total_score >= 50:
            return total_score, RiskLevel.HIGH
        elif total_score >= 25:
            return total_score, RiskLevel.MEDIUM
        return total_score, RiskLevel.LOW
    
    def _generate_recommendations(
        self, 
        risk_level: RiskLevel,
        indicators: List[RiskIndicator]
    ) -> List[str]:
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "⛔ NO PROCEDER sin aprobación de compliance senior",
                "Escalar inmediatamente al Oficial de Cumplimiento",
                "Documentar todos los hallazgos para posible reporte SAR/STR",
                "Considerar reporte a autoridades competentes"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ Requiere Enhanced Due Diligence (EDD)",
                "Obtener documentación adicional de UBO",
                "Verificar fuente de fondos",
                "Revisión por comité de cumplimiento"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Completar Customer Due Diligence estándar",
                "Monitoreo continuo recomendado",
                "Documentar justificación de negocio"
            ])
        else:
            recommendations.extend([
                "✅ Riesgo bajo - proceder con due diligence estándar",
                "Mantener documentación actualizada",
                "Revisión periódica recomendada"
            ])
        
        # Recomendaciones específicas por indicador
        for indicator in indicators:
            if indicator.category == "sanctions":
                recommendations.append(f"Verificar match de sanciones con datos adicionales")
            elif indicator.category == "adverse_media":
                recommendations.append("Revisar menciones de medios adversos manualmente")
        
        return recommendations
    
    def _calculate_confidence(self, results: List) -> float:
        """Calcula confianza del análisis."""
        successful = sum(1 for r in results if not isinstance(r, Exception))
        return (successful / len(results)) * 100 if results else 0
    
    async def close_all(self):
        """Cierra todos los servicios."""
        await asyncio.gather(
            self.sanctions_checker.close(),
            self.pep_checker.close(),
            self.adverse_media.close(),
            self.structure_analyzer.close()
        )


# Singleton
_dd_service: Optional[DueDiligenceService] = None

def get_due_diligence_service() -> DueDiligenceService:
    """Obtiene instancia singleton del servicio."""
    global _dd_service
    if _dd_service is None:
        _dd_service = DueDiligenceService()
    return _dd_service
