"""
Google APIs Service
====================
Integración con Google Custom Search API y otros servicios de Google.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings

logger = logging.getLogger("market_intelligence.services.google")


@dataclass
class SearchResult:
    """Resultado de búsqueda de Google."""
    title: str
    link: str
    snippet: str
    source: str = "google_search"
    published_date: Optional[str] = None
    page_map: Optional[Dict] = None
    relevance_score: float = 0.0


@dataclass
class GoogleSearchResponse:
    """Respuesta completa de Google Search."""
    query: str
    total_results: int
    results: List[SearchResult]
    search_time: float
    searched_at: datetime = field(default_factory=datetime.utcnow)


class GoogleSearchService:
    """
    Servicio de Google Custom Search API.
    Proporciona búsqueda web potente para inteligencia de mercado.
    """
    
    BASE_URL = "https://www.googleapis.com/customsearch/v1"
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0)
        self._validate_config()
    
    def _validate_config(self):
        """Valida configuración de Google API."""
        if not self.settings.google_api_key:
            logger.warning("Google API key not configured")
        if not self.settings.google_search_engine_id:
            logger.warning("Google Search Engine ID not configured")
    
    @property
    def is_configured(self) -> bool:
        """Verifica si el servicio está configurado."""
        return bool(
            self.settings.google_api_key and 
            self.settings.google_search_engine_id
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def search(
        self,
        query: str,
        num_results: int = 10,
        language: str = "es",
        country: str = "co",
        date_restrict: Optional[str] = None,
        site_search: Optional[str] = None,
        exclude_terms: Optional[str] = None,
        exact_terms: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> GoogleSearchResponse:
        """
        Realiza búsqueda en Google.
        
        Args:
            query: Consulta de búsqueda
            num_results: Número de resultados (máx 10 por request)
            language: Código de idioma (es, en, etc.)
            country: Código de país (co, us, etc.)
            date_restrict: Restricción de fecha (d1, w1, m1, y1)
            site_search: Limitar a sitio específico
            exclude_terms: Términos a excluir
            exact_terms: Términos exactos requeridos
            file_type: Tipo de archivo (pdf, doc, etc.)
        
        Returns:
            Respuesta de búsqueda con resultados
        """
        if not self.is_configured:
            logger.error("Google Search not configured")
            return GoogleSearchResponse(
                query=query,
                total_results=0,
                results=[],
                search_time=0
            )
        
        # Construir parámetros
        params = {
            "key": self.settings.google_api_key,
            "cx": self.settings.google_search_engine_id,
            "q": query,
            "num": min(num_results, 10),  # Máximo 10 por request
            "lr": f"lang_{language}",
            "gl": country,
        }
        
        # Parámetros opcionales
        if date_restrict:
            params["dateRestrict"] = date_restrict
        if site_search:
            params["siteSearch"] = site_search
        if exclude_terms:
            params["excludeTerms"] = exclude_terms
        if exact_terms:
            params["exactTerms"] = exact_terms
        if file_type:
            params["fileType"] = file_type
        
        try:
            start_time = datetime.utcnow()
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            search_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Parsear resultados
            results = []
            for item in data.get("items", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google_search",
                    page_map=item.get("pagemap"),
                    relevance_score=self._calculate_relevance(item, query)
                )
                results.append(result)
            
            total_results = int(
                data.get("searchInformation", {}).get("totalResults", 0)
            )
            
            logger.info(f"Google search for '{query}': {len(results)} results")
            
            return GoogleSearchResponse(
                query=query,
                total_results=total_results,
                results=results,
                search_time=search_time
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Google Search API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Google Search error: {str(e)}")
            raise
    
    async def search_news(
        self,
        query: str,
        days_back: int = 7,
        num_results: int = 10
    ) -> GoogleSearchResponse:
        """
        Búsqueda específica de noticias recientes.
        
        Args:
            query: Consulta de búsqueda
            days_back: Días hacia atrás para buscar
            num_results: Número de resultados
        
        Returns:
            Resultados de noticias
        """
        # Usar restricción de fecha
        date_restrict = f"d{days_back}"
        
        # Añadir términos de noticias
        news_query = f"{query} (news OR noticias OR article OR artículo)"
        
        return await self.search(
            query=news_query,
            num_results=num_results,
            date_restrict=date_restrict
        )
    
    async def search_company(
        self,
        company_name: str,
        search_type: str = "all"
    ) -> Dict[str, GoogleSearchResponse]:
        """
        Búsqueda completa de información de empresa.
        
        Args:
            company_name: Nombre de la empresa
            search_type: Tipo de búsqueda (all, news, financial, legal)
        
        Returns:
            Diccionario con resultados por categoría
        """
        results = {}
        
        searches = {
            "general": f'"{company_name}" empresa información',
            "news": f'"{company_name}" noticias últimas novedades',
            "financial": f'"{company_name}" financiero resultados ingresos',
            "legal": f'"{company_name}" legal demanda regulación',
            "executive": f'"{company_name}" CEO director ejecutivo gerente'
        }
        
        if search_type != "all":
            searches = {search_type: searches.get(search_type, searches["general"])}
        
        # Ejecutar búsquedas en paralelo
        tasks = {
            key: self.search(query, num_results=5)
            for key, query in searches.items()
        }
        
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"Error in {key} search: {str(e)}")
                results[key] = GoogleSearchResponse(
                    query=searches[key],
                    total_results=0,
                    results=[],
                    search_time=0
                )
        
        return results
    
    async def search_market(
        self,
        market_name: str,
        aspects: List[str] = None
    ) -> Dict[str, GoogleSearchResponse]:
        """
        Búsqueda de información de mercado.
        
        Args:
            market_name: Nombre del mercado
            aspects: Aspectos específicos a buscar
        
        Returns:
            Resultados por aspecto del mercado
        """
        default_aspects = ["size", "trends", "players", "forecast"]
        aspects = aspects or default_aspects
        
        queries = {
            "size": f'"{market_name}" market size tamaño mercado 2024 2025',
            "trends": f'"{market_name}" trends tendencias crecimiento growth',
            "players": f'"{market_name}" principales empresas key players companies',
            "forecast": f'"{market_name}" forecast pronóstico proyección 2025 2030',
            "regulation": f'"{market_name}" regulation regulación normativa',
            "technology": f'"{market_name}" technology tecnología innovación'
        }
        
        results = {}
        for aspect in aspects:
            if aspect in queries:
                try:
                    results[aspect] = await self.search(
                        queries[aspect],
                        num_results=5,
                        date_restrict="y1"  # Último año
                    )
                except Exception as e:
                    logger.error(f"Error in market {aspect} search: {str(e)}")
        
        return results
    
    def _calculate_relevance(self, item: Dict, query: str) -> float:
        """Calcula score de relevancia para un resultado."""
        score = 0.5  # Base score
        
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()
        query_terms = query.lower().split()
        
        # Términos en título (mayor peso)
        for term in query_terms:
            if term in title:
                score += 0.1
        
        # Términos en snippet
        for term in query_terms:
            if term in snippet:
                score += 0.05
        
        # Bonus por fuentes confiables
        link = item.get("link", "").lower()
        trusted_domains = [
            "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
            "forbes.com", "businessinsider.com", "cnbc.com",
            "gov.co", "gov", "edu", "org"
        ]
        for domain in trusted_domains:
            if domain in link:
                score += 0.1
                break
        
        return min(1.0, score)
    
    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()


class GoogleTrendsService:
    """
    Servicio para análisis de Google Trends.
    (Nota: Google Trends no tiene API oficial, usa scraping o pytrends)
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    async def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "today 12-m",
        geo: str = "CO"
    ) -> Dict[str, Any]:
        """
        Obtiene interés a lo largo del tiempo para keywords.
        
        Args:
            keywords: Lista de palabras clave (máx 5)
            timeframe: Período de tiempo
            geo: Código de país
        
        Returns:
            Datos de tendencias
        """
        # Nota: En producción, usar pytrends library
        # pip install pytrends
        
        return {
            "keywords": keywords[:5],
            "timeframe": timeframe,
            "geo": geo,
            "note": "Implementar con pytrends en producción",
            "data": []
        }


class GoogleIntegrationService:
    """
    Servicio integrado de Google APIs.
    Combina Search, Trends y otros servicios de Google.
    """
    
    def __init__(self):
        self.search = GoogleSearchService()
        self.trends = GoogleTrendsService()
        self.settings = get_settings()
    
    async def comprehensive_search(
        self,
        query: str,
        include_news: bool = True,
        include_trends: bool = False
    ) -> Dict[str, Any]:
        """
        Búsqueda comprehensiva usando múltiples servicios de Google.
        
        Args:
            query: Consulta de búsqueda
            include_news: Incluir búsqueda de noticias
            include_trends: Incluir datos de tendencias
        
        Returns:
            Resultados combinados
        """
        results = {
            "query": query,
            "searched_at": datetime.utcnow().isoformat(),
            "sources": {}
        }
        
        # Búsqueda web general
        try:
            web_results = await self.search.search(query)
            results["sources"]["web"] = {
                "total": web_results.total_results,
                "results": [
                    {
                        "title": r.title,
                        "link": r.link,
                        "snippet": r.snippet,
                        "relevance": r.relevance_score
                    }
                    for r in web_results.results
                ]
            }
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            results["sources"]["web"] = {"error": str(e)}
        
        # Búsqueda de noticias
        if include_news:
            try:
                news_results = await self.search.search_news(query)
                results["sources"]["news"] = {
                    "total": news_results.total_results,
                    "results": [
                        {
                            "title": r.title,
                            "link": r.link,
                            "snippet": r.snippet
                        }
                        for r in news_results.results
                    ]
                }
            except Exception as e:
                logger.error(f"News search error: {str(e)}")
                results["sources"]["news"] = {"error": str(e)}
        
        # Tendencias
        if include_trends:
            try:
                trends_data = await self.trends.get_interest_over_time([query])
                results["sources"]["trends"] = trends_data
            except Exception as e:
                logger.error(f"Trends error: {str(e)}")
                results["sources"]["trends"] = {"error": str(e)}
        
        return results
    
    async def close_all(self):
        """Cierra todos los servicios."""
        await self.search.close()


# Singleton
_google_service: Optional[GoogleIntegrationService] = None

def get_google_service() -> GoogleIntegrationService:
    """Obtiene instancia singleton del servicio."""
    global _google_service
    if _google_service is None:
        _google_service = GoogleIntegrationService()
    return _google_service
