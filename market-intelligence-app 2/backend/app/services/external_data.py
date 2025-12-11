"""
External Data Services - Financial Data Integration
====================================================
Servicios para integración con APIs de datos financieros externos.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings

logger = logging.getLogger("market_intelligence.services.external")


@dataclass
class DataPoint:
    """Punto de datos de una fuente externa."""
    source: str
    data_type: str
    value: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class BaseDataService(ABC):
    """Clase base para servicios de datos externos."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
    
    def _cache_key(self, *args) -> str:
        """Genera clave de caché."""
        return f"{self.__class__.__name__}:{':'.join(str(a) for a in args)}"
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Obtiene valor de caché si es válido."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cached(self, key: str, data: Any):
        """Guarda valor en caché."""
        self.cache[key] = (data, datetime.utcnow())
    
    @abstractmethod
    async def get_data(self, identifier: str) -> Dict[str, Any]:
        """Obtiene datos para un identificador."""
        pass
    
    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()


class FinnhubService(BaseDataService):
    """
    Servicio de integración con Finnhub API.
    Proporciona datos de mercado, noticias y fundamentales.
    
    API Docs: https://finnhub.io/docs/api
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self):
        super().__init__()
        self.api_key = self.settings.FINNHUB_API_KEY
        self.is_configured = bool(self.api_key)
    
    def _headers(self) -> Dict[str, str]:
        """Headers para las requests."""
        return {"X-Finnhub-Token": self.api_key or ""}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_data(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene datos completos de una acción.
        
        Args:
            symbol: Símbolo de la acción (ej: "AAPL")
        
        Returns:
            Datos completos incluyendo precio, perfil, noticias
        """
        if not self.is_configured:
            return {"error": "Finnhub API key not configured"}
        
        cache_key = self._cache_key(symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Obtener múltiples tipos de datos en paralelo
            tasks = [
                self._get_quote(symbol),
                self._get_company_profile(symbol),
                self._get_company_news(symbol),
                self._get_basic_financials(symbol),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data = {
                "symbol": symbol,
                "quote": results[0] if not isinstance(results[0], Exception) else None,
                "profile": results[1] if not isinstance(results[1], Exception) else None,
                "news": results[2] if not isinstance(results[2], Exception) else None,
                "financials": results[3] if not isinstance(results[3], Exception) else None,
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
            self._set_cached(cache_key, data)
            return data
            
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _get_quote(self, symbol: str) -> Dict[str, Any]:
        """Obtiene cotización actual."""
        response = await self.client.get(
            f"{self.BASE_URL}/quote",
            params={"symbol": symbol},
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def _get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Obtiene perfil de la empresa."""
        response = await self.client.get(
            f"{self.BASE_URL}/stock/profile2",
            params={"symbol": symbol},
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def _get_company_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Obtiene noticias recientes de la empresa."""
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        response = await self.client.get(
            f"{self.BASE_URL}/company-news",
            params={
                "symbol": symbol,
                "from": from_date,
                "to": to_date
            },
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()[:10]  # Limitar a 10 noticias
    
    async def _get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Obtiene métricas financieras básicas."""
        response = await self.client.get(
            f"{self.BASE_URL}/stock/metric",
            params={"symbol": symbol, "metric": "all"},
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()
    
    async def search_companies(self, query: str) -> List[Dict[str, Any]]:
        """Busca empresas por nombre o símbolo."""
        if not self.is_configured:
            return []
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/search",
                params={"q": query},
                headers=self._headers()
            )
            response.raise_for_status()
            return response.json().get("result", [])[:20]
        except Exception as e:
            logger.error(f"Finnhub search error: {e}")
            return []
    
    async def get_market_news(self, category: str = "general") -> List[Dict[str, Any]]:
        """Obtiene noticias generales del mercado."""
        if not self.is_configured:
            return []
        
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/news",
                params={"category": category},
                headers=self._headers()
            )
            response.raise_for_status()
            return response.json()[:20]
        except Exception as e:
            logger.error(f"Finnhub news error: {e}")
            return []


class SECEdgarService(BaseDataService):
    """
    Servicio de integración con SEC EDGAR.
    Proporciona filings y documentos regulatorios.
    
    API Docs: https://www.sec.gov/developer
    """
    
    BASE_URL = "https://data.sec.gov"
    FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"
    
    def __init__(self):
        super().__init__()
        self.user_agent = self.settings.SEC_EDGAR_USER_AGENT
        self.cache_ttl = timedelta(hours=1)  # SEC data changes less frequently
    
    def _headers(self) -> Dict[str, str]:
        """Headers requeridos por SEC."""
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_data(self, cik: str) -> Dict[str, Any]:
        """
        Obtiene datos de una empresa por CIK.
        
        Args:
            cik: Central Index Key de la empresa
        
        Returns:
            Información de filings y datos de la empresa
        """
        cache_key = self._cache_key(cik)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Normalizar CIK a 10 dígitos
            cik_padded = cik.zfill(10)
            
            # Obtener submissions
            response = await self.client.get(
                f"{self.BASE_URL}/submissions/CIK{cik_padded}.json",
                headers=self._headers()
            )
            response.raise_for_status()
            data = response.json()
            
            # Procesar filings recientes
            recent_filings = self._process_filings(data.get("filings", {}).get("recent", {}))
            
            result = {
                "cik": cik,
                "name": data.get("name"),
                "sic": data.get("sic"),
                "sic_description": data.get("sicDescription"),
                "category": data.get("category"),
                "fiscal_year_end": data.get("fiscalYearEnd"),
                "state": data.get("stateOfIncorporation"),
                "recent_filings": recent_filings[:20],
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"SEC EDGAR error for CIK {cik}: {e}")
            return {"error": str(e)}
    
    def _process_filings(self, filings: Dict) -> List[Dict[str, Any]]:
        """Procesa filings en formato estructurado."""
        if not filings:
            return []
        
        processed = []
        accession_numbers = filings.get("accessionNumber", [])
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])
        primary_documents = filings.get("primaryDocument", [])
        
        for i in range(min(len(accession_numbers), 50)):
            processed.append({
                "accession_number": accession_numbers[i] if i < len(accession_numbers) else None,
                "form": forms[i] if i < len(forms) else None,
                "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                "primary_document": primary_documents[i] if i < len(primary_documents) else None,
            })
        
        return processed
    
    async def search_companies(self, query: str) -> List[Dict[str, Any]]:
        """Busca empresas en SEC."""
        try:
            # Usar el endpoint de company search
            response = await self.client.get(
                f"{self.BASE_URL}/cgi-bin/browse-edgar",
                params={
                    "company": query,
                    "type": "",
                    "owner": "include",
                    "count": 20,
                    "action": "getcompany",
                    "output": "atom"
                },
                headers=self._headers()
            )
            response.raise_for_status()
            
            # Parsear XML response (simplified)
            # En producción, usar un parser XML apropiado
            return []  # TODO: Implementar parsing XML
            
        except Exception as e:
            logger.error(f"SEC search error: {e}")
            return []
    
    async def get_filing_document(self, accession_number: str, document: str) -> str:
        """Obtiene el contenido de un documento de filing."""
        try:
            # Formatear accession number
            accession_formatted = accession_number.replace("-", "")
            
            response = await self.client.get(
                f"{self.BASE_URL}/Archives/edgar/data/{accession_formatted}/{document}",
                headers=self._headers()
            )
            response.raise_for_status()
            return response.text
            
        except Exception as e:
            logger.error(f"SEC document error: {e}")
            return ""


class NewsAggregatorService(BaseDataService):
    """
    Servicio de agregación de noticias.
    Integra múltiples fuentes de noticias.
    """
    
    def __init__(self):
        super().__init__()
        self.news_api_key = self.settings.NEWS_API_KEY
        self.is_configured = bool(self.news_api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_data(self, query: str) -> Dict[str, Any]:
        """Obtiene noticias para una query."""
        if not self.is_configured:
            return {"error": "News API key not configured", "articles": []}
        
        cache_key = self._cache_key(query)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            response = await self.client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 20
                },
                headers={"X-Api-Key": self.news_api_key}
            )
            response.raise_for_status()
            data = response.json()
            
            result = {
                "query": query,
                "total_results": data.get("totalResults", 0),
                "articles": [
                    {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "source": article.get("source", {}).get("name"),
                        "url": article.get("url"),
                        "published_at": article.get("publishedAt"),
                    }
                    for article in data.get("articles", [])
                ],
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"News API error: {e}")
            return {"error": str(e), "articles": []}


class ExternalDataOrchestrator:
    """
    Orquestador de servicios de datos externos.
    Coordina la obtención de datos de múltiples fuentes.
    """
    
    def __init__(self):
        self.finnhub = FinnhubService()
        self.sec = SECEdgarService()
        self.news = NewsAggregatorService()
    
    async def get_company_intelligence(self, identifier: str) -> Dict[str, Any]:
        """
        Obtiene inteligencia completa de una empresa.
        
        Args:
            identifier: Símbolo de acción o nombre de empresa
        
        Returns:
            Datos agregados de múltiples fuentes
        """
        logger.info(f"Gathering intelligence for: {identifier}")
        
        tasks = [
            self.finnhub.get_data(identifier.upper()),
            self.news.get_data(identifier),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "identifier": identifier,
            "market_data": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
            "news": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
            "retrieved_at": datetime.utcnow().isoformat()
        }
    
    async def search_all(self, query: str) -> Dict[str, Any]:
        """Busca en todas las fuentes disponibles."""
        tasks = [
            self.finnhub.search_companies(query),
            self.finnhub.get_market_news(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "query": query,
            "companies": results[0] if not isinstance(results[0], Exception) else [],
            "market_news": results[1] if not isinstance(results[1], Exception) else [],
            "retrieved_at": datetime.utcnow().isoformat()
        }
    
    async def close_all(self):
        """Cierra todos los servicios."""
        await asyncio.gather(
            self.finnhub.close(),
            self.sec.close(),
            self.news.close()
        )


# Singleton instance
_data_orchestrator: Optional[ExternalDataOrchestrator] = None

def get_data_orchestrator() -> ExternalDataOrchestrator:
    """Obtiene la instancia singleton del orquestador de datos."""
    global _data_orchestrator
    if _data_orchestrator is None:
        _data_orchestrator = ExternalDataOrchestrator()
    return _data_orchestrator
