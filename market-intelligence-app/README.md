# 🚀 Market Intelligence AI Platform

Plataforma de inteligencia de mercado impulsada por IA con **RAG (Retrieval-Augmented Generation)**, sistema **multi-agente** y **triple verificación** para información 100% confiable.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Características Principales

### Inteligencia Confiable
- ✅ **Triple Verificación**: Verificación primaria, secundaria y terciaria de toda información
- 🎯 **RAG Avanzado**: Retrieval-Augmented Generation con ChromaDB
- 🤖 **Sistema Multi-Agente**: Agentes especializados para búsqueda, análisis, verificación y síntesis
- 📊 **Confianza Cuantificada**: Score de confianza para cada respuesta

### Capacidades de Inteligencia
- 🔍 **Búsqueda Semántica**: Búsqueda inteligente en múltiples fuentes
- 📈 **Análisis de Mercado**: Tamaño, crecimiento, tendencias, competidores
- 🏢 **Inteligencia Corporativa**: Información de empresas, directivos, financieros
- ⚠️ **Detección de Señales**: Patentes, funding, M&A, regulaciones
- 🎯 **Inteligencia Predictiva**: Tendencias emergentes y oportunidades

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend React                            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Agent Orchestrator                         │ │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │ │
│  │   │ Search   │ │ Analysis │ │ Verify   │ │ Synthesis│ │ │
│  │   │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │ │ │
│  │   └──────────┘ └──────────┘ └──────────┘ └──────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   RAG Engine                            │ │
│  │   ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │ │
│  │   │ Retriever  │ │ Augmenter  │ │ Generator (LLM)    │ │ │
│  │   └────────────┘ └────────────┘ └────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
│  │ ChromaDB     │ │ External     │ │ Document           │  │
│  │ (Vectors)    │ │ APIs         │ │ Store              │  │
│  └──────────────┘ └──────────────┘ └────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Opción 1: Instalación Local

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd market-intelligence-app

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows

# 3. Instalar dependencias
cd backend
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# 5. Ejecutar servidor
uvicorn app.main:app --reload
```

### Opción 2: Docker

```bash
# 1. Configurar variables de entorno
cp backend/.env.example .env

# 2. Construir y ejecutar
docker-compose up -d

# 3. Ver logs
docker-compose logs -f backend
```

## 📝 API Endpoints

### Consulta de Inteligencia
```bash
POST /api/v1/query
{
    "query": "¿Cuál es el tamaño del mercado de IA en 2025?",
    "include_verification": true
}
```

### Búsqueda Rápida
```bash
POST /api/v1/search
{
    "query": "plataformas RAG finanzas",
    "top_k": 10
}
```

### Verificación de Información
```bash
POST /api/v1/verify
{
    "claim": "El mercado de IA alcanzará $400B en 2025"
}
```

### Ingestar Documentos
```bash
POST /api/v1/ingest
{
    "content": "Contenido del documento...",
    "source_name": "MarketsandMarkets Report 2025",
    "source_type": "primary"
}
```

## 🔧 Configuración

### Variables de Entorno Requeridas

| Variable | Descripción | Obligatorio |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | API key de Anthropic Claude | Sí* |
| `OPENAI_API_KEY` | API key de OpenAI | Sí* |
| `FINNHUB_API_KEY` | API key de Finnhub (gratuita) | No |
| `ALPHA_VANTAGE_API_KEY` | API key de Alpha Vantage | No |

*Al menos una de las dos

### Obtener API Keys

1. **Anthropic Claude**: https://console.anthropic.com/
2. **OpenAI**: https://platform.openai.com/api-keys
3. **Finnhub** (Gratuita): https://finnhub.io/register
4. **Alpha Vantage** (Gratuita): https://www.alphavantage.co/support/#api-key

## 📊 Sistema de Verificación Triple

```
┌─────────────────────────────────────────────────────────────┐
│                 TRIPLE VERIFICATION SYSTEM                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NIVEL 1: Verificación Primaria                             │
│  └─ Fuentes oficiales: SEC, reguladores, empresas           │
│                                                              │
│  NIVEL 2: Verificación Secundaria                           │
│  └─ Cross-reference: Múltiples fuentes independientes       │
│                                                              │
│  NIVEL 3: Verificación Terciaria                            │
│  └─ Análisis de consenso y detección de discrepancias       │
│                                                              │
│  RESULTADO: Confidence Score + Estado de Verificación       │
│  └─ verified | partial | unverified | failed                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Agentes Disponibles

| Agente | Función |
|--------|---------|
| **SearchAgent** | Búsqueda semántica en múltiples fuentes |
| **AnalysisAgent** | Análisis de mercado y competencia |
| **VerificationAgent** | Triple verificación de información |
| **SynthesisAgent** | Síntesis y consolidación de resultados |

## 📁 Estructura del Proyecto

```
market-intelligence-app/
├── backend/
│   ├── app/
│   │   ├── agents/           # Sistema multi-agente
│   │   ├── core/             # RAG Engine, configuración
│   │   ├── api/              # Endpoints adicionales
│   │   ├── services/         # Servicios externos
│   │   ├── models/           # Modelos de datos
│   │   └── main.py           # FastAPI app
│   ├── data/                 # ChromaDB storage
│   ├── tests/                # Tests
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
├── docs/
├── docker-compose.yml
└── README.md
```

## 🧪 Testing

```bash
# Ejecutar tests
cd backend
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=app --cov-report=html
```

## 📈 Roadmap

- [x] MVP con RAG básico
- [x] Sistema multi-agente
- [x] Triple verificación
- [ ] Integración APIs financieras en tiempo real
- [ ] Knowledge Graph con Neo4j
- [ ] Modelos predictivos
- [ ] Sistema de alertas
- [ ] API pública

## 🤝 Contribuir

1. Fork el repositorio
2. Crear branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push al branch (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

## 📞 Soporte

- **Documentación**: `/docs`
- **API Docs**: `http://localhost:8000/docs`
- **Issues**: GitHub Issues

---

Desarrollado con ❤️ para inteligencia de mercado confiable.
