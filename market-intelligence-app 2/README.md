# 🎯 Market Intelligence Platform v2.0

Plataforma de Inteligencia de Mercado potenciada por IA con verificación triple, análisis predictivo y módulos de compliance.

## 🌟 Características

| Módulo | Descripción |
|--------|-------------|
| **Intelligence** | RAG Engine + Verificación Triple + Multi-fuente |
| **Due Diligence** | AML, Sanciones, PEP, Medios Adversos |
| **Competitive** | Monitoreo de competidores, señales de mercado |
| **Predictive** | Tendencias, señales débiles, escenarios |

## 🚀 Quick Start

### 1. Clonar y Configurar

```bash
# Clonar repositorio
git clone <repo-url>
cd market-intelligence-app

# Copiar y editar variables de entorno
cp .env.example .env
nano .env  # Agregar tus API keys
```

### 2. API Keys Requeridas

| API | Requerido | Obtener en |
|-----|-----------|------------|
| Anthropic Claude | ✅ | https://console.anthropic.com |
| Google Search | 📌 | https://console.cloud.google.com |
| Finnhub | 📌 | https://finnhub.io (gratuito) |
| OpenAI | ⭕ | https://platform.openai.com |

### 3. Desplegar

```bash
# Dar permisos al script
chmod +x scripts/deploy.sh

# Verificar configuración
./scripts/deploy.sh check

# Despliegue local (desarrollo)
./scripts/deploy.sh local

# Despliegue completo (con frontend)
./scripts/deploy.sh full

# Despliegue producción
./scripts/deploy.sh production
```

### 4. Acceder

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000

## 📚 API Reference

### Intelligence (RAG + Verificación)

```bash
# Consulta con verificación
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuál es el tamaño del mercado de IA en 2025?",
    "include_verification": true
  }'

# Búsqueda multi-fuente
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence market",
    "sources": ["rag", "google", "finnhub"]
  }'

# Indexar documento
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "El mercado de IA alcanzará USD 400B en 2027...",
    "source_name": "Industry Report 2025",
    "source_type": "primary"
  }'
```

### Due Diligence / AML

```bash
# Verificación completa de empresa
curl -X POST http://localhost:8000/api/v1/due-diligence/check \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "Empresa XYZ S.A.",
    "entity_type": "company",
    "include_sanctions": true,
    "include_adverse_media": true
  }'

# Check de sanciones rápido
curl -X POST http://localhost:8000/api/v1/due-diligence/sanctions \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Smith",
    "entity_type": "person"
  }'

# Screening por lotes
curl -X POST http://localhost:8000/api/v1/due-diligence/batch-screening \
  -H "Content-Type: application/json" \
  -d '{
    "entities": ["Empresa A", "Empresa B", "Persona C"],
    "check_type": "full"
  }'
```

### Competitive Intelligence

```bash
# Análisis de competidor
curl -X POST http://localhost:8000/api/v1/competitive/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Mi Empresa",
    "competitors": ["Competidor 1", "Competidor 2"],
    "market": "SaaS"
  }'

# Comparar competidores
curl -X POST http://localhost:8000/api/v1/competitive/compare \
  -H "Content-Type: application/json" \
  -d '{
    "competitors": ["Microsoft", "Google", "Amazon"]
  }'

# Monitoreo de mercado
curl -X POST http://localhost:8000/api/v1/competitive/market \
  -H "Content-Type: application/json" \
  -d '{
    "market_name": "Cloud Computing",
    "key_players": ["AWS", "Azure", "GCP"]
  }'

# Obtener señales competitivas
curl "http://localhost:8000/api/v1/competitive/signals?competitors=Microsoft,Google&min_strength=moderate"
```

### Predictive Intelligence

```bash
# Pronóstico de mercado
curl -X POST http://localhost:8000/api/v1/predictive/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "market": "Artificial Intelligence",
    "time_horizon": "medium_term",
    "include_weak_signals": true
  }'

# Predicción rápida
curl -X POST http://localhost:8000/api/v1/predictive/predict \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Mercado de vehículos eléctricos",
    "time_horizon": "long_term"
  }'

# Detectar señales débiles
curl -X POST http://localhost:8000/api/v1/predictive/weak-signals \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "Fintech",
    "categories": ["technology", "regulatory"]
  }'
```

### Google Search

```bash
# Búsqueda web
curl "http://localhost:8000/api/v1/google/search?q=AI%20market%202025&num=10&lang=es"

# Búsqueda de noticias
curl "http://localhost:8000/api/v1/google/news?q=artificial%20intelligence&days=7"
```

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         NGINX (Reverse Proxy)                    │
└─────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend     │    │      Redis      │
│    (React)      │    │    (FastAPI)    │    │     (Cache)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Engine    │    │   Multi-Agent   │    │  External APIs  │
│   (ChromaDB)    │    │    System       │    │ Google/Finnhub  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Estructura del Proyecto

```
market-intelligence-app/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # Configuración
│   │   │   └── rag_engine.py      # Motor RAG
│   │   ├── agents/
│   │   │   └── intelligence_agents.py  # Sistema multi-agente
│   │   ├── services/
│   │   │   ├── google_service.py       # Google APIs
│   │   │   ├── due_diligence.py        # DD/AML
│   │   │   ├── competitive_intelligence.py  # CI
│   │   │   ├── predictive_intelligence.py   # Predictivo
│   │   │   └── external_data.py        # Finnhub, etc.
│   │   └── main.py                # API FastAPI
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   └── package.json
├── nginx/
│   └── nginx.conf
├── scripts/
│   └── deploy.sh
├── docker-compose.yml
├── docker-compose.cloud.yml
├── .env.example
└── README.md
```

## ☁️ Despliegue en Cloud

### AWS ECS

```bash
# Configurar AWS CLI
aws configure

# Crear repositorio ECR
aws ecr create-repository --repository-name mi-backend
aws ecr create-repository --repository-name mi-frontend

# Desplegar
./scripts/deploy.sh aws
```

### Google Cloud Run

```bash
# Configurar gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Desplegar
./scripts/deploy.sh gcp
```

## 📊 Monitoreo

```bash
# Desplegar stack de monitoreo
./scripts/deploy.sh monitoring

# Acceder a:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001
```

## 🔧 Configuración Avanzada

### Variables de Entorno Clave

| Variable | Descripción | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | API key de Claude | - |
| `GOOGLE_API_KEY` | Google Custom Search | - |
| `FINNHUB_API_KEY` | Datos financieros | - |
| `VERIFICATION_ENABLED` | Triple verificación | true |
| `RAG_TOP_K` | Docs a recuperar | 5 |
| `CACHE_TTL_MARKET_DATA` | Cache datos mercado | 300s |

### Feature Flags

```env
FEATURE_DUE_DILIGENCE=true
FEATURE_COMPETITIVE_INTEL=true
FEATURE_PREDICTIVE=true
FEATURE_WEAK_SIGNALS=true
```

## 🛡️ Seguridad

- **Rate Limiting**: 100 req/min por IP
- **CORS**: Configurable por entorno
- **SSL/TLS**: Requerido en producción
- **Secrets**: Usar AWS Secrets Manager / GCP Secret Manager

## 📈 Roadmap

- [x] RAG Engine con verificación triple
- [x] Due Diligence / AML
- [x] Competitive Intelligence
- [x] Predictive Intelligence
- [x] Google Search Integration
- [x] Finnhub Integration
- [x] Docker deployment
- [x] Cloud deployment (AWS/GCP)
- [ ] Knowledge Graph (Neo4j)
- [ ] Bloomberg Terminal API
- [ ] ML Predictive Models
- [ ] Real-time WebSocket alerts

## 📝 Licencia

MIT License

## 🤝 Soporte

Para soporte técnico o consultas de implementación, contactar al equipo de desarrollo.
