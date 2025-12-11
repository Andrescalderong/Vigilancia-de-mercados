# 🚀 GUÍA DE INICIO RÁPIDO

## Inicio en 5 Minutos

### Paso 1: Configurar Variables de Entorno

```bash
cd backend
cp .env.example .env
```

Edita `backend/.env` y configura al menos una API key de LLM:

```env
# Opción A: Anthropic Claude (Recomendado)
ANTHROPIC_API_KEY=tu_api_key_de_anthropic

# Opción B: OpenAI
OPENAI_API_KEY=tu_api_key_de_openai
```

### Paso 2: Instalar y Ejecutar Backend

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn app.main:app --reload
```

El backend estará disponible en: **http://localhost:8000**

- Documentación API: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Paso 3: (Opcional) Ejecutar Frontend

```bash
cd frontend
npm install
npm start
```

El frontend estará en: **http://localhost:3000**

---

## Probar la API

### Consulta de Inteligencia
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Cuál es el tamaño del mercado de IA en 2025?"}'
```

### Indexar un Documento
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "El mercado de inteligencia artificial alcanzará $400 billones en 2025...",
    "source_name": "Market Report 2025",
    "source_type": "primary"
  }'
```

### Verificar Información
```bash
curl -X POST http://localhost:8000/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "El mercado de IA crecerá 40% en 2025"}'
```

---

## Obtener API Keys (Gratuitas)

| Servicio | URL | Uso |
|----------|-----|-----|
| Anthropic | https://console.anthropic.com | LLM principal |
| OpenAI | https://platform.openai.com/api-keys | LLM alternativo |
| Finnhub | https://finnhub.io/register | Datos financieros |
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | Datos de acciones |

---

## Docker (Alternativa)

```bash
# Configurar .env primero
docker-compose up -d

# Ver logs
docker-compose logs -f backend
```

---

## Estructura del Proyecto

```
market-intelligence-app/
├── backend/
│   ├── app/
│   │   ├── agents/          # Sistema multi-agente
│   │   ├── core/            # RAG Engine
│   │   ├── services/        # APIs externas
│   │   └── main.py          # FastAPI
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   └── src/
│       └── App.jsx          # React App
├── docker-compose.yml
└── README.md
```

---

## Troubleshooting

### Error: "No module named..."
```bash
pip install -r requirements.txt
```

### Error: "API key not configured"
Verifica que `.env` existe y tiene las API keys correctas.

### Error de CORS
Asegúrate de que el frontend apunte a `http://localhost:8000`

---

**¿Preguntas?** Revisa la documentación completa en `/docs` o abre un issue.
