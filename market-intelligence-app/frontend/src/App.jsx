import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, AlertTriangle, Building2, FileText, Globe, Zap, Shield, BarChart3, Activity, ChevronRight, Loader2, CheckCircle2, Clock, Filter, Download, Bell, Settings, Database, Cpu, RefreshCw } from 'lucide-react';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API Client
const api = {
  async query(queryText, includeVerification = true) {
    const response = await fetch(`${API_BASE_URL}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: queryText,
        include_verification: includeVerification,
        max_results: 10
      })
    });
    if (!response.ok) throw new Error('Query failed');
    return response.json();
  },

  async search(query, topK = 10) {
    const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK })
    });
    if (!response.ok) throw new Error('Search failed');
    return response.json();
  },

  async verify(claim) {
    const response = await fetch(`${API_BASE_URL}/api/v1/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ claim })
    });
    if (!response.ok) throw new Error('Verification failed');
    return response.json();
  },

  async getStats() {
    const response = await fetch(`${API_BASE_URL}/api/v1/stats`);
    if (!response.ok) throw new Error('Stats failed');
    return response.json();
  },

  async health() {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  },

  async ingestDocument(content, sourceName, sourceType = 'primary') {
    const response = await fetch(`${API_BASE_URL}/api/v1/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content,
        source_name: sourceName,
        source_type: sourceType
      })
    });
    if (!response.ok) throw new Error('Ingestion failed');
    return response.json();
  }
};

// Demo data for when API is not available
const demoMarketData = [
  { id: 1, name: 'Artificial Intelligence Platforms', size: '$54.8B', growth: '+39.5%', status: 'high-growth', confidence: 98 },
  { id: 2, name: 'Enterprise Data Integration', size: '$18.2B', growth: '+28.3%', status: 'high-growth', confidence: 95 },
  { id: 3, name: 'RAG Systems for Enterprise', size: '$2.1B', growth: '+67.2%', status: 'emerging', confidence: 88 },
  { id: 4, name: 'Agentic AI Applications', size: '$5.5B', growth: '+44.8%', status: 'high-growth', confidence: 94 },
  { id: 5, name: 'Market Intelligence Software', size: '$4.2B', growth: '+31.5%', status: 'high-growth', confidence: 91 },
];

const demoSignals = [
  { type: 'patent', title: 'Nueva patente de OpenAI en verificación automatizada', impact: 'high', date: '2025-12-10' },
  { type: 'funding', title: 'Anthropic cierra ronda de $2B Serie D', impact: 'high', date: '2025-12-09' },
  { type: 'regulatory', title: 'EU AI Act entra en vigor Q1 2026', impact: 'medium', date: '2025-12-08' },
  { type: 'm&a', title: 'Rumores: Databricks evalúa adquisición de startup RAG', impact: 'medium', date: '2025-12-07' },
];

// Verification Badge Component
function VerificationBadge({ status }) {
  const configs = {
    pending: { icon: Clock, color: 'text-gray-400', bg: 'bg-gray-400/10', text: 'Pendiente' },
    verifying: { icon: Loader2, color: 'text-amber-400', bg: 'bg-amber-400/10', text: 'Verificando...' },
    cross_checking: { icon: RefreshCw, color: 'text-blue-400', bg: 'bg-blue-400/10', text: 'Cross-validando...' },
    verified: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-400/10', text: 'Verificado ✓' },
    partial: { icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-400/10', text: 'Parcial' },
    failed: { icon: AlertTriangle, color: 'text-red-400', bg: 'bg-red-400/10', text: 'No verificado' },
  };
  const config = configs[status] || configs.pending;
  const Icon = config.icon;
  
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bg} ${config.color} text-sm font-medium`}>
      <Icon className={`w-4 h-4 ${status === 'verifying' ? 'animate-spin' : ''}`} />
      {config.text}
    </div>
  );
}

// Main Application Component
export default function MarketIntelligenceApp() {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [systemStatus, setSystemStatus] = useState(null);
  const [systemStats, setSystemStats] = useState(null);
  const [error, setError] = useState(null);

  // Check system status on mount
  useEffect(() => {
    checkSystemStatus();
    loadSystemStats();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const health = await api.health();
      setSystemStatus(health);
      setError(null);
    } catch (e) {
      setSystemStatus({ status: 'offline' });
      setError('Backend no disponible. Mostrando datos de demostración.');
    }
  };

  const loadSystemStats = async () => {
    try {
      const stats = await api.getStats();
      setSystemStats(stats);
    } catch (e) {
      console.log('Stats not available');
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setError(null);
    
    try {
      const result = await api.query(query, true);
      setSearchResults({
        query: query,
        answer: result.answer,
        confidence: result.confidence_score,
        verificationStatus: result.verification_status,
        sources: result.sources || [],
        executionTime: result.execution_time_ms,
        timestamp: result.timestamp
      });
    } catch (e) {
      setError('Error al procesar la consulta. Verifique que el backend esté activo.');
      // Show demo response
      setSearchResults({
        query: query,
        answer: `[MODO DEMO] Esta es una respuesta de demostración para: "${query}"\n\nPara obtener respuestas reales basadas en IA:\n1. Inicie el backend: cd backend && uvicorn app.main:app --reload\n2. Configure sus API keys en backend/.env\n3. Indexe documentos usando el endpoint /api/v1/ingest`,
        confidence: 0.5,
        verificationStatus: 'pending',
        sources: [{ name: 'Demo Source', type: 'demo' }],
        executionTime: 0,
        timestamp: new Date().toISOString()
      });
    }
    
    setIsSearching(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100" style={{ fontFamily: "'Inter', -apple-system, sans-serif" }}>
      {/* Background gradient */}
      <div className="fixed inset-0 bg-gradient-to-br from-indigo-950/30 via-transparent to-emerald-950/20 pointer-events-none" />
      
      {/* Header */}
      <header className="relative border-b border-white/5 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-emerald-500 flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">Market Intelligence AI</h1>
                <p className="text-xs text-gray-500">RAG + Multi-Agent + Triple Verification</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <button onClick={checkSystemStatus} className="p-2 rounded-lg hover:bg-white/5 transition-colors" title="Refresh status">
                <RefreshCw className="w-5 h-5 text-gray-400" />
              </button>
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Bell className="w-5 h-5 text-gray-400" />
              </button>
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Settings className="w-5 h-5 text-gray-400" />
              </button>
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                systemStatus?.status === 'healthy' 
                  ? 'bg-emerald-500/10 text-emerald-400' 
                  : 'bg-amber-500/10 text-amber-400'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus?.status === 'healthy' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
                }`} />
                {systemStatus?.status === 'healthy' ? 'Sistema Activo' : 'Demo Mode'}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-6 py-8">
        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Search Section */}
        <section className="mb-10">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
              Inteligencia de Mercado Confiable
            </h2>
            <p className="text-gray-500">Consultas verificadas con triple validación de fuentes</p>
          </div>
          
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 to-emerald-500/20 rounded-2xl blur-xl" />
              <div className="relative bg-gray-900/80 backdrop-blur-xl rounded-2xl border border-white/10 p-2">
                <div className="flex items-center gap-3">
                  <Search className="w-5 h-5 text-gray-500 ml-4" />
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Ej: ¿Cuál es el tamaño del mercado de plataformas RAG en 2025?"
                    className="flex-1 bg-transparent py-4 text-lg outline-none placeholder-gray-600"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isSearching || !query.trim()}
                    className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-emerald-500 rounded-xl font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center gap-2"
                  >
                    {isSearching ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Analizando...
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5" />
                        Consultar
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
            
            {/* Quick queries */}
            <div className="flex flex-wrap gap-2 mt-4 justify-center">
              {['Mercado IA 2025', 'Competidores RAG', 'Tendencias FinTech', 'Due Diligence AI'].map((q) => (
                <button
                  key={q}
                  onClick={() => setQuery(q)}
                  className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-sm text-gray-400 hover:text-white transition-all border border-white/5"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Search Results */}
        {searchResults && (
          <section className="mb-10">
            <div className="bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
              <div className="p-6 border-b border-white/5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Resultado de Inteligencia</h3>
                  <VerificationBadge status={searchResults.verificationStatus} />
                </div>
                
                <div className="text-gray-300 leading-relaxed whitespace-pre-wrap">
                  {searchResults.answer}
                </div>
                
                <div className="mt-4 flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm text-gray-400">
                      Confianza: <span className={`font-semibold ${searchResults.confidence > 0.8 ? 'text-emerald-400' : 'text-amber-400'}`}>
                        {(searchResults.confidence * 100).toFixed(1)}%
                      </span>
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-indigo-400" />
                    <span className="text-sm text-gray-400">{searchResults.sources.length} fuentes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-400">{searchResults.executionTime?.toFixed(0) || 0}ms</span>
                  </div>
                </div>
              </div>
              
              {searchResults.sources.length > 0 && (
                <div className="p-6 bg-black/20">
                  <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Fuentes Citadas</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {searchResults.sources.map((source, i) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5">
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-emerald-400" />
                          <span className="text-sm">{source.name}</span>
                        </div>
                        <span className="text-xs text-gray-500 px-2 py-1 rounded bg-white/5">{source.type}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Navigation Tabs */}
        <div className="flex gap-1 mb-6 p-1 bg-gray-900/50 rounded-xl w-fit">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'markets', label: 'Mercados', icon: TrendingUp },
            { id: 'signals', label: 'Señales', icon: AlertTriangle },
            { id: 'system', label: 'Sistema', icon: Cpu },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-indigo-500/20 to-emerald-500/20 text-white'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-3 gap-6">
          {/* Markets Panel */}
          <div className="col-span-2 bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
            <div className="p-5 border-b border-white/5 flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-emerald-400" />
                Mercados de Alto Crecimiento
              </h3>
              <button className="text-sm text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
                Ver todos <ChevronRight className="w-4 h-4" />
              </button>
            </div>
            
            <div className="divide-y divide-white/5">
              {demoMarketData.map((market) => (
                <div key={market.id} className="p-4 hover:bg-white/5 transition-colors cursor-pointer">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium mb-1">{market.name}</h4>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-gray-500">Tamaño: <span className="text-white">{market.size}</span></span>
                        <span className="font-semibold text-emerald-400">{market.growth} CAGR</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        market.status === 'high-growth' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                      }`}>
                        {market.status === 'high-growth' ? 'Alto Crecimiento' : 'Emergente'}
                      </span>
                      <div className="text-right">
                        <div className="text-xs text-gray-500">Confianza</div>
                        <div className="text-sm font-semibold text-emerald-400">{market.confidence}%</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Signals Panel */}
          <div className="bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
            <div className="p-5 border-b border-white/5">
              <h3 className="font-semibold flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                Señales Detectadas
              </h3>
            </div>
            
            <div className="divide-y divide-white/5">
              {demoSignals.map((signal, i) => (
                <div key={i} className="p-4 hover:bg-white/5 transition-colors cursor-pointer">
                  <div className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                      signal.type === 'patent' ? 'bg-purple-500/20 text-purple-400' :
                      signal.type === 'funding' ? 'bg-emerald-500/20 text-emerald-400' :
                      signal.type === 'regulatory' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-amber-500/20 text-amber-400'
                    }`}>
                      {signal.type === 'patent' ? <FileText className="w-4 h-4" /> :
                       signal.type === 'funding' ? <TrendingUp className="w-4 h-4" /> :
                       signal.type === 'regulatory' ? <Shield className="w-4 h-4" /> :
                       <Building2 className="w-4 h-4" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium leading-snug">{signal.title}</p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-gray-500">{signal.date}</span>
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          signal.impact === 'high' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'
                        }`}>
                          {signal.impact === 'high' ? 'Alto Impacto' : 'Medio'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* System Stats */}
          {activeTab === 'system' && (
            <div className="col-span-3 bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-white/10 p-6">
              <h3 className="font-semibold flex items-center gap-2 mb-4">
                <Cpu className="w-5 h-5 text-indigo-400" />
                Estado del Sistema
              </h3>
              
              <div className="grid grid-cols-4 gap-4">
                <div className="p-4 rounded-xl bg-white/5">
                  <div className="text-xs text-gray-500 mb-1">Documentos Indexados</div>
                  <div className="text-2xl font-bold">{systemStats?.documents_indexed || 0}</div>
                </div>
                <div className="p-4 rounded-xl bg-white/5">
                  <div className="text-xs text-gray-500 mb-1">Proveedor LLM</div>
                  <div className="text-lg font-semibold">{systemStats?.llm_provider || 'No configurado'}</div>
                </div>
                <div className="p-4 rounded-xl bg-white/5">
                  <div className="text-xs text-gray-500 mb-1">Modelo</div>
                  <div className="text-sm font-medium truncate">{systemStats?.llm_model || 'N/A'}</div>
                </div>
                <div className="p-4 rounded-xl bg-white/5">
                  <div className="text-xs text-gray-500 mb-1">Vector Store</div>
                  <div className="text-lg font-semibold">{systemStats?.vector_store_status || 'N/A'}</div>
                </div>
              </div>
              
              {systemStats?.rag_config && (
                <div className="mt-4 p-4 rounded-xl bg-white/5">
                  <div className="text-xs text-gray-500 mb-2">Configuración RAG</div>
                  <div className="grid grid-cols-4 gap-4 text-sm">
                    <div>Chunk Size: <span className="text-white">{systemStats.rag_config.chunk_size}</span></div>
                    <div>Overlap: <span className="text-white">{systemStats.rag_config.chunk_overlap}</span></div>
                    <div>Top K: <span className="text-white">{systemStats.rag_config.top_k}</span></div>
                    <div>Threshold: <span className="text-white">{systemStats.rag_config.similarity_threshold}</span></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Stats Cards */}
          <div className="col-span-3 grid grid-cols-4 gap-4">
            {[
              { label: 'Mercados Monitoreados', value: '200,000+', icon: Globe, color: 'indigo' },
              { label: 'Fuentes Verificadas', value: '1.2M', icon: Database, color: 'emerald' },
              { label: 'Precisión Sistema', value: '96.4%', icon: Activity, color: 'amber' },
              { label: 'Agentes Activos', value: '4', icon: Cpu, color: 'purple' },
            ].map((stat, i) => (
              <div key={i} className="bg-gray-900/50 backdrop-blur-xl rounded-xl border border-white/10 p-4">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    stat.color === 'indigo' ? 'bg-indigo-500/20' :
                    stat.color === 'emerald' ? 'bg-emerald-500/20' :
                    stat.color === 'amber' ? 'bg-amber-500/20' : 'bg-purple-500/20'
                  }`}>
                    <stat.icon className={`w-5 h-5 ${
                      stat.color === 'indigo' ? 'text-indigo-400' :
                      stat.color === 'emerald' ? 'text-emerald-400' :
                      stat.color === 'amber' ? 'text-amber-400' : 'text-purple-400'
                    }`} />
                  </div>
                  <div>
                    <div className="text-2xl font-bold">{stat.value}</div>
                    <div className="text-xs text-gray-500">{stat.label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative border-t border-white/5 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" />
              Sistema de Triple Verificación Activo
            </div>
            <div>
              API: {API_BASE_URL} | Última actualización: {new Date().toLocaleString('es-ES')}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
