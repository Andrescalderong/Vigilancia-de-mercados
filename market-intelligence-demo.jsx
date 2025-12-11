import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, AlertTriangle, Building2, Users, FileText, Globe, Zap, Shield, BarChart3, PieChart, Activity, ChevronRight, Loader2, CheckCircle2, XCircle, Clock, Filter, Download, Bell, Settings } from 'lucide-react';

// Mock data for demonstration
const mockMarketData = [
  { id: 1, name: 'Artificial Intelligence Platforms', size: '$54.8B', growth: '+39.5%', status: 'high-growth', confidence: 98 },
  { id: 2, name: 'Enterprise Data Integration', size: '$18.2B', growth: '+28.3%', status: 'high-growth', confidence: 95 },
  { id: 3, name: 'Customer Data Platforms', size: '$6.4B', growth: '+35.9%', status: 'emerging', confidence: 92 },
  { id: 4, name: 'RAG Systems', size: '$2.1B', growth: '+67.2%', status: 'emerging', confidence: 88 },
  { id: 5, name: 'Agentic AI', size: '$5.5B', growth: '+44.8%', status: 'high-growth', confidence: 94 },
];

const mockCompanies = [
  { name: 'MarketsandMarkets', revenue: '$150M', employees: '850+', sector: 'Market Intelligence', risk: 'low' },
  { name: 'Gartner Inc.', revenue: '$5.9B', employees: '19,500', sector: 'Research & Advisory', risk: 'low' },
  { name: 'Bloomberg LP', revenue: '$12.2B', employees: '20,000+', sector: 'Financial Data', risk: 'low' },
  { name: 'AlphaSense', revenue: '$200M', employees: '1,200', sector: 'AI Search', risk: 'medium' },
];

const mockSignals = [
  { type: 'patent', title: 'Nueva patente de OpenAI en verificación automatizada', impact: 'high', date: '2025-12-10' },
  { type: 'funding', title: 'Anthropic levanta $2B en Serie D', impact: 'high', date: '2025-12-09' },
  { type: 'regulatory', title: 'EU AI Act entra en vigor Q1 2026', impact: 'medium', date: '2025-12-08' },
  { type: 'm&a', title: 'Rumores de adquisición: Databricks evalúa comprar startup RAG', impact: 'medium', date: '2025-12-07' },
];

export default function MarketIntelligenceDashboard() {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [verificationStatus, setVerificationStatus] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setVerificationStatus('verifying');
    
    // Simulate RAG search with verification
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setVerificationStatus('cross-checking');
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setVerificationStatus('verified');
    
    setSearchResults({
      query: query,
      answer: `Basado en el análisis de 47 fuentes verificadas, el mercado de ${query} muestra un crecimiento sostenido del 35-40% CAGR. Los principales drivers incluyen: (1) adopción enterprise acelerada post-2024, (2) reducción de costos de infraestructura cloud, (3) madurez de frameworks RAG. Se identifican 3 oportunidades de alto valor en segmentos adyacentes.`,
      sources: [
        { name: 'MarketsandMarkets Report 2025', type: 'primary', confidence: 98 },
        { name: 'Gartner Hype Cycle 2025', type: 'primary', confidence: 96 },
        { name: 'SEC Filings - NVDA, MSFT', type: 'regulatory', confidence: 99 },
        { name: 'Patent Analysis USPTO', type: 'alternative', confidence: 94 },
      ],
      confidence: 96,
      timestamp: new Date().toISOString(),
    });
    
    setIsSearching(false);
  };

  const VerificationBadge = ({ status }) => {
    const configs = {
      verifying: { icon: Loader2, color: 'text-amber-400', bg: 'bg-amber-400/10', text: 'Verificando fuentes primarias...' },
      'cross-checking': { icon: Clock, color: 'text-blue-400', bg: 'bg-blue-400/10', text: 'Cross-validando datos...' },
      verified: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-400/10', text: 'Verificado ✓' },
    };
    const config = configs[status];
    if (!config) return null;
    
    return (
      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bg} ${config.color} text-sm font-medium`}>
        <config.icon className={`w-4 h-4 ${status === 'verifying' ? 'animate-spin' : ''}`} />
        {config.text}
      </div>
    );
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
                <p className="text-xs text-gray-500">Powered by RAG + Triple Verification</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Bell className="w-5 h-5 text-gray-400" />
              </button>
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Settings className="w-5 h-5 text-gray-400" />
              </button>
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 text-sm">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Sistema Activo
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-6 py-8">
        {/* Search Section */}
        <section className="mb-10">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
              Inteligencia de Mercado Confiable
            </h2>
            <p className="text-gray-500">Haga cualquier pregunta comercial y obtenga inteligencia verificada en segundos</p>
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
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Ej: ¿Cuál es el tamaño del mercado de plataformas RAG en 2025?"
                    className="flex-1 bg-transparent py-4 text-lg outline-none placeholder-gray-600"
                  />
                  <button
                    onClick={handleSearch}
                    disabled={isSearching}
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
              {['Mercado AI 2025', 'Competidores Bloomberg', 'Tendencias FinTech', 'Due Diligence Startups'].map((q) => (
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
                  <VerificationBadge status={verificationStatus} />
                </div>
                
                <p className="text-gray-300 leading-relaxed">{searchResults.answer}</p>
                
                <div className="mt-4 flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm text-gray-400">Confianza: <span className="text-emerald-400 font-semibold">{searchResults.confidence}%</span></span>
                  </div>
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-indigo-400" />
                    <span className="text-sm text-gray-400">{searchResults.sources.length} fuentes verificadas</span>
                  </div>
                </div>
              </div>
              
              <div className="p-6 bg-black/20">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Fuentes Citadas</h4>
                <div className="grid grid-cols-2 gap-3">
                  {searchResults.sources.map((source, i) => (
                    <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${source.confidence > 95 ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                        <span className="text-sm">{source.name}</span>
                      </div>
                      <span className="text-xs text-gray-500 px-2 py-1 rounded bg-white/5">{source.type}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Navigation Tabs */}
        <div className="flex gap-1 mb-6 p-1 bg-gray-900/50 rounded-xl w-fit">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'markets', label: 'Mercados', icon: TrendingUp },
            { id: 'companies', label: 'Empresas', icon: Building2 },
            { id: 'signals', label: 'Señales', icon: AlertTriangle },
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
              {mockMarketData.map((market) => (
                <div key={market.id} className="p-4 hover:bg-white/5 transition-colors cursor-pointer">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="font-medium mb-1">{market.name}</h4>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-gray-500">Tamaño: <span className="text-white">{market.size}</span></span>
                        <span className={`font-semibold ${market.growth.startsWith('+') ? 'text-emerald-400' : 'text-red-400'}`}>
                          {market.growth} CAGR
                        </span>
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
              {mockSignals.map((signal, i) => (
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

          {/* Companies Panel */}
          <div className="col-span-2 bg-gray-900/50 backdrop-blur-xl rounded-2xl border border-white/10 overflow-hidden">
            <div className="p-5 border-b border-white/5 flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2">
                <Building2 className="w-5 h-5 text-indigo-400" />
                Empresas Monitoreadas
              </h3>
              <div className="flex gap-2">
                <button className="px-3 py-1.5 rounded-lg bg-white/5 text-sm flex items-center gap-1 hover:bg-white/10">
                  <Filter className="w-4 h-4" /> Filtrar
                </button>
                <button className="px-3 py-1.5 rounded-lg bg-white/5 text-sm flex items-center gap-1 hover:bg-white/10">
                  <Download className="w-4 h-4" /> Exportar
                </button>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-xs text-gray-500 uppercase tracking-wider">
                    <th className="px-4 py-3">Empresa</th>
                    <th className="px-4 py-3">Revenue</th>
                    <th className="px-4 py-3">Empleados</th>
                    <th className="px-4 py-3">Sector</th>
                    <th className="px-4 py-3">Riesgo</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {mockCompanies.map((company, i) => (
                    <tr key={i} className="hover:bg-white/5 transition-colors cursor-pointer">
                      <td className="px-4 py-3 font-medium">{company.name}</td>
                      <td className="px-4 py-3 text-gray-400">{company.revenue}</td>
                      <td className="px-4 py-3 text-gray-400">{company.employees}</td>
                      <td className="px-4 py-3 text-gray-400">{company.sector}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          company.risk === 'low' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                        }`}>
                          {company.risk === 'low' ? 'Bajo' : 'Medio'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Stats Panel */}
          <div className="space-y-4">
            {[
              { label: 'Mercados Monitoreados', value: '200,000+', icon: Globe, color: 'indigo' },
              { label: 'Fuentes Verificadas', value: '1.2M', icon: Shield, color: 'emerald' },
              { label: 'Precisión Promedio', value: '96.4%', icon: Activity, color: 'amber' },
            ].map((stat, i) => (
              <div key={i} className="bg-gray-900/50 backdrop-blur-xl rounded-xl border border-white/10 p-4">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg bg-${stat.color}-500/20 flex items-center justify-center`}>
                    <stat.icon className={`w-5 h-5 text-${stat.color}-400`} />
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
              Última actualización: {new Date().toLocaleString('es-ES')}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
