"""
RAG Engine - Retrieval-Augmented Generation Core
=================================================
Motor principal de RAG para inteligencia de mercado.
Implementa recuperación, aumento y generación con verificación.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

# Vector store
import chromadb
from chromadb.config import Settings as ChromaSettings

# Embeddings
from sentence_transformers import SentenceTransformer

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..core.config import get_settings, Constants

logger = logging.getLogger("market_intelligence.rag")


@dataclass
class Source:
    """Representa una fuente de datos citada."""
    name: str
    url: Optional[str] = None
    source_type: str = "primary"
    confidence: float = 0.0
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Resultado de una operación de recuperación."""
    documents: List[Document]
    sources: List[Source]
    query: str
    relevance_scores: List[float]
    retrieval_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Resultado de generación con verificación."""
    answer: str
    sources: List[Source]
    confidence_score: float
    verification_status: str
    query: str
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EmbeddingEngine:
    """Motor de embeddings usando Sentence Transformers."""
    
    def __init__(self, model_name: str = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de embeddings."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def embed_text(self, text: str) -> List[float]:
        """Genera embedding para un texto."""
        return self.model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para múltiples textos."""
        return self.model.encode(texts).tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud coseno entre dos textos."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        emb1 = np.array(self.embed_text(text1)).reshape(1, -1)
        emb2 = np.array(self.embed_text(text2)).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0][0])


class VectorStore:
    """Almacén vectorial usando ChromaDB."""
    
    def __init__(self, embedding_engine: EmbeddingEngine = None):
        settings = get_settings()
        self.settings = settings
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Inicializa ChromaDB."""
        logger.info("Initializing ChromaDB vector store")
        
        self.client = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.settings.CHROMA_PERSIST_DIRECTORY,
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=self.settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized with {self.collection.count()} documents")
    
    def add_documents(
        self, 
        documents: List[Document],
        source_info: Dict[str, Any] = None
    ) -> int:
        """Añade documentos al vector store."""
        if not documents:
            return 0
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_engine.embed_texts(texts)
        
        ids = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = hashlib.md5(
                (doc.page_content + str(datetime.utcnow())).encode()
            ).hexdigest()
            ids.append(doc_id)
            
            metadata = {
                "source": doc.metadata.get("source", "unknown"),
                "source_type": source_info.get("type", "primary") if source_info else "primary",
                "indexed_at": datetime.utcnow().isoformat(),
                **doc.metadata
            }
            metadatas.append(metadata)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return len(documents)
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filter_criteria: Dict[str, Any] = None
    ) -> Tuple[List[Document], List[float]]:
        """Búsqueda semántica en el vector store."""
        settings = get_settings()
        top_k = top_k or settings.RAG_TOP_K
        
        query_embedding = self.embedding_engine.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_criteria
        )
        
        documents = []
        scores = []
        
        if results and results['documents']:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append(Document(
                    page_content=doc_text,
                    metadata=metadata
                ))
                
                # ChromaDB retorna distancias, convertir a similitud
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # Convertir distancia coseno a similitud
                scores.append(similarity)
        
        return documents, scores
    
    def delete_by_source(self, source_name: str) -> int:
        """Elimina documentos por fuente."""
        # Obtener IDs de documentos con esa fuente
        results = self.collection.get(
            where={"source": source_name}
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0


class TextChunker:
    """Procesador de texto para chunking."""
    
    def __init__(self):
        settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """Divide texto en chunks."""
        chunks = self.splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    def chunk_documents(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """Divide una lista de documentos en chunks."""
        chunked_docs = []
        for doc in documents:
            chunks = self.chunk_text(doc.page_content, doc.metadata)
            chunked_docs.extend(chunks)
        return chunked_docs


class RAGEngine:
    """
    Motor principal de Retrieval-Augmented Generation.
    
    Implementa el pipeline completo:
    1. Recuperación de documentos relevantes
    2. Aumento del contexto
    3. Generación con LLM
    4. Verificación de respuestas
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore(self.embedding_engine)
        self.text_chunker = TextChunker()
        self.llm_client = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Inicializa el cliente LLM."""
        if self.settings.LLM_PROVIDER == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(
                    api_key=self.settings.ANTHROPIC_API_KEY
                )
                logger.info("Anthropic LLM client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e}")
        elif self.settings.LLM_PROVIDER == "openai":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=self.settings.OPENAI_API_KEY
                )
                logger.info("OpenAI LLM client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
    
    def ingest_document(
        self, 
        content: str, 
        source_name: str,
        source_type: str = "primary",
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Ingesta un documento al sistema.
        
        Args:
            content: Contenido del documento
            source_name: Nombre de la fuente
            source_type: Tipo de fuente (primary, secondary, alternative, regulatory)
            metadata: Metadatos adicionales
        
        Returns:
            Número de chunks indexados
        """
        doc_metadata = {
            "source": source_name,
            "source_type": source_type,
            "ingested_at": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # Chunk del documento
        chunks = self.text_chunker.chunk_text(content, doc_metadata)
        
        # Indexar en vector store
        source_info = {"type": source_type, "name": source_name}
        count = self.vector_store.add_documents(chunks, source_info)
        
        logger.info(f"Ingested document '{source_name}' with {count} chunks")
        return count
    
    def retrieve(
        self, 
        query: str,
        top_k: int = None,
        source_types: List[str] = None
    ) -> RetrievalResult:
        """
        Recupera documentos relevantes para una consulta.
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de resultados a retornar
            source_types: Filtrar por tipos de fuente
        
        Returns:
            RetrievalResult con documentos y metadatos
        """
        import time
        start_time = time.time()
        
        filter_criteria = None
        if source_types:
            filter_criteria = {"source_type": {"$in": source_types}}
        
        documents, scores = self.vector_store.search(
            query=query,
            top_k=top_k or self.settings.RAG_TOP_K,
            filter_criteria=filter_criteria
        )
        
        # Filtrar por umbral de similitud
        filtered_docs = []
        filtered_scores = []
        sources = []
        
        for doc, score in zip(documents, scores):
            if score >= self.settings.RAG_SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                filtered_scores.append(score)
                
                sources.append(Source(
                    name=doc.metadata.get("source", "unknown"),
                    source_type=doc.metadata.get("source_type", "primary"),
                    confidence=score,
                    metadata=doc.metadata
                ))
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            documents=filtered_docs,
            sources=sources,
            query=query,
            relevance_scores=filtered_scores,
            retrieval_time_ms=retrieval_time,
            metadata={
                "total_candidates": len(documents),
                "filtered_count": len(filtered_docs),
                "threshold": self.settings.RAG_SIMILARITY_THRESHOLD
            }
        )
    
    def _build_context(self, documents: List[Document]) -> str:
        """Construye el contexto a partir de documentos recuperados."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            source_type = doc.metadata.get("source_type", "primary")
            
            context_parts.append(
                f"[Source {i}: {source} ({source_type})]\n{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _generate_with_llm(
        self, 
        query: str, 
        context: str,
        system_prompt: str = None
    ) -> Tuple[str, List[str]]:
        """
        Genera respuesta usando el LLM.
        
        Returns:
            Tuple de (respuesta, cadena de razonamiento)
        """
        default_system_prompt = """Eres un analista experto en inteligencia de mercado. 
Tu trabajo es proporcionar respuestas precisas, verificables y basadas en evidencia.

REGLAS CRÍTICAS:
1. SOLO usa información de las fuentes proporcionadas en el contexto
2. SIEMPRE cita las fuentes específicas que respaldan cada afirmación
3. Si la información no está en el contexto, indica claramente "No tengo información suficiente"
4. Proporciona números específicos, fechas y datos cuando estén disponibles
5. Indica el nivel de confianza de cada afirmación (alto/medio/bajo)
6. Nunca inventes datos o estadísticas

FORMATO DE RESPUESTA:
- Comienza con un resumen ejecutivo (2-3 oraciones)
- Proporciona el análisis detallado con citas
- Concluye con las limitaciones o datos faltantes"""
        
        system = system_prompt or default_system_prompt
        
        user_message = f"""CONTEXTO DE FUENTES VERIFICADAS:
{context}

---

CONSULTA DEL USUARIO:
{query}

---

Proporciona una respuesta completa basada ÚNICAMENTE en las fuentes anteriores.
Cita específicamente qué fuente respalda cada afirmación."""
        
        if not self.llm_client:
            # Respuesta de fallback si no hay LLM configurado
            return self._generate_fallback_response(query, context), []
        
        try:
            if self.settings.LLM_PROVIDER == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.settings.LLM_MODEL,
                    max_tokens=self.settings.LLM_MAX_TOKENS,
                    temperature=self.settings.LLM_TEMPERATURE,
                    system=system,
                    messages=[{"role": "user", "content": user_message}]
                )
                return response.content[0].text, []
            
            elif self.settings.LLM_PROVIDER == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.settings.LLM_TEMPERATURE,
                    max_tokens=self.settings.LLM_MAX_TOKENS
                )
                return response.choices[0].message.content, []
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._generate_fallback_response(query, context), []
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Genera respuesta de fallback cuando no hay LLM."""
        return f"""**Modo de demostración activo** (LLM no configurado)

Se encontraron {context.count('[Source')} fuentes relevantes para su consulta: "{query}"

Para obtener respuestas completas con análisis de IA:
1. Configure ANTHROPIC_API_KEY o OPENAI_API_KEY en el archivo .env
2. Reinicie el servidor

Contexto recuperado (primeros 500 caracteres):
{context[:500]}..."""
    
    def query(
        self, 
        query: str,
        include_verification: bool = True
    ) -> GenerationResult:
        """
        Ejecuta el pipeline RAG completo.
        
        Args:
            query: Pregunta del usuario
            include_verification: Si incluir verificación de respuesta
        
        Returns:
            GenerationResult con respuesta y metadatos
        """
        # 1. Recuperación
        retrieval = self.retrieve(query)
        
        if not retrieval.documents:
            return GenerationResult(
                answer="No se encontró información relevante en las fuentes disponibles para responder esta consulta.",
                sources=[],
                confidence_score=0.0,
                verification_status=Constants.VERIFICATION_STATUS["FAILED"],
                query=query,
                metadata={"reason": "no_relevant_documents"}
            )
        
        # 2. Construcción de contexto
        context = self._build_context(retrieval.documents)
        
        # 3. Generación
        answer, reasoning = self._generate_with_llm(query, context)
        
        # 4. Cálculo de confianza
        avg_relevance = sum(retrieval.relevance_scores) / len(retrieval.relevance_scores)
        source_diversity = len(set(s.source_type for s in retrieval.sources)) / 4  # Max 4 types
        confidence = (avg_relevance * 0.6 + source_diversity * 0.4)
        
        # 5. Verificación (si está habilitada)
        verification_status = Constants.VERIFICATION_STATUS["VERIFIED"]
        if include_verification:
            if len(retrieval.sources) < self.settings.VERIFICATION_MIN_SOURCES:
                verification_status = Constants.VERIFICATION_STATUS["PENDING"]
                confidence *= 0.8
            elif confidence < self.settings.VERIFICATION_CONFIDENCE_THRESHOLD:
                verification_status = Constants.VERIFICATION_STATUS["CROSS_CHECKING"]
        
        return GenerationResult(
            answer=answer,
            sources=retrieval.sources,
            confidence_score=round(confidence, 4),
            verification_status=verification_status,
            query=query,
            reasoning_chain=reasoning,
            metadata={
                "retrieval_time_ms": retrieval.retrieval_time_ms,
                "documents_used": len(retrieval.documents),
                "avg_relevance": round(avg_relevance, 4),
                "source_diversity": round(source_diversity, 4),
            }
        )


# Singleton instance
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    """Obtiene la instancia singleton del RAG Engine."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
