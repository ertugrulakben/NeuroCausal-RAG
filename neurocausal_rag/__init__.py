"""
NeuroCausal RAG - Neuro-Causal Graph-Based Intelligent Information Retrieval System

Façade Pattern: Single entry point for the entire system.

Yazar: Ertugrul Akben
E-posta: i@ertugrulakben.com
Website: https://ertugrulakben.com
Versiyon: 6.0.0
Lisans: MIT

v6.0 Features:
- Enterprise backends (Neo4j, Milvus)
- Funnel discovery (O(N²) → O(50))
- LangGraph agentic RAG
- Docker deployment
- Multi-hop retrieval with bridge document discovery
- Hybrid search optimization (6 modes)
- Query decomposition for complex questions
- Persistent memory system with RLHF
- Contradiction detection & temporal reasoning
- Entity linking with alias resolution

Kullanım:
    from neurocausal_rag import NeuroCausalRAG

    rag = NeuroCausalRAG()
    rag.add_document("doc1", "İklim değişikliği küresel bir sorundur.")
    rag.add_causal_link("doc1", "doc2", "causes")

    results = rag.search("iklim değişikliği etkileri")
    answer = rag.generate_answer("İklim değişikliğinin etkileri nelerdir?")

    # Agentic RAG (v5.0)
    from neurocausal_rag.agents import create_agent
    agent = create_agent(rag._retriever, rag._graph, rag._llm)
    result = agent.run("What causes global warming?")
"""

__version__ = "6.0.0"
__author__ = "Ertugrul Akben"
__email__ = "i@ertugrulakben.com"

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .config import (
    NeuroCausalConfig,
    get_config,
    set_config,
    load_config
)
from .interfaces import SearchResult, EvaluationResult, EntityResolution

logger = logging.getLogger(__name__)


class NeuroCausalRAG:
    """
    NeuroCausal RAG Façade

    Single entry point for all NeuroCausal RAG operations.
    Implements Façade Pattern for simplified API access.

    Attributes:
        config: Configuration object
        graph: Graph engine instance
        embedding: Embedding engine instance
        retriever: Retriever instance
        llm: LLM client instance (optional)
    """

    def __init__(
        self,
        config: Optional[NeuroCausalConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize NeuroCausal RAG system.

        Args:
            config: Pre-built configuration object
            config_path: Path to YAML config file
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = get_config()

        set_config(self.config)

        # Lazy initialization flags
        self._graph = None
        self._embedding = None
        self._retriever = None
        self._llm = None
        self._learning = None
        self._entity_linker = None
        self._initialized = False

        logger.info(f"NeuroCausal RAG v{__version__} initialized")

    def _ensure_initialized(self) -> None:
        """Lazy initialization of components"""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from .core.graph import GraphEngine
        from .embedding.text import TextEmbedding
        from .search.retriever import Retriever
        from .entity import EntityLinker

        self._embedding = TextEmbedding(self.config.embedding)
        self._graph = GraphEngine(self.config.graph)
        self._entity_linker = EntityLinker()
        self._retriever = Retriever(
            graph=self._graph,
            embedding=self._embedding,
            config=self.config.search,
            index_config=self.config.index,
            entity_linker=self._entity_linker
        )

        self._initialized = True
        logger.info("NeuroCausal RAG components initialized (with Entity Linking)")

    # =========================================================================
    # DOCUMENT MANAGEMENT
    # =========================================================================
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a document to the knowledge base.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata dictionary
        """
        self._ensure_initialized()
        embedding = self._embedding.get_text_embedding(content)
        self._graph.add_node(doc_id, content, embedding, metadata)
        self._retriever.rebuild_index()
        logger.debug(f"Added document: {doc_id}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add multiple documents at once.

        Args:
            documents: List of dicts with 'id', 'content', and optional 'metadata'

        Returns:
            Number of documents added
        """
        self._ensure_initialized()
        count = 0
        for doc in documents:
            embedding = self._embedding.get_text_embedding(doc['content'])
            self._graph.add_node(
                doc['id'],
                doc['content'],
                embedding,
                doc.get('metadata')
            )
            count += 1
        self._retriever.rebuild_index()
        logger.info(f"Added {count} documents")
        return count

    def add_causal_link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "causes",
        strength: float = 1.0
    ) -> None:
        """
        Add a causal relationship between documents.

        Args:
            source_id: Source document ID
            target_id: Target document ID
            relation_type: Type of relation (causes, requires, supports, etc.)
            strength: Relationship strength (0.0 to 1.0)
        """
        self._ensure_initialized()
        self._graph.add_edge(source_id, target_id, relation_type, strength)
        logger.debug(f"Added edge: {source_id} --{relation_type}--> {target_id}")

    # =========================================================================
    # SEARCH & RETRIEVAL
    # =========================================================================
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Search for relevant documents.

        Args:
            query: Search query text
            top_k: Number of results to return (default from config)

        Returns:
            List of SearchResult objects
        """
        self._ensure_initialized()
        k = top_k or self.config.search.top_k
        return self._retriever.search(query, k)

    def get_causal_chain(
        self,
        doc_id: str,
        max_depth: int = 3
    ) -> List[str]:
        """
        Get the causal chain starting from a document.

        Args:
            doc_id: Starting document ID
            max_depth: Maximum chain depth

        Returns:
            List of document IDs in causal chain
        """
        self._ensure_initialized()
        return self._graph.get_causal_chain(doc_id, max_depth)

    # =========================================================================
    # LLM INTEGRATION
    # =========================================================================
    def _ensure_llm(self) -> None:
        """Initialize LLM client if not already done"""
        if self._llm is None:
            from .llm.client import LLMClient
            self._llm = LLMClient(self.config.llm)

    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        Generate an answer using RAG pipeline.

        Args:
            query: User question
            top_k: Number of context documents

        Returns:
            Generated answer string
        """
        self._ensure_initialized()
        self._ensure_llm()

        # Retrieve relevant documents
        results = self.search(query, top_k)

        # Build context
        context_parts = []
        for i, r in enumerate(results, 1):
            ctx = f"[{i}] {r.content}"
            if r.causal_chain and len(r.causal_chain) > 1:
                ctx += f"\n  Causal chain: {' -> '.join(r.causal_chain)}"
            context_parts.append(ctx)

        context = "\n\n".join(context_parts)

        # Generate answer
        return self._llm.generate(query, context)

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        context: str
    ) -> EvaluationResult:
        """
        Evaluate answer quality using LLM.

        Args:
            query: Original query
            answer: Generated answer
            context: Context used for generation

        Returns:
            EvaluationResult with score and reasoning
        """
        self._ensure_llm()
        return self._llm.evaluate(query, answer, context)

    # =========================================================================
    # LEARNING & FEEDBACK
    # =========================================================================
    def _ensure_learning(self) -> None:
        """Initialize learning engine if not already done"""
        if self._learning is None and self.config.learning.enabled:
            from .learning.learner import LearningEngine
            self._learning = LearningEngine(
                graph=self._graph,
                config=self.config.learning
            )

    def record_feedback(
        self,
        query: str,
        result_ids: List[str],
        rating: float,
        comment: Optional[str] = None
    ) -> None:
        """
        Record user feedback for learning.

        Args:
            query: Original query
            result_ids: List of result document IDs
            rating: Rating (0.0 to 1.0)
            comment: Optional user comment
        """
        self._ensure_learning()
        if self._learning:
            self._learning.record_feedback(query, result_ids, rating, comment)

    def discover_links(self, min_confidence: float = 0.7) -> List[Dict]:
        """
        Discover potential new causal links.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of potential link dictionaries
        """
        self._ensure_learning()
        if self._learning:
            return self._learning.discover_links(min_confidence)
        return []

    # =========================================================================
    # ENTITY LINKING (v5.1)
    # =========================================================================
    def learn_aliases(self, documents: Optional[List[Dict]] = None) -> int:
        """
        Dokümanlardaki alias/kod adı ilişkilerini öğren.

        Args:
            documents: Optional document list. If None, uses graph documents.

        Returns:
            Number of aliases learned
        """
        self._ensure_initialized()

        if documents is None:
            # Graf'taki tüm dökümanları kullan
            documents = []
            for node_id in self._graph.graph.nodes():
                node = self._graph.get_node(node_id)
                if node:
                    documents.append({
                        'id': node_id,
                        'content': node.get('content', '')
                    })

        return self._entity_linker.learn_aliases_from_documents(documents)

    def add_alias(self, alias: str, canonical: str, confidence: float = 0.9) -> None:
        """
        Manuel alias tanımla.

        Args:
            alias: Alias (örn: "Mavi Ufuk")
            canonical: Gerçek isim (örn: "Güneş Enerjisi A.Ş.")
            confidence: Güven skoru (0-1)
        """
        self._ensure_initialized()
        self._entity_linker.add_alias(alias, canonical, confidence)
        logger.info(f"Alias added: '{alias}' -> '{canonical}'")

    def resolve_entity(self, text: str) -> str:
        """
        Metindeki alias'ları canonical isimlere çevir.

        Args:
            text: İşlenecek metin

        Returns:
            Alias'ları çözümlenmiş metin
        """
        self._ensure_initialized()
        resolved = self._entity_linker.resolve_text(text)
        # Return the resolved version of the text
        result = text
        for alias, canonical in resolved.items():
            result = result.replace(alias, canonical)
        return result

    def get_aliases(self) -> Dict[str, str]:
        """
        Tüm alias eşleştirmelerini döndür.

        Returns:
            {alias: canonical} dictionary
        """
        self._ensure_initialized()
        return dict(self._entity_linker.alias_store.aliases)

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Metinden entity'leri çıkar (NER).

        Args:
            text: Analiz edilecek metin

        Returns:
            List of entity dictionaries
        """
        self._ensure_initialized()
        from .entity import EntityExtractor
        extractor = EntityExtractor()
        entities = extractor.extract_entities(text)
        return [
            {
                'text': e.text,
                'type': e.entity_type,
                'start': e.start,
                'end': e.end,
                'confidence': e.confidence
            }
            for e in entities
        ]

    # =========================================================================
    # REASONING (v5.1 - FAZ 1.2/1.3)
    # =========================================================================
    def detect_contradictions(
        self,
        documents: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Belgeler arasindaki celiskileri tespit et.

        Args:
            documents: Belge listesi. None ise arama sonuclarini kullanir.

        Returns:
            Celiski listesi
        """
        self._ensure_initialized()
        from .reasoning import ContradictionDetector

        detector = ContradictionDetector()

        if documents is None:
            # Graf'taki tum belgeleri kullan
            documents = []
            for node_id in self._graph.graph.nodes():
                node = self._graph.get_node(node_id)
                if node:
                    documents.append({
                        'id': node_id,
                        'content': node.get('content', '')
                    })

        contradictions = detector.detect_all(documents)

        return [
            {
                'doc1_id': c.doc1_id,
                'doc2_id': c.doc2_id,
                'type': c.contradiction_type.value,
                'confidence': c.confidence,
                'details': c.details,
                'values': c.conflicting_values,
                'suggested_resolution': c.suggested_resolution.value
            }
            for c in contradictions
        ]

    def validate_causal_order(
        self,
        cause_doc_id: str,
        effect_doc_id: str
    ) -> Dict:
        """
        Nedensel iliskinin zamansal gecerliligini kontrol et.

        Args:
            cause_doc_id: Neden belgesi ID
            effect_doc_id: Sonuc belgesi ID

        Returns:
            Dogrulama sonucu
        """
        self._ensure_initialized()
        from .reasoning import TemporalReasoner

        cause_node = self._graph.get_node(cause_doc_id)
        effect_node = self._graph.get_node(effect_doc_id)

        if not cause_node or not effect_node:
            return {'valid': False, 'error': 'Belge bulunamadi'}

        reasoner = TemporalReasoner()
        validation = reasoner.validate_causal_order(
            {'id': cause_doc_id, 'content': cause_node.get('content', '')},
            {'id': effect_doc_id, 'content': effect_node.get('content', '')}
        )

        return {
            'valid': validation.is_valid(),
            'order': validation.order.value,
            'cause_date': validation.cause_date,
            'effect_date': validation.effect_date,
            'confidence': validation.confidence,
            'explanation': validation.explanation
        }

    def check_contradictions_in_results(
        self,
        results: List[Any]
    ) -> List[Dict]:
        """
        Arama sonuclari arasindaki celiskileri kontrol et.

        Args:
            results: SearchResult listesi

        Returns:
            Celiski listesi
        """
        from .reasoning import detect_contradictions_in_results

        # SearchResult veya dict olabilir
        formatted_results = []
        for r in results:
            if hasattr(r, 'node_id'):
                formatted_results.append({'id': r.node_id, 'content': r.content})
            elif isinstance(r, dict):
                formatted_results.append({
                    'id': r.get('id', r.get('node_id', 'unknown')),
                    'content': r.get('content', '')
                })

        contradictions = detect_contradictions_in_results(formatted_results)

        return [
            {
                'doc1_id': c.doc1_id,
                'doc2_id': c.doc2_id,
                'type': c.contradiction_type.value,
                'confidence': c.confidence,
                'details': c.details
            }
            for c in contradictions
        ]

    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    def save(self, path: str) -> None:
        """
        Save the entire system state.

        Args:
            path: Directory path to save to
        """
        self._ensure_initialized()
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self._graph.export(str(save_dir / "graph.json"))
        self._retriever.save_index(str(save_dir / "index"))
        self.config.to_yaml(save_dir / "config.yaml")

        # Save entity aliases (v5.1)
        if self._entity_linker:
            self._entity_linker.alias_store.save(str(save_dir / "aliases.json"))

        logger.info(f"System saved to {path}")

    def load(self, path: str) -> None:
        """
        Load system state from disk.

        Args:
            path: Directory path to load from
        """
        load_dir = Path(path)

        if (load_dir / "config.yaml").exists():
            self.config = NeuroCausalConfig.from_yaml(load_dir / "config.yaml")
            set_config(self.config)

        self._ensure_initialized()

        if (load_dir / "graph.json").exists():
            self._graph.load(str(load_dir / "graph.json"))

        if (load_dir / "index").exists():
            self._retriever.load_index(str(load_dir / "index"))

        # Load entity aliases (v5.1)
        if (load_dir / "aliases.json").exists() and self._entity_linker:
            self._entity_linker.alias_store.load(str(load_dir / "aliases.json"))

        logger.info(f"System loaded from {path}")

    # =========================================================================
    # STATISTICS & INFO
    # =========================================================================
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        self._ensure_initialized()
        stats = {
            "version": __version__,
            "config_version": self.config.version,
            "node_count": self._graph.node_count,
            "edge_count": self._graph.edge_count,
            "index_backend": self.config.index.backend,
            "search_weights": {
                "alpha": self.config.search.alpha,
                "beta": self.config.search.beta,
                "gamma": self.config.search.gamma
            }
        }

        # Entity linking stats (v5.1)
        if self._entity_linker:
            stats["entity_linking"] = {
                "alias_count": len(self._entity_linker.alias_store.aliases),
                "enabled": True
            }

        return stats

    def __repr__(self) -> str:
        return f"NeuroCausalRAG(version={__version__}, config={self.config.version})"


# Convenience exports
__all__ = [
    "NeuroCausalRAG",
    "NeuroCausalConfig",
    "SearchResult",
    "EvaluationResult",
    "EntityResolution",
    "get_config",
    "set_config",
    "load_config",
    "__version__"
]
