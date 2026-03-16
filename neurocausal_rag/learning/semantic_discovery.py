"""
NeuroCausal RAG - Semantic Causal Discovery Engine
Regex limitasyonunu aşan gelişmiş nedensellik keşif sistemi

Bu modül, basit regex kalıpları yerine:
1. Asymmetric Embedding Analysis - A→B vs B→A asimetrisi
2. Cluster-based Discovery - Semantik kümelerdeki yapı
3. Graph Propagation - Transitive ilişki çıkarımı
4. Multi-Signal Fusion - Çoklu sinyal birleştirme

Yazar: Ertuğrul Akben
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalSignal:
    """Tek bir nedensellik sinyali"""
    source_id: str
    target_id: str
    signal_type: str  # 'asymmetric', 'cluster', 'lexical', 'structural'
    strength: float  # 0-1
    evidence: str  # Neden bu sinyal algılandı


class SemanticCausalDiscovery:
    """
    Regex'i aşan semantik nedensellik keşfi.

    Temel prensip: Nedensellik, tek bir sinyal değil,
    birden fazla zayıf sinyalin birleşimidir.

    Sinyaller:
    1. Asymmetric Similarity: A→B benzerliği B→A'dan farklıysa
    2. Lexical Causality: Metin içinde nedensel kelimeler
    3. Structural Position: Graf yapısındaki konum
    4. Category Coherence: Aynı kategorideki dokümanlar
    5. Temporal Markers: Zaman belirteçleri
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        asymmetry_threshold: float = 0.1,
        min_confidence: float = 0.6
    ):
        self.similarity_threshold = similarity_threshold
        self.asymmetry_threshold = asymmetry_threshold
        self.min_confidence = min_confidence

        # Nedensellik anahtar kelimeleri (dil bağımsız skorlama)
        self.cause_markers = {
            # Türkçe
            'neden': 0.9, 'sebep': 0.9, 'kaynak': 0.7, 'etken': 0.8,
            'faktör': 0.8, 'tetikler': 0.9, 'başlatır': 0.8, 'oluşturur': 0.7,
            # İngilizce
            'cause': 0.9, 'source': 0.7, 'factor': 0.8, 'trigger': 0.9,
            'origin': 0.7, 'driver': 0.8, 'leads': 0.8, 'produces': 0.7,
        }

        self.effect_markers = {
            # Türkçe
            'sonuç': 0.9, 'etki': 0.9, 'netice': 0.8, 'çıktı': 0.7,
            'oluşur': 0.7, 'meydana': 0.8, 'gerçekleşir': 0.7,
            # İngilizce
            'effect': 0.9, 'result': 0.9, 'outcome': 0.8, 'impact': 0.9,
            'consequence': 0.9, 'occurs': 0.7, 'happens': 0.6,
        }

        self.temporal_cause_markers = {
            'önce': 0.7, 'başlangıç': 0.8, 'ilk': 0.6,
            'before': 0.7, 'initial': 0.8, 'first': 0.6, 'primary': 0.7,
        }

        self.temporal_effect_markers = {
            'sonra': 0.7, 'ardından': 0.8, 'sonuç': 0.7,
            'after': 0.7, 'then': 0.6, 'subsequently': 0.8, 'resulting': 0.8,
        }

    def discover(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Ana keşif fonksiyonu.

        Args:
            documents: [{'id': str, 'content': str, 'category': str}, ...]
            embeddings: (n_docs, dim) embedding matrisi

        Returns:
            Keşfedilen nedensel ilişkiler
        """
        n = len(documents)
        if n < 2:
            return []

        logger.info(f"SemanticCausalDiscovery: {n} doküman analiz ediliyor...")

        # Tüm sinyalleri topla
        all_signals: List[CausalSignal] = []

        # 1. Asymmetric Similarity Analysis
        logger.info("  1. Asimetrik benzerlik analizi...")
        asymmetric_signals = self._analyze_asymmetric_similarity(documents, embeddings)
        all_signals.extend(asymmetric_signals)

        # 2. Lexical Causality Signals
        logger.info("  2. Leksikal nedensellik analizi...")
        lexical_signals = self._analyze_lexical_causality(documents)
        all_signals.extend(lexical_signals)

        # 3. Category-based Signals
        logger.info("  3. Kategori bazlı analiz...")
        category_signals = self._analyze_category_structure(documents, embeddings)
        all_signals.extend(category_signals)

        # 4. Cluster-based Discovery
        logger.info("  4. Küme bazlı analiz...")
        cluster_signals = self._analyze_clusters(documents, embeddings)
        all_signals.extend(cluster_signals)

        # Multi-Signal Fusion
        logger.info("  5. Sinyal birleştirme...")
        fused_relations = self._fuse_signals(all_signals)

        # Filter by confidence
        confident_relations = [
            r for r in fused_relations
            if r['confidence'] >= self.min_confidence
        ]

        logger.info(f"  Toplam {len(confident_relations)} ilişki keşfedildi")
        return confident_relations

    def _analyze_asymmetric_similarity(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[CausalSignal]:
        """
        Asimetrik benzerlik analizi.

        Fikir: A'nın B'ye benzerliği, B'nin A'ya benzerliğinden farklıysa,
        bir yönlü ilişki olabilir (nedensellik genellikle yönlüdür).

        Örnek: "Sera gazları" → "Küresel ısınma" ilişkisinde,
        sera gazları küresel ısınmayı anlatır ama tersi değil.
        """
        signals = []
        n = len(documents)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms

        # Compute similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                sim_ij = sim_matrix[i, j]
                sim_ji = sim_matrix[j, i]  # Aslında aynı olmalı ama...

                # Tek yönlü "bağlam" analizi
                # A'nın B'yi kapsama oranı vs B'nin A'yı kapsama oranı
                content_i = documents[i]['content'].lower()
                content_j = documents[j]['content'].lower()

                # Kelime kesişimi analizi (asimetri tespiti için)
                words_i = set(content_i.split())
                words_j = set(content_j.split())

                # i'nin j'yi kapsama oranı
                if len(words_j) > 0:
                    coverage_i_to_j = len(words_i & words_j) / len(words_j)
                else:
                    coverage_i_to_j = 0

                # j'nin i'yi kapsama oranı
                if len(words_i) > 0:
                    coverage_j_to_i = len(words_i & words_j) / len(words_i)
                else:
                    coverage_j_to_i = 0

                # Asimetri: i, j'yi daha çok kapsıyorsa, i→j ilişkisi olabilir
                asymmetry = coverage_i_to_j - coverage_j_to_i

                if sim_ij > self.similarity_threshold and asymmetry > self.asymmetry_threshold:
                    signals.append(CausalSignal(
                        source_id=documents[i]['id'],
                        target_id=documents[j]['id'],
                        signal_type='asymmetric',
                        strength=min(1.0, (sim_ij + asymmetry) / 2),
                        evidence=f"Sim={sim_ij:.2f}, Asimetri={asymmetry:.2f}"
                    ))

        return signals

    def _analyze_lexical_causality(
        self,
        documents: List[Dict]
    ) -> List[CausalSignal]:
        """
        Leksikal nedensellik analizi.

        Metin içindeki neden/sonuç kelimelerinin varlığına göre
        dokümanları "neden" veya "sonuç" olarak sınıflandır.
        """
        signals = []

        # Her doküman için neden/sonuç skoru hesapla
        cause_scores = {}
        effect_scores = {}

        for doc in documents:
            content = doc['content'].lower()

            cause_score = 0.0
            for marker, weight in self.cause_markers.items():
                if marker in content:
                    cause_score += weight
            cause_score += sum(
                w for m, w in self.temporal_cause_markers.items() if m in content
            )

            effect_score = 0.0
            for marker, weight in self.effect_markers.items():
                if marker in content:
                    effect_score += weight
            effect_score += sum(
                w for m, w in self.temporal_effect_markers.items() if m in content
            )

            cause_scores[doc['id']] = cause_score
            effect_scores[doc['id']] = effect_score

        # Neden skorlu dokümanları sonuç skorlu dokümanlarla eşleştir
        for doc_i in documents:
            for doc_j in documents:
                if doc_i['id'] == doc_j['id']:
                    continue

                cause_i = cause_scores[doc_i['id']]
                effect_j = effect_scores[doc_j['id']]

                # i neden, j sonuç ise
                if cause_i > 0.5 and effect_j > 0.5:
                    strength = min(1.0, (cause_i + effect_j) / 4)  # Normalize
                    signals.append(CausalSignal(
                        source_id=doc_i['id'],
                        target_id=doc_j['id'],
                        signal_type='lexical',
                        strength=strength,
                        evidence=f"Cause={cause_i:.1f}, Effect={effect_j:.1f}"
                    ))

        return signals

    def _analyze_category_structure(
        self,
        documents: List[Dict],
        embeddings: np.ndarray
    ) -> List[CausalSignal]:
        """
        Kategori bazlı yapısal analiz.

        Aynı kategorideki dokümanlar arasında yapısal nedensellik olabilir.
        Örnek: "iklim" kategorisinde sera→ısınma→buzul→deniz zinciri.
        """
        signals = []

        # Kategorilere göre grupla
        categories = defaultdict(list)
        for i, doc in enumerate(documents):
            cat = doc.get('category', 'unknown')
            categories[cat].append((i, doc))

        # Her kategori içinde analiz
        for cat, cat_docs in categories.items():
            if len(cat_docs) < 2:
                continue

            # Kategori içi benzerlik matrisi
            indices = [i for i, _ in cat_docs]
            cat_embeddings = embeddings[indices]

            norms = np.linalg.norm(cat_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = cat_embeddings / norms
            sim_matrix = np.dot(normalized, normalized.T)

            # En yüksek benzerlik çiftlerini bul
            for i in range(len(cat_docs)):
                for j in range(len(cat_docs)):
                    if i >= j:
                        continue

                    sim = sim_matrix[i, j]
                    if sim > self.similarity_threshold:
                        doc_i = cat_docs[i][1]
                        doc_j = cat_docs[j][1]

                        # Hangisi önce? (alfabetik sıra heuristiği - geliştirilebilir)
                        if doc_i['id'] < doc_j['id']:
                            source, target = doc_i, doc_j
                        else:
                            source, target = doc_j, doc_i

                        signals.append(CausalSignal(
                            source_id=source['id'],
                            target_id=target['id'],
                            signal_type='category',
                            strength=sim * 0.7,  # Kategori sinyali daha zayıf
                            evidence=f"Kategori={cat}, Sim={sim:.2f}"
                        ))

        return signals

    def _analyze_clusters(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> List[CausalSignal]:
        """
        Küme bazlı nedensellik analizi.

        Dokümanları embedding uzayında kümele.
        Küme merkezlerine yakınlık → önem
        Kümeler arası geçiş → nedensellik zinciri
        """
        signals = []
        n = len(documents)

        if n < n_clusters:
            return signals

        # Simple k-means clustering
        # (Gerçek uygulamada sklearn kullanılabilir)
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=min(n_clusters, n//2), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Küme merkezleri
            centers = kmeans.cluster_centers_

            # Her dokümanın merkeze uzaklığı (önem skoru)
            importance = []
            for i, emb in enumerate(embeddings):
                center = centers[cluster_labels[i]]
                dist = np.linalg.norm(emb - center)
                importance.append(1.0 / (1.0 + dist))

            # Önemli dokümanlardan daha az önemli olanlara sinyal
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue

                    # Aynı kümedeyse ve i daha önemliyse
                    if cluster_labels[i] == cluster_labels[j]:
                        if importance[i] > importance[j] + 0.1:
                            signals.append(CausalSignal(
                                source_id=documents[i]['id'],
                                target_id=documents[j]['id'],
                                signal_type='cluster',
                                strength=(importance[i] - importance[j]) * 0.8,
                                evidence=f"Küme={cluster_labels[i]}, Imp_diff={importance[i]-importance[j]:.2f}"
                            ))
        except ImportError:
            # sklearn yoksa basit kümeleme
            pass

        return signals

    def _fuse_signals(
        self,
        signals: List[CausalSignal]
    ) -> List[Dict]:
        """
        Multi-Signal Fusion.

        Birden fazla zayıf sinyal → güçlü ilişki

        Fusion stratejisi:
        - Aynı (source, target) için birden fazla sinyal varsa, güçlendir
        - Farklı tipte sinyaller daha değerli
        - Maximum değil, weighted sum kullan
        """
        # Sinyalleri (source, target) çiftlerine göre grupla
        pair_signals: Dict[Tuple[str, str], List[CausalSignal]] = defaultdict(list)

        for signal in signals:
            key = (signal.source_id, signal.target_id)
            pair_signals[key].append(signal)

        # Her çift için fusion
        relations = []
        for (source, target), signals_list in pair_signals.items():
            if not signals_list:
                continue

            # Sinyal tiplerini say
            signal_types = set(s.signal_type for s in signals_list)
            type_diversity = len(signal_types) / 4  # 4 farklı tip var

            # Weighted sum
            total_strength = sum(s.strength for s in signals_list)
            avg_strength = total_strength / len(signals_list)

            # Diversity bonus
            confidence = min(1.0, avg_strength * (1 + type_diversity))

            # İlişki tipini belirle
            relation_type = self._infer_relation_type(signals_list)

            relations.append({
                'source': source,
                'target': target,
                'relation_type': relation_type,
                'confidence': confidence,
                'signal_count': len(signals_list),
                'signal_types': list(signal_types),
                'evidence': [s.evidence for s in signals_list[:3]],  # İlk 3 kanıt
                'method': 'semantic_fusion'
            })

        return sorted(relations, key=lambda x: x['confidence'], reverse=True)

    def _infer_relation_type(self, signals: List[CausalSignal]) -> str:
        """Sinyal tiplerinden ilişki tipini çıkar"""
        # Lexical sinyal varsa ve güçlüyse → causes
        lexical_signals = [s for s in signals if s.signal_type == 'lexical']
        if lexical_signals and max(s.strength for s in lexical_signals) > 0.6:
            return 'causes'

        # Asymmetric sinyal varsa → supports
        asymmetric_signals = [s for s in signals if s.signal_type == 'asymmetric']
        if asymmetric_signals:
            return 'supports'

        # Kategori veya küme sinyali → related
        return 'related'


class GraphPropagation:
    """
    Graf üzerinde transitive ilişki çıkarımı.

    A→B ve B→C varsa, A→C de çıkarılabilir (decay ile).
    """

    def __init__(self, decay_factor: float = 0.7):
        self.decay_factor = decay_factor

    def propagate(
        self,
        relations: List[Dict],
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Mevcut ilişkilerden transitive ilişkiler çıkar.
        """
        # Adjacency list oluştur
        adj = defaultdict(list)
        edge_weights = {}

        for rel in relations:
            src, tgt = rel['source'], rel['target']
            conf = rel['confidence']
            adj[src].append(tgt)
            edge_weights[(src, tgt)] = conf

        # Yeni ilişkiler
        new_relations = []
        existing_pairs = set((r['source'], r['target']) for r in relations)

        # Her düğümden BFS
        all_nodes = list(adj.keys())  # Copy to avoid mutation issues
        for start_node in all_nodes:
            visited = {start_node: 1.0}  # node -> strength
            queue = [(start_node, 1.0, 0)]  # (node, strength, depth)

            while queue:
                current, strength, depth = queue.pop(0)

                if depth >= max_depth:
                    continue

                for neighbor in adj[current]:
                    edge_strength = edge_weights.get((current, neighbor), 0.5)
                    new_strength = strength * edge_strength * self.decay_factor

                    if neighbor not in visited or visited[neighbor] < new_strength:
                        visited[neighbor] = new_strength
                        queue.append((neighbor, new_strength, depth + 1))

                        # Yeni transitive ilişki
                        if (start_node, neighbor) not in existing_pairs:
                            existing_pairs.add((start_node, neighbor))
                            new_relations.append({
                                'source': start_node,
                                'target': neighbor,
                                'relation_type': 'causes',  # Transitive → causes
                                'confidence': new_strength,
                                'depth': depth + 1,
                                'method': 'propagation'
                            })

        return new_relations


def enhanced_causal_discovery(
    documents: List[Dict],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.5,
    min_confidence: float = 0.55
) -> List[Dict]:
    """
    Gelişmiş nedensellik keşfi - tek fonksiyon API.

    Args:
        documents: [{'id': str, 'content': str, 'category': str}, ...]
        embeddings: (n_docs, dim) embedding matrisi
        similarity_threshold: Benzerlik eşiği
        min_confidence: Minimum güven skoru

    Returns:
        Keşfedilen nedensel ilişkiler listesi
    """
    # 1. Semantic Discovery
    semantic = SemanticCausalDiscovery(
        similarity_threshold=similarity_threshold,
        min_confidence=min_confidence
    )
    initial_relations = semantic.discover(documents, embeddings)

    # 2. Graph Propagation
    propagator = GraphPropagation(decay_factor=0.7)
    transitive_relations = propagator.propagate(initial_relations, max_depth=2)

    # 3. Birleştir ve sırala
    all_relations = initial_relations + transitive_relations

    # Duplicate'leri kaldır
    seen = set()
    unique_relations = []
    for rel in sorted(all_relations, key=lambda x: x['confidence'], reverse=True):
        key = (rel['source'], rel['target'])
        if key not in seen:
            seen.add(key)
            unique_relations.append(rel)

    return unique_relations
