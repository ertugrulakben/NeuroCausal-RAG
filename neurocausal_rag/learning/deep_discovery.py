"""
NeuroCausal RAG - Deep Causal Discovery
NLI (Natural Language Inference) tabanlı derin nedensellik keşfi

Bu modül, basit regex kalıpları yerine transformer modelleri kullanarak
anlamsal nedensellik ilişkilerini keşfeder.

Temel fikir:
- "Sera gazları atmosferde birikir" → "Dünya ısınır"
- İki cümle arasında ENTAILMENT (çıkarım) varsa, nedensellik olabilir

Yazar: Ertuğrul Akben
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalPair:
    """Nedensel çift"""
    source_id: str
    target_id: str
    source_text: str
    target_text: str
    entailment_score: float
    causal_score: float
    evidence: str


class DeepCausalDiscovery:
    """
    NLI (Natural Language Inference) tabanlı derin nedensellik keşfi.

    Çalışma prensibi:
    1. Her doküman çifti için "X neden olur Y" hipotezini test et
    2. NLI modeli ile entailment skoru hesapla
    3. Yüksek entailment = potansiyel nedensellik

    NOT: Bu yöntem, regex kalıplarına göre çok daha güçlüdür çünkü:
    - Anlamsal benzerliği değil, mantıksal çıkarımı kullanır
    - "Sera gazı" ve "ısınma" arasında kelime eşleşmesi olmasa bile
      mantıksal bağlantıyı yakalayabilir
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        threshold: float = 0.6
    ):
        """
        Args:
            model_name: NLI model ismi (HuggingFace)
            device: "cpu" veya "cuda"
            threshold: Entailment eşiği
        """
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._model = None
        self._use_simple = False

    def _load_model(self):
        """Model'i lazy load et"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"DeepCausalDiscovery: {self.model_name} yüklendi")
        except ImportError:
            logger.warning("sentence-transformers yüklü değil, basit mod kullanılacak")
            self._use_simple = True
        except Exception as e:
            logger.warning(f"Model yüklenemedi: {e}, basit mod kullanılacak")
            self._use_simple = True

    def discover(
        self,
        documents: List[Dict],
        max_pairs: int = 500
    ) -> List[Dict]:
        """
        Dokümanlar arasında nedensel ilişkileri keşfet.

        Args:
            documents: [{'id': str, 'content': str}, ...]
            max_pairs: Maksimum çift sayısı (performans için)

        Returns:
            Keşfedilen nedensel ilişkiler
        """
        self._load_model()

        if self._use_simple:
            return self._simple_discovery(documents)

        n = len(documents)
        pairs = []

        # Tüm çiftleri oluştur (sınırlı)
        all_pairs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    all_pairs.append((i, j))

        # Rastgele örnekle
        if len(all_pairs) > max_pairs:
            import random
            random.shuffle(all_pairs)
            all_pairs = all_pairs[:max_pairs]

        logger.info(f"DeepCausalDiscovery: {len(all_pairs)} çift analiz ediliyor...")

        # Batch işleme için hazırla
        batch_size = 32
        results = []

        for batch_start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[batch_start:batch_start + batch_size]

            # NLI formatında çiftler
            nli_pairs = []
            for i, j in batch:
                # Premise: source content (özet)
                premise = documents[i]['content'][:200]
                # Hypothesis: "Bu nedenle {target} oluşur"
                target_text = documents[j]['content'][:100]
                hypothesis = f"Bu nedenle {target_text}"
                nli_pairs.append((premise, hypothesis))

            try:
                # NLI skorları (entailment, neutral, contradiction)
                scores = self._model.predict(nli_pairs)

                for idx, (i, j) in enumerate(batch):
                    # CrossEncoder returns logits [contradiction, entailment, neutral]
                    if isinstance(scores[idx], (list, np.ndarray)):
                        entailment = float(scores[idx][1])  # entailment score
                    else:
                        entailment = float(scores[idx])

                    if entailment > self.threshold:
                        results.append({
                            'source': documents[i]['id'],
                            'target': documents[j]['id'],
                            'relation_type': 'causes',
                            'confidence': entailment,
                            'method': 'nli_deep',
                            'evidence': f"NLI Entailment: {entailment:.3f}"
                        })
            except Exception as e:
                logger.error(f"NLI batch hata: {e}")
                continue

        logger.info(f"DeepCausalDiscovery: {len(results)} ilişki bulundu")
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _simple_discovery(self, documents: List[Dict]) -> List[Dict]:
        """
        Basit keşif modu (model yoksa).
        Embedding similarity + keyword matching kullanır.
        """
        results = []
        n = len(documents)

        # Nedensellik anahtar kelimeleri
        cause_words = {'neden', 'sebep', 'kaynak', 'etken', 'tetikler',
                       'cause', 'source', 'trigger', 'leads'}
        effect_words = {'sonuç', 'etki', 'netice', 'oluşur', 'meydana',
                        'effect', 'result', 'outcome', 'leads to'}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                text_i = documents[i]['content'].lower()
                text_j = documents[j]['content'].lower()

                # i neden içeriyorsa ve j sonuç içeriyorsa
                has_cause = any(w in text_i for w in cause_words)
                has_effect = any(w in text_j for w in effect_words)

                if has_cause and has_effect:
                    # Basit skor
                    score = 0.6 + 0.1 * (sum(1 for w in cause_words if w in text_i) +
                                         sum(1 for w in effect_words if w in text_j))
                    score = min(0.9, score)

                    results.append({
                        'source': documents[i]['id'],
                        'target': documents[j]['id'],
                        'relation_type': 'causes',
                        'confidence': score,
                        'method': 'keyword_simple',
                        'evidence': f"Cause keywords in source, effect keywords in target"
                    })

        return sorted(results, key=lambda x: x['confidence'], reverse=True)[:100]


class CausalStrengthEstimator:
    """
    Embedding tabanlı nedensel güç tahmini.

    İki doküman arasındaki nedensel ilişki gücünü tahmin eder.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._projection = None

    def estimate_strength(
        self,
        source_emb: np.ndarray,
        target_emb: np.ndarray
    ) -> float:
        """
        İki embedding arasındaki nedensel güç tahmini.

        Yöntem:
        1. Cosine similarity (temel benzerlik)
        2. Asymmetry score (yön bilgisi)
        3. Projection score (nedensel uzay projeksiyonu)
        """
        # 1. Cosine similarity
        cos_sim = np.dot(source_emb, target_emb) / (
            np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
        )

        # 2. Asymmetry: source → target güçlüyse, nedensellik olabilir
        # Vektör farkının yönü
        diff = target_emb - source_emb
        diff_norm = np.linalg.norm(diff)

        # Source'un diff yönündeki projeksiyonu
        if diff_norm > 0:
            projection = np.dot(source_emb, diff) / diff_norm
            asymmetry = max(0, projection / (np.linalg.norm(source_emb) + 1e-8))
        else:
            asymmetry = 0

        # 3. Final score
        strength = 0.5 * cos_sim + 0.5 * asymmetry
        return float(np.clip(strength, 0, 1))


def deep_causal_discovery(
    documents: List[Dict],
    embeddings: Optional[np.ndarray] = None,
    use_nli: bool = True,
    max_pairs: int = 500
) -> List[Dict]:
    """
    Derin nedensellik keşfi - tek fonksiyon API.

    Args:
        documents: [{'id': str, 'content': str}, ...]
        embeddings: Opsiyonel embedding matrisi
        use_nli: NLI modeli kullanılsın mı?
        max_pairs: Maksimum çift sayısı

    Returns:
        Keşfedilen nedensel ilişkiler
    """
    all_relations = []

    # 1. NLI-based discovery (varsa)
    if use_nli:
        try:
            nli_discovery = DeepCausalDiscovery()
            nli_relations = nli_discovery.discover(documents, max_pairs)
            all_relations.extend(nli_relations)
        except Exception as e:
            logger.warning(f"NLI discovery atlandı: {e}")

    # 2. Embedding-based strength estimation (varsa)
    if embeddings is not None:
        estimator = CausalStrengthEstimator()
        n = len(documents)

        for i in range(min(n, 50)):  # İlk 50 doküman için
            for j in range(min(n, 50)):
                if i == j:
                    continue

                strength = estimator.estimate_strength(embeddings[i], embeddings[j])
                if strength > 0.6:
                    all_relations.append({
                        'source': documents[i]['id'],
                        'target': documents[j]['id'],
                        'relation_type': 'supports',
                        'confidence': strength,
                        'method': 'embedding_strength',
                        'evidence': f"Embedding strength: {strength:.3f}"
                    })

    # Deduplicate
    seen = set()
    unique = []
    for rel in sorted(all_relations, key=lambda x: x['confidence'], reverse=True):
        key = (rel['source'], rel['target'])
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    return unique
