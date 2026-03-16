"""
NeuroCausal RAG - Automatic Causal Discovery
LLM-based and embedding-based causal relationship extraction

Bu modül, manuel tanımlama yerine otomatik nedensellik keşfi yapar.
İki ana yaklaşım:
1. LLM-based: GPT-4o-mini ile metinlerden nedensel ilişkileri çıkarma
2. Embedding-based: Vektör uzayında nedensel yapı analizi
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re

from ..config import LearningConfig
from ..embedding.text import cosine_similarity

logger = logging.getLogger(__name__)


# Türkçe nedensellik kalıpları
CAUSAL_PATTERNS_TR = [
    (r'(.+?)\s+nedeniyle\s+(.+)', 'causes'),
    (r'(.+?)\s+yüzünden\s+(.+)', 'causes'),
    (r'(.+?)\s+sonucunda\s+(.+)', 'causes'),
    (r'(.+?)\s+sebebiyle\s+(.+)', 'causes'),
    (r'(.+?)\s+neden ol(ur|du|abilir)\s+(.+)', 'causes'),
    (r'(.+?)\s+etkiler\s+(.+)', 'causes'),
    (r'(.+?)\s+arttır(ır|dı|abilir)\s+(.+)', 'causes'),
    (r'(.+?)\s+azalt(ır|tı|abilir)\s+(.+)', 'causes'),
    (r'(.+?)\s+gerektirir\s+(.+)', 'requires'),
    (r'(.+?)\s+için\s+(.+)\s+gerekli', 'requires'),
    (r'(.+?)\s+ile\s+çelişir\s+(.+)', 'contradicts'),
    (r'(.+?)\s+destekler\s+(.+)', 'supports'),
]

# İngilizce nedensellik kalıpları
CAUSAL_PATTERNS_EN = [
    (r'(.+?)\s+causes?\s+(.+)', 'causes'),
    (r'(.+?)\s+leads?\s+to\s+(.+)', 'causes'),
    (r'(.+?)\s+results?\s+in\s+(.+)', 'causes'),
    (r'(.+?)\s+increases?\s+(.+)', 'causes'),
    (r'(.+?)\s+decreases?\s+(.+)', 'causes'),
    (r'(.+?)\s+affects?\s+(.+)', 'causes'),
    (r'(.+?)\s+requires?\s+(.+)', 'requires'),
    (r'(.+?)\s+contradicts?\s+(.+)', 'contradicts'),
    (r'(.+?)\s+supports?\s+(.+)', 'supports'),
]


class AutoCausalDiscovery:
    """
    Automatic Causal Relationship Discovery Engine.

    İki ana keşif yöntemi:
    1. Pattern-based: Regex ile nedensellik kalıpları arama
    2. LLM-based: GPT-4o-mini ile nedensel analiz (daha doğru ama maliyetli)
    3. Embedding-based: Vektör benzerliği + temporal/logical analiz

    Bu, manuel nedensellik tanımına alternatif olarak sistemin
    kendi kendine öğrenmesini sağlar.
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self._llm_client = None

    def discover_from_text(self, text: str) -> List[Dict]:
        """
        Tek bir metinden nedensel ilişkileri çıkar.
        Pattern-based extraction.
        """
        relations = []

        # Türkçe kalıplar
        for pattern, rel_type in CAUSAL_PATTERNS_TR:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    relations.append({
                        'cause': match[0].strip(),
                        'effect': match[-1].strip(),
                        'relation_type': rel_type,
                        'confidence': 0.7,
                        'method': 'pattern_tr'
                    })

        # İngilizce kalıplar
        for pattern, rel_type in CAUSAL_PATTERNS_EN:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    relations.append({
                        'cause': match[0].strip(),
                        'effect': match[-1].strip(),
                        'relation_type': rel_type,
                        'confidence': 0.7,
                        'method': 'pattern_en'
                    })

        return relations

    def discover_from_corpus(
        self,
        documents: List[Dict[str, str]],
        embeddings: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Döküman koleksiyonundan nedensel ilişkileri keşfet.

        Args:
            documents: List of {'id': str, 'content': str}
            embeddings: Optional pre-computed embeddings

        Returns:
            List of discovered causal relationships
        """
        all_relations = []

        # 1. Pattern-based discovery from each document
        for doc in documents:
            text_relations = self.discover_from_text(doc['content'])
            for rel in text_relations:
                rel['source_doc'] = doc['id']
                all_relations.append(rel)

        # 2. Cross-document causal discovery using embeddings
        if embeddings is not None and len(embeddings) > 1:
            cross_doc_relations = self._discover_cross_document(documents, embeddings)
            all_relations.extend(cross_doc_relations)

        # Deduplicate and rank by confidence
        unique_relations = self._deduplicate_relations(all_relations)
        return sorted(unique_relations, key=lambda x: x['confidence'], reverse=True)

    def _discover_cross_document(
        self,
        documents: List[Dict[str, str]],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Dökümanlar arası nedensel ilişkileri embedding benzerliğine göre keşfet.

        Hipotez: Semantik olarak benzer ve belirli kalıplara uyan
        dökümanlar arasında nedensel ilişki olabilir.
        """
        relations = []

        # Compute pairwise similarities
        n = len(documents)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                sim = cosine_similarity(embeddings[i], embeddings[j])

                # High similarity suggests relationship
                if sim > self.config.discovery_threshold:
                    # Check if doc_i could be cause of doc_j
                    cause_score = self._compute_cause_score(
                        documents[i]['content'],
                        documents[j]['content']
                    )

                    if cause_score > 0.5:
                        relations.append({
                            'source': documents[i]['id'],
                            'target': documents[j]['id'],
                            'cause': documents[i]['content'][:100],
                            'effect': documents[j]['content'][:100],
                            'relation_type': 'causes',
                            'confidence': (sim + cause_score) / 2,
                            'method': 'embedding_cross_doc'
                        })

        return relations

    def _compute_cause_score(self, text1: str, text2: str) -> float:
        """
        İki metin arasında nedensellik skoru hesapla.

        Basit heuristikler:
        - text1 bir "neden" içeriyorsa
        - text2 bir "sonuç" içeriyorsa
        """
        cause_keywords = ['neden', 'sebep', 'kaynak', 'etken', 'faktör',
                          'cause', 'source', 'factor', 'reason']
        effect_keywords = ['sonuç', 'etki', 'netice', 'çıktı',
                          'effect', 'result', 'outcome', 'impact']

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        cause_in_text1 = any(kw in text1_lower for kw in cause_keywords)
        effect_in_text2 = any(kw in text2_lower for kw in effect_keywords)

        if cause_in_text1 and effect_in_text2:
            return 0.8
        elif cause_in_text1 or effect_in_text2:
            return 0.6
        else:
            return 0.3

    def discover_with_llm(
        self,
        documents: List[Dict[str, str]],
        llm_client=None
    ) -> List[Dict]:
        """
        LLM kullanarak nedensel ilişkileri keşfet.
        En doğru ama en maliyetli yöntem.
        """
        if llm_client is None:
            logger.warning("LLM client not provided, skipping LLM-based discovery")
            return []

        relations = []

        # Batch documents for efficiency
        batch_size = 5
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for i in range(0, len(documents), batch_size):
            batch_num = i // batch_size + 1
            print(f"    LLM batch {batch_num}/{total_batches}...", end=" ", flush=True)

            batch = documents[i:i+batch_size]
            batch_text = "\n\n".join([
                f"[{doc['id']}] {doc['content']}"
                for doc in batch
            ])

            prompt = f"""Aşağıdaki dökümanlar arasındaki NEDENSEL İLİŞKİLERİ bul.

Dökümanlar:
{batch_text}

Her ilişki için şu formatı kullan:
- KAYNAK: [döküman_id]
- HEDEF: [döküman_id]
- İLİŞKİ: causes/requires/supports/contradicts
- GÜÇ: 0.0-1.0 arası
- AÇIKLAMA: Neden bu ilişkiyi tespit ettin?

Sadece güçlü ve açık nedensel ilişkileri listele."""

            try:
                # generate_raw kullan (Q&A wrapper olmadan)
                response = llm_client.generate_raw(prompt, max_tokens=2000)
                # DEBUG: İlk batch'te LLM yanıtını göster
                if batch_num == 1:
                    print(f"\n    [DEBUG] LLM yanıtı:\n{response[:800]}...\n")
                parsed = self._parse_llm_response(response)
                relations.extend(parsed)
                print(f"✓ ({len(parsed)} ilişki)")
            except Exception as e:
                print(f"✗ HATA: {e}")
                logger.error(f"LLM discovery failed: {e}")

        return relations

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM response to extract relations"""
        relations = []
        lines = response.split('\n')

        def clean_id(text: str) -> str:
            """Remove brackets and clean document ID"""
            text = text.strip()
            # Remove [] brackets
            if text.startswith('[') and text.endswith(']'):
                text = text[1:-1]
            return text.strip()

        current_rel = {}
        for line in lines:
            line = line.strip()
            if line.startswith('- KAYNAK:'):
                if current_rel:
                    relations.append(current_rel)
                current_rel = {'source': clean_id(line.replace('- KAYNAK:', ''))}
            elif line.startswith('- HEDEF:'):
                current_rel['target'] = clean_id(line.replace('- HEDEF:', ''))
            elif line.startswith('- İLİŞKİ:'):
                current_rel['relation_type'] = line.replace('- İLİŞKİ:', '').strip().lower()
            elif line.startswith('- GÜÇ:'):
                try:
                    current_rel['confidence'] = float(line.replace('- GÜÇ:', '').strip())
                except ValueError:
                    current_rel['confidence'] = 0.5
            elif line.startswith('- AÇIKLAMA:'):
                current_rel['explanation'] = line.replace('- AÇIKLAMA:', '').strip()
                current_rel['method'] = 'llm'

        if current_rel:
            relations.append(current_rel)

        return relations

    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """Remove duplicate relations, keeping highest confidence"""
        seen = {}
        for rel in relations:
            key = (
                rel.get('source', rel.get('cause', '')),
                rel.get('target', rel.get('effect', '')),
                rel.get('relation_type', '')
            )
            if key not in seen or rel['confidence'] > seen[key]['confidence']:
                seen[key] = rel
        return list(seen.values())


class CausalInferenceEngine:
    """
    Pearl's Do-Calculus based causal inference.

    Bu, korelasyon vs nedensellik ayrımını matematiksel olarak yapar.
    P(Y|do(X)) vs P(Y|X) farkını hesaplar.

    NOT: Bu tam implementasyon için DoWhy veya CausalNex gibi
    kütüphaneler gerekir. Bu temel bir iskelet.
    """

    def __init__(self, graph):
        self.graph = graph

    def do_intervention(
        self,
        target_node: str,
        intervention_node: str,
        intervention_value: float = 1.0
    ) -> Dict:
        """
        do(X=x) müdahalesi simüle et.

        Grafta X'in ebeveynlerini kes ve X'i sabit değere ayarla.
        Sonra Y üzerindeki etkiyi hesapla.
        """
        # Get nodes affected by intervention
        affected = self._get_descendants(intervention_node)

        # Compute causal effect (simplified)
        if target_node in affected:
            # Direct causal path exists
            path, strength = self.graph.find_causal_path(intervention_node, target_node)
            return {
                'intervention': intervention_node,
                'target': target_node,
                'causal_effect': strength,
                'path': path,
                'is_causal': len(path) > 0
            }
        else:
            return {
                'intervention': intervention_node,
                'target': target_node,
                'causal_effect': 0.0,
                'path': [],
                'is_causal': False
            }

    def counterfactual_query(
        self,
        observed: Dict[str, float],
        intervention: Dict[str, float],
        query_node: str
    ) -> Dict:
        """
        Counterfactual sorgu: "X olmasaydı Y ne olurdu?"

        Args:
            observed: Gözlemlenen değerler {node_id: value}
            intervention: Varsayımsal müdahale {node_id: value}
            query_node: Sorgulanacak değişken

        Returns:
            Counterfactual tahmin
        """
        # This is a simplified implementation
        # Full implementation requires structural causal models (SCM)

        result = {
            'observed': observed,
            'intervention': intervention,
            'query': query_node,
            'counterfactual_value': None,
            'explanation': ''
        }

        # Check if intervention affects query through causal path
        for int_node in intervention.keys():
            path, strength = self.graph.find_causal_path(int_node, query_node)
            if path:
                result['counterfactual_value'] = intervention[int_node] * strength
                result['explanation'] = f"Causal path: {' -> '.join(path)}"
                break

        if result['counterfactual_value'] is None:
            result['counterfactual_value'] = observed.get(query_node, 0.0)
            result['explanation'] = "No causal path found, value unchanged"

        return result

    def _get_descendants(self, node_id: str) -> List[str]:
        """Get all descendants of a node"""
        descendants = []
        to_visit = [node_id]
        visited = set()

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)

            neighbors = self.graph.get_neighbors(current, ['causes'])
            to_visit.extend(neighbors)

        return descendants
