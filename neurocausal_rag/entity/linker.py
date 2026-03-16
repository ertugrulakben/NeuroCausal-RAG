"""
NeuroCausal RAG - Entity Linker
Kod adları ve takma adları gerçek entity'lerle eşleştirir.

Örnek Problem:
- Belge A: "Mavi Ufuk projesi başlatıldı"
- Belge B: "Güneş Enerjisi A.Ş. satın alındı"
- Belge C: "Mavi Ufuk = Güneş Enerjisi A.Ş. kod adıdır"

EntityLinker bu üç belgeyi birleştirerek:
"Mavi Ufuk" = "Güneş Enerjisi A.Ş." alias'ını öğrenir.

Yazar: Ertugrul Akben
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Bir entity (varlık) temsili"""
    name: str
    entity_type: str  # PERSON, ORG, PROJECT, PRODUCT, LOCATION, etc.
    aliases: Set[str] = field(default_factory=set)
    source_doc_id: Optional[str] = None
    confidence: float = 1.0

    def __hash__(self):
        return hash(self.name.lower())

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.name.lower() == other.name.lower()
        return False

    def matches(self, text: str) -> bool:
        """Bu entity verilen metinle eşleşiyor mu?"""
        text_lower = text.lower()
        if self.name.lower() in text_lower:
            return True
        return any(alias.lower() in text_lower for alias in self.aliases)


class AliasStore:
    """
    Entity alias'larını saklayan ve sorgulayan store.
    "Mavi Ufuk" → "Güneş Enerjisi A.Ş." gibi eşleştirmeleri tutar.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.aliases: Dict[str, str] = {}  # alias → canonical_name
        self.reverse_aliases: Dict[str, Set[str]] = defaultdict(set)  # canonical → {aliases}
        self.confidence_scores: Dict[Tuple[str, str], float] = {}  # (alias, canonical) → confidence
        self.persist_path = persist_path

        if persist_path and Path(persist_path).exists():
            self._load()

    def add_alias(self, alias: str, canonical_name: str, confidence: float = 0.8):
        """Yeni alias ekle"""
        alias_lower = alias.lower().strip()
        canonical_lower = canonical_name.lower().strip()

        # Kendisiyle eşleştirme yapma
        if alias_lower == canonical_lower:
            return

        self.aliases[alias_lower] = canonical_lower
        self.reverse_aliases[canonical_lower].add(alias_lower)
        self.confidence_scores[(alias_lower, canonical_lower)] = confidence

        logger.info(f"Alias eklendi: '{alias}' → '{canonical_name}' (güven: {confidence:.2f})")

    def resolve(self, text: str) -> Optional[str]:
        """Verilen metni canonical forma çözümle"""
        text_lower = text.lower().strip()
        return self.aliases.get(text_lower)

    def get_aliases(self, canonical_name: str) -> Set[str]:
        """Bir canonical name'in tüm alias'larını getir"""
        return self.reverse_aliases.get(canonical_name.lower(), set())

    def get_confidence(self, alias: str, canonical: str) -> float:
        """Alias-canonical eşleşmesinin güven skorunu getir"""
        return self.confidence_scores.get((alias.lower(), canonical.lower()), 0.0)

    def find_in_text(self, text: str) -> List[Tuple[str, str, float]]:
        """Metinde geçen tüm alias'ları bul ve çözümle"""
        found = []
        text_lower = text.lower()

        for alias, canonical in self.aliases.items():
            if alias in text_lower:
                confidence = self.confidence_scores.get((alias, canonical), 0.5)
                found.append((alias, canonical, confidence))

        return found

    def save(self, path: Optional[str] = None):
        """Alias'ları dosyaya kaydet (public method)"""
        save_path = path or self.persist_path
        if not save_path:
            return

        data = {
            'aliases': self.aliases,
            'confidence_scores': {f"{k[0]}|{k[1]}": v for k, v in self.confidence_scores.items()}
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Alias store kaydedildi: {len(self.aliases)} alias -> {save_path}")

    def load(self, path: Optional[str] = None):
        """Alias'ları dosyadan yükle (public method)"""
        load_path = path or self.persist_path
        if not load_path:
            return

        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.aliases = data.get('aliases', {})

            # Reverse aliases'ı yeniden oluştur
            for alias, canonical in self.aliases.items():
                self.reverse_aliases[canonical].add(alias)

            # Confidence scores'u yeniden oluştur
            for key, conf in data.get('confidence_scores', {}).items():
                parts = key.split('|')
                if len(parts) == 2:
                    self.confidence_scores[(parts[0], parts[1])] = conf

            logger.info(f"Alias store yüklendi: {len(self.aliases)} alias <- {load_path}")
        except Exception as e:
            logger.warning(f"Alias store yüklenemedi: {e}")

    def _save(self):
        """Internal save - backward compat"""
        self.save()

    def _load(self):
        """Internal load - backward compat"""
        self.load()


class EntityLinker:
    """
    Entity Linking ana sınıfı.

    Görevleri:
    1. Belgelerden entity'leri çıkar (NER)
    2. Farklı belgelerdeki entity'leri eşleştir (Coreference)
    3. Alias'ları öğren ve sakla
    4. Sorgu zamanında alias'ları çözümle
    """

    # Alias tespit kalıpları (Türkçe + İngilizce)
    ALIAS_PATTERNS = [
        # "X, Y olarak da bilinir"
        r'["\']?([^"\']+)["\']?\s*(?:olarak da bilinir|olarak bilinen|kod adı|kod adlı|takma adı|lakabı)',
        # "Y (X)"
        r'([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)\s*\(\s*["\']?([^)"\']+)["\']?\s*\)',
        # "X = Y"
        r'["\']?([^"\'=]+)["\']?\s*=\s*["\']?([^"\']+)["\']?',
        # "X, yani Y"
        r'["\']?([^"\']+)["\']?\s*,?\s*yani\s*["\']?([^"\']+)["\']?',
        # "X (also known as Y)"
        r'([A-Za-z\s]+)\s*\(\s*(?:also known as|aka|a\.k\.a\.)\s*([^)]+)\s*\)',
        # "Project X is Y"
        r'(?:proje|project)\s+["\']?([^"\']+)["\']?\s+(?:is|=|olarak)\s+["\']?([^"\'\.]+)["\']?',
    ]

    def __init__(self, alias_store: Optional[AliasStore] = None):
        self.alias_store = alias_store or AliasStore()
        self.entities: Dict[str, Entity] = {}  # entity_name → Entity
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.ALIAS_PATTERNS]

    def extract_aliases_from_text(self, text: str, doc_id: str = None) -> List[Tuple[str, str, float]]:
        """
        Metinden alias kalıplarını çıkar.

        Returns:
            List of (alias, canonical_name, confidence) tuples
        """
        found_aliases = []

        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    alias = match[0].strip()
                    canonical = match[1].strip()

                    # Çok kısa veya çok uzun eşleşmeleri filtrele
                    if 2 <= len(alias) <= 100 and 2 <= len(canonical) <= 100:
                        # Aynı kelime değilse ekle
                        if alias.lower() != canonical.lower():
                            found_aliases.append((alias, canonical, 0.85))

        return found_aliases

    def learn_aliases_from_documents(self, documents: List[Dict]) -> int:
        """
        Doküman listesinden alias'ları öğren.

        Args:
            documents: [{'id': str, 'content': str, ...}, ...]

        Returns:
            Öğrenilen alias sayısı
        """
        learned_count = 0

        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            content = doc.get('content', '')

            aliases = self.extract_aliases_from_text(content, doc_id)

            for alias, canonical, confidence in aliases:
                self.alias_store.add_alias(alias, canonical, confidence)
                learned_count += 1

        logger.info(f"{learned_count} alias öğrenildi (toplam: {len(self.alias_store.aliases)})")
        return learned_count

    def learn_alias(self, alias: str, canonical: str, confidence: float = 0.9):
        """Manuel olarak alias ekle (kullanıcı feedback'i veya LLM'den)"""
        self.alias_store.add_alias(alias, canonical, confidence)

    def add_alias(self, alias: str, canonical: str, confidence: float = 0.9):
        """Manuel olarak alias ekle (learn_alias ile aynı, daha kısa isim)"""
        self.alias_store.add_alias(alias, canonical, confidence)

    def resolve_text(self, text: str) -> Dict[str, str]:
        """
        Metindeki alias'ları tespit et ve canonical karşılıklarını döndür.

        Args:
            text: İşlenecek metin

        Returns:
            {alias: canonical} eşleştirmeleri

        Örnek:
            Input: "Mavi Ufuk projesi başarılı oldu"
            Output: {"Mavi Ufuk": "Güneş Enerjisi A.Ş."}
        """
        found = self.alias_store.find_in_text(text)
        result = {}

        for alias, canonical, confidence in found:
            result[alias] = canonical

        return result

    def resolve_text_full(self, text: str) -> str:
        """
        Metindeki tüm alias'ları canonical formlarıyla değiştir.

        Örnek:
            Input: "Mavi Ufuk projesi başarılı oldu"
            Output: "Güneş Enerjisi A.Ş. (Mavi Ufuk) projesi başarılı oldu"
        """
        resolved_text = text
        found = self.alias_store.find_in_text(text)

        # Uzundan kısaya sırala (overlapping match'leri önlemek için)
        found.sort(key=lambda x: len(x[0]), reverse=True)

        for alias, canonical, confidence in found:
            # Büyük/küçük harf uyumlu değiştirme
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            replacement = f"{canonical} ({alias})"
            resolved_text = pattern.sub(replacement, resolved_text, count=1)

        return resolved_text

    def find_entity_connections(self, doc1: Dict, doc2: Dict) -> List[Tuple[str, str, float]]:
        """
        İki doküman arasında entity bağlantılarını bul.

        Returns:
            List of (entity_in_doc1, entity_in_doc2, confidence) tuples
        """
        connections = []

        content1 = doc1.get('content', '')
        content2 = doc2.get('content', '')

        # Her iki dokümandaki alias'ları bul
        aliases1 = self.alias_store.find_in_text(content1)
        aliases2 = self.alias_store.find_in_text(content2)

        # Aynı canonical'a işaret eden alias'ları bul
        canonical_to_aliases1 = defaultdict(list)
        canonical_to_aliases2 = defaultdict(list)

        for alias, canonical, conf in aliases1:
            canonical_to_aliases1[canonical].append((alias, conf))

        for alias, canonical, conf in aliases2:
            canonical_to_aliases2[canonical].append((alias, conf))

        # Ortak canonical'ları bul
        common_canonicals = set(canonical_to_aliases1.keys()) & set(canonical_to_aliases2.keys())

        for canonical in common_canonicals:
            for alias1, conf1 in canonical_to_aliases1[canonical]:
                for alias2, conf2 in canonical_to_aliases2[canonical]:
                    combined_confidence = (conf1 + conf2) / 2
                    connections.append((alias1, alias2, combined_confidence))

        return connections

    def enrich_query(self, query: str) -> str:
        """
        Sorguyu alias bilgisiyle zenginleştir.

        Örnek:
            Input: "Mavi Ufuk kaç dolar?"
            Output: "Mavi Ufuk (Güneş Enerjisi A.Ş. satın alması) kaç dolar?"
        """
        enriched = query
        found = self.alias_store.find_in_text(query)

        for alias, canonical, confidence in found:
            if confidence >= 0.7:  # Yüksek güvenli alias'ları ekle
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                enriched = pattern.sub(f"{alias} ({canonical})", enriched, count=1)

        return enriched

    def get_all_aliases(self) -> Dict[str, List[str]]:
        """Tüm canonical → aliases eşleştirmelerini döndür"""
        return {
            canonical: list(aliases)
            for canonical, aliases in self.alias_store.reverse_aliases.items()
        }

    def to_dict(self) -> Dict:
        """Serileştirme için dict'e dönüştür"""
        return {
            'aliases': self.alias_store.aliases,
            'confidence_scores': {
                f"{k[0]}|{k[1]}": v
                for k, v in self.alias_store.confidence_scores.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EntityLinker':
        """Dict'ten EntityLinker oluştur"""
        linker = cls()

        for alias, canonical in data.get('aliases', {}).items():
            conf_key = f"{alias}|{canonical}"
            confidence = data.get('confidence_scores', {}).get(conf_key, 0.8)
            linker.alias_store.add_alias(alias, canonical, confidence)

        return linker


def find_aliases_with_llm(documents: List[Dict], llm_client) -> List[Tuple[str, str, float]]:
    """
    LLM kullanarak belgeler arasındaki alias'ları bul.

    Bu fonksiyon daha yüksek doğruluk için LLM'i kullanır.
    """
    if not documents or not llm_client:
        return []

    # İlk 5 belgenin içeriğini birleştir
    combined_content = "\n---\n".join([
        f"[{doc.get('id', 'unknown')}]: {doc.get('content', '')[:500]}"
        for doc in documents[:5]
    ])

    prompt = f"""Aşağıdaki belgelerde geçen kod adlarını, takma adları ve gerçek isimlerini bul.

BELGELER:
{combined_content}

GÖREV: Her belgedeki "kod adı = gerçek isim" eşleştirmelerini bul.
Örnek çıktı formatı:
- "Mavi Ufuk" = "Güneş Enerjisi A.Ş."
- "Proje X" = "Yeni Ürün Lansmanı"

Sadece kesin eşleştirmeleri yaz. Belirsizse yazma."""

    try:
        response = llm_client.generate_raw(prompt, max_tokens=500)

        # Parse response
        aliases = []
        for line in response.split('\n'):
            line = line.strip()
            if '=' in line and '"' in line:
                # Parse "X" = "Y" format
                match = re.search(r'"([^"]+)"\s*=\s*"([^"]+)"', line)
                if match:
                    aliases.append((match.group(1), match.group(2), 0.9))

        return aliases
    except Exception as e:
        logger.warning(f"LLM alias bulma hatası: {e}")
        return []
