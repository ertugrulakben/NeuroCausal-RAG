"""
NeuroCausal RAG - Named Entity Recognition
Metinden entity'leri (varlıkları) çıkarır.

Desteklenen entity türleri:
- PERSON: Kişi isimleri (Ahmet Yılmaz, Elon Musk)
- ORG: Organizasyonlar (Güneş Enerjisi A.Ş., Tesla)
- PROJECT: Proje isimleri (Mavi Ufuk, Project X)
- PRODUCT: Ürün isimleri
- LOCATION: Yerler (İstanbul, Avrupa)
- DATE: Tarihler (2025, Mart 2024)
- MONEY: Para miktarları (1.2 Milyar $)

Yazar: Ertugrul Akben
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Çıkarılan entity"""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 0.8

    def __repr__(self):
        return f"Entity({self.text!r}, {self.entity_type}, conf={self.confidence:.2f})"


class EntityExtractor:
    """
    Regex ve pattern-based Named Entity Recognition.

    spaCy veya Transformers kullanmadan basit ama etkili NER.
    Türkçe ve İngilizce destekli.
    """

    # Türkçe isim kalıpları
    TURKISH_NAME_PATTERN = r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\b'

    # Organizasyon kalıpları
    ORG_PATTERNS = [
        r'\b([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)\s+(?:A\.Ş\.|AŞ|Ltd\.|Şti\.|Inc\.|Corp\.|LLC|GmbH)\b',
        r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)*)\s+(?:Holding|Grubu|Group|Bank|Bankası)\b',
        r'\b(?:Şirket|Company|Firma)\s+([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)\b',
    ]

    # Proje kalıpları
    PROJECT_PATTERNS = [
        r'\b(?:Proje|Project)\s+["\']?([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)["\']?\b',
        r'\b["\']?([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)*)["\']?\s+(?:projesi|projesinin|project)\b',
        r'\b(?:kod adı|code name)\s*:?\s*["\']?([A-ZÇĞİÖŞÜa-zçğıöşü\s]+)["\']?\b',
    ]

    # Para miktarı kalıpları
    MONEY_PATTERNS = [
        r'(\d+(?:[.,]\d+)*)\s*(?:Milyar|Milyon|Bin|K|M|B)?\s*(?:\$|USD|EUR|€|TL|TRY|dolar|euro|lira)',
        r'(?:\$|USD|EUR|€|TL)\s*(\d+(?:[.,]\d+)*)\s*(?:Milyar|Milyon|Bin|K|M|B)?',
    ]

    # Tarih kalıpları
    DATE_PATTERNS = [
        r'\b(\d{1,2})\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})\b',
        r'\b(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})\b',
        r'\b(?:Q[1-4])\s+(\d{4})\b',
        r'\b(\d{4})\b',  # Yıl
    ]

    # Konum kalıpları
    LOCATION_PATTERNS = [
        r'\b(İstanbul|Ankara|İzmir|Bursa|Antalya|Adana)\b',
        r'\b(Türkiye|Turkey|Almanya|Germany|Fransa|France|ABD|USA|Avrupa|Europe)\b',
        r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\s+(?:pazarı|pazarında|bölgesi|bölgesinde)\b',
    ]

    # Unvan/Rol kalıpları (kişi tespiti için)
    TITLE_PATTERNS = [
        r'\b(?:Sayın|Bay|Bayan|Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\b',
        r'\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\s+(?:CEO|CFO|CTO|CISO|Başkan|Müdür|Yönetici)\b',
    ]

    def __init__(self, use_spacy: bool = False):
        """
        Args:
            use_spacy: True ise spaCy modelini kullan (daha doğru ama yavaş)
        """
        self.use_spacy = use_spacy
        self._spacy_nlp = None

        if use_spacy:
            self._load_spacy()

    def _load_spacy(self):
        """spaCy modelini yükle (opsiyonel)"""
        try:
            import spacy
            try:
                self._spacy_nlp = spacy.load("tr_core_news_md")
                logger.info("spaCy Türkçe modeli yüklendi")
            except OSError:
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy İngilizce modeli yüklendi")
                except OSError:
                    logger.warning("spaCy modeli bulunamadı, regex-based NER kullanılacak")
                    self.use_spacy = False
        except ImportError:
            logger.warning("spaCy kurulu değil, regex-based NER kullanılacak")
            self.use_spacy = False

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Metinden tüm entity'leri çıkar.

        Args:
            text: Analiz edilecek metin

        Returns:
            List of ExtractedEntity objects
        """
        if self.use_spacy and self._spacy_nlp:
            return self._extract_with_spacy(text)
        return self._extract_with_regex(text)

    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """spaCy ile entity çıkarma"""
        entities = []
        doc = self._spacy_nlp(text)

        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entities.append(ExtractedEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9
                ))

        return entities

    def _map_spacy_label(self, label: str) -> Optional[str]:
        """spaCy label'ını bizim formatımıza çevir"""
        mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'DATE': 'DATE',
            'MONEY': 'MONEY',
            'PRODUCT': 'PRODUCT',
        }
        return mapping.get(label)

    def _extract_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Regex ile entity çıkarma"""
        entities = []
        seen_spans = set()  # Overlapping'i önlemek için

        # Kişi isimleri (unvanlı)
        for pattern in self.TITLE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if not self._overlaps(span, seen_spans):
                    name = match.group(1) if match.lastindex else match.group(0)
                    entities.append(ExtractedEntity(
                        text=name.strip(),
                        entity_type='PERSON',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))
                    seen_spans.add(span)

        # Organizasyonlar
        for pattern in self.ORG_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if not self._overlaps(span, seen_spans):
                    org_name = match.group(1) if match.lastindex else match.group(0)
                    entities.append(ExtractedEntity(
                        text=org_name.strip(),
                        entity_type='ORG',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    ))
                    seen_spans.add(span)

        # Projeler
        for pattern in self.PROJECT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if not self._overlaps(span, seen_spans):
                    project_name = match.group(1) if match.lastindex else match.group(0)
                    entities.append(ExtractedEntity(
                        text=project_name.strip(),
                        entity_type='PROJECT',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    ))
                    seen_spans.add(span)

        # Para miktarları
        for pattern in self.MONEY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if not self._overlaps(span, seen_spans):
                    entities.append(ExtractedEntity(
                        text=match.group(0).strip(),
                        entity_type='MONEY',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95
                    ))
                    seen_spans.add(span)

        # Tarihler
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if not self._overlaps(span, seen_spans):
                    entities.append(ExtractedEntity(
                        text=match.group(0).strip(),
                        entity_type='DATE',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))
                    seen_spans.add(span)

        # Konumlar
        for pattern in self.LOCATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span = (match.start(), match.end())
                if not self._overlaps(span, seen_spans):
                    location = match.group(1) if match.lastindex else match.group(0)
                    entities.append(ExtractedEntity(
                        text=location.strip(),
                        entity_type='LOCATION',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85
                    ))
                    seen_spans.add(span)

        return entities

    def _overlaps(self, span: Tuple[int, int], seen_spans: set) -> bool:
        """Verilen span mevcut span'larla çakışıyor mu?"""
        start, end = span
        for s_start, s_end in seen_spans:
            if not (end <= s_start or start >= s_end):
                return True
        return False

    def extract_entities_by_type(self, text: str, entity_type: str) -> List[ExtractedEntity]:
        """Belirli bir türdeki entity'leri çıkar"""
        all_entities = self.extract_entities(text)
        return [e for e in all_entities if e.entity_type == entity_type]

    def find_persons(self, text: str) -> List[str]:
        """Metindeki kişi isimlerini bul"""
        entities = self.extract_entities_by_type(text, 'PERSON')
        return [e.text for e in entities]

    def find_organizations(self, text: str) -> List[str]:
        """Metindeki organizasyonları bul"""
        entities = self.extract_entities_by_type(text, 'ORG')
        return [e.text for e in entities]

    def find_projects(self, text: str) -> List[str]:
        """Metindeki proje isimlerini bul"""
        entities = self.extract_entities_by_type(text, 'PROJECT')
        return [e.text for e in entities]

    def find_money_amounts(self, text: str) -> List[str]:
        """Metindeki para miktarlarını bul"""
        entities = self.extract_entities_by_type(text, 'MONEY')
        return [e.text for e in entities]


def extract_all_entities(documents: List[Dict], use_spacy: bool = False) -> Dict[str, List[ExtractedEntity]]:
    """
    Tüm dokümanlardan entity'leri çıkar.

    Args:
        documents: [{'id': str, 'content': str}, ...]
        use_spacy: spaCy kullan mı?

    Returns:
        {doc_id: [entities], ...}
    """
    extractor = EntityExtractor(use_spacy=use_spacy)
    results = {}

    for doc in documents:
        doc_id = doc.get('id', 'unknown')
        content = doc.get('content', '')
        entities = extractor.extract_entities(content)
        results[doc_id] = entities

    return results


def build_entity_graph(documents: List[Dict]) -> Dict[str, set]:
    """
    Dokümanlardan entity co-occurrence grafiği oluştur.

    Returns:
        {entity_name: {related_entities}, ...}
    """
    extractor = EntityExtractor()
    entity_docs = {}  # entity → [doc_ids]

    # Her entity'nin hangi dokümanlarda geçtiğini bul
    for doc in documents:
        doc_id = doc.get('id', 'unknown')
        content = doc.get('content', '')
        entities = extractor.extract_entities(content)

        for entity in entities:
            if entity.text not in entity_docs:
                entity_docs[entity.text] = []
            entity_docs[entity.text].append(doc_id)

    # Aynı dokümanda geçen entity'leri ilişkilendir
    entity_relations = {name: set() for name in entity_docs}

    for doc in documents:
        doc_id = doc.get('id', 'unknown')
        doc_entities = [name for name, docs in entity_docs.items() if doc_id in docs]

        for i, e1 in enumerate(doc_entities):
            for e2 in doc_entities[i+1:]:
                entity_relations[e1].add(e2)
                entity_relations[e2].add(e1)

    return entity_relations
