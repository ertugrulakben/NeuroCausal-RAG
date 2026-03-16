"""
NeuroCausal RAG - Entity Extraction Engine
NER-based entity extraction ve entity-relation mapping

Bu modül:
1. SpaCy veya regex ile entity extraction
2. Entity normalization (alias → canonical)
3. Entity-to-document mapping
4. Entity relationship discovery

Kullanım:
    >>> extractor = EntityExtractor()
    >>> entities = extractor.extract(documents)
    >>> relations = extractor.discover_entity_relations(entities)

Yazar: Ertuğrul Akben
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity"""
    text: str  # Original text
    canonical: str  # Normalized form
    entity_type: str  # CONCEPT, PERSON, ORG, LOCATION, EVENT
    doc_ids: Set[str] = field(default_factory=set)  # Documents containing this entity
    frequency: int = 0
    context_snippets: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.canonical)

    def __eq__(self, other):
        return self.canonical == other.canonical


@dataclass
class EntityRelation:
    """Relation between two entities"""
    source_entity: str  # Canonical form
    target_entity: str  # Canonical form
    relation_type: str  # causes, supports, co_occurs
    strength: float
    evidence: List[str] = field(default_factory=list)


class EntityExtractor:
    """
    Entity extraction with optional SpaCy support.

    Features:
    - Domain-specific entity recognition (climate, science, etc.)
    - Alias normalization (CO2 → Karbondioksit)
    - Multi-word entity detection
    - Fallback to regex when SpaCy not available
    """

    def __init__(
        self,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        domain: str = "climate"
    ):
        """
        Args:
            use_spacy: SpaCy kullan (varsa)
            spacy_model: SpaCy model ismi
            domain: Domain-specific patterns (climate, medical, etc.)
        """
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self.domain = domain

        self._nlp = None
        self._patterns = self._load_domain_patterns(domain)
        self._aliases = self._load_aliases(domain)

    def _load_domain_patterns(self, domain: str) -> Dict[str, List[str]]:
        """Load domain-specific entity patterns"""
        patterns = {
            "climate": {
                "CONCEPT": [
                    r"\b(küresel\s+ısınma|global\s+warming)\b",
                    r"\b(sera\s+gaz[ıi]|greenhouse\s+gas)\b",
                    r"\b(karbon\s+ayak\s*izi|carbon\s+footprint)\b",
                    r"\b(iklim\s+değişikliği|climate\s+change)\b",
                    r"\b(deniz\s+seviyesi|sea\s+level)\b",
                    r"\b(buzul\s+erimesi|glacier\s+melting)\b",
                    r"\b(yenilenebilir\s+enerji|renewable\s+energy)\b",
                    r"\b(fosil\s+yakıt|fossil\s+fuel)\b",
                    r"\b(atmosfer|atmosphere)\b",
                    r"\b(ekosistem|ecosystem)\b",
                    r"\b(biyoçeşitlilik|biodiversity)\b",
                    r"\b(karbon\s+emisyon|carbon\s+emission)\b",
                    r"\b(paris\s+anlaşması|paris\s+agreement)\b",
                ],
                "CHEMICAL": [
                    r"\b(CO2|karbondioksit|carbon\s+dioxide)\b",
                    r"\b(CH4|metan|methane)\b",
                    r"\b(N2O|diazot\s+monoksit|nitrous\s+oxide)\b",
                    r"\b(O3|ozon|ozone)\b",
                    r"\b(CFC|kloroflorokarbon)\b",
                ],
                "MEASUREMENT": [
                    r"\b\d+(\.\d+)?\s*(°C|derece|ppm|ppb|Gt|Mt|km²)\b",
                    r"\b(yüzde|percent)\s*\d+(\.\d+)?\b",
                ],
            },
            "general": {
                "CONCEPT": [
                    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Title Case phrases
                ],
                "NUMBER": [
                    r"\b\d+(\.\d+)?%?\b",
                ],
            }
        }

        return patterns.get(domain, patterns["general"])

    def _load_aliases(self, domain: str) -> Dict[str, str]:
        """Load entity alias mappings (alias → canonical)"""
        aliases = {
            "climate": {
                # Turkish aliases
                "küresel ısınma": "küresel_ısınma",
                "global warming": "küresel_ısınma",
                "sera gazı": "sera_gazları",
                "sera gazları": "sera_gazları",
                "greenhouse gas": "sera_gazları",
                "greenhouse gases": "sera_gazları",
                "ghg": "sera_gazları",
                "co2": "karbondioksit",
                "carbondioxide": "karbondioksit",
                "carbon dioxide": "karbondioksit",
                "karbondioksit": "karbondioksit",
                "ch4": "metan",
                "methane": "metan",
                "iklim değişikliği": "iklim_değişikliği",
                "climate change": "iklim_değişikliği",
                "deniz seviyesi": "deniz_seviyesi",
                "sea level": "deniz_seviyesi",
                "buzul": "buzul",
                "glacier": "buzul",
                "buzul erimesi": "buzul_erimesi",
                "glacier melting": "buzul_erimesi",
                "fosil yakıt": "fosil_yakıt",
                "fossil fuel": "fosil_yakıt",
                "yenilenebilir enerji": "yenilenebilir_enerji",
                "renewable energy": "yenilenebilir_enerji",
                "paris anlaşması": "paris_anlaşması",
                "paris agreement": "paris_anlaşması",
            }
        }

        return aliases.get(domain, {})

    def _get_nlp(self):
        """Lazy load SpaCy model"""
        if self._nlp is not None:
            return self._nlp

        if not self.use_spacy:
            return None

        try:
            import spacy
            self._nlp = spacy.load(self.spacy_model)
            logger.info(f"SpaCy model loaded: {self.spacy_model}")
            return self._nlp
        except ImportError:
            logger.warning("SpaCy not installed, using regex fallback")
            self.use_spacy = False
            return None
        except OSError:
            logger.warning(f"SpaCy model {self.spacy_model} not found, using regex")
            self.use_spacy = False
            return None

    def extract(
        self,
        documents: List[Dict],
        min_frequency: int = 1
    ) -> Dict[str, Entity]:
        """
        Extract entities from documents.

        Args:
            documents: [{'id': str, 'content': str}, ...]
            min_frequency: Minimum entity frequency

        Returns:
            {canonical: Entity} dictionary
        """
        entities: Dict[str, Entity] = {}

        for doc in documents:
            doc_id = doc['id']
            content = doc['content']

            # Extract entities from this document
            doc_entities = self._extract_from_text(content, doc_id)

            # Merge into global entities
            for entity in doc_entities:
                canonical = entity.canonical

                if canonical in entities:
                    entities[canonical].doc_ids.add(doc_id)
                    entities[canonical].frequency += 1
                    if len(entities[canonical].context_snippets) < 5:
                        entities[canonical].context_snippets.extend(entity.context_snippets)
                else:
                    entity.doc_ids.add(doc_id)
                    entity.frequency = 1
                    entities[canonical] = entity

        # Filter by frequency
        filtered = {
            k: v for k, v in entities.items()
            if v.frequency >= min_frequency
        }

        logger.info(f"Extracted {len(filtered)} unique entities from {len(documents)} docs")
        return filtered

    def _extract_from_text(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities from single text"""
        entities = []
        text_lower = text.lower()

        # 1. Try SpaCy first
        nlp = self._get_nlp()
        if nlp:
            try:
                doc = nlp(text[:5000])  # Limit for performance
                for ent in doc.ents:
                    canonical = self._normalize(ent.text)
                    entities.append(Entity(
                        text=ent.text,
                        canonical=canonical,
                        entity_type=ent.label_,
                        context_snippets=[self._get_context(text, ent.start_char)]
                    ))
            except Exception as e:
                logger.debug(f"SpaCy extraction error: {e}")

        # 2. Domain-specific patterns (always run)
        for entity_type, patterns in self._patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        matched_text = match.group(0)
                        canonical = self._normalize(matched_text)

                        # Skip if already found
                        if any(e.canonical == canonical for e in entities):
                            continue

                        entities.append(Entity(
                            text=matched_text,
                            canonical=canonical,
                            entity_type=entity_type,
                            context_snippets=[self._get_context(text, match.start())]
                        ))
                except re.error:
                    continue

        # 3. Check aliases
        for alias, canonical in self._aliases.items():
            if alias in text_lower:
                if not any(e.canonical == canonical for e in entities):
                    idx = text_lower.find(alias)
                    entities.append(Entity(
                        text=alias,
                        canonical=canonical,
                        entity_type="CONCEPT",
                        context_snippets=[self._get_context(text, idx)]
                    ))

        return entities

    def _normalize(self, text: str) -> str:
        """Normalize entity text to canonical form"""
        text_lower = text.lower().strip()

        # Check aliases first
        if text_lower in self._aliases:
            return self._aliases[text_lower]

        # Default normalization: lowercase, replace spaces with underscore
        normalized = re.sub(r'\s+', '_', text_lower)
        normalized = re.sub(r'[^\w_]', '', normalized)

        return normalized

    def _get_context(self, text: str, position: int, window: int = 50) -> str:
        """Get context around entity"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()


class EntityRelationDiscovery:
    """
    Discover relations between entities based on co-occurrence and context.
    """

    def __init__(
        self,
        co_occurrence_threshold: int = 2,
        context_similarity_threshold: float = 0.5
    ):
        self.co_occurrence_threshold = co_occurrence_threshold
        self.context_similarity_threshold = context_similarity_threshold

    def discover(
        self,
        entities: Dict[str, Entity],
        documents: List[Dict],
        embeddings: Optional[dict] = None
    ) -> List[EntityRelation]:
        """
        Discover relations between entities.

        Methods:
        1. Co-occurrence: Same document → related
        2. Context similarity: Similar context → related
        3. Causal patterns: "A causes B" patterns

        Args:
            entities: {canonical: Entity} from EntityExtractor
            documents: Original documents
            embeddings: Optional {doc_id: embedding} for similarity

        Returns:
            List of EntityRelation
        """
        relations = []

        # 1. Co-occurrence based relations
        relations.extend(self._discover_co_occurrence(entities))

        # 2. Causal pattern based relations
        relations.extend(self._discover_causal_patterns(entities, documents))

        # Deduplicate and sort
        seen = set()
        unique_relations = []
        for rel in sorted(relations, key=lambda x: x.strength, reverse=True):
            key = (rel.source_entity, rel.target_entity)
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        logger.info(f"Discovered {len(unique_relations)} entity relations")
        return unique_relations

    def _discover_co_occurrence(
        self,
        entities: Dict[str, Entity]
    ) -> List[EntityRelation]:
        """Find entities that co-occur in same documents"""
        relations = []
        entity_list = list(entities.values())

        for i, e1 in enumerate(entity_list):
            for e2 in entity_list[i+1:]:
                # Find common documents
                common_docs = e1.doc_ids & e2.doc_ids

                if len(common_docs) >= self.co_occurrence_threshold:
                    # Strength based on co-occurrence frequency
                    strength = min(1.0, len(common_docs) / 5.0)

                    relations.append(EntityRelation(
                        source_entity=e1.canonical,
                        target_entity=e2.canonical,
                        relation_type="co_occurs",
                        strength=strength,
                        evidence=[f"Co-occur in {len(common_docs)} documents"]
                    ))

        return relations

    def _discover_causal_patterns(
        self,
        entities: Dict[str, Entity],
        documents: List[Dict]
    ) -> List[EntityRelation]:
        """Find causal relations using text patterns"""
        relations = []

        # Causal patterns (entity A) causes/leads to (entity B)
        causal_patterns = [
            r"(\w+)\s+(?:neden\s+olur|causes?|leads?\s+to|triggers?)\s+(\w+)",
            r"(\w+)\s+(?:sonucunda|results?\s+in|produces?)\s+(\w+)",
            r"(\w+)\s+(?:arttırır|increases?|boosts?)\s+(\w+)",
            r"(\w+)\s+(?:azaltır|decreases?|reduces?)\s+(\w+)",
        ]

        entity_canonicals = set(entities.keys())

        for doc in documents:
            content = doc['content'].lower()

            for pattern in causal_patterns:
                try:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        source_text = match.group(1).lower()
                        target_text = match.group(2).lower()

                        # Normalize and check if they are known entities
                        source_norm = re.sub(r'\s+', '_', source_text)
                        target_norm = re.sub(r'\s+', '_', target_text)

                        # Fuzzy match to known entities
                        source_entity = self._fuzzy_match(source_norm, entity_canonicals)
                        target_entity = self._fuzzy_match(target_norm, entity_canonicals)

                        if source_entity and target_entity and source_entity != target_entity:
                            relations.append(EntityRelation(
                                source_entity=source_entity,
                                target_entity=target_entity,
                                relation_type="causes",
                                strength=0.8,
                                evidence=[f"Pattern match: {match.group(0)[:50]}"]
                            ))
                except re.error:
                    continue

        return relations

    def _fuzzy_match(
        self,
        text: str,
        candidates: Set[str],
        threshold: float = 0.6
    ) -> Optional[str]:
        """Fuzzy match text to known entity canonicals"""
        text_clean = text.replace('_', ' ').lower()

        for candidate in candidates:
            candidate_clean = candidate.replace('_', ' ').lower()

            # Exact match
            if text_clean == candidate_clean:
                return candidate

            # Substring match
            if text_clean in candidate_clean or candidate_clean in text_clean:
                return candidate

            # Simple Jaccard similarity
            set1 = set(text_clean.split())
            set2 = set(candidate_clean.split())
            if set1 and set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
                if jaccard >= threshold:
                    return candidate

        return None


def extract_entities_and_relations(
    documents: List[Dict],
    domain: str = "climate",
    min_frequency: int = 2
) -> Tuple[Dict[str, Entity], List[EntityRelation]]:
    """
    Combined entity extraction and relation discovery.

    Args:
        documents: [{'id': str, 'content': str}, ...]
        domain: Domain for specialized patterns
        min_frequency: Minimum entity frequency

    Returns:
        (entities, relations) tuple

    Example:
        >>> entities, relations = extract_entities_and_relations(documents)
        >>> print(f"Found {len(entities)} entities, {len(relations)} relations")
    """
    # Extract entities
    extractor = EntityExtractor(domain=domain)
    entities = extractor.extract(documents, min_frequency=min_frequency)

    # Discover relations
    discovery = EntityRelationDiscovery()
    relations = discovery.discover(entities, documents)

    return entities, relations
