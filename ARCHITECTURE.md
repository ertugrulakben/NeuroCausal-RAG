# NeuroCausal RAG v5.2 - Mimari Dokumantasyonu

## Sistem Genel Bakis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NeuroCausal RAG v5.2                             │
│                   "Nedensel Zincirlerle Akilli Arama"                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        PRESENTATION LAYER                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │ Streamlit   │  │  FastAPI    │  │   CLI       │                   │  │
│  │  │    UI       │  │  REST API   │  │  Scripts    │                   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         REASONING LAYER (v5.1)                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │   Entity    │  │Contradiction│  │  Temporal   │                   │  │
│  │  │   Linking   │  │  Detection  │  │  Reasoning  │                   │  │
│  │  │  (FAZ 1.1)  │  │  (FAZ 1.2)  │  │  (FAZ 1.3)  │                   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         RETRIEVAL LAYER                               │  │
│  │  ┌───────────────────────────────────────────────────────────────┐   │  │
│  │  │                    Hybrid Retriever                            │   │  │
│  │  │  Score = α×Similarity + β×Causal + γ×Importance               │   │  │
│  │  │         (0.5)           (0.3)        (0.2)                    │   │  │
│  │  └───────────────────────────────────────────────────────────────┘   │  │
│  │                              │                                        │  │
│  │           ┌──────────────────┼──────────────────┐                    │  │
│  │           ▼                  ▼                  ▼                    │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │  │
│  │  │  Semantic   │    │   Causal    │    │  PageRank   │              │  │
│  │  │  Search     │    │   Chain     │    │ Importance  │              │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          CORE LAYER                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │    Graph    │  │  Embedding  │  │    Index    │                   │  │
│  │  │   Engine    │  │   Engine    │  │   Backend   │                   │  │
│  │  │ (NetworkX/  │  │ (S-BERT)    │  │ (FAISS/     │                   │  │
│  │  │  Neo4j)     │  │             │  │  Milvus)    │                   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        PERSISTENCE LAYER                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │   Neo4j     │  │   Milvus    │  │   SQLite    │                   │  │
│  │  │  (Graph)    │  │  (Vectors)  │  │ (Feedback)  │                   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Modul Yapisi

```
neurocausal_rag/
│
├── __init__.py              # NeuroCausalRAG facade class
├── config.py                # Pydantic konfigurasyonlar
├── interfaces.py            # Arayuz tanimlari (IRetriever, etc.)
│
├── core/                    # Temel motorlar
│   ├── graph.py             # GraphEngine (NetworkX), Neo4jGraphEngine
│   ├── node.py              # NeuroCausalNode dataclass
│   ├── edge.py              # NeuroCausalEdge, RelationType enum
│   └── models.py            # Temel veri modelleri
│
├── entity/                  # Entity Linking (FAZ 1.1)
│   ├── linker.py            # EntityLinker, AliasStore, Entity
│   └── ner.py               # Named Entity Recognition
│
├── reasoning/               # Reasoning Layer (FAZ 1.2, 1.3)
│   ├── contradiction.py     # ContradictionDetector
│   └── temporal.py          # TemporalEngine
│
├── search/                  # Arama motorlari
│   ├── index.py             # Index backends (FAISS, Milvus, BruteForce)
│   └── retriever.py         # Hybrid Retriever + Chain Injection
│
├── embedding/               # Vektor embedding
│   └── text.py              # TextEmbedding (Sentence-BERT)
│
├── learning/                # Discovery & Learning
│   ├── semantic_discovery.py    # Multi-signal causal discovery
│   ├── deep_discovery.py        # NLI-based verification
│   ├── funnel_discovery.py      # Funnel optimization
│   ├── entity_extraction.py     # NER extraction
│   ├── pipeline.py              # Unified discovery pipeline
│   ├── feedback.py              # RLHF feedback loop
│   └── discovery.py             # Legacy (deprecated)
│
├── agents/                  # Agentic RAG
│   ├── graph_agent.py       # Self-correcting agent
│   └── tools.py             # Agent tools
│
├── api/                     # REST API
│   ├── app.py               # FastAPI application factory
│   ├── routes.py            # API endpoints
│   └── models.py            # Pydantic request/response models
│
├── visualization/           # Gorsellestirme
│   └── graph_viz.py         # PyVis graph rendering
│
└── llm/                     # LLM istemcisi
    └── client.py            # OpenAI client wrapper
```

---

## Bilesen Detaylari

### 1. Entity Linking (FAZ 1.1)

**Amac:** Kod adlarini ve alias'lari gercek entity isimlerine cozumle.

```
┌─────────────────────────────────────────────────────────────┐
│                      EntityLinker                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ornek:                                                      │
│    "Mavi Ufuk projesi baslatildi"                           │
│         │                                                    │
│         ▼                                                    │
│    AliasStore.resolve("Mavi Ufuk")                          │
│         │                                                    │
│         ▼                                                    │
│    "Gunes Enerjisi A.S." (canonical name)                   │
│                                                              │
│  Patterns:                                                   │
│    - "X = Y"                                                 │
│    - "X (Y)"                                                 │
│    - "X olarak da bilinen Y"                                │
│    - "Project X is Y"                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**API:**
```python
from neurocausal_rag.entity.linker import EntityLinker

linker = EntityLinker()
linker.add_alias("Mavi Ufuk", "Gunes Enerjisi A.S.", confidence=0.95)
linker.learn_aliases_from_documents(documents)

# Sorgu zenginlestirme
enriched = linker.enrich_query("Mavi Ufuk projesi?")
# -> "Mavi Ufuk (Gunes Enerjisi A.S.) projesi?"

# Alias cozumleme
resolved = linker.resolve_text("Mavi Ufuk hakkinda bilgi")
# -> {"mavi ufuk": "gunes enerjisi a.s."}
```

---

### 2. Contradiction Detection (FAZ 1.2)

**Amac:** Belgeler arasindaki celiskileri tespit et.

```
┌─────────────────────────────────────────────────────────────┐
│                  ContradictionDetector                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Tespit Edilen Celiski Turleri:                             │
│                                                              │
│  1. Kar/Zarar Celiskisi                                     │
│     "Sirket kar etti" vs "Sirket zarar etti"                │
│     Score: 0.9                                               │
│                                                              │
│  2. Artis/Azalis Celiskisi                                  │
│     "Satislar artti" vs "Gelir azaldi"                      │
│     Score: 0.85                                              │
│                                                              │
│  3. Sayisal Celiski (gelecek)                               │
│     "100 milyon" vs "50 milyon"                             │
│                                                              │
│  4. Zamansal Celiski (gelecek)                              │
│     "2023'te basladi" vs "2024'te planlanmisti"             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**API:**
```python
from neurocausal_rag.reasoning.contradiction import ContradictionDetector

detector = ContradictionDetector()

score = detector.detect_conflict(
    "Sirket 100 milyon kar etti",
    "Sirket ciddi zarar etti"
)
# -> 0.9 (yuksek celiski)
```

---

### 3. Temporal Reasoning (FAZ 1.3)

**Amac:** Nedensel iliskilerin zamansal gecerliligini dogrula.

```
┌─────────────────────────────────────────────────────────────┐
│                     TemporalEngine                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Kural: Neden, sonuctan ONCE olmalidir.                     │
│                                                              │
│  Gecerli:                                                    │
│    [2022-01-01] Arastirma → [2023-06-01] Urun lansmani      │
│                                                              │
│  Gecersiz:                                                   │
│    [2023-12-01] Sonuc → [2022-01-01] Neden  (IMKANSIZ!)     │
│                                                              │
│  Tarih Formatlari:                                           │
│    - YYYY-MM-DD                                              │
│    - YYYY                                                    │
│    - Ocak/Subat/... YYYY (Turkce)                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**API:**
```python
from neurocausal_rag.reasoning.temporal import TemporalEngine

engine = TemporalEngine()

# Tarih cikarimi
date = engine.extract_date("Proje 2023-06-15'te basladi")
# -> datetime(2023, 6, 15)

# Nedensel sira dogrulamasi
is_valid = engine.validate_causal_order(
    "2022-01-01: Arastirma basladi",
    "2023-06-01: Urun piyasaya cikti"
)
# -> True (neden sonuctan once)
```

---

### 4. Hybrid Retriever

**Amac:** Semantic, causal ve importance skorlarini birlestir.

```
┌─────────────────────────────────────────────────────────────┐
│                     Hybrid Retriever                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Final Score = α × Similarity + β × Causal + γ × Importance │
│                                                              │
│  Varsayilan Agirliklar:                                      │
│    α = 0.5 (Semantic benzerlik - embedding cosine)          │
│    β = 0.3 (Nedensel skor - graf bazli)                     │
│    γ = 0.2 (Onem skoru - PageRank)                          │
│                                                              │
│  Iliski Agirliklari:                                         │
│    causes   = 1.0  (Dogrudan neden)                         │
│    supports = 0.8  (Destekleyici)                           │
│    requires = 0.7  (On kosul)                               │
│    related  = 0.5  (Genel iliski)                           │
│                                                              │
│  Chain Injection:                                            │
│    Nedensel zincirdeki bagli dokumanlar otomatik eklenir    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. Graph Engine

**Amac:** Bilgi grafi yonetimi.

```
┌─────────────────────────────────────────────────────────────┐
│                      GraphEngine                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Backends:                                                   │
│    - NetworkX (in-memory, gelistirme)                       │
│    - Neo4j (production, olceklenebilir)                     │
│                                                              │
│  Ozellikler:                                                 │
│    - Node: id, content, embedding, metadata, importance     │
│    - Edge: source, target, relation_type, strength          │
│    - PageRank importance hesaplama                          │
│    - Causal chain traversal                                 │
│    - Shortest path bulma                                    │
│    - Graph export/import (JSON)                             │
│                                                              │
│  Iliski Tipleri (RelationType):                             │
│    - CAUSES:   "X neden olur Y"                             │
│    - SUPPORTS: "X destekler Y"                              │
│    - REQUIRES: "X on kosul Y"                               │
│    - RELATED:  "X iliskili Y"                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Veri Akisi

```
                          KULLANICI SORGUSU
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Entity Linking     │
                    │  (Alias Cozumleme)    │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Query Enrichment    │
                    │  (Sorgu Zenginles.)   │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Embedding Model     │
                    │  (Sentence-BERT)      │
                    └───────────────────────┘
                                │
                                ▼
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Vector Index  │   │   Graph Engine  │   │    PageRank     │
│   (Similarity)  │   │    (Causal)     │   │  (Importance)   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Score Fusion        │
                    │  α×S + β×C + γ×I      │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Contradiction Check   │
                    │  (Celiski Kontrolu)   │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Temporal Validation  │
                    │ (Zamansal Dogrulama)  │
                    └───────────────────────┘
                                │
                                ▼
                         SONUC LISTESI
                    (+ Celiski Uyarilari)
                    (+ Zamansal Dogrulama)
```

---

## Test Mimarisi

```
tests/
├── conftest.py              # Paylasilan fixtures
├── unit/                    # Unit testler (85+)
│   ├── test_graph.py        # GraphEngine testleri
│   ├── test_entity_linker.py # EntityLinker testleri
│   ├── test_contradiction.py # ContradictionDetector testleri
│   ├── test_temporal.py     # TemporalEngine testleri
│   └── test_retriever.py    # Retriever testleri
│
├── integration/             # Entegrasyon testleri (18+)
│   ├── test_end_to_end.py   # Tam pipeline testleri
│   └── test_pipeline.py     # Reasoning pipeline testleri
│
└── fixtures/                # Test verileri
    └── sample_docs.json     # Ornek dokumanlar

Toplam: 103 test
```

---

## Performans Ozellikleri

| Metrik | v5.0 | v5.1 | v5.2 |
|--------|------|------|------|
| Max Dokuman | 10M+ | 10M+ | 10M+ |
| Graf Sorgu | O(log N) | O(log N) | O(log N) |
| Entity Resolution | - | O(1) lookup | O(1) lookup |
| Contradiction Check | - | O(N²) | O(N²) |
| Temporal Validation | - | O(1) per pair | O(1) per pair |
| Multi-Hop Search | - | - | O(N×H) where H=hops |
| Query Decomposition | - | - | O(K) sub-queries |
| Memory Operations | - | - | O(1) CRUD |
| Test Coverage | ~50% | ~80% | ~90% |
| Toplam Test | ~50 | 103 | 187 |

---

## Faz Durumu

| Faz | Durum | Aciklama |
|-----|-------|----------|
| FAZ 1.1 | ✅ | Entity Linking |
| FAZ 1.2 | ✅ | Contradiction Detection |
| FAZ 1.3 | ✅ | Temporal Reasoning |
| FAZ 1.4 | ✅ | Test Framework (103 test) |
| FAZ 2.1 | ✅ | Multi-Hop Retrieval |
| FAZ 2.2 | ✅ | Hybrid Search Optimization |
| FAZ 2.3 | ✅ | Query Decomposition |
| FAZ 2.4 | ✅ | Memory System |
| FAZ 3.1 | 📋 | Advanced NLI |
| FAZ 3.2 | 📋 | Causal Strength Learning |

---

## v5.2 Yeni Moduller

```
neurocausal_rag/
│
├── search/                     # Arama motorlari
│   ├── multi_hop.py            # Multi-Hop Retriever (FAZ 2.1)
│   ├── optimizer.py            # Search Optimizer (FAZ 2.2)
│   └── decomposer.py           # Query Decomposer (FAZ 2.3)
│
└── memory/                     # Hafiza sistemi (FAZ 2.4)
    ├── __init__.py
    └── store.py                # MemoryStore (SQLite)
```

---

**Yazar:** Ertugrul Akben | i@ertugrulakben.com
**Versiyon:** 5.2.0
**Tarih:** 2025
