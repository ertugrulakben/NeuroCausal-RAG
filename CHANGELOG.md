# Changelog

All notable changes to this project are documented in this file.

---

## [6.1.0] - 2026-03-16

### Quality & Maturity Release

- **Single version source**: All version references now read from `__version__` (was scattered across 4 files)
- **pyproject.toml**: Modern PEP 517/518 packaging, ready for PyPI
- **py.typed**: PEP 561 type hint marker
- **New tests**: API routes, LLM client, config validation, import checks (~30+ new tests)
- **Expanded exports**: `__all__` now includes GraphEngine, Retriever, EntityLinker, SearchMode
- **Deprecation warnings**: `learning/discovery.py` now warns users to migrate
- **Configurable logging**: `NEUROCAUSAL_LOG_LEVEL` environment variable support
- **Fixed version inconsistencies**: config.py (4.2.0), api/app.py (5.2.0) → all unified to 6.1.0

---

## [6.0.0] - 2026-03-16

### Open Source Release

First public release of NeuroCausal RAG. Production-deployed 2 months before global academia published similar research (CC-RAG, UIUC June 2025).

#### Highlights
- **382 tests** with comprehensive coverage
- **11,382 LOC** across 45+ Python files
- **6 search modes** (Balanced, Encyclopedia, Detective, Hub, Explorer, Fact-Checker)
- **Multi-hop retrieval** with bridge document discovery
- **Contradiction detection** and temporal reasoning
- **Entity linking** with alias resolution
- **Persistent memory** with RLHF feedback loop
- **REST API** (FastAPI) with full OpenAPI documentation
- **Agentic RAG** via LangGraph state machine
- **Enterprise backends** (Neo4j, Milvus, FAISS)
- Clean codebase ready for community contributions
- GitHub Actions CI (Python 3.10/3.11/3.12)

---

## [5.2.0] - 2025-11-30

### Advanced Search Revolution (FAZ 2)

Gelismis arama ozellikleri: Multi-Hop Retrieval, Hybrid Search Optimization, Query Decomposition ve Memory System.

---

### FAZ 2.1: Multi-Hop Retrieval

Dolaylu baglantilar uzerinden dokuman bulma.

```python
from neurocausal_rag.search import MultiHopRetriever

retriever = MultiHopRetriever(graph, embedding, max_hops=3)

# Zincir uzerinden arama
results = retriever.search("sera gazlari buzullara etki", top_k=10)

# Yol aciklamasi
explanation = retriever.explain_connection("sera_gazi", "deniz_seviyesi")
```

**Ozellikler:**
- N-hop path finding (varsayilan 3 hop)
- Path-based scoring (decay factor)
- Bridge document discovery
- Bidirectional search

---

### FAZ 2.2: Hybrid Search Optimization

Sorgu tipine gore adaptif arama.

```python
from neurocausal_rag.search import SearchOptimizer, SearchMode

optimizer = SearchOptimizer(auto_analyze=True)

# Otomatik agirlik belirleme
weights = optimizer.get_weights("Sera gazlari neden olur?")
# Detective mode: alpha=0.3, beta=0.5, gamma=0.2

# Manuel mod
weights = optimizer.get_weights("test", mode=SearchMode.ENCYCLOPEDIA)
```

**Modlar:**
| Mod | Alpha | Beta | Gamma | Kullanim |
|-----|-------|------|-------|----------|
| BALANCED | 0.5 | 0.3 | 0.2 | Genel arama |
| ENCYCLOPEDIA | 0.7 | 0.2 | 0.1 | Bilgi odakli |
| DETECTIVE | 0.3 | 0.5 | 0.2 | Nedensellik odakli |
| HUB | 0.3 | 0.2 | 0.5 | Hub odakli |
| EXPLORER | 0.4 | 0.3 | 0.3 | Kesfedici |

**Ek Ozellikler:**
- MMR (Maximal Marginal Relevance) diversification
- Coverage-based re-ranking
- Multi-hop result combination

---

### FAZ 2.3: Query Decomposition

Karmasik sorgulari alt sorgulara ayirma.

```python
from neurocausal_rag.search import QueryDecomposer, DecomposedSearch

decomposer = QueryDecomposer()

# Ayristirma
result = decomposer.decompose("Sera gazlari ve buzul erimesi hakkinda ne biliyoruz?")
# 2 sub-query: "Sera gazlari?", "Buzul erimesi?"

# Tam arama pipeline
search = DecomposedSearch(retriever)
results, decomposition = search.search("karmasik sorgu")
```

**Stratejiler:**
- CONJUNCTION: "ve", "and" ile ayristirma
- CAUSAL_CHAIN: "nedeniyle", "sonucunda" ile ayristirma
- MULTI_ASPECT: Coklu soru isareti ile ayristirma

---

### FAZ 2.4: Memory System

Kalici hafiza ve geri bildirim sistemi.

```python
from neurocausal_rag.memory import MemoryStore

store = MemoryStore("memory.db")

# Not ekle
store.add_note(
    "Bu sorgu iyi calisti",
    related_docs=["doc1", "doc2"],
    tags=["basarili"]
)

# Manuel nedensellik ekle
store.add_causal_relation("doc_a", "doc_b", "causes", note="Kesin iliski")

# Geri bildirim ver
store.add_feedback("x", "y", "causes", is_positive=True)

# Istatistikler
stats = store.get_stats()

# Sifirla (dikkatli kullanin!)
store.reset(confirm=True)

# Yedekle/Yukle
store.export_to_json("backup.json")
store.import_from_json("backup.json")
```

**Ozellikler:**
- SQLite tabanli kalici depolama
- Kullanici notlari ve gozlemleri
- Manuel nedensellik ekleme/silme
- Olumlu/olumsuz geri bildirim
- Export/import (JSON)
- Sifirlama yetenegi

---

### Yeni Dosyalar (v5.2)

| Dosya | Aciklama |
|-------|----------|
| `search/multi_hop.py` | Multi-Hop Retriever (~350 satir) |
| `search/optimizer.py` | Search Optimizer (~400 satir) |
| `search/decomposer.py` | Query Decomposer (~450 satir) |
| `memory/__init__.py` | Memory module exports |
| `memory/store.py` | Memory Store (~500 satir) |

---

### Test Istatistikleri (v5.2)

| Metrik | Deger |
|--------|-------|
| Toplam Test | 187 |
| Unit Test | 145 |
| Integration Test | 42 |
| FAZ 2 Testleri | 68 |
| Memory Testleri | 24 |
| Test Suresi | ~1.3s |

---

## [5.1.0] - 2025-11-30

### Foundation Revolution (FAZ 1)

Akil yürütme, entity linking ve test altyapisi ile v5.0 enterprise altyapisinin üzerine insa edildi.

---

### FAZ 1.1: Contradiction Detection & Temporal Reasoning

#### ContradictionDetector
Dokuman ciftleri arasinda mantiksal celiskileri tespit eder.

```python
from neurocausal_rag.reasoning import ContradictionDetector

detector = ContradictionDetector()

# Celisiki tespit
score = detector.detect_conflict(
    "Sirket rekor kar acikladi.",
    "Sirket iflas basvurusu yapti."
)
# score > 0.5 = celisiki var

# Birden fazla dokuman
results = detector.check_consistency(docs_list)
```

**Ozellikler:**
- Keyword-based conflict detection (profit/loss, increase/decrease, etc.)
- Negation pattern matching
- 0-1 arasi celisiki skoru
- Batch consistency checking

#### TemporalEngine
Tarihsel tutarlilik ve nedensel sira dogrulama.

```python
from neurocausal_rag.reasoning import TemporalEngine

engine = TemporalEngine()

# Nedensel sira dogrulama
is_valid = engine.validate_causal_order(
    "Olay A 2019'da gerceklesti.",  # Neden
    "Olay B 2020'de gerceklesti."   # Sonuc
)
# True: 2019 → 2020 gecerli nedensel sira

# Tarih cikarma
date = engine.extract_date("Proje 2021-06-15'te basladi")
# datetime(2021, 6, 15)
```

**Ozellikler:**
- Regex-based tarih cikarma (YYYY-MM-DD, YYYY, etc.)
- Nedensel sira dogrulama (neden < sonuc)
- Temporal conflict detection
- Multi-format date support

---

### FAZ 1.2: Entity Linking

#### AliasStore
Alias-canonical name eslestirme deposu.

```python
from neurocausal_rag.entity import AliasStore

store = AliasStore()

# Alias ekle
store.add("Mavi Ufuk", "Gunes Enerjisi A.S.", confidence=0.9)
store.add("Proje Alpha", "Gunes Enerjisi A.S.", confidence=0.85)

# Resolve
canonical = store.resolve("Mavi Ufuk")  # "gunes enerjisi a.s."

# Ters lookup
aliases = store.get_aliases("gunes enerjisi a.s.")
# ["mavi ufuk", "proje alpha"]
```

#### EntityLinker
Metin icinde alias resolution ve sorgu zenginlestirme.

```python
from neurocausal_rag.entity import EntityLinker

linker = EntityLinker()
linker.add_alias("Fenix", "ERP Sistemi", 0.9)

# Sorgu zenginlestirme
enriched = linker.enrich_query("Fenix nedir?")
# "Fenix nedir? (ERP Sistemi)"

# Metin icinde resolution
resolved = linker.resolve_text("Fenix ve SAP karsilastirmasi")
# {"fenix": "erp sistemi"}
```

**Ozellikler:**
- Case-insensitive matching
- Confidence-based ranking
- Bulk alias import/export
- Query enrichment for better retrieval

---

### FAZ 1.3: Hybrid Retriever

Retriever sinifi entity linking entegrasyonu ile gelistirildi.

```python
from neurocausal_rag.search import Retriever
from neurocausal_rag.entity import EntityLinker

linker = EntityLinker()
linker.add_alias("Mavi Ufuk", "Gunes Enerjisi A.S.")

retriever = Retriever(
    graph=graph_engine,
    embedding=embedding_model,
    entity_linker=linker  # Opsiyonel
)

# Entity-aware arama
results = retriever.search(
    "Mavi Ufuk'un karbon ayak izi",
    top_k=5,
    alpha=0.5,  # Similarity weight
    beta=0.3,   # Causal weight
    gamma=0.2   # Importance weight
)
```

**Skor Formulü:**
```
Score = α×Similarity + β×Causal + γ×Importance
```

---

### FAZ 1.4: Test Framework

103 test ile kapsamli test altyapisi.

#### Test Yapisi
```
tests/
├── conftest.py          # Shared fixtures
├── pytest.ini           # Test configuration
├── unit/
│   ├── test_graph.py        # 16 tests - GraphEngine
│   ├── test_entity_linker.py # 21 tests - Entity Linking
│   ├── test_contradiction.py # 13 tests - ContradictionDetector
│   ├── test_temporal.py      # 18 tests - TemporalEngine
│   └── test_retriever.py     # 11 tests - Retriever
└── integration/
    ├── test_pipeline.py      # 5 tests - Full pipeline
    └── test_end_to_end.py    # 19 tests - E2E scenarios
```

#### Test Calistirma
```bash
# Tum testler
python -m pytest tests/ -v

# Belirli modul
python -m pytest tests/unit/test_graph.py -v

# Coverage
python -m pytest tests/ --cov=neurocausal_rag
```

#### Fixtures (conftest.py)
```python
@pytest.fixture
def mock_graph_engine():
    """Mock GraphEngine for unit tests"""
    engine = MagicMock()
    engine.get_all_embeddings.return_value = (np.array([]), [])
    return engine

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model"""
    model = MagicMock()
    model.encode.return_value = np.random.randn(768)
    return model
```

---

### Yeni Dosyalar

| Dosya | Aciklama |
|-------|----------|
| `reasoning/contradiction.py` | ContradictionDetector (~150 satir) |
| `reasoning/temporal.py` | TemporalEngine (~200 satir) |
| `entity/alias_store.py` | AliasStore (~180 satir) |
| `entity/linker.py` | EntityLinker (~220 satir) |
| `tests/conftest.py` | Pytest fixtures |
| `tests/pytest.ini` | Test configuration |
| `setup.bat` | Windows setup with venv option |
| `run_tests.bat` | Test runner script |

---

### Performans Metrikleri

| Metrik | Deger |
|--------|-------|
| Toplam Test | 103 |
| Unit Test | 79 |
| Integration Test | 24 |
| Test Coverage | ~85% |
| Ortalama Test Suresi | <2s |

---

### Migration Guide (v5.0 → v5.1)

```python
# Yeni import'lar
from neurocausal_rag.reasoning import ContradictionDetector, TemporalEngine
from neurocausal_rag.entity import EntityLinker, AliasStore

# Entity linking kullanimi
linker = EntityLinker()
linker.add_alias("alias", "canonical", 0.9)

# Retriever ile entegrasyon
retriever = Retriever(
    graph=graph,
    embedding=model,
    entity_linker=linker  # Yeni parametre
)

# Reasoning kullanimi
detector = ContradictionDetector()
temporal = TemporalEngine()
```

---

## [5.0.0] - 2025-11-30

### ⭐ Major: Enterprise Infrastructure Revolution

Research prototype'tan production-ready enterprise sistemine gecis.

#### Yeni Altyapi Bilesenleri

| Bilesen | Eski | Yeni | Fayda |
|---------|------|------|-------|
| Graf DB | NetworkX (in-memory) | **Neo4j 5.x** | Olceklenebilir, kalici, Cypher |
| Vector DB | NumPy/FAISS | **Milvus 2.3** | Dagitik, milyarlarca vektor |
| Deployment | Manuel | **Docker Compose** | Tek komutla kurulum |

### Added

- **Neo4jGraphEngine**: Production-grade graf motoru
  - Cypher query language ile guclu sorgulama
  - MERGE ile atomik node/edge operasyonlari
  - PageRank hesaplamasi (GDS veya fallback)
  - Connection pooling ve timeout yonetimi
  - Lazy driver initialization
  - JSON export/import ile migration desteği

- **MilvusIndex**: Dagitik vektor arama
  - IVF_FLAT index ile hizli arama
  - L2 distance metrik
  - Otomatik collection yonetimi
  - Embedding normalizasyonu

- **Docker Compose Orchestration**
  - `docker-compose.yml`: Neo4j + Milvus + App
  - `Dockerfile`: Python 3.11-slim, non-root user
  - `.env.example`: Tum konfigurasyonlar
  - Health check'ler ve service dependency'ler

- **Factory Pattern Functions**
  - `create_graph_engine()`: Backend'e gore GraphEngine veya Neo4jGraphEngine
  - `create_index_backend()`: BruteForce, FAISS veya Milvus

### Changed

- `GraphConfig.backend`: `"networkx"` | `"neo4j"` secimi
- `IndexConfig.backend`: `"brute_force"` | `"faiss"` | `"milvus"` secimi
- Config artik environment variable'lardan deger aliyor

### Technical

- Yeni moduller:
  - `neurocausal_rag/core/graph.py` → `Neo4jGraphEngine` class (~400 satir)
  - `neurocausal_rag/search/index.py` → `MilvusIndex` class (~200 satir)
- Yeni dosyalar:
  - `docker-compose.yml`
  - `Dockerfile`
  - `.dockerignore`
  - `.env.example` (genisletildi)

### Migration Guide

```bash
# 1. Docker kurulumu
docker-compose up -d

# 2. Environment ayarlari
cp .env.example .env
# USE_NEO4J=true
# USE_MILVUS=true

# 3. Uygulamayi baslat
streamlit run app.py
```

### Performans Karsilastirmasi

| Metrik | v4.x (In-Memory) | v5.0 (Enterprise) |
|--------|------------------|-------------------|
| Max Dokuman | ~10K | **10M+** |
| Graf Sorgu | O(N) | **O(log N)** |
| Vektor Arama | Single-node | **Dagitik** |
| Kalicilik | Yok | **Kalici** |
| Olcekleme | Dikey | **Yatay** |

---

### ⚡ Deep Discovery Optimization (AŞAMA 2)

O(N²) → O(50) optimizasyonu ile büyük veri setlerinde hızlı nedensellik keşfi.

#### Funnel Strategy

```
Stage 1: Semantic Pre-filter (Fast, O(N*K))
    └── Top-K similar pairs via embedding
Stage 2: NLI Verification (Medium, O(K))
    └── Cross-Encoder entailment scoring
Stage 3: LLM Confirmation (Optional, O(M))
    └── High-confidence validation
```

### Added

- **FunnelDiscovery**: 3 aşamalı huni stratejisi
  - Stage 1: Semantic filtering (embedding similarity)
  - Stage 2: NLI verification (Cross-Encoder)
  - Stage 3: LLM confirmation (optional)
  - O(N²) → O(50) kompleksite azaltımı

- **AsyncFunnelDiscovery**: Async/parallel NLI işleme
  - ThreadPoolExecutor ile batch processing
  - asyncio event loop desteği
  - ~3x hız artışı

- **EntityExtractor**: NER-based entity extraction
  - Domain-specific patterns (climate, general)
  - SpaCy entegrasyonu (opsiyonel)
  - Alias normalization (CO2 → karbondioksit)
  - Entity-to-document mapping

- **EntityRelationDiscovery**: Entity bazlı ilişki keşfi
  - Co-occurrence analysis
  - Causal pattern matching
  - Fuzzy entity matching

- **DiscoveryPipeline**: Unified discovery API
  - 4 mod: FAST, BALANCED, DEEP, FULL
  - Otomatik score fusion
  - Execution statistics

### Pipeline Modes

| Mode | Methods | Speed | Coverage |
|------|---------|-------|----------|
| FAST | Semantic only | ~100ms | Basic |
| BALANCED | Semantic + Funnel | ~500ms | Good |
| DEEP | + Entity extraction | ~1s | Comprehensive |
| FULL | All + LLM | ~2-5s | Maximum |

### Usage

```python
from neurocausal_rag.learning import run_discovery_pipeline

# Quick discovery
relations = run_discovery_pipeline(docs, embeddings, mode="balanced")

# Full pipeline with stats
from neurocausal_rag.learning import DiscoveryPipeline

pipeline = DiscoveryPipeline(mode="deep")
result = pipeline.run(documents, embeddings)
print(f"Found {len(result.relations)} in {result.execution_time_ms:.0f}ms")
```

### Technical

- Yeni modüller:
  - `learning/funnel_discovery.py` (~400 satır)
  - `learning/entity_extraction.py` (~350 satır)
  - `learning/pipeline.py` (~300 satır)

---

### 🤖 Agentic RAG (AŞAMA 3)

LangGraph tabanlı self-correcting multi-step reasoning agent.

#### Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CausalRAGAgent                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │ Analyze │ -> │ Retrieve│ -> │ Reason  │ -> │ Verify  │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                         │                                   │
│                    Self-Correct Loop                        │
│                   (if confidence < 0.7)                     │
└─────────────────────────────────────────────────────────────┘
```

### Added

- **CausalRAGAgent**: Self-correcting reasoning agent
  - LangGraph state machine (fallback to simple execution)
  - Multi-step reasoning: Analyze → Retrieve → Reason → Verify
  - Self-correction loop with max iterations
  - Confidence-based verification
  - Detailed reasoning chain tracking

- **Agent Tools**:
  - `SearchTool`: Semantic + causal search
  - `GraphTool`: Graph navigation (chains, paths, neighbors)
  - `VerifyTool`: Fact verification against sources
  - `ReasonTool`: LLM-powered multi-hop reasoning

- **AgentState**: TypedDict-based state management
  - Query tracking
  - Search results accumulation
  - Reasoning chain logging
  - Tool call history
  - Error tracking

### Agent Features

| Feature | Description |
|---------|-------------|
| Self-Correction | Retries with adjusted strategy if confidence < 0.7 |
| Causal Weighting | Increases causal weight on retry iterations |
| Chain Exploration | Follows causal chains for deeper understanding |
| Verification | Checks answer quality against multiple factors |

### Usage

```python
from neurocausal_rag.agents import create_agent

# Create agent
agent = create_agent(retriever, graph, llm)

# Run query
result = agent.run("What causes global warming?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Iterations: {result['iterations']}")
print(f"Reasoning: {result['reasoning_chain']}")
```

### Technical

- Yeni modüller:
  - `agents/__init__.py` - Module exports
  - `agents/graph_agent.py` (~400 satır) - Main agent
  - `agents/tools.py` (~350 satır) - Tool definitions
- LangGraph entegrasyonu (optional, fallback available)
- OpenAI function calling format support

---

### 🚀 Productization (AŞAMA 4)

Production-ready REST API ve continuous learning sistemi.

#### API Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                             │
├─────────────────────────────────────────────────────────────────┤
│  /api/v1/search     - Semantic + causal search                  │
│  /api/v1/documents  - Document CRUD                             │
│  /api/v1/agent      - Agentic RAG queries                       │
│  /api/v1/feedback   - User feedback collection                  │
│  /api/v1/discovery  - Causal discovery                          │
│  /api/v1/graph      - Graph operations                          │
│  /api/v1/health     - Health check & metrics                    │
└─────────────────────────────────────────────────────────────────┘
```

### Added

- **FastAPI REST API** (`api/` module)
  - Full CRUD endpoints for documents
  - Search with mode presets (balanced, encyclopedia, detective)
  - Agent queries with self-correction
  - OpenAPI documentation (/docs, /redoc)
  - CORS middleware

- **API Authentication**
  - API key via X-API-Key header
  - Environment-based key loading
  - Optional auth for development mode

- **Health & Metrics**
  - `/health` endpoint with component status
  - `/metrics` with request stats, timing, error rates
  - Uptime tracking

- **Pydantic Models** (`api/models.py`)
  - Request/response validation
  - SearchMode, RelationType, PipelineMode enums
  - Full OpenAPI schema generation

- **FeedbackLoop** (`learning/feedback.py`)
  - FeedbackStore: SQLite/JSON persistence
  - FeedbackRecord: Typed feedback data
  - WeightAdjuster: Automatic edge weight updates
  - Analytics dashboard data
  - Low-quality result detection

### Feedback Loop Features

| Feature | Description |
|---------|-------------|
| Persistence | SQLite (prod) / JSON (dev) storage |
| Auto-Adjustment | EMA-based edge weight updates |
| Analytics | Time-based stats, trend detection |
| Quality Tracking | Low-rated result identification |

### API Usage

```python
# Option 1: Run server directly
from neurocausal_rag.api import run_server
run_server(port=8000)

# Option 2: Create custom app
from neurocausal_rag.api import create_app
from neurocausal_rag import NeuroCausalRAG

rag = NeuroCausalRAG()
app = create_app(rag_instance=rag, api_keys=["my-secret-key"])

# Option 3: CLI
# uvicorn neurocausal_rag.api.app:app --reload
```

### API Examples

```bash
# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: my-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change effects", "top_k": 5}'

# Agent query
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "X-API-Key: my-key" \
  -d '{"query": "What causes global warming?", "max_iterations": 3}'

# Submit feedback
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "X-API-Key: my-key" \
  -d '{"query": "test", "result_ids": ["doc1"], "rating": 0.8}'
```

### Technical

- Yeni modüller:
  - `api/__init__.py` - Module exports
  - `api/models.py` (~280 satır) - Pydantic models
  - `api/routes.py` (~500 satır) - FastAPI routes
  - `api/app.py` (~300 satır) - Application factory
  - `learning/feedback.py` (~450 satır) - Feedback loop
- Yeni bağımlılıklar:
  - `fastapi>=0.100.0`
  - `uvicorn>=0.22.0`
  - `python-multipart>=0.0.6`
  - `httpx>=0.24.0`

---

## [4.2.0] - 2025-11-30

### Added
- **Graf Gorsellestirme**: Yeni "Graf" sekmesi ile interaktif nedensel zincir gorsellestirme
  - PyVis entegrasyonu
  - 3 gorunum modu: Arama Sonucu, Tam Graf, Zincir Takibi
  - Renk kodlama: Sonuc (yesil), Enjekte (turuncu), Query (kirmizi)
  - Edge kalinligi iliski gucune gore dinamik

- **Arama Modu Slider'lari**: Dinamik hibrit arama agirliklari
  - 4 preset mod: Ansiklopedi, Dedektif, Dengeli, Hub Odakli
  - Manuel slider'lar: alpha (benzerlik), beta (nedensellik), gamma (onem)
  - Canli toplam kontrolu (α + β + γ = 1.0)

- **Dinamik Weight Destegi**: Retriever artik arama sirasinda weight degistirebilir

### Changed
- UI 4 sekmeden 5 sekmeye genisledi
- Versiyon 4.1 → 4.2

### Technical
- Yeni modul: `neurocausal_rag/visualization/graph_viz.py`
- PyVis>=0.3.2 bagimliligi eklendi
- `Retriever.search()` ve `search_by_embedding()` alpha/beta/gamma parametreleri aldi

---

## [4.1.0] - 2025-11-30

### Added
- **Case Study sekmesi**: UI'da hazir test senaryolari
- **Hizli Benchmark**: Tek tikla tum testleri calistir
- **Objektiflik Garantisi**: Test sonuclarinda tarafsizlik dokumantasyonu
- **Durust Performans Dokumantasyonu**:
  - Deep Discovery (NLI) CPU/GPU maliyeti uyarisi
  - Olceklenebilirlik limitleri (~5K-10K dokuman)
  - Enterprise mimari yol haritasi (Neo4j, Milvus, Kafka, K8s)
  - Funnel stratejisi aciklamasi (O(N²) → O(50))

### Changed
- Skor dengesi guncellendi (Topic Drift fix):
  - alpha: 0.3 → 0.5 (Similarity)
  - beta: 0.5 → 0.3 (Causal)
- Versiyon 4.0 → 4.1

### Fixed
- Buzul testi: Dogrudan eslesen dokumanlar artik one cikiyor

### Documentation
- README.md: Performans ve Olceklenebilirlik bolumu eklendi
- ARCHITECTURE.md: Enterprise mimari karsilastirma tablosu eklendi
- Mermaid diyagramlari ile gorsel mimari aciklamasi

---

## [4.0.0] - 2025-11-30

### ⭐ Major: Semantic Causal Discovery

Regex-based nedensellik kesfinden **Multi-Signal Fusion** sistemine gecis.

#### Yeni Moduller
- **`semantic_discovery.py`**: Coklu sinyal fuzyonu ile nedensellik kesfi
  - Asymmetric Similarity (yon analizi)
  - Lexical Causality (genel anahtar kelimeler)
  - Cluster Analysis (k-means)
  - Graph Propagation (transitif cikarim)

- **`deep_discovery.py`**: NLI (Natural Language Inference) tabanli kesif
  - Cross-Encoder modeli (nli-deberta-v3-small)
  - Entailment skorlama

#### Performans Sonuclari

| Metrik | v3.0 (Regex) | v4.0 (Semantic) |
|--------|--------------|-----------------|
| Bulunan Iliski | ~10-20 | **236** |
| Nedensellik Skoru | 0.00 | **1.00** |
| Arama Suresi | 110ms | **82ms** |
| Zincir Derinligi | 0 | **3-4 node** |

#### Case Study: Gorunmez Baglanti
```
Sorgu: "Sera gazlari kuresel isinmaya nasil neden olur?"
Sonuc: cimento_uretimi dokumani Top-5'te bulundu

Neden? Kelime eslesmesi yok ama zincir kesfedildi:
Cimento → CO2 → Sera Gazi → Kuresel Isinma
```

### Added
- `SemanticCausalDiscovery` sinifi (multi-signal fusion)
- `GraphPropagation` sinifi (transitif cikarim, decay=0.7)
- `DeepCausalDiscovery` sinifi (NLI-based)
- `CausalStrengthEstimator` sinifi (embedding-based guc tahmini)
- `enhanced_causal_discovery()` fonksiyonu (tek API)
- Chain Injection mekanizmasi (retriever'da)

### Changed
- Varsayilan discovery `semantic_discovery.py` kullanir
- LLM discovery baslangicta calismiyor (performans icin)
- app.py: `enhanced_causal_discovery` entegrasyonu

### Fixed
- `RuntimeError: dictionary changed size during iteration`
  - GraphPropagation'da dict keys kopyalanarak cozuldu
- Streamlit port catismalari

### Deprecated
- `discovery.py` (eski regex-based sistem)
  - Hala calisiyor ama onerilmiyor
  - Gelecek surumde kaldirilacak

---

## [3.0.0] - 2025-11-29

### Added
- Pydantic-based configuration system
- Facade pattern (NeuroCausalRAG main class)
- Strategy pattern (index backends: FAISS/BruteForce)
- 115 Turkce iklim dokumani (climate_knowledge_base.py)
- Wikipedia data fetcher (145 makale)

### Changed
- Tum config'ler Pydantic modellere tasindi
- Index backend dinamik secim

---

## [2.0.0] - 2025-11-28

### Added
- FAISS index support (yuksek olceklilik)
- Pearl's Do-Calculus (counterfactual queries)
- Dual-channel embedding (text + metadata)
- PageRank-based importance scoring

### Changed
- Graf motoru NetworkX'e gecti
- Embedding boyutu 384-dim (all-MiniLM-L6-v2)

---

## [1.0.0] - 2025-11-27

### Added
- Initial release
- Basic RAG functionality
- Simple keyword-based causal patterns
- OpenAI LLM integration
- Streamlit UI prototype

---

## Iliski Agirliklari (Referans)

| Tip | Agirlik | Aciklama |
|-----|---------|----------|
| causes | 1.0 | Dogrudan nedensellik |
| supports | 0.8 | Destekleyici kanit |
| requires | 0.7 | On kosul |
| related | 0.5 | Genel iliski |

---

## Skor Formulu (Referans)

```
Final Score = 0.5 × Similarity + 0.3 × Causal + 0.2 × Importance
```

---

**Yazar:** Ertugrul Akben | i@ertugrulakben.com
