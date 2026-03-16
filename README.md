<p align="center">
  <img src="cover.png" alt="NeuroCausal RAG" width="100%">
</p>

<p align="center">
  <strong>Causality-Aware Retrieval-Augmented Generation</strong><br>
  <em>Find what keyword search can't — by understanding why things are connected.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-6.1.0-blue?style=for-the-badge" alt="v6.0.0" />
  <img src="https://img.shields.io/badge/python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/tests-382_passed-brightgreen?style=for-the-badge" alt="187 Tests" />
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" alt="MIT License" />
</p>

<p align="center">
  <a href="#the-problem">Problem</a> •
  <a href="#the-solution">Solution</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#search-modes">Search Modes</a> •
  <a href="#api">API</a> •
  <a href="#benchmarks">Benchmarks</a>
</p>

---

> **Research Context:** In June 2025, researchers at the University of Illinois Urbana-Champaign published "CC-RAG: Structured Multi-Hop Reasoning via Theme-Based Causal Graphs" — a breakthrough paper that brought causal reasoning into RAG systems. The academic world was excited: RAG could finally "understand and connect," not just "find and fetch."
>
> **We had already been doing this for two months.** NeuroCausal RAG v5.0 was deployed to production in April 2025. The causal engine, multi-hop retrieval, and chain injection were already running in real enterprise environments.
>
> This is how we work: we build for real clients first, battle-test in production, then open-source. Our personal AI system [JARVIS](https://github.com/ertugrulakben) has been alive for 5 years and operating as an autonomous agent for 3 years — months before platforms like OpenClaw existed. We plan to open-source that too.
>
> **Read more:** [Our 2025 AI R&D: NeuroCausal RAG, DSGMv2, and 100+ SaaS Projects](https://ertugrulakben.com/2025-yapay-zeka-ar-ge-calismalarimiz-neurocausal-rag-dsgmv2-ve-100-saas-projesi/)

---

## The Problem

Classic RAG systems retrieve documents by **keyword similarity**. Search for "stress" and you get documents containing the word "stress."

But real-world knowledge doesn't work that way:

```
Stress → Cortisol rises → Sleep disrupted → Attention drops → Workplace accident risk increases
```

If your system can't see this chain, it misses critical connections.

**The academic world noticed this in June 2025** when UIUC researchers published CC-RAG, introducing causal graphs into RAG.

**We deployed this to production in April 2025.** Two months earlier.

## The Solution

**NeuroCausal RAG** builds a causal knowledge graph on top of your documents and retrieves information by understanding *why* things are connected — not just *what words* they share.

| | Classic RAG | NeuroCausal RAG |
|---|---|---|
| **Retrieval** | Keyword similarity | Cause-effect relationships |
| **Search for "stress"** | Documents about stress | + Cortisol, sleep, workplace accidents |
| **Hops** | Single (1-hop) | N-degree (multi-hop) |
| **Scoring** | Vector distance | Hybrid: Similarity + Causal + PageRank |
| **Memory** | None | Persistent feedback loop |
| **Contradictions** | Ignored | Detected and flagged |

## Architecture

```
NeuroCausal RAG v6.0
├── Core Layer
│   ├── Causal Knowledge Graph (NetworkX / Neo4j)
│   ├── Multilingual Embeddings (Sentence-BERT)
│   └── Vector Index (BruteForce / FAISS / Milvus)
│
├── Search & Retrieval
│   ├── Hybrid Retriever (Similarity + Causal + Importance)
│   ├── Multi-Hop Search (N-hop path finding + bridge docs)
│   ├── Search Optimizer (6 adaptive modes)
│   └── Query Decomposer (complex → sub-queries)
│
├── Reasoning
│   ├── Contradiction Detector
│   ├── Temporal Reasoner
│   └── Entity Linker (alias resolution)
│
├── Learning
│   ├── Causal Discovery (semantic + NLI + funnel)
│   ├── Feedback Loop (RLHF)
│   └── Persistent Memory (SQLite)
│
├── Agentic RAG
│   └── LangGraph Self-Correcting Agent
│
└── API & UI
    ├── FastAPI REST API
    └── Streamlit Dashboard
```

## Scoring Formula

```
Final Score = α × Similarity + β × Causal + γ × Importance

Multi-Hop Decay: hop_score = base_score × (0.7 ^ hop_distance)
```

## Quick Start

### Installation

```bash
git clone https://github.com/ertugrulakben/NeuroCausal-RAG.git
cd NeuroCausal-RAG
pip install -r requirements.txt
```

### Basic Usage

```python
from neurocausal_rag import NeuroCausalRAG

rag = NeuroCausalRAG()

# Add documents
rag.add_document("cement", "Cement production is responsible for 8% of global CO2 emissions.")
rag.add_document("co2", "CO2 is the primary greenhouse gas driving climate change.")
rag.add_document("warming", "Global warming causes sea level rise and extreme weather.")

# Add causal links
rag.add_causal_link("cement", "co2", "causes")
rag.add_causal_link("co2", "warming", "causes")

# Search — finds cement even though query doesn't mention it
results = rag.search("What causes global warming?")
# → Returns: co2, warming, AND cement (via causal chain)
```

### Multi-Hop Search

```python
from neurocausal_rag.search import create_multi_hop_retriever

retriever = create_multi_hop_retriever(graph, embedding, max_hops=3)
results = retriever.search("How does cement affect sea levels?")

# Discovered chain:
# Cement Production → CO2 Emissions → Global Warming → Sea Level Rise
explanation = retriever.explain_connection("cement", "warming")
```

### Docker

```bash
docker-compose up -d
# API: http://localhost:8000
# UI: http://localhost:8501
```

## Search Modes

6 preset modes for different retrieval strategies:

| Mode | α (Similarity) | β (Causal) | γ (Importance) | Best For |
|------|:-:|:-:|:-:|---|
| **BALANCED** | 0.5 | 0.3 | 0.2 | General purpose |
| **ENCYCLOPEDIA** | 0.7 | 0.2 | 0.1 | Factual queries |
| **DETECTIVE** | 0.3 | 0.5 | 0.2 | Cause-effect investigation |
| **HUB** | 0.3 | 0.2 | 0.5 | Finding central documents |
| **EXPLORER** | 0.4 | 0.3 | 0.3 | Open-ended research |
| **FACT_CHECKER** | 0.6 | 0.3 | 0.1 | Verification tasks |

```python
from neurocausal_rag.search import create_optimizer

optimizer = create_optimizer(graph, embedding)
results = optimizer.search("Why did the bridge collapse?", mode="DETECTIVE")
```

## API

Full REST API via FastAPI:

```bash
uvicorn neurocausal_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Search with causal reasoning |
| `POST` | `/api/v1/documents` | Add documents |
| `GET` | `/api/v1/documents` | List documents |
| `POST` | `/api/v1/documents/links` | Add causal links |
| `POST` | `/api/v1/agent/query` | Agentic RAG query |
| `POST` | `/api/v1/feedback` | Submit feedback |
| `POST` | `/api/v1/discovery` | Auto-discover causal links |
| `GET` | `/api/v1/graph/stats` | Graph statistics |
| `POST` | `/api/v1/graph/chain` | Get causal chain |
| `GET` | `/api/v1/health` | Health check |

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What causes global warming?", "top_k": 5, "mode": "DETECTIVE"}'
```

## Benchmarks

### Case Study: The Invisible Connection

**Query:** "How do greenhouse gases cause global warming?"

| Metric | Classic RAG | NeuroCausal RAG |
|--------|:-:|:-:|
| Search Time | 37 ms | 22 ms |
| Documents Found | Greenhouse effect, Gases | + **Cement Production** |
| Causal Score | 0.00 | 1.00 |
| Multi-Hop | None | 3-hop chain |

**Discovered chain:**
```
Cement Production → CO2 Emissions → Greenhouse Gas → Global Warming
```

The word "cement" appears nowhere in the query — but the causal chain reveals the connection.

### vs. CC-RAG (UIUC, June 2025)

| Feature | CC-RAG (June 2025) | NeuroCausal RAG (April 2025) |
|---------|:-:|:-:|
| Causal Graph | DAG structure | NetworkX + Neo4j |
| Multi-Hop | Theme-based chaining | N-hop + bridge documents |
| Bidirectional Search | Yes | Yes |
| Memory System | No | Persistent (SQLite) |
| Query Decomposition | No | Sub-query system |
| Contradiction Detection | No | Yes |
| Temporal Reasoning | No | Yes |
| Entity Linking | No | Yes (alias resolution) |
| Enterprise Ready | Academic | Production deployed |
| **Published** | **June 2025** | **April 2025** |

## Testing

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=neurocausal_rag --cov-report=html
```

```
Test Distribution (v6.1)
├── Core (graph, node, edge): 35 tests
├── Search (retriever, multi_hop, optimizer, decomposer): 66 tests
├── Learning (discovery, entity, temporal, contradiction): 42 tests
├── Memory: 24 tests
├── Integration: 20 tests
├── API Routes: 58 tests
├── Config Validation: 70 tests
├── LLM Client: 34 tests
└── Imports & Exports: 33 tests
─────────────────────────────
Total: 382 tests, 0 failures
```

## Project Structure

```
neurocausal_rag/
├── core/           # Graph engine, nodes, edges
├── embedding/      # Sentence-BERT multilingual
├── search/         # Retriever, multi-hop, optimizer, decomposer
├── learning/       # Causal discovery, feedback, pipeline
├── entity/         # Entity linking, NER
├── reasoning/      # Contradiction detection, temporal reasoning
├── memory/         # Persistent memory store
├── agents/         # LangGraph agentic RAG
├── api/            # FastAPI REST endpoints
├── llm/            # LLM client (OpenAI)
├── visualization/  # Graph visualization (PyVis)
└── ui/             # Streamlit components
```

## Configuration

```bash
cp .env.example .env
# Set your API keys in .env
```

## Roadmap

- [x] Causal knowledge graph
- [x] Multi-hop retrieval
- [x] 6 search modes
- [x] Contradiction detection
- [x] Temporal reasoning
- [x] Entity linking
- [x] Persistent memory (RLHF)
- [x] REST API (FastAPI)
- [x] Agentic RAG (LangGraph)
- [x] Enterprise backends (Neo4j, Milvus)
- [x] 187 tests
- [ ] Batch processing
- [ ] Advanced UI
- [ ] PyPI package

## Built By

[Ertugrul Akben](https://ertugrulakben.com) — AI & Systems Strategist

## License

[MIT](LICENSE)

---

<p align="center">
  <em>Because knowing "what" is not enough — you need to know "why."</em>
</p>
