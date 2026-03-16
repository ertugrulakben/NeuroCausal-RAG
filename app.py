"""
NeuroCausal RAG - Streamlit UI
Chatbot + Benchmark + Comparison + Case Study Interface

Author: Ertugrul Akben
"""

import os
import sys
import time
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Encoding fix
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Page config
st.set_page_config(
    page_title="NeuroCausal RAG v6.1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import visualization module
try:
    from neurocausal_rag.visualization import CausalGraphVisualizer
    VISUALIZATION_AVAILABLE = True
    VISUALIZATION_ERROR = None
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    VISUALIZATION_ERROR = str(e)

# Custom CSS - Modern Design
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #e91e63);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .result-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }

    .classic-card {
        border-left: 4px solid #ff6b6b;
    }

    .neuro-card {
        border-left: 4px solid #4ecdc4;
    }

    /* Chat bubbles */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
    }

    .bot-msg {
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Chain badge */
    .chain-badge {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem 0;
    }

    .injected-badge {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem 0;
    }

    /* Metric cards */
    .metric-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ecdc4;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.5rem;
    }

    /* Preset questions */
    .preset-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: all 0.3s;
        color: #ccc;
        width: 100%;
        text-align: left;
    }

    .preset-btn:hover {
        background: rgba(255,255,255,0.1);
        border-color: #4ecdc4;
    }

    /* Score bars */
    .score-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 8px;
        margin: 0.3rem 0;
    }

    .score-bar-classic {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8e8e 100%);
        height: 8px;
        border-radius: 10px;
    }

    .score-bar-neuro {
        background: linear-gradient(90deg, #4ecdc4 0%, #44e5db 100%);
        height: 8px;
        border-radius: 10px;
    }

    /* Comparison section */
    .compare-winner {
        background: linear-gradient(135deg, rgba(78,205,196,0.2) 0%, rgba(68,229,219,0.2) 100%);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 2px solid #4ecdc4;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def cosine_similarity(vec1, vec2):
    """Cosine similarity hesapla"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


class ClassicRAG:
    """Classic RAG: Sadece cosine similarity"""
    def __init__(self, embedding_engine):
        self.docs = {}
        self.embeddings = {}
        self.emb = embedding_engine

    def add(self, doc_id, content, metadata=None):
        self.docs[doc_id] = {'content': content, 'metadata': metadata or {}}
        self.embeddings[doc_id] = self.emb.get_text_embedding(content)

    def search(self, query, k=5):
        q_emb = self.emb.get_text_embedding(query)
        results = []
        for doc_id, doc_emb in self.embeddings.items():
            sim = cosine_similarity(q_emb, doc_emb)
            results.append({
                'id': doc_id,
                'content': self.docs[doc_id]['content'],
                'score': sim,
                'metadata': self.docs[doc_id]['metadata']
            })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]


@st.cache_resource
def load_systems(data_source='climate'):
    """Sistemleri yukle (cached)"""
    from neurocausal_rag.embedding.text import TextEmbedding
    from neurocausal_rag.core.graph import GraphEngine
    from neurocausal_rag.search.retriever import Retriever
    from neurocausal_rag.config import SearchConfig, IndexConfig, EmbeddingConfig, GraphConfig, LLMConfig
    from neurocausal_rag.learning.discovery import AutoCausalDiscovery

    # Embedding
    emb_config = EmbeddingConfig()
    embedding = TextEmbedding(emb_config)

    # Classic RAG
    classic_rag = ClassicRAG(embedding)

    # NeuroCausal RAG
    graph_config = GraphConfig()
    graph = GraphEngine(graph_config)

    search_config = SearchConfig(top_k=5, alpha=0.5, beta=0.3, gamma=0.2)
    index_config = IndexConfig(backend="brute_force")
    neuro_retriever = Retriever(
        graph=graph,
        embedding=embedding,
        config=search_config,
        index_config=index_config
    )

    # Load documents based on source
    docs = []

    try:
        from data.example_datasets import load_dataset
        docs = load_dataset(data_source)
    except Exception as e:
        st.warning(f"Veri seti yuklenemedi: {e}")
        # Fallback to climate if available
        try:
            from data.climate_knowledge_base import get_documents
            docs = get_documents()
            for doc in docs:
                doc['source'] = 'climate'
        except:
            pass


    # Add documents
    for doc in docs:
        emb = embedding.get_text_embedding(doc['content'])
        graph.add_node(
            doc['id'],
            doc['content'],
            emb,
            {'category': doc.get('category'), 'source': doc.get('source')}
        )
        classic_rag.add(doc['id'], doc['content'], {'source': doc.get('source')})

    # Auto-discover relations - Gelismis Semantic Discovery
    try:
        from neurocausal_rag.learning.semantic_discovery import enhanced_causal_discovery

        embeddings_array = np.array([embedding.get_text_embedding(d['content']) for d in docs])

        # Gelismis nedensellik kesfi
        relations = enhanced_causal_discovery(
            docs,
            embeddings_array,
            similarity_threshold=0.5,
            min_confidence=0.55
        )

        added_count = 0
        for rel in relations:
            src = rel.get('source')
            tgt = rel.get('target')
            if src and tgt and src != tgt:
                try:
                    graph.add_edge(
                        src, tgt,
                        rel.get('relation_type', 'related'),
                        rel['confidence']
                    )
                    added_count += 1
                except:
                    pass

        print(f"    Semantic Discovery: {added_count} iliski eklendi")
    except Exception as e:
        # Fallback to old method
        try:
            discovery = AutoCausalDiscovery()
            embeddings_array = np.array([embedding.get_text_embedding(d['content']) for d in docs])
            cross_relations = discovery.discover_from_corpus(docs, embeddings_array)

            for rel in cross_relations:
                if rel.get('confidence', 0) > 0.7:
                    src = rel.get('source')
                    tgt = rel.get('target')
                    if src and tgt and src != tgt:
                        try:
                            graph.add_edge(src, tgt, rel.get('relation_type', 'related'), rel['confidence'])
                        except:
                            pass
        except:
            pass

    # NOT: LLM discovery cok yavas - UI'da ayri butonla calistirilacak
    # Sadece embedding-based discovery kullaniyoruz (hizli)

    neuro_retriever.rebuild_index()

    return {
        'classic': classic_rag,
        'neuro': neuro_retriever,
        'graph': graph,
        'embedding': embedding,
        'docs': docs,
        'data_source': data_source
    }


def search_classic(systems, query, k=5):
    """Classic RAG arama"""
    start = time.time()
    results = systems['classic'].search(query, k)
    elapsed = time.time() - start
    return results, elapsed


def search_neuro(systems, query, k=5, alpha=None, beta=None, gamma=None):
    """NeuroCausal RAG arama (dinamik weight destekli + entity linking)"""
    start = time.time()
    results = systems['neuro'].search(query, k, alpha=alpha, beta=beta, gamma=gamma)
    elapsed = time.time() - start

    # Entity resolution info
    resolved_entities = []
    if results and results[0].resolved_entities:
        resolved_entities = results[0].resolved_entities

    # Format results
    formatted = []
    for r in results:
        item = {
            'id': r.node_id,
            'content': r.content,
            'score': r.score,
            'similarity': r.similarity_score,
            'causal': r.causal_score,
            'importance': r.importance_score,
            'chain': r.causal_chain,
            'injected': r.metadata.get('injected_from') if r.metadata else None,
            'metadata': r.metadata,
            'resolved_entities': r.resolved_entities  # Entity linking sonuclari
        }
        formatted.append(item)

    return formatted, elapsed, resolved_entities


def generate_answer(systems, query, context, is_neuro=False):
    """LLM ile cevap uret"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        return "API key bulunamadi. .env dosyasina OPENAI_API_KEY ekleyin.", 0, 0

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = """Sen yardimci bir asistansin. Verilen baglama gore sorulari cevapla.
Turkce cevap ver. Kisa ve oz cevaplar ver.
Eger baglamda bilgi yoksa, bilmedigini belirt."""

        if is_neuro:
            system_prompt += """
ONEMLI: Baglamda NEDENSEL ZINCIR bilgisi var. Bu zincirleri kullanarak
neden-sonuc iliskilerini acikla. Ornegin: "X, Y'ye neden olur cunku..."."""

        start = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Baglam:\n{context}\n\nSoru: {query}"}
            ],
            max_completion_tokens=500
        )
        elapsed = time.time() - start

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        return answer, elapsed, tokens
    except Exception as e:
        return f"Hata: {e}", 0, 0


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("# 🧠 NeuroCausal RAG")
    st.markdown("**v6.1** | Nedensel Bilgi Getirme + Multi-Hop + Memory")
    st.markdown("---")

    # Data source selection
    st.markdown("### 📁 Veri Seti Secimi")

    # Dataset options
    try:
        from data.example_datasets import DATASETS, get_dataset_raw_content
        dataset_options = list(DATASETS.keys())
        dataset_labels = {k: f"{v['icon']} {v['name']}" for k, v in DATASETS.items()}
    except:
        dataset_options = ["climate"]
        dataset_labels = {"climate": "🌍 Iklim Verileri"}
        DATASETS = {"climate": {"description": "Iklim degisikligi verileri", "doc_count": 115}}

    data_source = st.selectbox(
        "Veri seti:",
        dataset_options,
        format_func=lambda x: dataset_labels.get(x, x),
        key="dataset_selector"
    )

    # Show dataset info
    if data_source in DATASETS:
        ds_info = DATASETS[data_source]
        st.caption(f"{ds_info.get('description', '')}")
        st.caption(f"📄 {ds_info.get('doc_count', '?')} dokuman")

    # Show raw data button
    with st.expander("📄 Ham Veriyi Gor"):
        try:
            raw_content = get_dataset_raw_content(data_source)
            st.code(raw_content[:2000] + "..." if len(raw_content) > 2000 else raw_content, language="markdown")
        except Exception as e:
            st.warning(f"Ham veri yuklenemedi: {e}")

    st.markdown("---")

    # Preset questions - dataset'e gore dinamik
    st.markdown("### 💡 Hazir Sorular")

    # Dataset'e ozel sorular
    DATASET_QUESTIONS = {
        "climate": [
            "Sera gazlari nelerdir?",
            "Fosil yakitlarin iklim etkisi nedir?",
            "Buzul erimesi deniz seviyesini nasil etkiler?",
            "Cimento uretimi karbon emisyonuna nasil etki eder?",
            "Kuresel isinma yanginlari nasil artirir?",
        ],
        "supply_chain": [
            "Yangin urun lansmanini nasil etkiledi?",
            "HGM-X modulu neden kritik?",
            "Visionary Pro neden ertelendi?",
            "Tesis-7 ne uretiyordu?",
            "Tedarik zinciri riski nedir?",
        ],
        "secret_merger": [
            "Mavi Ufuk projesi nedir?",
            "Gunes Enerjisi A.S. satin almasi ne kadar?",
            "Yenilenebilir enerji yatirimi kim liderliginde?",
            "1.2 Milyar dolar ne icin ayrildi?",
            "Kod adi ile gercek isim arasindaki baglanti?",
        ],
        "legal_impact": [
            "DMA-2025 yasasi ne getirdi?",
            "Pazarlama performansi neden dustu?",
            "Reklam dönüsüm orani neden azaldi?",
            "Retargeting neden durduruldu?",
            "Avrupa satislari neden dustu?",
        ],
        "stress_chain": [
            "Stres is kazalarini nasil etkiler?",
            "Kortizol uykuyu nasil bozar?",
            "Uykusuzluk dikkati nasil etkiler?",
            "Dikkat eksikligi kaza riskini artirir mi?",
        ],
    }

    questions = DATASET_QUESTIONS.get(data_source, DATASET_QUESTIONS["climate"])

    # Hazir soru butonlari - tiklaninca direkt text_input'a yazilir
    for q in questions:
        if st.button(f"💬 {q[:35]}...", key=f"q_{hash(q)}", use_container_width=True):
            st.session_state['preset_query'] = q
            st.toast("Soru secildi! Chatbot sekmesine gidin.", icon="💬")
            st.rerun()  # Sayfayi yenile ki soru input'a yazilsin

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Ayarlar")
    num_results = st.slider("Sonuc sayisi", 3, 10, 5)
    show_scores = st.checkbox("Detayli skorlari goster", value=True)
    generate_llm = st.checkbox("LLM cevabi uret", value=True)

    st.markdown("---")

    # Search Mode - Hybrid Weights
    st.markdown("### 🎛️ Arama Modu")

    # Initialize session state for weights
    if 'search_alpha' not in st.session_state:
        st.session_state.search_alpha = 0.5
    if 'search_beta' not in st.session_state:
        st.session_state.search_beta = 0.3
    if 'search_gamma' not in st.session_state:
        st.session_state.search_gamma = 0.2

    # Preset buttons
    col_preset1, col_preset2 = st.columns(2)
    with col_preset1:
        if st.button("🔍 Ansiklopedi", use_container_width=True, help="Dogrudan cevap, yuksek benzerlik"):
            st.session_state.search_alpha = 0.7
            st.session_state.search_beta = 0.2
            st.session_state.search_gamma = 0.1
            st.rerun()
    with col_preset2:
        if st.button("🕵️ Dedektif", use_container_width=True, help="Gizli baglantilar, nedensel kesif"):
            st.session_state.search_alpha = 0.3
            st.session_state.search_beta = 0.5
            st.session_state.search_gamma = 0.2
            st.rerun()

    col_preset3, col_preset4 = st.columns(2)
    with col_preset3:
        if st.button("⚖️ Dengeli", use_container_width=True, help="Varsayilan mod"):
            st.session_state.search_alpha = 0.5
            st.session_state.search_beta = 0.3
            st.session_state.search_gamma = 0.2
            st.rerun()
    with col_preset4:
        if st.button("⭐ Hub Odakli", use_container_width=True, help="Onemli dokumanlar one cikar"):
            st.session_state.search_alpha = 0.4
            st.session_state.search_beta = 0.2
            st.session_state.search_gamma = 0.4
            st.rerun()

    # Manual sliders
    with st.expander("🔧 Gelismis Ayarlar", expanded=False):
        search_alpha = st.slider(
            "α Benzerlik",
            0.0, 1.0,
            st.session_state.search_alpha,
            0.1,
            key="slider_alpha",
            help="Vektor benzerlik agirligi"
        )
        search_beta = st.slider(
            "β Nedensellik",
            0.0, 1.0,
            st.session_state.search_beta,
            0.1,
            key="slider_beta",
            help="Nedensel iliski agirligi"
        )
        search_gamma = st.slider(
            "γ Onem",
            0.0, 1.0,
            st.session_state.search_gamma,
            0.1,
            key="slider_gamma",
            help="PageRank onem agirligi"
        )

        # Update session state
        st.session_state.search_alpha = search_alpha
        st.session_state.search_beta = search_beta
        st.session_state.search_gamma = search_gamma

        # Total check
        total = search_alpha + search_beta + search_gamma
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Toplam: {total:.1f} (1.0 olmali)")
        else:
            st.success(f"✅ Toplam: {total:.1f}")

        # Current formula display
        st.caption(f"**Formul:** {search_alpha:.1f}×Sim + {search_beta:.1f}×Causal + {search_gamma:.1f}×Imp")

    st.markdown("---")


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown('<h1 class="main-header">🧠 NeuroCausal RAG v6.1</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Nedensel Zincirlerle Zenginlestirilmis Bilgi Getirme Sistemi</p>', unsafe_allow_html=True)

# Load systems
with st.spinner("🔄 Sistem yukleniyor..."):
    systems = load_systems(data_source)

# System info
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{len(systems['docs'])}</div>
        <div class="metric-label">Dokuman</div>
    </div>
    """, unsafe_allow_html=True)
with col_info2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{systems['graph'].edge_count}</div>
        <div class="metric-label">Iliski</div>
    </div>
    """, unsafe_allow_html=True)
with col_info3:
    source_icon = "🌍"  # Climate data
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{source_icon}</div>
        <div class="metric-label">Kaynak</div>
    </div>
    """, unsafe_allow_html=True)
with col_info4:
    api_status = "✅" if os.environ.get("OPENAI_API_KEY", "").startswith("sk-") else "❌"
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{api_status}</div>
        <div class="metric-label">LLM</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SISTEM REHBERI ICIN DOKUMAN YUKLEME
# ============================================================================
@st.cache_data
def load_system_docs():
    """README, ARCHITECTURE ve CHANGELOG dosyalarini yukle"""
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))

    docs = {}
    files = ['README.md', 'ARCHITECTURE.md', 'CHANGELOG.md']

    for filename in files:
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                docs[filename] = f.read()
        except Exception:
            docs[filename] = ""

    return docs


def search_system_docs(query, docs, embedding_engine):
    """Sistem dokumanlarinda arama yap"""
    query_lower = query.lower()
    results = []

    # Her dokumani paragraflara bol ve ara
    for doc_name, content in docs.items():
        if not content:
            continue

        # ## basliklarla bol
        sections = content.split('\n## ')

        for i, section in enumerate(sections):
            if len(section.strip()) < 20:
                continue

            # Baslik ve icerik ayir
            lines = section.strip().split('\n')
            title = lines[0].replace('#', '').strip() if lines else "Bolum"
            body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else section

            # Basit anahtar kelime eslesmesi
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2:
                    if word in section.lower():
                        score += 0.3
                    if word in title.lower():
                        score += 0.5

            # Ozel anahtar kelime eslesmesi
            keywords = {
                'nedir': ['nedir', 'ne yapar', 'ozellikleri', 'temel'],
                'nasil': ['nasil', 'kurulum', 'baslangic', 'adim'],
                'mimari': ['mimari', 'architecture', 'yapi', 'sistem'],
                'versiyon': ['versiyon', 'version', 'changelog', 'yeni'],
                'api': ['api', 'endpoint', 'rest', 'fastapi'],
                'arama': ['arama', 'search', 'retriever', 'sorgu'],
                'graf': ['graf', 'graph', 'iliski', 'nedensel'],
                'discovery': ['discovery', 'kesif', 'otomatik'],
                'multi-hop': ['multi-hop', 'hop', 'atlama'],
                'memory': ['memory', 'hafiza', 'not', 'geri bildirim'],
            }

            for key, words in keywords.items():
                if key in query_lower:
                    for w in words:
                        if w in section.lower():
                            score += 0.2

            if score > 0.2:
                results.append({
                    'doc': doc_name,
                    'title': title[:50],
                    'content': body[:500],
                    'score': min(score, 1.0)
                })

    # Skorla sirala ve en iyi 3 sonuc dondur
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    return results


def generate_system_answer(query, results):
    """Sistem hakkinda yanit olustur"""
    if not results:
        return "Bu konuda bilgi bulunamadi. Lutfen farkli bir soru sorun veya Hakkinda sekmesine bakin."

    # En iyi sonuclari birlestir
    answer_parts = []

    for r in results:
        doc_name = r['doc'].replace('.md', '')
        answer_parts.append(f"**{doc_name} - {r['title']}:**\n{r['content'][:300]}...")

    return '\n\n---\n\n'.join(answer_parts)


# Tabs (7 tab: Sistem Rehberi + 6 mevcut)
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🤖 Sistem Rehberi", "💬 Chatbot", "📊 Karsilastirma", "🕸️ Graf", "🧪 Case Study", "🔧 Yonetim", "ℹ️ Hakkinda"])

# ============================================================================
# TAB 0: SISTEM REHBERI - Tanitim Botu
# ============================================================================
with tab0:
    st.markdown("### 🤖 NeuroCausal RAG Sistem Rehberi")
    st.markdown("*Sistemi tanimak icin sorular sorun! README, ARCHITECTURE ve CHANGELOG bilgilerine hakimim.*")

    # Chat gecmisi
    if 'system_chat_history' not in st.session_state:
        st.session_state.system_chat_history = []

    # Sistem dokumanlarini yukle
    sys_docs = load_system_docs()

    def run_system_search(query):
        """Sistem dokumanlarda ara ve sonuc ekle"""
        results = search_system_docs(query, sys_docs, None)
        answer = generate_system_answer(query, results)
        st.session_state.system_chat_history.append({
            'question': query,
            'answer': answer,
            'sources': [r['doc'] for r in results] if results else ['Bilgi bulunamadi']
        })

    # Ornek sorular
    st.markdown("**Ornek sorular:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        if st.button("NeuroCausal RAG nedir?", key="sys_q1", use_container_width=True):
            run_system_search("NeuroCausal RAG nedir?")
            st.rerun()
    with col_q2:
        if st.button("Yeni ozellikler neler?", key="sys_q2", use_container_width=True):
            run_system_search("v6.1 yeni ozellikler neler?")
            st.rerun()
    with col_q3:
        if st.button("API nasil kullanilir?", key="sys_q3", use_container_width=True):
            run_system_search("API nasil kullanilir?")
            st.rerun()

    col_q4, col_q5, col_q6 = st.columns(3)
    with col_q4:
        if st.button("Multi-hop nedir?", key="sys_q4", use_container_width=True):
            run_system_search("Multi-hop retrieval nedir?")
            st.rerun()
    with col_q5:
        if st.button("Mimari nasil?", key="sys_q5", use_container_width=True):
            run_system_search("Sistem mimarisi nasil?")
            st.rerun()
    with col_q6:
        if st.button("Memory sistemi?", key="sys_q6", use_container_width=True):
            run_system_search("Memory hafiza sistemi nedir?")
            st.rerun()

    st.divider()

    # Sorgu alani - Form kullan
    with st.form(key="system_query_form"):
        system_query = st.text_input(
            "Sistem hakkinda sorunuzu yazin:",
            placeholder="Ornek: NeuroCausal RAG ne ise yarar?",
            key="system_query_input"
        )
        submit_btn = st.form_submit_button("🔍 Sor", use_container_width=True)

        if submit_btn and system_query.strip():
            run_system_search(system_query.strip())
            st.rerun()

    # Chat gecmisini goster
    if st.session_state.system_chat_history:
        st.markdown("---")
        st.markdown("### Sohbet Gecmisi")
        for chat in reversed(st.session_state.system_chat_history[-5:]):
            st.markdown(f"""
            <div class="user-msg">
                <strong>Siz:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="bot-msg">
                <strong>Sistem Rehberi:</strong><br>{chat['answer']}
                <br><br><small>📚 Kaynaklar: {', '.join(chat['sources'])}</small>
            </div>
            """, unsafe_allow_html=True)

        # Gecmisi temizle butonu
        if st.button("🗑️ Gecmisi Temizle", key="clear_sys_history"):
            st.session_state.system_chat_history = []
            st.rerun()
    else:
        st.info("👆 Yukardaki ornek sorulardan birini secin veya kendi sorunuzu yazin!")

# ============================================================================
# TAB 1: CHATBOT
# ============================================================================
with tab1:
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Query input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        # Preset sorular veya manuel giris
        default_query = st.session_state.pop('preset_query', '')  # pop ile al ve sil
        query = st.text_input(
            "Sorunuzu yazin:",
            value=default_query,
            placeholder="Ornegin: Sera gazlari nelerdir?",
            label_visibility="collapsed",
            key="chatbot_query_input"
        )
    with col_btn:
        search_btn = st.button("🔍 Ara", type="primary", use_container_width=True)

    if search_btn and query:
        # Add to history
        st.session_state.chat_history.append({'role': 'user', 'content': query})

        # Search with dynamic weights
        with st.spinner("Araniyor..."):
            neuro_results, neuro_time, resolved_entities = search_neuro(
                systems, query, num_results,
                alpha=st.session_state.search_alpha,
                beta=st.session_state.search_beta,
                gamma=st.session_state.search_gamma
            )

        # Build context with chain info
        context_parts = []
        for i, r in enumerate(neuro_results[:3], 1):
            ctx = f"[{i}] {r['content'][:300]}"
            if r['chain'] and len(r['chain']) > 1:
                chain_texts = []
                for nid in r['chain'][1:3]:
                    node = systems['graph'].get_node(nid)
                    if node:
                        chain_texts.append(node['content'][:150])
                if chain_texts:
                    ctx += f"\n   NEDENSEL ZINCIR: {' -> '.join(chain_texts)}"
            context_parts.append(ctx)
        context = "\n\n".join(context_parts)

        # Generate answer
        if generate_llm:
            with st.spinner("LLM cevap uretiyor..."):
                answer, llm_time, tokens = generate_answer(systems, query, context, is_neuro=True)
        else:
            answer = f"Bulunan sonuclar:\n" + "\n".join([f"- {r['content'][:150]}..." for r in neuro_results[:3]])
            llm_time, tokens = 0, 0

        # Celiski kontrolu (FAZ 1.2)
        contradictions = []
        try:
            contradictions = systems['neuro'].check_contradictions_in_results(neuro_results[:5])
        except Exception as e:
            pass  # Celiski kontrolu basarisiz olsa da devam et

        # Add to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer,
            'results': neuro_results,
            'time': neuro_time,
            'llm_time': llm_time,
            'tokens': tokens,
            'resolved_entities': resolved_entities,  # Entity linking sonuclari
            'contradictions': contradictions  # Celiski tespiti
        })

    # Display chat history
    st.markdown("---")
    for msg in st.session_state.chat_history[-10:]:  # Last 10 messages
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

            # ENTITY LINKING BILGISI
            if msg.get('resolved_entities') and len(msg['resolved_entities']) > 0:
                entities_text = ", ".join(msg['resolved_entities'][:3])
                st.markdown(f"""
                <div style="background: rgba(255,193,7,0.15); padding: 8px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #ffc107;">
                    <small>🔗 <b>Entity Linking:</b> {entities_text}</small>
                </div>
                """, unsafe_allow_html=True)

            # CELISKI UYARISI (FAZ 1.2)
            if msg.get('contradictions') and len(msg['contradictions']) > 0:
                for contr in msg['contradictions'][:2]:  # Max 2 celiski goster
                    st.markdown(f"""
                    <div style="background: rgba(255,87,34,0.15); padding: 8px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #ff5722;">
                        <small>⚠️ <b>Celiski Tespit Edildi:</b> [{contr['doc1_id']}] vs [{contr['doc2_id']}]</small><br>
                        <small>{contr['details']}</small>
                    </div>
                    """, unsafe_allow_html=True)

            # CITATIONS - Kaynak Gosterme (Her zaman gorunur)
            if 'results' in msg and msg['results']:
                st.markdown("**📚 Kaynaklar (Bu cevap su dokumanlara dayanmaktadir):**")
                citations_cols = st.columns(min(3, len(msg['results'][:3])))
                for idx, r in enumerate(msg['results'][:3]):
                    with citations_cols[idx]:
                        conf_color = "🟢" if r['score'] > 0.7 else "🟡" if r['score'] > 0.4 else "🔴"
                        chain_icon = "🔗" if r['chain'] and len(r['chain']) > 1 else ""
                        st.markdown(f"""
                        <div style="background: rgba(78,205,196,0.1); padding: 8px; border-radius: 8px; border-left: 3px solid #4ecdc4;">
                            <small><b>[{idx+1}] {r['id']}</b> {chain_icon}</small><br>
                            <small>{conf_color} Guven: {r['score']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)

                # Detailed results in expander
                with st.expander(f"📄 Detayli Dokumanlar ({len(msg['results'])} sonuc)"):
                    for i, r in enumerate(msg['results'], 1):
                        col_r1, col_r2 = st.columns([4, 1])
                        with col_r1:
                            st.markdown(f"**[{i}] {r['id']}**")
                            st.write(r['content'][:200] + "..." if len(r['content']) > 200 else r['content'])
                        with col_r2:
                            st.metric("Skor", f"{r['score']:.3f}")
                            if r['chain'] and len(r['chain']) > 1:
                                st.markdown('<span class="chain-badge">📊 Zincir</span>', unsafe_allow_html=True)
                            if r['injected']:
                                st.markdown('<span class="injected-badge">💉 Enjekte</span>', unsafe_allow_html=True)
                        st.markdown("---")

                st.caption(f"⏱️ Arama: {msg['time']*1000:.0f}ms | LLM: {msg['llm_time']:.2f}s | 🎫 {msg['tokens']} token")

    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Gecmisi Temizle"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# TAB 2: KARSILASTIRMA
# ============================================================================
with tab2:
    st.markdown("### 📊 Classic RAG vs NeuroCausal RAG")
    st.markdown("Ayni sorgu icin iki sistemi yan yana karsilastirin")

    # Form kullanarak tek tikla calismasini sagla
    with st.form(key="compare_form"):
        compare_query = st.text_input(
            "Karsilastirma sorgusu:",
            value="Sera gazlari kuresel isinmaya nasil neden olur?",
        )
        compare_submitted = st.form_submit_button("🔬 Karsilastir", type="primary")

    if compare_submitted and compare_query:
        col1, col2 = st.columns(2)

        # Classic RAG
        with col1:
            st.markdown("### 📕 Classic RAG")
            with st.spinner("Araniyor..."):
                classic_results, classic_time = search_classic(systems, compare_query, num_results)

            st.metric("Arama Suresi", f"{classic_time*1000:.0f} ms")

            for i, r in enumerate(classic_results, 1):
                st.markdown(f"""
                <div class="result-card classic-card">
                    <strong>[{i}] {r['id']}</strong><br>
                    <small>Skor: {r['score']:.3f}</small><br>
                    <p>{r['content'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)

            # LLM answer
            if generate_llm:
                context = "\n".join([r['content'][:200] for r in classic_results[:3]])
                with st.spinner("LLM cevap uretiyor..."):
                    answer, llm_time, tokens = generate_answer(systems, compare_query, context)
                st.markdown("#### 🤖 Cevap")
                st.info(answer)
                st.caption(f"⏱️ {llm_time:.2f}s | 🎫 {tokens} token")

        # NeuroCausal RAG
        with col2:
            st.markdown("### 📗 NeuroCausal RAG")
            with st.spinner("Araniyor..."):
                neuro_results, neuro_time, _ = search_neuro(systems, compare_query, num_results)

            st.metric("Arama Suresi", f"{neuro_time*1000:.0f} ms")

            for i, r in enumerate(neuro_results, 1):
                badges = ""
                if r['chain'] and len(r['chain']) > 1:
                    badges += '<span class="chain-badge">📊 Zincir</span> '
                if r['injected']:
                    badges += '<span class="injected-badge">💉 Enjekte</span>'

                scores_html = ""
                if show_scores:
                    scores_html = f"<small>Sim: {r['similarity']:.2f} | Causal: {r['causal']:.2f} | Imp: {r['importance']:.3f}</small><br>"

                st.markdown(f"""
                <div class="result-card neuro-card">
                    <strong>[{i}] {r['id']}</strong> {badges}<br>
                    <small>Skor: {r['score']:.3f}</small><br>
                    {scores_html}
                    <p>{r['content'][:200]}...</p>
                </div>
                """, unsafe_allow_html=True)

            # LLM answer with chains
            if generate_llm:
                context_parts = []
                for r in neuro_results[:3]:
                    ctx = r['content'][:200]
                    if r['chain'] and len(r['chain']) > 1:
                        for nid in r['chain'][1:3]:
                            node = systems['graph'].get_node(nid)
                            if node:
                                ctx += f"\n[Nedensel: {node['content'][:100]}]"
                    context_parts.append(ctx)
                context = "\n".join(context_parts)

                with st.spinner("LLM cevap uretiyor..."):
                    answer, llm_time, tokens = generate_answer(systems, compare_query, context, is_neuro=True)
                st.markdown("#### 🤖 Cevap")
                st.success(answer)
                st.caption(f"⏱️ {llm_time:.2f}s | 🎫 {tokens} token")

        # Comparison summary
        st.markdown("---")
        st.markdown("### 📈 Karsilastirma Ozeti")

        comp1, comp2, comp3, comp4 = st.columns(4)

        with comp1:
            time_diff = classic_time - neuro_time
            winner = "NeuroCausal" if time_diff > 0 else "Classic"
            st.metric("Daha Hizli", winner, f"{abs(time_diff)*1000:.0f} ms")

        with comp2:
            chain_count = sum(1 for r in neuro_results if r['chain'] and len(r['chain']) > 1)
            st.metric("Nedensel Zincir", f"{chain_count}/{len(neuro_results)}")

        with comp3:
            injected_count = sum(1 for r in neuro_results if r['injected'])
            st.metric("Enjekte Edilen", f"{injected_count}/{len(neuro_results)}")

        with comp4:
            avg_neuro = np.mean([r['score'] for r in neuro_results]) if neuro_results else 0
            avg_classic = np.mean([r['score'] for r in classic_results]) if classic_results else 0
            st.metric("Ort. Skor Fark", f"{(avg_neuro - avg_classic):.3f}")

        # Visual Score Comparison
        st.markdown("### 📊 Skor Karsilastirmasi")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <div style="flex: 1; text-align: center;">
                    <span style="color: #ff6b6b; font-size: 1.5rem; font-weight: bold;">Classic RAG</span>
                </div>
                <div style="flex: 1; text-align: center;">
                    <span style="color: #4ecdc4; font-size: 1.5rem; font-weight: bold;">NeuroCausal RAG</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Score bars for each result
        max_results = min(len(classic_results), len(neuro_results), 5)
        for i in range(max_results):
            c_score = classic_results[i]['score'] if i < len(classic_results) else 0
            n_score = neuro_results[i]['score'] if i < len(neuro_results) else 0
            c_width = int(c_score * 100)
            n_width = int(n_score * 100)

            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="flex: 1; text-align: right;">
                        <div style="background: rgba(255,107,107,0.3); height: 25px; border-radius: 5px; display: flex; justify-content: flex-end; align-items: center;">
                            <div style="background: linear-gradient(90deg, #ff8e8e, #ff6b6b); width: {c_width}%; height: 100%; border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-size: 0.8rem; font-weight: bold;">{c_score:.2f}</span>
                            </div>
                        </div>
                    </div>
                    <div style="width: 30px; text-align: center; color: #888;">[{i+1}]</div>
                    <div style="flex: 1;">
                        <div style="background: rgba(78,205,196,0.3); height: 25px; border-radius: 5px; display: flex; align-items: center;">
                            <div style="background: linear-gradient(90deg, #4ecdc4, #44e5db); width: {n_width}%; height: 100%; border-radius: 5px; display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-size: 0.8rem; font-weight: bold;">{n_score:.2f}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # SEMANTIC WINNER DETERMINATION
        # Sadece ham skora degil, anlamsal degere bakiyoruz
        st.markdown("### 🧠 Anlamsal Degerlendirme")

        # Classic RAG puani (sadece benzerlik)
        classic_semantic_score = sum(r['score'] for r in classic_results) / len(classic_results) if classic_results else 0

        # NeuroCausal RAG puani (benzerlik + nedensellik bonuslari)
        neuro_base_score = sum(r['score'] for r in neuro_results) / len(neuro_results) if neuro_results else 0
        chain_bonus = chain_count * 0.15  # Her zincir icin %15 bonus
        injection_bonus = injected_count * 0.10  # Her enjeksiyon icin %10 bonus
        causal_bonus = sum(r['causal'] for r in neuro_results) / len(neuro_results) * 0.20 if neuro_results else 0  # Nedensellik skoru bonusu

        neuro_semantic_score = neuro_base_score + chain_bonus + injection_bonus + causal_bonus

        # Gorsellestir
        col_sem1, col_sem2 = st.columns(2)
        with col_sem1:
            st.markdown(f"""
            <div style="background: rgba(255,107,107,0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #ff6b6b;">
                <h4 style="color: #ff6b6b;">📕 Classic RAG</h4>
                <p><b>Temel Skor:</b> {classic_semantic_score:.3f}</p>
                <p><b>Zincir Bonusu:</b> 0.000</p>
                <p><b>Enjeksiyon Bonusu:</b> 0.000</p>
                <p><b>Nedensellik Bonusu:</b> 0.000</p>
                <hr>
                <p style="font-size: 1.3rem;"><b>TOPLAM: {classic_semantic_score:.3f}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_sem2:
            st.markdown(f"""
            <div style="background: rgba(78,205,196,0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #4ecdc4;">
                <h4 style="color: #4ecdc4;">📗 NeuroCausal RAG</h4>
                <p><b>Temel Skor:</b> {neuro_base_score:.3f}</p>
                <p><b>Zincir Bonusu:</b> +{chain_bonus:.3f} ({chain_count} zincir)</p>
                <p><b>Enjeksiyon Bonusu:</b> +{injection_bonus:.3f} ({injected_count} enjekte)</p>
                <p><b>Nedensellik Bonusu:</b> +{causal_bonus:.3f}</p>
                <hr>
                <p style="font-size: 1.3rem;"><b>TOPLAM: {neuro_semantic_score:.3f}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Winner announcement (SEMANTIC)
        score_diff = neuro_semantic_score - classic_semantic_score
        if score_diff > 0.05:  # NeuroCausal en az %5 daha iyi
            st.markdown(f"""
            <div class="compare-winner">
                <h3>🏆 NeuroCausal RAG Kazandi!</h3>
                <p>Anlamsal skor: <b>{neuro_semantic_score:.3f}</b> vs <b>{classic_semantic_score:.3f}</b> (+{score_diff:.3f})</p>
                <p>💡 <i>Nedensel zincirleri kesfederek daha kapsamli sonuclar buldu!</i></p>
            </div>
            """, unsafe_allow_html=True)
        elif score_diff < -0.05:  # Classic en az %5 daha iyi
            st.markdown(f"""
            <div style="background: rgba(255,107,107,0.2); border-radius: 15px; padding: 1rem; text-align: center; border: 2px solid #ff6b6b;">
                <h3>📕 Classic RAG Kazandi</h3>
                <p>Anlamsal skor: <b>{classic_semantic_score:.3f}</b> vs <b>{neuro_semantic_score:.3f}</b></p>
                <p>💡 <i>Bu sorguda dogrudan kelime eslesmesi daha etkili oldu.</i></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 1rem; text-align: center; border: 2px solid #888;">
                <h3>🤝 Yakin Sonuc</h3>
                <p>Anlamsal skorlar: <b>{neuro_semantic_score:.3f}</b> vs <b>{classic_semantic_score:.3f}</b></p>
                <p>💡 <i>Her iki sistem de benzer performans gosterdi.</i></p>
            </div>
            """, unsafe_allow_html=True)

        # Aciklama
        with st.expander("❓ Anlamsal Puanlama Nasil Calisir?"):
            st.markdown("""
            **Anlamsal Puanlama Sistemi:**

            Classic RAG sadece **kelime benzerligine** bakar. NeuroCausal RAG ise ek deger saglar:

            | Bonus | Aciklama | Deger |
            |-------|----------|-------|
            | 🔗 Zincir | Her bulunan nedensel zincir | +0.15 |
            | 💉 Enjeksiyon | Her enjekte edilen dokuman | +0.10 |
            | 📊 Nedensellik | Ortalama nedensellik skoru | +%20 |

            **Ornek:**
            - Stres → Kortizol → Uyku Bozuklugu → Dikkat → Is Kazasi zincirini bulmak
            - Kelime eslesmesi olmadan iliskili dokumanlari getirmek
            - Gizli baglantilari kesfetmek

            Bu bonuslar sayesinde, ham skor dusuk olsa bile **gercek deger** olculur.
            """)


# ============================================================================
# TAB 3: GRAF GORSELLESTIRME
# ============================================================================
with tab3:
    st.markdown("### 🕸️ Nedensel Graf Gorsellestirme")

    if not VISUALIZATION_AVAILABLE:
        st.error(f"⚠️ Gorsellestirme modulu yuklenemedi: {VISUALIZATION_ERROR}")
        st.info("💡 Cozum: `run_app.bat` ile baslatmayi deneyin veya PYTHONPATH ayarlayin.")
    else:
        # Create visualizer
        visualizer = CausalGraphVisualizer(systems['graph'])

        # View mode selection
        view_mode = st.radio(
            "Gorunum Modu:",
            ["📊 Arama Sonucu", "🌐 Tam Graf", "🔗 Zincir Takibi"],
            horizontal=True
        )

        # Show legend
        st.markdown(visualizer.get_legend_html(), unsafe_allow_html=True)

        if view_mode == "📊 Arama Sonucu":
            st.markdown("---")
            col_query, col_btn = st.columns([4, 1])
            with col_query:
                viz_query = st.text_input(
                    "Sorgu:",
                    value="Sera gazlari kuresel isinmaya nasil neden olur?",
                    key="viz_query",
                    label_visibility="collapsed"
                )
            with col_btn:
                viz_search_btn = st.button("🔍 Ara ve Gorsellestir", type="primary")

            if viz_search_btn and viz_query:
                with st.spinner("Arama ve gorsellestirme..."):
                    # Search
                    results, search_time, _ = search_neuro(
                        systems, viz_query, 5,
                        alpha=st.session_state.search_alpha,
                        beta=st.session_state.search_beta,
                        gamma=st.session_state.search_gamma
                    )

                    # Get chains for visualization
                    chains = {}
                    for r in results:
                        if r['chain']:
                            chains[r['id']] = r['chain']

                    # Create result objects for visualizer
                    class ResultObj:
                        def __init__(self, d):
                            self.node_id = d['id']
                            self.score = d['score']
                            self.metadata = d.get('metadata', {})

                    result_objs = [ResultObj(r) for r in results]

                    # Visualize
                    html = visualizer.visualize_search_results(
                        query=viz_query,
                        results=result_objs,
                        chains=chains
                    )

                    st.caption(f"⏱️ Arama suresi: {search_time*1000:.0f}ms | Sonuc: {len(results)}")
                    components.html(html, height=650, scrolling=True)

                    # Show results list
                    with st.expander("📄 Bulunan Dokumanlar"):
                        for i, r in enumerate(results, 1):
                            badges = ""
                            if r['chain'] and len(r['chain']) > 1:
                                badges += "🔗 "
                            if r['injected']:
                                badges += "💉 "
                            st.markdown(f"**[{i}] {r['id']}** {badges} - Skor: {r['score']:.3f}")

        elif view_mode == "🌐 Tam Graf":
            st.markdown("---")
            max_nodes = st.slider("Maksimum node sayisi:", 10, 200, 50)

            if st.button("🌐 Grafi Goster", type="primary"):
                with st.spinner("Graf olusturuluyor..."):
                    html = visualizer.visualize_full_graph(max_nodes=max_nodes)
                    components.html(html, height=700, scrolling=True)

                st.info(f"📊 Toplam: {systems['graph'].node_count} node, {systems['graph'].edge_count} edge")

        elif view_mode == "🔗 Zincir Takibi":
            st.markdown("---")
            st.markdown("Iki dokuman arasindaki nedensel yolu gorsellestirin.")

            # Get node list
            node_ids = list(systems['graph'].nodes.keys())

            col_src, col_tgt = st.columns(2)
            with col_src:
                source_node = st.selectbox(
                    "Baslangic Node:",
                    node_ids[:50],  # First 50 for performance
                    index=0,
                    key="chain_source"
                )
            with col_tgt:
                target_node = st.selectbox(
                    "Hedef Node (opsiyonel):",
                    ["(Forward chain)"] + node_ids[:50],
                    index=0,
                    key="chain_target"
                )

            if st.button("🔗 Zinciri Goster", type="primary"):
                with st.spinner("Zincir hesaplaniyor..."):
                    if target_node == "(Forward chain)":
                        html = visualizer.visualize_causal_chain(source_node)
                        chain = systems['graph'].get_causal_chain(source_node, max_depth=4)
                    else:
                        html = visualizer.visualize_causal_chain(source_node, target_node)
                        chain, score = systems['graph'].find_causal_path(source_node, target_node)

                    if chain:
                        st.success(f"🔗 Zincir: {' → '.join(chain)}")
                        components.html(html, height=400, scrolling=True)
                    else:
                        st.warning("⚠️ Bu iki node arasinda yol bulunamadi.")


# ============================================================================
# TAB 4: CASE STUDY
# ============================================================================
with tab4:
    st.markdown("### 🧪 Case Study: Gorunmez Baglantilari Yakalamak")
    st.markdown("""
    Bu bolumde NeuroCausal RAG'in **gizli nedensellik iliskilerini** nasil kesfettigini gorebilirsiniz.
    Hazir test sorgulari ile sistemin gucunu test edin.
    """)

    st.markdown("---")

    # Case Study Selection
    case_studies = {
        "🏭 Cimento Testi": {
            "query": "Sera gazlari kuresel isinmaya nasil neden olur?",
            "expected": "cimento_uretimi",
            "explanation": """
            **Beklenen:** `cimento_uretimi` dokumani bulunmali

            **Neden onemli?** Sorguda "cimento" kelimesi yok! Ama NeuroCausal RAG,
            nedensel zinciri kesfederek bu dokumani buluyor:

            ```
            Cimento Uretimi → CO2 Salimi → Sera Gazi → Kuresel Isinma
            ```

            Classic RAG bu dokumani bulamaz cunku kelime eslesmesi yok.
            """
        },
        "🧊 Buzul Testi": {
            "query": "Buzul erimesi deniz seviyesini nasil etkiler?",
            "expected": "buzul_erimesi",
            "explanation": """
            **Beklenen:** `buzul_erimesi` veya `kriyosfer` dokumanlari ilk siralarda

            **Bu test:** Dogrudan eslesen dokumanlarin one cikmasi gerekir.
            Yeni skor dengelemesi (alpha=0.5) ile bu test gecmeli.
            """
        },
        "🌡️ Isı Dalgası Testi": {
            "query": "Kuresel isinma saglik sorunlarina nasil yol acar?",
            "expected": "isi_dalgasi",
            "explanation": """
            **Beklenen:** `isi_dalgasi` veya `saglik_etkileri` dokumanlari

            **Zincir:** Kuresel Isinma → Isi Dalgasi → Saglik Sorunlari
            """
        },
        "🌾 Gıda Güvenliği Testi": {
            "query": "Iklim degisikligi tarim uretimini nasil etkiler?",
            "expected": "kuraklik",
            "explanation": """
            **Beklenen:** `kuraklik`, `tarim_verimi` veya `gida_guvenligi` dokumanlari

            **Zincir:** Iklim Degisikligi → Kuraklik → Tarim → Gida Guvenligi
            """
        }
    }

    selected_case = st.selectbox(
        "Test secin:",
        list(case_studies.keys()),
        index=0
    )

    case = case_studies[selected_case]

    # Show explanation
    with st.expander("📖 Bu test ne yapiyor?", expanded=True):
        st.markdown(case["explanation"])

    # Run test button
    if st.button("🚀 Testi Calistir", type="primary", use_container_width=True):
        st.markdown("---")

        col_classic, col_neuro = st.columns(2)

        with col_classic:
            st.markdown("### 📕 Classic RAG")
            with st.spinner("Araniyor..."):
                classic_results, classic_time = search_classic(systems, case["query"], 5)

            st.metric("Arama Suresi", f"{classic_time*1000:.0f} ms")

            found_expected = False
            for i, r in enumerate(classic_results, 1):
                is_expected = case["expected"] in r['id'].lower()
                if is_expected:
                    found_expected = True

                badge = "✅" if is_expected else ""
                st.markdown(f"""
                <div class="result-card classic-card">
                    <strong>[{i}] {r['id']}</strong> {badge}<br>
                    <small>Benzerlik: {r['score']:.3f}</small><br>
                    <p style="font-size: 0.85rem;">{r['content'][:150]}...</p>
                </div>
                """, unsafe_allow_html=True)

            if found_expected:
                st.success(f"✅ `{case['expected']}` bulundu!")
            else:
                st.error(f"❌ `{case['expected']}` bulunamadi!")

        with col_neuro:
            st.markdown("### 📗 NeuroCausal RAG")
            with st.spinner("Araniyor..."):
                neuro_results, neuro_time, _ = search_neuro(systems, case["query"], 5)

            st.metric("Arama Suresi", f"{neuro_time*1000:.0f} ms")

            found_expected = False
            for i, r in enumerate(neuro_results, 1):
                is_expected = case["expected"] in r['id'].lower()
                if is_expected:
                    found_expected = True

                badges = "✅ " if is_expected else ""
                if r['chain'] and len(r['chain']) > 1:
                    badges += "🔗 "
                if r['injected']:
                    badges += "💉 "

                st.markdown(f"""
                <div class="result-card neuro-card">
                    <strong>[{i}] {r['id']}</strong> {badges}<br>
                    <small>Skor: {r['score']:.3f} | Sim: {r['similarity']:.2f} | Causal: {r['causal']:.2f}</small><br>
                    <p style="font-size: 0.85rem;">{r['content'][:150]}...</p>
                </div>
                """, unsafe_allow_html=True)

                # Show chain if exists
                if r['chain'] and len(r['chain']) > 1:
                    chain_str = " → ".join(r['chain'][:4])
                    st.caption(f"🔗 Zincir: {chain_str}")

            if found_expected:
                st.success(f"✅ `{case['expected']}` bulundu!")
            else:
                st.warning(f"⚠️ `{case['expected']}` ilk 5'te yok (Chain Injection ile gelebilir)")

        # Summary
        st.markdown("---")
        st.markdown("### 📊 Test Sonucu")

        sum1, sum2, sum3 = st.columns(3)
        with sum1:
            time_winner = "NeuroCausal" if neuro_time < classic_time else "Classic"
            st.metric("Daha Hizli", time_winner, f"{abs(classic_time - neuro_time)*1000:.0f} ms")
        with sum2:
            chain_count = sum(1 for r in neuro_results if r['chain'] and len(r['chain']) > 1)
            st.metric("Zincir Sayisi", f"{chain_count}/5")
        with sum3:
            max_causal = max((r['causal'] for r in neuro_results), default=0)
            st.metric("Max Nedensellik", f"{max_causal:.2f}")

    st.markdown("---")

    # Quick benchmark
    st.markdown("### ⚡ Hizli Benchmark")
    if st.button("🔬 Tum Testleri Calistir", use_container_width=True):
        results_table = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (name, case) in enumerate(case_studies.items()):
            status_text.text(f"Test ediliyor: {name}...")
            progress_bar.progress((idx + 1) / len(case_studies))

            # Classic
            classic_results, classic_time = search_classic(systems, case["query"], 5)
            classic_found = any(case["expected"] in r['id'].lower() for r in classic_results)

            # Neuro
            neuro_results, neuro_time, _ = search_neuro(systems, case["query"], 5)
            neuro_found = any(case["expected"] in r['id'].lower() for r in neuro_results)

            results_table.append({
                "Test": name,
                "Beklenen": case["expected"],
                "Classic": "✅" if classic_found else "❌",
                "NeuroCausal": "✅" if neuro_found else "❌",
                "Classic (ms)": f"{classic_time*1000:.0f}",
                "Neuro (ms)": f"{neuro_time*1000:.0f}",
            })

        progress_bar.empty()
        status_text.empty()

        # Show results table
        import pandas as pd
        df = pd.DataFrame(results_table)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary
        classic_wins = sum(1 for r in results_table if r["Classic"] == "✅")
        neuro_wins = sum(1 for r in results_table if r["NeuroCausal"] == "✅")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Classic RAG Basari", f"{classic_wins}/{len(case_studies)}")
        with col_s2:
            st.metric("NeuroCausal RAG Basari", f"{neuro_wins}/{len(case_studies)}")


# ============================================================================
# TAB 5: YONETIM (Admin Panel)
# ============================================================================
with tab5:
    st.markdown("### 🔧 Graf Yonetimi & Iliski Duzenleme")
    st.markdown("""
    **Neden Onemli?** Yapay zeka hata yapabilir, ama bu sistemde hatayi gorup duzeltebilirsiniz.
    Diger sistemlerde bu sansiniz yok!
    """)

    st.markdown("---")

    # Stats
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Toplam Dokuman", systems['graph'].node_count)
    with col_stat2:
        st.metric("Toplam Iliski", systems['graph'].edge_count)
    with col_stat3:
        # Calculate average confidence
        edges = list(systems['graph'].edges.values()) if hasattr(systems['graph'], 'edges') else []
        if edges:
            avg_conf = sum(e.get('weight', 0.5) for e in edges) / len(edges)
            st.metric("Ort. Guven", f"{avg_conf:.2f}")
        else:
            st.metric("Ort. Guven", "N/A")

    st.markdown("---")

    # Sub-tabs for management
    admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5, admin_tab6 = st.tabs(["📋 Iliskileri Gor", "🗑️ Iliski Sil", "➕ Iliski Ekle", "🔗 Entity Linking", "🧠 Reasoning", "🧠 Hafiza"])

    with admin_tab1:
        st.markdown("#### Mevcut Iliskiler")

        # Get all edges
        try:
            graph = systems['graph']
            edge_list = []

            # Collect edges from NetworkX graph
            if hasattr(graph, '_graph'):
                import networkx as nx
                for u, v, data in graph._graph.edges(data=True):
                    edge_list.append({
                        'Kaynak': u,
                        'Hedef': v,
                        'Tip': data.get('relation_type', 'related'),
                        'Guven': f"{data.get('weight', 0.5):.2f}",
                    })

            if edge_list:
                # Filter options
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    min_conf = st.slider("Min Guven Skoru:", 0.0, 1.0, 0.0, 0.1)
                with col_f2:
                    relation_filter = st.multiselect(
                        "Iliski Tipi:",
                        ["causes", "supports", "requires", "related", "contradicts"],
                        default=["causes", "supports", "requires", "related"]
                    )

                # Filter edges
                filtered_edges = [
                    e for e in edge_list
                    if float(e['Guven']) >= min_conf and e['Tip'] in relation_filter
                ]

                st.caption(f"Gosterilen: {len(filtered_edges)} / {len(edge_list)} iliski")

                # Show as dataframe
                if filtered_edges:
                    import pandas as pd
                    df = pd.DataFrame(filtered_edges)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Filtreye uyan iliski bulunamadi.")
            else:
                st.info("Henuz iliski yok.")
        except Exception as e:
            st.error(f"Iliskiler yuklenemedi: {e}")

    with admin_tab2:
        st.markdown("#### Hatali Iliski Sil")
        st.warning("⚠️ Dikkat: Silinen iliskiler geri alinamaz!")

        # Edge selection
        try:
            graph = systems['graph']
            if hasattr(graph, '_graph'):
                import networkx as nx
                edges_to_delete = list(graph._graph.edges())

                if edges_to_delete:
                    # Create readable edge list
                    edge_options = [f"{u} → {v}" for u, v in edges_to_delete]

                    selected_edge = st.selectbox(
                        "Silinecek iliski:",
                        edge_options,
                        key="delete_edge_select"
                    )

                    if selected_edge:
                        # Parse selection
                        parts = selected_edge.split(" → ")
                        if len(parts) == 2:
                            src, tgt = parts

                            # Show edge details
                            edge_data = graph._graph.get_edge_data(src, tgt)
                            if edge_data:
                                st.info(f"""
                                **Kaynak:** {src}
                                **Hedef:** {tgt}
                                **Tip:** {edge_data.get('relation_type', 'N/A')}
                                **Guven:** {edge_data.get('weight', 'N/A')}
                                """)

                            # Delete button
                            col_del1, col_del2 = st.columns([1, 3])
                            with col_del1:
                                if st.button("🗑️ Sil", type="primary", key="delete_edge_btn"):
                                    try:
                                        graph._graph.remove_edge(src, tgt)
                                        st.success(f"'{src} → {tgt}' iliskisi silindi!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Silme hatasi: {e}")
                else:
                    st.info("Silinecek iliski yok.")
        except Exception as e:
            st.error(f"Hata: {e}")

    with admin_tab3:
        st.markdown("#### Yeni Iliski Ekle")

        try:
            graph = systems['graph']
            node_ids = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []

            if len(node_ids) >= 2:
                col_add1, col_add2 = st.columns(2)

                with col_add1:
                    new_source = st.selectbox("Kaynak Dokuman:", node_ids, key="new_edge_source")
                with col_add2:
                    new_target = st.selectbox("Hedef Dokuman:", node_ids, key="new_edge_target")

                col_add3, col_add4 = st.columns(2)
                with col_add3:
                    new_rel_type = st.selectbox(
                        "Iliski Tipi:",
                        ["causes", "supports", "requires", "related"],
                        key="new_edge_type"
                    )
                with col_add4:
                    new_confidence = st.slider("Guven Skoru:", 0.1, 1.0, 0.7, 0.1, key="new_edge_conf")

                if new_source != new_target:
                    if st.button("➕ Iliski Ekle", type="primary", key="add_edge_btn"):
                        try:
                            graph.add_edge(new_source, new_target, new_rel_type, new_confidence)
                            st.success(f"'{new_source} → {new_target}' iliskisi eklendi!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ekleme hatasi: {e}")
                else:
                    st.warning("Kaynak ve hedef ayni olamaz!")
            else:
                st.info("Iliski eklemek icin en az 2 dokuman gerekli.")
        except Exception as e:
            st.error(f"Hata: {e}")

    with admin_tab4:
        st.markdown("#### 🔗 Entity Linking Yonetimi")
        st.markdown("""
        **Entity Linking Nedir?** Kod adlarini gercek isimlerle eslestirir.
        Ornegin: "Mavi Ufuk" → "Gunes Enerjisi A.S. Satin Alma Projesi"
        """)

        # Mevcut alias'ları göster
        st.markdown("##### Mevcut Alias'lar")
        try:
            aliases = systems['neuro'].get_aliases()
            if aliases:
                alias_data = [{"Alias": k, "Gercek Isim": v} for k, v in aliases.items()]
                st.dataframe(alias_data, use_container_width=True)
            else:
                st.info("Henuz alias tanimlanmadi. Asagidan ekleyebilirsiniz.")
        except Exception as e:
            st.warning(f"Alias'lar yuklenemedi: {e}")

        st.markdown("---")

        # Otomatik alias öğrenme
        st.markdown("##### Otomatik Alias Ogrenme")
        if st.button("🧠 Dokumanlardan Alias Ogren", use_container_width=True):
            try:
                learned = systems['neuro'].learn_aliases()
                st.success(f"✅ {learned} alias ogrendi!")
                st.rerun()
            except Exception as e:
                st.error(f"Ogrenme hatasi: {e}")

        st.markdown("---")

        # Manuel alias ekleme
        st.markdown("##### Manuel Alias Ekle")
        col_alias1, col_alias2 = st.columns(2)
        with col_alias1:
            new_alias = st.text_input("Alias (Kod Adi):", placeholder="Orn: Mavi Ufuk", key="new_alias")
        with col_alias2:
            new_canonical = st.text_input("Gercek Isim:", placeholder="Orn: Gunes Enerjisi A.S.", key="new_canonical")

        if st.button("➕ Alias Ekle", type="primary", key="add_alias_btn"):
            if new_alias and new_canonical:
                try:
                    systems['neuro'].add_alias(new_alias, new_canonical)
                    st.success(f"✅ Alias eklendi: '{new_alias}' → '{new_canonical}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ekleme hatasi: {e}")
            else:
                st.warning("Alias ve gercek isim alanlarini doldurun!")

        st.markdown("---")

        # NER Test
        st.markdown("##### 🔍 NER Test (Entity Cikarma)")
        ner_test_text = st.text_area(
            "Metin:",
            value="CEO Ahmet Yilmaz, 2025 yilinda Istanbul'da Mavi Ufuk projesini duyurdu.",
            key="ner_test_text"
        )
        if st.button("🔬 Entity'leri Cikar", key="ner_test_btn"):
            try:
                entities = systems['neuro'].extract_entities(ner_test_text)
                if entities:
                    st.markdown("**Bulunan Entity'ler:**")
                    for e in entities:
                        color_map = {"PERSON": "🟢", "ORG": "🔵", "DATE": "🟡", "LOCATION": "🟣", "MONEY": "🟠"}
                        color = color_map.get(e['type'], "⚪")
                        st.markdown(f"- {color} **{e['text']}** ({e['type']}) - Guven: {e['confidence']:.0%}")
                else:
                    st.info("Entity bulunamadi.")
            except Exception as e:
                st.error(f"NER hatasi: {e}")

    with admin_tab5:
        st.markdown("#### 🧠 Reasoning Analizi")
        st.markdown("""
        **Celiski Tespiti:** Belgeler arasindaki celiskili bilgileri bulur.
        **Zamansal Dogrulama:** Nedensel iliskilerin zaman sirasini kontrol eder.
        """)

        # Celiski Tespiti
        st.markdown("##### ⚠️ Celiski Tespiti")
        if st.button("🔍 Celiskileri Tara", use_container_width=True, key="scan_contradictions"):
            try:
                with st.spinner("Belgeler taraniyor..."):
                    contradictions = systems['neuro'].detect_contradictions()

                if contradictions:
                    st.warning(f"**{len(contradictions)} celiski tespit edildi!**")
                    for i, c in enumerate(contradictions[:5], 1):  # Max 5 goster
                        type_icon = {"numeric": "🔢", "temporal": "📅", "negation": "⚡"}.get(c['type'], "⚠️")
                        st.markdown(f"""
                        <div style="background: rgba(255,87,34,0.1); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid #ff5722;">
                            <b>{type_icon} Celiski {i}:</b> [{c['doc1_id']}] vs [{c['doc2_id']}]<br>
                            <small>Tip: {c['type']} | Guven: {c['confidence']:.0%}</small><br>
                            <small>Detay: {c['details']}</small><br>
                            <small>Degerler: {c['values'][0]} vs {c['values'][1]}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ Celiski bulunamadi!")
            except Exception as e:
                st.error(f"Celiski tarama hatasi: {e}")

        st.markdown("---")

        # Zamansal Dogrulama
        st.markdown("##### 📅 Zamansal Iliski Dogrulama")
        st.markdown("Bir nedensel iliskinin zaman sirasini kontrol edin:")

        try:
            graph = systems['graph']
            node_ids = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []

            if len(node_ids) >= 2:
                col_temp1, col_temp2 = st.columns(2)
                with col_temp1:
                    cause_id = st.selectbox("Neden Belgesi:", node_ids, key="temporal_cause")
                with col_temp2:
                    effect_id = st.selectbox("Sonuc Belgesi:", node_ids, key="temporal_effect")

                if st.button("🕐 Zamansal Sirayi Dogrula", key="validate_temporal"):
                    try:
                        validation = systems['neuro'].validate_causal_order(cause_id, effect_id)

                        if validation.get('valid'):
                            st.success(f"✅ **Gecerli Siralama**")
                        elif validation.get('order') == 'invalid':
                            st.error(f"❌ **Gecersiz Siralama** - Sonuc nedenden once!")
                        else:
                            st.warning(f"❓ **Belirsiz** - Tarih bilgisi yetersiz")

                        st.markdown(f"""
                        - **Neden tarihi:** {validation.get('cause_date', 'Bulunamadi')}
                        - **Sonuc tarihi:** {validation.get('effect_date', 'Bulunamadi')}
                        - **Guven:** {validation.get('confidence', 0):.0%}
                        - **Aciklama:** {validation.get('explanation', '')}
                        """)
                    except Exception as e:
                        st.error(f"Dogrulama hatasi: {e}")
            else:
                st.info("Dogrulama icin en az 2 belge gerekli.")
        except Exception as e:
            st.error(f"Hata: {e}")

    with admin_tab6:
        st.markdown("#### 🧠 Hafiza Sistemi")
        st.markdown("""
        **Hafiza Sistemi Nedir?** Arama sonuclari, notlar ve nedensellik geri bildirimlerini kalici olarak saklar.
        Model ogrensin, hatalar duzeltilsin, notlar alinsin - her sey hafizada kalir.
        """)

        # Memory Store import
        try:
            from neurocausal_rag.memory import create_memory_store
            import tempfile
            import os

            # Get or create memory store path
            if 'memory_store_path' not in st.session_state:
                memory_dir = os.path.join(tempfile.gettempdir(), 'neurocausal_memory')
                os.makedirs(memory_dir, exist_ok=True)
                st.session_state.memory_store_path = os.path.join(memory_dir, 'memory.db')

            memory = create_memory_store(st.session_state.memory_store_path)

            # Stats
            stats = memory.get_stats()
            col_mem1, col_mem2, col_mem3, col_mem4 = st.columns(4)
            with col_mem1:
                st.metric("Notlar", stats.total_notes)
            with col_mem2:
                st.metric("Geri Bildirimler", stats.total_feedbacks)
            with col_mem3:
                st.metric("Olumlu", stats.positive_feedbacks)
            with col_mem4:
                st.metric("Olumsuz", stats.negative_feedbacks)

            st.markdown("---")

            # Memory sub-sections
            mem_section = st.radio(
                "Bolum Sec:",
                ["📝 Notlar", "👍 Geri Bildirim", "📤 Export/Import", "🔄 Sifirla"],
                horizontal=True
            )

            if mem_section == "📝 Notlar":
                st.markdown("##### Notlariniz")

                # Not ekleme
                with st.expander("➕ Yeni Not Ekle", expanded=False):
                    note_content = st.text_area("Not icerigi:", placeholder="Bu sonuc cok faydali oldu...")
                    note_tags = st.text_input("Etiketler (virgul ile):", placeholder="onemli, iklim, proje")

                    if st.button("💾 Notu Kaydet", key="save_note_btn"):
                        if note_content:
                            tags = [t.strip() for t in note_tags.split(",") if t.strip()]
                            note = memory.add_note(content=note_content, tags=tags)
                            st.success(f"✅ Not kaydedildi: {note.id}")
                            st.rerun()
                        else:
                            st.warning("Not icerigi bos olamaz!")

                # Mevcut notlar
                notes = memory.get_notes()
                if notes:
                    for note in notes[:10]:  # Son 10 not
                        with st.container():
                            col_n1, col_n2 = st.columns([4, 1])
                            with col_n1:
                                st.markdown(f"**{note.id}** ({note.created_at[:10]})")
                                st.write(note.content[:200] + "..." if len(note.content) > 200 else note.content)
                                if note.tags:
                                    st.caption(f"Etiketler: {', '.join(note.tags)}")
                            with col_n2:
                                if st.button("🗑️", key=f"del_note_{note.id}"):
                                    memory.delete_note(note.id)
                                    st.rerun()
                            st.markdown("---")
                else:
                    st.info("Henuz not yok. Yukaridaki formu kullanarak not ekleyebilirsiniz.")

            elif mem_section == "👍 Geri Bildirim":
                st.markdown("##### Nedensellik Geri Bildirimi")
                st.markdown("Yanlis tespit edilen nedensellik iliskilerini duzeltebilirsiniz.")

                # Mevcut dokümanlar
                try:
                    graph = systems['graph']
                    node_ids = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []

                    if len(node_ids) >= 2:
                        col_fb1, col_fb2 = st.columns(2)
                        with col_fb1:
                            fb_source = st.selectbox("Kaynak Dokuman:", node_ids, key="fb_source")
                        with col_fb2:
                            fb_target = st.selectbox("Hedef Dokuman:", node_ids, key="fb_target")

                        col_fb3, col_fb4 = st.columns(2)
                        with col_fb3:
                            fb_type = st.selectbox("Islem:", ["✅ Iliski dogru (olumlu)", "❌ Iliski yanlis (olumsuz)", "➕ Yeni iliski ekle", "➖ Iliski kaldir"], key="fb_type")
                        with col_fb4:
                            fb_note = st.text_input("Aciklama:", placeholder="Bu iliski kesinlikle var...", key="fb_note")

                        if st.button("💾 Geri Bildirimi Kaydet", type="primary", key="save_fb_btn"):
                            if fb_source != fb_target:
                                if "olumlu" in fb_type:
                                    memory.add_feedback(fb_source, fb_target, "causes", True, fb_note)
                                    st.success("✅ Olumlu geri bildirim kaydedildi!")
                                elif "olumsuz" in fb_type:
                                    memory.add_feedback(fb_source, fb_target, "causes", False, fb_note)
                                    st.success("❌ Olumsuz geri bildirim kaydedildi!")
                                elif "ekle" in fb_type:
                                    memory.add_causal_relation(fb_source, fb_target, "causes", fb_note)
                                    st.success("➕ Yeni iliski teklifi kaydedildi!")
                                elif "kaldir" in fb_type:
                                    memory.remove_causal_relation(fb_source, fb_target, fb_note)
                                    st.success("➖ Iliski kaldirma teklifi kaydedildi!")
                                st.rerun()
                            else:
                                st.warning("Kaynak ve hedef ayni olamaz!")
                    else:
                        st.info("Geri bildirim icin en az 2 dokuman gerekli.")
                except Exception as e:
                    st.error(f"Hata: {e}")

                # Son geri bildirimler
                st.markdown("---")
                st.markdown("##### Son Geri Bildirimler")
                feedbacks = memory.get_feedbacks()
                if feedbacks:
                    for fb in feedbacks[:5]:
                        fb_icon = {"positive": "👍", "negative": "👎", "manual_add": "➕", "manual_del": "➖"}.get(fb.feedback_type, "📝")
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0;">
                            {fb_icon} <b>{fb.source_id}</b> → <b>{fb.target_id}</b> ({fb.relation_type})<br>
                            <small>Not: {fb.user_note or 'Yok'} | Guven: {fb.confidence:.1f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Henuz geri bildirim yok.")

            elif mem_section == "📤 Export/Import":
                st.markdown("##### Hafiza Yedekleme")

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    if st.button("📥 Hafizayi Indir (JSON)", use_container_width=True, key="export_memory"):
                        try:
                            export_path = os.path.join(tempfile.gettempdir(), 'memory_export.json')
                            memory.export_to_json(export_path)
                            with open(export_path, 'r', encoding='utf-8') as f:
                                json_content = f.read()
                            st.download_button(
                                "⬇️ JSON Indir",
                                json_content,
                                "neurocausal_memory.json",
                                "application/json",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Export hatasi: {e}")

                with col_exp2:
                    uploaded_file = st.file_uploader("📤 JSON Yukle", type=['json'], key="import_memory")
                    if uploaded_file:
                        if st.button("📥 Iceri Aktar", key="import_btn"):
                            try:
                                import_path = os.path.join(tempfile.gettempdir(), 'memory_import.json')
                                with open(import_path, 'wb') as f:
                                    f.write(uploaded_file.getvalue())
                                notes, feedbacks = memory.import_from_json(import_path)
                                st.success(f"✅ Aktarildi: {notes} not, {feedbacks} geri bildirim")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Import hatasi: {e}")

            elif mem_section == "🔄 Sifirla":
                st.markdown("##### Hafizayi Sifirla")
                st.warning("⚠️ DIKKAT: Bu islem tum notlari ve geri bildirimleri kalici olarak siler!")

                confirm = st.checkbox("Hafizayi sifirlamak istedigimden eminim", key="reset_confirm")
                if confirm:
                    if st.button("🗑️ Hafizayi Sifirla", type="primary", key="reset_memory_btn"):
                        if memory.reset(confirm=True):
                            st.success("✅ Hafiza sifirlandi!")
                            st.rerun()
                        else:
                            st.error("Sifirlama basarisiz!")

        except ImportError as e:
            st.error(f"Hafiza modulu yuklenemedi: {e}")
        except Exception as e:
            st.error(f"Hafiza sistemi hatasi: {e}")

    st.markdown("---")

    # Export/Import section
    st.markdown("#### 💾 Graf Yedekleme")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📥 Grafi Indir (JSON)", use_container_width=True):
            try:
                import json
                graph = systems['graph']
                export_data = {
                    'nodes': [],
                    'edges': []
                }

                # Export nodes
                for node_id, node_data in graph.nodes.items():
                    export_data['nodes'].append({
                        'id': node_id,
                        'content': node_data.get('content', '')[:200],
                    })

                # Export edges
                if hasattr(graph, '_graph'):
                    for u, v, data in graph._graph.edges(data=True):
                        export_data['edges'].append({
                            'source': u,
                            'target': v,
                            'type': data.get('relation_type', 'related'),
                            'weight': data.get('weight', 0.5)
                        })

                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ JSON Indir",
                    json_str,
                    "neurocausal_graph.json",
                    "application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Export hatasi: {e}")

    with col_exp2:
        st.info("Graf yedegi JSON formatinda indirilebilir. Ilerde import ozelligi eklenecek.")


# ============================================================================
# TAB 6: HAKKINDA
# ============================================================================
with tab6:
    # Hakkinda - README tarzi kapsamli bilgi
    hakkinda_tab1, hakkinda_tab2, hakkinda_tab3, hakkinda_tab4 = st.tabs([
        "📖 Genel Bakis", "🚀 Hizli Baslangic", "🧪 Ornek Senaryolar", "⚙️ Teknik Detaylar"
    ])

    with hakkinda_tab1:
        st.markdown("""
        # 🧠 NeuroCausal RAG v6.1

        > **"Sadece benzer degil, NEDENSEL olarak iliskili bilgileri bulan akilli arama sistemi"**
        >
        > **v6.1 Yenilikler:** Multi-Hop Retrieval, Search Optimizer, Query Decomposition, Memory System

        ---

        ## 🎯 Bu Sistem Ne Yapar?

        Klasik RAG (Retrieval Augmented Generation) sistemleri sadece **kelime benzerligine** bakar.
        NeuroCausal RAG ise **neden-sonuc iliskilerini** de anlayarak cok daha akilli aramalar yapar.

        ### Ornek: "Stres" Sorgusu

        | Sistem | Bulunan Dokumanlar |
        |--------|-------------------|
        | **Klasik RAG** | "Stres nedir?", "Stres belirtileri" |
        | **NeuroCausal RAG** | "Stres nedir?" + **"Kortizol hormonu"** + **"Uyku bozuklugu"** + **"Dikkat dagilmasi"** + **"Is kazasi riski"** |

        NeuroCausal RAG, "stres" kelimesi gecmese bile **nedensel zinciri** takip ederek iliskili tum bilgileri bulur!

        ---

        ## 💡 Temel Ozellikler

        | Ozellik | Aciklama |
        |---------|----------|
        | 🔗 **Auto-Causal Discovery** | LLM ile otomatik nedensellik kesfi |
        | 💉 **Chain Injection** | Nedensel zincir dokumanlarini sonuclara enjekte etme |
        | 🎚️ **Ayarlanabilir Agirliklar** | α, β, γ slider'lari ile arama modunu secme |
        | 📊 **Hibrit Skorlama** | Benzerlik + Nedensellik + Onem |
        | 🕸️ **Graf Gorsellestirme** | Interaktif nedensellik grafi |
        | 📚 **Kaynak Gosterme** | Her cevap icin kanit dokumanlari |

        ---

        ## 👨‍💻 Gelistirici

        **Ertugrul Akben**
        - 📧 Email: i@ertugrulakben.com
        - 🌐 Web: ertugrulakben.com
        - 📅 Version: 6.1.0 (2026)
        """)

    with hakkinda_tab2:
        st.markdown("""
        # 🚀 Hizli Baslangic

        ## Adim 1: Kurulum

        ```bash
        # Repoyu klonla
        git clone <repo-url>
        cd NeuroCausalRAG

        # Bagimliliklari yukle
        pip install -r requirements.txt

        # OpenAI API key'i ayarla
        echo "OPENAI_API_KEY=sk-your-key-here" > .env
        ```

        ## Adim 2: Uygulamayi Baslat

        ### Windows:
        ```bash
        run_app.bat
        ```

        ### Linux/Mac:
        ```bash
        export PYTHONPATH=$(pwd)
        streamlit run app.py
        ```

        ## Adim 3: Ilk Aramanizi Yapin

        1. Sol panelden bir **veri seti** secin (ornek: "Stres Zinciri")
        2. "📥 Yukle" butonuna basin
        3. "🔍 Discovery" ile iliskileri kesfettirin
        4. Chatbot'a soru sorun: "Stres is kazasina neden olur mu?"

        ---

        ## 🎮 Arama Modlari

        | Mod | Ne Zaman Kullanilir? |
        |-----|---------------------|
        | 🔍 **Ansiklopedi** | Dogrudan cevap ariyorsaniz |
        | 🕵️ **Dedektif** | Gizli baglantilari kesfetmek istiyorsaniz |
        | ⚖️ **Dengeli** | Genel amacli arama |
        | 🎯 **Hub Odakli** | Onemli/merkezi dokumanlari bulmak icin |

        ---

        ## 📡 API Kullanimi

        ```bash
        # API'yi baslat
        run_api.bat  # veya: uvicorn api:app --port 8000

        # Arama yap
        curl -X POST http://localhost:8000/api/v1/search \\
          -H "Content-Type: application/json" \\
          -d '{"query": "Stres etkileri nelerdir?", "top_k": 5}'
        ```
        """)

    with hakkinda_tab3:
        st.markdown("""
        # 🧪 Ornek Senaryolar

        ## Senaryo 1: Stres Zinciri 😰

        **Soru:** "Stres is kazasina neden olur mu?"

        **Klasik RAG:** ❌ "Bu konuda bilgi bulunamadi" (cunku "stres" ve "kaza" arasinda dogrudan kelime eslesmesi yok)

        **NeuroCausal RAG:** ✅ Zinciri takip eder:
        ```
        Stres → Kortizol artisi → Uyku bozuklugu → Dikkat dagilmasi → Is kazasi riski artar
        ```

        ---

        ## Senaryo 2: Gizli Satin Alma 🔐

        **Soru:** "Ahmet Yilmaz kac dolar yonetiyor?"

        **Belgeler:**
        - Belge A: "Ahmet Yilmaz, Mavi Ufuk projesinin lideridir"
        - Belge B: "Mavi Ufuk = Gunes Enerjisi A.S. satin alimi"
        - Belge C: "Gunes Enerjisi A.S. icin 1.2 Milyar $ ayrildi"

        **Klasik RAG:** ❌ "Ahmet Yilmaz'in yonettigi tutar bilgisi yok"

        **NeuroCausal RAG:** ✅ Kod adini cozer:
        ```
        Ahmet Yilmaz → Mavi Ufuk → Gunes Enerjisi A.S. → 1.2 Milyar $
        Cevap: "Ahmet Yilmaz, 1.2 Milyar dolarlik projeyi yonetiyor"
        ```

        ---

        ## Senaryo 3: Hukuki Domino Etkisi ⚖️

        **Soru:** "Pazarlama performansi neden dustu?"

        NeuroCausal RAG zinciri bulur:
        ```
        DMA-2025 yasasi → Cerez politikasi degisikligi → Hedefleme algoritmasi calismaz →
        Retargeting durduruldu → ROI %40 dustu
        ```

        ---

        ## 🎯 Demo Nasil Yapilir?

        1. Sol panelden **"🔐 Gizli Satin Alma"** veri setini secin
        2. "📥 Yukle" basin
        3. "🔍 Discovery" basin (iliskiler otomatik bulunacak)
        4. Chatbot'a sorun: **"Ahmet Yilmaz kac dolar yonetiyor?"**
        5. Sistemin zinciri nasil cozduguny izleyin!
        """)

    with hakkinda_tab4:
        st.markdown("""
        # ⚙️ Teknik Detaylar

        ## 📐 Skor Formulu

        ```
        Score = α × similarity + β × causal + γ × importance

        Varsayilan degerler:
        α = 0.5 (benzerlik agirligi)
        β = 0.3 (nedensellik agirligi)
        γ = 0.2 (onem agirligi)
        ```

        ---

        ## 🔗 Iliski Tipleri ve Agirliklari

        | Iliski | Agirlik | Anlam |
        |--------|---------|-------|
        | `causes` | 1.0 | A, B'ye neden olur |
        | `supports` | 0.8 | A, B'yi destekler |
        | `requires` | 0.7 | A, B'yi gerektirir |
        | `related` | 0.5 | A ve B iliskilidir |
        | `contradicts` | 0.3 | A ve B celisir |

        ---

        ## 💉 Chain Injection Mekanizmasi

        ```
        1. Sorgu icin en iyi N sonuc bulunur
        2. Her sonuc icin nedensel zincir cikarilir
        3. Zincirdeki dokumanlar sonuclara enjekte edilir
        4. Enjekte edilen dokumanlar %80 skor cezasi alir
        5. Tum sonuclar yeniden siralanir
        ```

        ---

        ## 🏗️ Sistem Mimarisi

        ```
        ┌─────────────────────────────────────────────┐
        │            Streamlit UI (app.py)            │
        ├─────────────────────────────────────────────┤
        │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
        │  │ Chatbot │  │ Compare │  │ Graph View  │  │
        │  └────┬────┘  └────┬────┘  └──────┬──────┘  │
        │       │            │              │         │
        │  ┌────┴────────────┴──────────────┴────┐    │
        │  │         NeuroCausal Engine          │    │
        │  │  ┌──────────┐ ┌───────────────────┐ │    │
        │  │  │ Retriever│ │ Causal Discovery  │ │    │
        │  │  └────┬─────┘ └─────────┬─────────┘ │    │
        │  │       │                 │           │    │
        │  │  ┌────┴────┐   ┌────────┴────────┐  │    │
        │  │  │  Index  │   │  Causal Graph   │  │    │
        │  │  │ (FAISS) │   │   (NetworkX)    │  │    │
        │  │  └─────────┘   └─────────────────┘  │    │
        │  └─────────────────────────────────────┘    │
        └─────────────────────────────────────────────┘
        ```

        ---

        ## 📁 Dosya Yapisi

        ```
        NeuroCausalRAG/
        ├── app.py              # Streamlit UI
        ├── api.py              # FastAPI REST API
        ├── config.yaml         # Yapilandirma
        ├── requirements.txt    # Bagimliliklar
        ├── neurocausal_rag/    # Ana modül
        │   ├── core/           # Graf, Node, Edge
        │   ├── search/         # Retriever, Index
        │   ├── learning/       # Discovery algoritmalari
        │   ├── llm/            # LLM client
        │   └── visualization/  # PyVis grafik
        ├── data/               # Veri setleri
        └── examples/           # Ornek senaryolar
        ```
        """)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p>NeuroCausal RAG v6.1 | Ertugrul Akben | 2026</p>
    <p><small>Nedensel Bilgi Getirme ile Daha Akilli Aramalar</small></p>
</div>
""", unsafe_allow_html=True)
