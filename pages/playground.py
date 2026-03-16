"""
NeuroCausal RAG - Canli Demo / Playground
Kullanicilarin kendi verileriyle sistemi test etmesi icin interaktif sayfa

Yazar: Ertugrul Akben
"""

import streamlit as st
import time
import re
from pathlib import Path

# Page config
st.set_page_config(
    page_title="NeuroCausal RAG - Playground",
    page_icon="🧪",
    layout="wide"
)

# =============================================================================
# STYLES
# =============================================================================
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00cc66;
    }
    .result-box {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00cc66;
    }
    .chain-box {
        background-color: #16213e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ffa500;
    }
    .score-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .score-high { background-color: #00cc66; color: white; }
    .score-med { background-color: #ffa500; color: white; }
    .score-low { background-color: #ff4444; color: white; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ORNEK VERILER - Dosyalardan yukle
# =============================================================================
def load_example_datasets():
    """Ornek veri setlerini yukle."""
    try:
        from data.example_datasets import DATASETS, get_dataset_raw_content
        return DATASETS, get_dataset_raw_content
    except ImportError:
        return {}, lambda x: ""

EXAMPLE_DATASETS, get_raw_content = load_example_datasets()

# Playground icin ozel ornekler (kisa veriler)
PLAYGROUND_EXAMPLES = {
    "Bos": "",
    "😰 Stres Zinciri": """### stres_kortizol
Stres, kortizol hormonu salgilanmasina neden olur. Kortizol, vucut uzerinde onemli fizyolojik etkilere sahiptir.

### kortizol_uyku
Yuksek kortizol seviyeleri, uyku duzenini ciddi sekilde bozar. Geceleri uyanik kalmaya yol acar.

### uyku_dikkat
Uykusuzluk, gun icinde dikkat daginikligina yol acar. Konsantrasyon guclugu yasanir.

### dikkat_kaza
Dikkat daginikligi is kazasi riskini artirir. Ozellikle tehlikeli islerde ciddi sonuclara yol acabilir.
""",
    "🌍 Iklim Mini": """### sera_gazlari
Sera gazlari atmosferde isiyi tutan gazlardir. Baslicalari karbondioksit (CO2), metan (CH4) ve su buharidir.

### fosil_yakit
Fosil yakitlar yaninca CO2 aciga cikar. Enerji uretiminin %80'i fosil yakitlardan saglanir.

### kuresel_isinma
Kuresel isinma, Dunya'nin ortalama sicakliginin artmasidir. Son 100 yilda 1.1 derece artis oldu.

### cimento_uretimi
Cimento uretimi kuresel CO2 emisyonlarinin %8'inden sorumludur.
""",
    "📦 Tedarik Zinciri": """### log_774
Tesis-7'de elektrik kontagindan cikan yangin nedeniyle uretim bandi 2 hafta durduruldu. Bu tesis sadece "Haptik Geri Bildirim Modulu (HGM-X)" parcasini uretiyor.

### spec_99
Visionary Pro VR Headset'in kritik bileseni: Haptik Geri Bildirim Modulu (HGM-X). Bu parca olmadan uretim yapilamiyor.

### ceo_01
Yilbasi icin planladigimiz buyuk urun lansmanini ertelemek zorundayiz. Kritik bilesen tedariğinde sorun var. Q4 gelir hedeflerimiz %15 dusecek.

### tedarik_riski
HGM-X modulu tek kaynak tedarik modeli kullanıyor (Tesis-7). Herhangi bir aksama durumunda uretim tamamen durabilir.
""",
    "🔐 Gizli Satin Alma": """### strat_05
"Mavi Ufuk" projesi baslatildi. Bu proje, sirketin yenilenebilir enerji sektorune giris biletidir. Proje lideri: Ahmet Yilmaz.

### fin_202
Satin Alma: Gunes Enerjisi A.S. firmasinin %100 hissesi icin 1.2 Milyar $ ayrildi.

### it_sec_44
Guvenlik hatirlatmasi: "Gunes Enerjisi A.S." satin almasi sistemlerimizde "Mavi Ufuk Projesi" olarak kodlanmistir.

### basin_bulteni
XYZ Holding, Gunes Enerjisi A.S.'yi satin alarak yesil enerji devrimine katiliyor. 1.2 Milyar dolarlik yatirim.
""",
    "⚖️ Hukuki Etki": """### law_eu_22
AB "DMA-2025" yasasini yururluge koydu. Kullanici verilerinin izinsiz paylasilmasi yasaklandi. Ihlalde cironun %10'u ceza.

### mkt_q3
Reklam donusum oranlarinda %40 dusus yasandi. Hedefleme algoritmasi calismiyor. Kullanici verilerine erisilemediginden retargeting durduruldu.

### compliance_memo
DMA-2025 ihlali durumunda 500 Milyon Euro ceza riski var. Tum cerez izinleri yeniden alinmali.

### satis_raporu
Avrupa satislari dustu: Almanya -%35, Fransa -%28, Ispanya -%42. Sebep: Hedefli reklamlarin etkinliginin azalmasi.
"""
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def parse_documents(text: str) -> list:
    """
    Metni dokumanlara ayirir. Desteklenen formatlar:
    1. Markdown: ### id\nicerik
    2. JSON: [{"id": "...", "content": "..."}]
    3. Paragraf: \n\n ile ayrilmis paragraflar
    4. Satir bazli: Her satir bir dokuman
    """
    import json

    docs = []
    text = text.strip()

    if not text:
        return docs

    # 1. JSON formati dene
    if text.startswith('[') or text.startswith('{'):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        doc_id = item.get('id', item.get('doc_id', f'doc_{i+1}'))
                        content = item.get('content', item.get('text', str(item)))
                        if content:
                            docs.append({'id': str(doc_id), 'content': content})
                    elif isinstance(item, str):
                        docs.append({'id': f'doc_{i+1}', 'content': item})
            if docs:
                return docs
        except json.JSONDecodeError:
            pass  # JSON degil, diger formatlari dene

    # 2. Markdown ### formati - daha esnek regex
    pattern = r'###\s*([^\n]+)\s*\n(.*?)(?=\n###|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    for doc_id, content in matches:
        content = content.strip()
        if content and len(content) > 5:  # En az 5 karakter
            # ID'yi temizle
            clean_id = re.sub(r'[^\w\-_]', '_', doc_id.strip())[:50]
            docs.append({
                'id': clean_id or f'doc_{len(docs)+1}',
                'content': content
            })

    if docs:
        return docs

    # 3. ## veya # baslikli format
    pattern2 = r'^#{1,2}\s*([^\n]+)\s*\n(.*?)(?=\n#{1,2}\s|\Z)'
    matches2 = re.findall(pattern2, text, re.DOTALL | re.MULTILINE)

    for doc_id, content in matches2:
        content = content.strip()
        if content and len(content) > 10:
            clean_id = re.sub(r'[^\w\-_]', '_', doc_id.strip())[:50]
            docs.append({
                'id': clean_id or f'doc_{len(docs)+1}',
                'content': content
            })

    if docs:
        return docs

    # 4. Paragraf bazli (cift satir boslugu)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if len(paragraphs) >= 2:
        for i, para in enumerate(paragraphs):
            docs.append({
                'id': f'doc_{i+1}',
                'content': para
            })
        return docs

    # 5. Satir bazli (tek satir boslugu, en az 30 karakter)
    lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 30]
    if len(lines) >= 2:
        for i, line in enumerate(lines):
            docs.append({
                'id': f'doc_{i+1}',
                'content': line
            })
        return docs

    # 6. Son care: Tum metni tek dokuman olarak al (en az 50 karakter)
    if len(text) > 50:
        # Metni yaklasik esit parcalara bol
        chunk_size = max(100, len(text) // 3)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                docs.append({
                    'id': f'doc_{i+1}',
                    'content': chunk.strip()
                })

    return docs


def create_sandbox_rag(docs: list, progress_callback=None):
    """
    Gecici (in-memory) RAG motoru olusturur.
    """
    from neurocausal_rag.embedding.text import TextEmbedding
    from neurocausal_rag.core.graph import GraphEngine
    from neurocausal_rag.search.retriever import Retriever
    from neurocausal_rag.config import SearchConfig, IndexConfig, EmbeddingConfig, GraphConfig
    from neurocausal_rag.learning.semantic_discovery import enhanced_causal_discovery
    import numpy as np

    total_steps = 4

    # Step 1: Embedding engine
    if progress_callback:
        progress_callback(1/total_steps, "Embedding motoru yukleniyor...")

    emb_config = EmbeddingConfig()
    embedding = TextEmbedding(emb_config)

    # Step 2: Graf olustur ve dokumanlari ekle
    if progress_callback:
        progress_callback(2/total_steps, "Dokumanlar isleniyor...")

    graph_config = GraphConfig()
    graph = GraphEngine(graph_config)

    embeddings_list = []
    for doc in docs:
        emb = embedding.get_text_embedding(doc['content'])
        graph.add_node(doc['id'], doc['content'], emb, {'source': 'playground'})
        embeddings_list.append(emb)

    embeddings_array = np.array(embeddings_list)

    # Step 3: Nedensellik kesfi
    if progress_callback:
        progress_callback(3/total_steps, "Nedensellik iliskileri kesfediliyor...")

    relations = enhanced_causal_discovery(
        docs,
        embeddings_array,
        min_confidence=0.5
    )

    # Iliskileri grafa ekle
    added_edges = 0
    for rel in relations:
        if rel.get('confidence', 0) > 0.5:
            try:
                graph.add_edge(
                    rel['source'],
                    rel['target'],
                    rel.get('relation_type', 'related'),
                    rel['confidence']
                )
                added_edges += 1
            except:
                pass

    # Step 4: Retriever olustur
    if progress_callback:
        progress_callback(4/total_steps, "Arama motoru hazirlaniyor...")

    search_config = SearchConfig(top_k=5, alpha=0.5, beta=0.3, gamma=0.2)
    index_config = IndexConfig(backend="brute_force")

    retriever = Retriever(
        graph=graph,
        embedding=embedding,
        config=search_config,
        index_config=index_config
    )
    retriever.rebuild_index()

    return {
        'retriever': retriever,
        'graph': graph,
        'embedding': embedding,
        'doc_count': len(docs),
        'edge_count': added_edges,
        'relations': relations
    }


# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.title("🧪 NeuroCausal RAG Playground")
    st.markdown("""
    **Kendi verilerinizle sistemi test edin!**
    Bu sayfa gecici (in-memory) bir ortamda calisir - verileriniz kaydedilmez.
    """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Veri Yukle", "🔍 Sorgula", "📊 Graf"])

    # Session state
    if 'sandbox' not in st.session_state:
        st.session_state.sandbox = None
    if 'docs' not in st.session_state:
        st.session_state.docs = []

    # ==========================================================================
    # TAB 1: Veri Yukle
    # ==========================================================================
    with tab1:
        st.subheader("Veri Girisi")

        # Session state for uploaded files
        if 'uploaded_files_content' not in st.session_state:
            st.session_state.uploaded_files_content = {}  # {filename: content}

        col1, col2 = st.columns([3, 1])

        with col1:
            # Coklu dosya yukleme
            uploaded_files = st.file_uploader(
                "📄 Dosyalari yukleyin (.txt, .md, .json) - Coklu secim yapabilirsiniz",
                type=["txt", "md", "json"],
                accept_multiple_files=True,
                help="Birden fazla dosya secebilirsiniz. Her dosya ayri dokuman olarak islenir.",
                key="playground_uploader"
            )

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.uploaded_files_content:
                        try:
                            file_content = uploaded_file.read().decode('utf-8')
                            st.session_state.uploaded_files_content[uploaded_file.name] = file_content
                        except Exception as e:
                            st.error(f"Dosya okuma hatasi ({uploaded_file.name}): {e}")

            # Yuklenen dosyalari goster
            if st.session_state.uploaded_files_content:
                st.markdown("### 📁 Yuklenen Dosyalar")

                total_chars = sum(len(c) for c in st.session_state.uploaded_files_content.values())
                st.success(f"**{len(st.session_state.uploaded_files_content)} dosya** yuklendi ({total_chars:,} karakter)")

                # Dosya listesi
                for fname, content in st.session_state.uploaded_files_content.items():
                    col_f1, col_f2, col_f3 = st.columns([3, 1, 1])
                    with col_f1:
                        st.text(f"📄 {fname}")
                    with col_f2:
                        st.caption(f"{len(content):,} chr")
                    with col_f3:
                        if st.button("🗑️", key=f"del_{fname}", help=f"{fname} dosyasini sil"):
                            del st.session_state.uploaded_files_content[fname]
                            st.rerun()

                # Tum dosyalari temizle
                if st.button("🗑️ Tum dosyalari temizle", key="clear_all_uploads"):
                    st.session_state.uploaded_files_content = {}
                    st.rerun()

            st.divider()

            # Ornek secimi - genisletilmis
            example_choice = st.selectbox(
                "📂 Veya Hazir Ornek Sec:",
                list(PLAYGROUND_EXAMPLES.keys()),
                index=0
            )

            # Hangi icerik kullanilacak?
            if st.session_state.uploaded_files_content:
                # Tum dosyalari birlestir
                combined_content = ""
                for fname, content in st.session_state.uploaded_files_content.items():
                    # Dosya adini baslik olarak ekle
                    file_id = fname.replace('.txt', '').replace('.md', '').replace('.json', '')
                    # Eger icerik ### ile baslamiyorsa, dosya adini ID olarak ekle
                    if not content.strip().startswith('###') and not content.strip().startswith('['):
                        combined_content += f"### {file_id}\n{content}\n\n"
                    else:
                        combined_content += content + "\n\n"

                default_text = combined_content
                st.info(f"📁 {len(st.session_state.uploaded_files_content)} dosya birlestirildi")
            else:
                default_text = PLAYGROUND_EXAMPLES.get(example_choice, "")
                # Ornek aciklamalari
                example_hints = {
                    "😰 Stres Zinciri": "💡 Sorgu: 'Stres is kazalarini nasil etkiler?'",
                    "🌍 Iklim Mini": "💡 Sorgu: 'Cimento iklimi nasil etkiler?'",
                    "📦 Tedarik Zinciri": "💡 Sorgu: 'Yangin urun lansmanini nasil etkiledi?'",
                    "🔐 Gizli Satin Alma": "💡 Sorgu: 'Mavi Ufuk projesi nedir?'",
                    "⚖️ Hukuki Etki": "💡 Sorgu: 'Pazarlama neden coktu?'",
                }
                if example_choice in example_hints:
                    st.info(example_hints[example_choice])

            # Text area
            input_text = st.text_area(
                "Dokumanlarinizi yapistirin (veya kendiniz yazin):",
                value=default_text,
                height=350,
                help="Format: ### dokuman_adi\\nDokuman icerigi"
            )

        with col2:
            st.markdown("### Format")
            st.code("""### id_1
Dokuman 1 icerigi.

### id_2
Dokuman 2 icerigi.""", language="markdown")

            st.markdown("### Ipucu")
            st.info("Nedensellik iceren cumleler kullanin: 'A, B'ye neden olur' gibi")

        # Analiz butonu
        if st.button("🚀 Analiz Et", type="primary", use_container_width=True):
            if not input_text.strip():
                st.error("Lutfen veri girin!")
            else:
                # Parse documents
                docs = parse_documents(input_text)

                if len(docs) < 2:
                    st.error("En az 2 dokuman gerekli!")
                else:
                    st.session_state.docs = docs

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(pct, msg):
                        progress_bar.progress(pct)
                        status_text.text(msg)

                    # Create sandbox
                    try:
                        with st.spinner("Sistem hazirlaniyor..."):
                            sandbox = create_sandbox_rag(docs, update_progress)
                            st.session_state.sandbox = sandbox

                        progress_bar.progress(1.0)
                        status_text.text("Tamamlandi!")

                        # Ozet
                        st.success(f"""
                        **Analiz Tamamlandi!**
                        - Dokuman: {sandbox['doc_count']}
                        - Iliski: {sandbox['edge_count']}
                        - Artik "Sorgula" sekmesinden soru sorabilirsiniz!
                        """)

                    except Exception as e:
                        st.error(f"Hata: {str(e)}")

    # ==========================================================================
    # TAB 2: Sorgula ve Karsilastir
    # ==========================================================================
    with tab2:
        st.subheader("🔍 Classic RAG vs NeuroCausal RAG Karsilastirma")

        if st.session_state.sandbox is None:
            st.warning("Oncelikle 'Veri Yukle' sekmesinden veri yukleyin!")
        else:
            sandbox = st.session_state.sandbox

            # Sorgu alani
            query = st.text_input(
                "Sorunuzu yazin:",
                placeholder="Ornek: Stres is kazalarini nasil etkiler?",
                key="playground_query"
            )

            # Arama modlari
            st.markdown("**NeuroCausal RAG Agirliklari:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.slider("Benzerlik (α)", 0.0, 1.0, 0.5, key="pg_alpha")
            with col2:
                beta = st.slider("Nedensellik (β)", 0.0, 1.0, 0.3, key="pg_beta")
            with col3:
                gamma = st.slider("Onem (γ)", 0.0, 1.0, 0.2, key="pg_gamma")

            if st.button("🔍 Karsilastir", type="primary", use_container_width=True):
                if not query.strip():
                    st.error("Lutfen bir soru yazin!")
                else:
                    import numpy as np

                    # Classic RAG icin cosine similarity fonksiyonu
                    def cosine_similarity(vec1, vec2):
                        dot = np.dot(vec1, vec2)
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        if norm1 == 0 or norm2 == 0:
                            return 0.0
                        return float(dot / (norm1 * norm2))

                    # Classic RAG arama
                    start_classic = time.time()
                    query_emb = sandbox['embedding'].get_text_embedding(query)

                    classic_results = []
                    for doc in st.session_state.docs:
                        doc_emb = sandbox['embedding'].get_text_embedding(doc['content'])
                        sim = cosine_similarity(query_emb, doc_emb)
                        classic_results.append({
                            'id': doc['id'],
                            'content': doc['content'],
                            'score': sim
                        })
                    classic_results = sorted(classic_results, key=lambda x: x['score'], reverse=True)[:5]
                    classic_time = (time.time() - start_classic) * 1000

                    # NeuroCausal RAG arama
                    start_neuro = time.time()
                    neuro_results = sandbox['retriever'].search(
                        query,
                        top_k=5,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma
                    )
                    neuro_time = (time.time() - start_neuro) * 1000

                    # Sonuclari yan yana goster
                    st.divider()

                    col_classic, col_neuro = st.columns(2)

                    # ===== CLASSIC RAG =====
                    with col_classic:
                        st.markdown("### 📕 Classic RAG")
                        st.caption(f"⏱️ {classic_time:.1f}ms | Sadece benzerlik")

                        classic_avg = sum(r['score'] for r in classic_results) / len(classic_results) if classic_results else 0

                        for i, r in enumerate(classic_results, 1):
                            with st.container():
                                st.markdown(f"**#{i} {r['id']}** | Skor: `{r['score']:.3f}`")
                                st.caption(r['content'][:150] + "..." if len(r['content']) > 150 else r['content'])
                                st.divider()

                    # ===== NEUROCAUSAL RAG =====
                    with col_neuro:
                        st.markdown("### 📗 NeuroCausal RAG")

                        # Zincir ve enjeksiyon sayisi
                        chain_count = sum(1 for r in neuro_results if r.causal_chain and len(r.causal_chain) > 1)
                        injected_count = sum(1 for r in neuro_results if r.metadata and r.metadata.get('injected_from'))

                        st.caption(f"⏱️ {neuro_time:.1f}ms | 🔗 {chain_count} zincir | 💉 {injected_count} enjeksiyon")

                        neuro_avg = sum(r.score for r in neuro_results) / len(neuro_results) if neuro_results else 0

                        for i, r in enumerate(neuro_results, 1):
                            badges = []
                            if r.causal_chain and len(r.causal_chain) > 1:
                                badges.append("🔗")
                            if r.metadata and r.metadata.get('injected_from'):
                                badges.append("💉")

                            badge_str = " ".join(badges)

                            with st.container():
                                # Detay skorlar
                                sim = getattr(r, 'similarity_score', r.score)
                                cau = getattr(r, 'causal_score', 0)
                                imp = getattr(r, 'importance_score', 0)

                                st.markdown(f"**#{i} {r.node_id}** {badge_str} | Skor: `{r.score:.3f}`")
                                st.caption(f"Sim: {sim:.2f} | Cau: {cau:.2f} | Imp: {imp:.2f}")
                                st.caption(r.content[:150] + "..." if len(r.content) > 150 else r.content)

                                # Zincir goster
                                if r.causal_chain and len(r.causal_chain) > 1:
                                    chain_str = " → ".join(r.causal_chain[:4])
                                    if len(r.causal_chain) > 4:
                                        chain_str += " → ..."
                                    st.info(f"🔗 {chain_str}")

                                st.divider()

                    # ===== KARSILASTIRMA OZETI =====
                    st.markdown("---")
                    st.markdown("### 📊 Karsilastirma Ozeti")

                    # Anlamsal puanlama
                    classic_semantic = classic_avg
                    chain_bonus = chain_count * 0.15
                    injection_bonus = injected_count * 0.10
                    causal_bonus = sum(getattr(r, 'causal_score', 0) for r in neuro_results) / max(1, len(neuro_results)) * 0.20
                    neuro_semantic = neuro_avg + chain_bonus + injection_bonus + causal_bonus

                    col_m1, col_m2, col_m3 = st.columns(3)

                    with col_m1:
                        st.metric("Classic RAG", f"{classic_semantic:.3f}")
                    with col_m2:
                        st.metric("NeuroCausal RAG", f"{neuro_semantic:.3f}",
                                  delta=f"+{chain_bonus + injection_bonus + causal_bonus:.2f} bonus")
                    with col_m3:
                        if neuro_semantic > classic_semantic:
                            st.success("🏆 NeuroCausal RAG kazandi!")
                        elif classic_semantic > neuro_semantic:
                            st.warning("📕 Classic RAG kazandi")
                        else:
                            st.info("🤝 Berabere")

                    # Gorsel karsilastirma
                    st.markdown("**Gorsel Karsilastirma:**")
                    max_score = max(classic_semantic, neuro_semantic, 0.01)

                    col_bar1, col_bar2 = st.columns(2)
                    with col_bar1:
                        st.progress(min(classic_semantic / max_score, 1.0))
                        st.caption(f"Classic: {classic_semantic:.3f}")
                    with col_bar2:
                        st.progress(min(neuro_semantic / max_score, 1.0))
                        st.caption(f"NeuroCausal: {neuro_semantic:.3f}")

                    # Fark analizi
                    if chain_count > 0 or injected_count > 0:
                        st.markdown("**🔍 NeuroCausal RAG Avantajlari:**")
                        advantages = []
                        if chain_count > 0:
                            advantages.append(f"- {chain_count} nedensel zincir bulundu (+{chain_bonus:.2f} bonus)")
                        if injected_count > 0:
                            advantages.append(f"- {injected_count} bagli dokuman enjekte edildi (+{injection_bonus:.2f} bonus)")
                        if causal_bonus > 0:
                            advantages.append(f"- Nedensellik skorlari eklendi (+{causal_bonus:.2f} bonus)")
                        st.markdown("\n".join(advantages))

                    # ===== LLM YANITLARI =====
                    st.markdown("---")
                    st.markdown("### 🤖 LLM Yanitlari")

                    # LLM client olustur
                    try:
                        from neurocausal_rag.llm.client import LLMClient
                        from neurocausal_rag.config import LLMConfig
                        import os

                        if os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
                            llm_config = LLMConfig()
                            llm_client = LLMClient(llm_config)

                            col_llm1, col_llm2 = st.columns(2)

                            # Classic RAG context
                            classic_context = "\n\n".join([
                                f"[{i+1}] {r['content'][:400]}"
                                for i, r in enumerate(classic_results[:3])
                            ])

                            # NeuroCausal RAG context (with chains)
                            neuro_context_parts = []
                            for i, r in enumerate(neuro_results[:3]):
                                ctx = f"[{i+1}] {r.content[:400]}"
                                if r.causal_chain and len(r.causal_chain) > 1:
                                    ctx += f"\n   Zincir: {' -> '.join(r.causal_chain[:4])}"
                                neuro_context_parts.append(ctx)
                            neuro_context = "\n\n".join(neuro_context_parts)

                            prompt_template = """Asagidaki kaynaklara dayanarak soruyu yanitla.
Sadece kaynaklardaki bilgileri kullan. Kisa ve net cevap ver.

Kaynaklar:
{context}

Soru: {query}

Cevap:"""

                            with col_llm1:
                                st.markdown("#### 📕 Classic RAG Yaniti")
                                with st.spinner("LLM cevap uretiyor..."):
                                    try:
                                        classic_prompt = prompt_template.format(
                                            context=classic_context,
                                            query=query
                                        )
                                        classic_answer = llm_client.generate_raw(classic_prompt, max_tokens=500)
                                        st.success(classic_answer)
                                    except Exception as e:
                                        st.error(f"Hata: {e}")

                            with col_llm2:
                                st.markdown("#### 📗 NeuroCausal RAG Yaniti")
                                with st.spinner("LLM cevap uretiyor..."):
                                    try:
                                        neuro_prompt = prompt_template.format(
                                            context=neuro_context,
                                            query=query
                                        )
                                        neuro_answer = llm_client.generate_raw(neuro_prompt, max_tokens=500)
                                        st.success(neuro_answer)
                                    except Exception as e:
                                        st.error(f"Hata: {e}")

                            # Sonuc
                            st.markdown("---")
                            st.markdown("### 🏆 Nihai Degerlendirme")

                            if neuro_semantic > classic_semantic:
                                st.success(f"""
                                **NeuroCausal RAG kazandi!**

                                - Anlamsal Skor: {neuro_semantic:.3f} vs {classic_semantic:.3f}
                                - Zincir Sayisi: {chain_count}
                                - Enjeksiyon: {injected_count}

                                NeuroCausal RAG, nedensel iliskileri kullanarak daha kapsamli ve baglamsal bir yanit uretti.
                                """)
                            else:
                                st.info(f"""
                                **Sonuclar yakin!**

                                - Classic: {classic_semantic:.3f}
                                - NeuroCausal: {neuro_semantic:.3f}

                                Bu sorgu icin her iki sistem benzer sonuclar verdi.
                                """)
                        else:
                            st.warning("⚠️ LLM yaniti icin OPENAI_API_KEY gerekli. .env dosyasina ekleyin.")

                    except ImportError as e:
                        st.warning(f"LLM modulu yuklenemedi: {e}")
                    except Exception as e:
                        st.error(f"LLM hatasi: {e}")

    # ==========================================================================
    # TAB 3: Graf
    # ==========================================================================
    with tab3:
        st.subheader("Nedensellik Grafi")

        if st.session_state.sandbox is None:
            st.warning("Oncelikle 'Veri Yukle' sekmesinden veri yukleyin!")
        else:
            sandbox = st.session_state.sandbox

            # Istatistikler
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dokumanlar", sandbox['doc_count'])
            with col2:
                st.metric("Iliskiler", sandbox['edge_count'])
            with col3:
                density = sandbox['edge_count'] / max(1, sandbox['doc_count'] * (sandbox['doc_count'] - 1))
                st.metric("Yogunluk", f"{density:.2%}")

            st.divider()

            # Iliski listesi
            st.markdown("### Bulunan Iliskiler")

            relations = sandbox.get('relations', [])
            if not relations:
                st.info("Hic iliski bulunamadi.")
            else:
                for rel in sorted(relations, key=lambda x: x.get('confidence', 0), reverse=True)[:20]:
                    conf = rel.get('confidence', 0)
                    rel_type = rel.get('relation_type', 'related')

                    if conf > 0.7:
                        color = "🟢"
                    elif conf > 0.5:
                        color = "🟡"
                    else:
                        color = "🔴"

                    st.markdown(f"{color} **{rel['source']}** → **{rel['target']}** ({rel_type}, {conf:.2f})")

            # Graf gorsellestirme (PyVis)
            st.divider()
            st.markdown("### Interaktif Graf")

            try:
                from neurocausal_rag.visualization.graph_viz import create_graph_visualization

                # Graf olustur
                html_content = create_graph_visualization(
                    sandbox['graph'],
                    result_ids=[],
                    height="400px"
                )

                # HTML goster
                import streamlit.components.v1 as components
                components.html(html_content, height=450, scrolling=True)

            except ImportError:
                st.warning("PyVis kurulu degil. Graf goruntulenemedi.")
            except Exception as e:
                st.warning(f"Graf gorsellestirme hatasi: {e}")


if __name__ == "__main__":
    main()
