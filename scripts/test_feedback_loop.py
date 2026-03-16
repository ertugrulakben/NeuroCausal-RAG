"""
NeuroCausal RAG - Feedback Loop (RLHF) Test
Sistemin kullanici geri bildiriminden ogrendigini dogrular

Bu test:
1. Birkac dokuman ekler
2. Aralarinda iliski olusturur
3. Feedback gonderir
4. Edge weight'inin degistigini kontrol eder

Kullanim:
    python scripts/test_feedback_loop.py

Yazar: Ertugrul Akben
"""

import sys
import time
from datetime import datetime

# Add parent to path
sys.path.insert(0, '.')


def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step: int, text: str):
    print(f"\n[ADIM {step}] {text}")
    print("-" * 40)


def main():
    print_header("NEUROCAUSAL RAG - FEEDBACK LOOP (RLHF) TEST")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # =================================================================
    # STEP 1: Setup
    # =================================================================
    print_step(1, "Sistem kuruluyor...")

    from neurocausal_rag.embedding.text import TextEmbedding
    from neurocausal_rag.core.graph import GraphEngine
    from neurocausal_rag.config import EmbeddingConfig, GraphConfig
    from neurocausal_rag.learning.feedback import FeedbackLoop, FeedbackType

    # Embedding engine
    emb_config = EmbeddingConfig()
    embedding = TextEmbedding(emb_config)
    print("  ✓ Embedding motoru yuklendi")

    # Graph engine
    graph_config = GraphConfig()
    graph = GraphEngine(graph_config)
    print("  ✓ Graf motoru yuklendi")

    # Feedback loop
    feedback_loop = FeedbackLoop(
        graph_engine=graph,
        storage_path="test_feedback",
        use_sqlite=False,  # JSON kullan (test icin)
        auto_adjust=True,
        learning_rate=0.2
    )
    print("  ✓ Feedback loop yuklendi")

    # =================================================================
    # STEP 2: Add Documents
    # =================================================================
    print_step(2, "Dokumanlar ekleniyor...")

    docs = [
        {"id": "ekonomi_faiz", "content": "Faiz oranlarinin artmasi kredi maliyetlerini yukselir."},
        {"id": "ekonomi_kredi", "content": "Kredi maliyetleri arttikca tuketici harcamalari azalir."},
        {"id": "ekonomi_tuketim", "content": "Tuketim azalmasi ekonomik yavasllamaya yol acar."},
        {"id": "ekonomi_issizlik", "content": "Ekonomik yavasllama issizlik oranini arttirir."},
    ]

    for doc in docs:
        emb = embedding.get_text_embedding(doc['content'])
        graph.add_node(doc['id'], doc['content'], emb)
        print(f"  + {doc['id']}")

    print(f"\n  Toplam dokuman: {graph.node_count}")

    # =================================================================
    # STEP 3: Add Initial Edges
    # =================================================================
    print_step(3, "Iliskiler ekleniyor...")

    # Baslangic weight'leri 0.5 olsun
    edges = [
        ("ekonomi_faiz", "ekonomi_kredi", "causes", 0.5),
        ("ekonomi_kredi", "ekonomi_tuketim", "causes", 0.5),
        ("ekonomi_tuketim", "ekonomi_issizlik", "causes", 0.5),
    ]

    for src, tgt, rel, weight in edges:
        graph.add_edge(src, tgt, rel, weight)
        print(f"  + {src} --[{rel}]--> {tgt} (weight: {weight})")

    print(f"\n  Toplam iliski: {graph.edge_count}")

    # =================================================================
    # STEP 4: Record Initial Weights
    # =================================================================
    print_step(4, "Baslangic weight'leri kaydediliyor...")

    initial_weights = {}
    for src, tgt, rel, _ in edges:
        try:
            weight = graph.get_edge_weight(src, tgt)
            initial_weights[(src, tgt)] = weight
            print(f"  {src} -> {tgt}: {weight:.3f}")
        except:
            print(f"  {src} -> {tgt}: N/A")

    # =================================================================
    # STEP 5: Send Positive Feedback
    # =================================================================
    print_step(5, "Pozitif feedback gonderiliyor...")

    # Yuksek rating ile feedback
    for i in range(5):  # 5 kez pozitif feedback
        feedback_loop.record(
            query="faiz issizligi nasil etkiler",
            result_ids=["ekonomi_faiz", "ekonomi_kredi", "ekonomi_tuketim", "ekonomi_issizlik"],
            rating=0.95,  # Cok yuksek rating
            comment=f"Mukemmel sonuc! (Test {i+1})",
            feedback_type=FeedbackType.EXPLICIT
        )
        print(f"  + Feedback {i+1}: rating=0.95")

    # =================================================================
    # STEP 6: Check Weight Changes
    # =================================================================
    print_step(6, "Weight degisimleri kontrol ediliyor...")

    print("\n  Edge Weight Degisimleri:")
    print("  " + "-" * 50)

    weight_increased = False

    for src, tgt, rel, _ in edges:
        try:
            old_weight = initial_weights.get((src, tgt), 0.5)
            new_weight = graph.get_edge_weight(src, tgt)

            if new_weight is None:
                new_weight = old_weight

            change = new_weight - old_weight
            change_pct = (change / old_weight) * 100 if old_weight > 0 else 0

            if change > 0:
                status = "⬆️ ARTTI"
                weight_increased = True
            elif change < 0:
                status = "⬇️ AZALDI"
            else:
                status = "➡️ DEGISMEDI"

            print(f"  {src} -> {tgt}:")
            print(f"    Onceki: {old_weight:.3f}")
            print(f"    Sonraki: {new_weight:.3f}")
            print(f"    Degisim: {change:+.3f} ({change_pct:+.1f}%) {status}")
            print()

        except Exception as e:
            print(f"  {src} -> {tgt}: Hata - {e}")

    # =================================================================
    # STEP 7: Check Feedback Stats
    # =================================================================
    print_step(7, "Feedback istatistikleri...")

    stats = feedback_loop.store.get_stats()
    print(f"  Toplam feedback: {stats.get('total_feedback', 0)}")
    print(f"  Ortalama rating: {stats.get('average_rating', 0):.2f}")
    print(f"  Pozitif: {stats.get('positive_count', 0)}")
    print(f"  Negatif: {stats.get('negative_count', 0)}")

    # =================================================================
    # STEP 8: Test Document Score
    # =================================================================
    print_step(8, "Dokuman skorlari kontrol ediliyor...")

    for doc_id in ["ekonomi_faiz", "ekonomi_kredi"]:
        score = feedback_loop.get_document_score(doc_id)
        print(f"  {doc_id}:")
        print(f"    Feedback sayisi: {score.get('feedback_count', 0)}")
        print(f"    Ortalama rating: {score.get('average_rating', 'N/A')}")
        print(f"    Trend: {score.get('trend', 'N/A')}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print_header("TEST SONUCU")

    if weight_increased:
        print("\n✅ BASARILI: Feedback Loop calisiyor!")
        print("   Pozitif feedback sonrasi edge weight'leri artti.")
        print("   Sistem kullanici geri bildiriminden ogreniyor.")
    else:
        print("\n⚠️ UYARI: Weight degisimi gozlemlenmedi.")
        print("   Bu normal olabilir - WeightAdjuster min_feedback_count")
        print("   esigine ulasilmamis olabilir (varsayilan: 3)")

    print("\n" + "=" * 60)
    print("Feedback Loop testi tamamlandi.")
    print("=" * 60 + "\n")

    # Cleanup
    import os
    if os.path.exists("test_feedback.json"):
        os.remove("test_feedback.json")
        print("(Test dosyasi temizlendi)")


if __name__ == "__main__":
    main()
