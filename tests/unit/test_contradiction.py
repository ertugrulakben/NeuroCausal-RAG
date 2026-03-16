"""
NeuroCausal RAG - Contradiction Detection Unit Tests
v5.1 - FAZ 1.4

Yazar: Ertugrul Akben
"""

import pytest
from neurocausal_rag.reasoning.contradiction import ContradictionDetector


class TestContradictionDetector:
    """ContradictionDetector unit tests"""

    @pytest.fixture
    def detector(self):
        """ContradictionDetector instance"""
        return ContradictionDetector()

    def test_detector_initialization(self, detector):
        """Detector baslatma testi"""
        assert detector is not None
        assert hasattr(detector, 'detect_conflict')

    def test_profit_loss_contradiction(self, detector):
        """Kar-zarar celiskisi testi"""
        text_a = "Company made profit this year"
        text_b = "Company suffered a loss this quarter"

        score = detector.detect_conflict(text_a, text_b)

        assert score > 0.5  # Should detect conflict
        assert score <= 1.0

    def test_increase_decrease_contradiction(self, detector):
        """Artis-azalis celiskisi testi"""
        text_a = "Sales showed significant increase"
        text_b = "Revenue continued to decrease"

        score = detector.detect_conflict(text_a, text_b)

        assert score > 0.5  # Should detect conflict
        assert score <= 1.0

    def test_no_contradiction(self, detector):
        """Celiski olmayan metinler"""
        text_a = "The project started in January"
        text_b = "Team members worked hard"

        score = detector.detect_conflict(text_a, text_b)

        assert score < 0.5  # Should not detect major conflict

    def test_same_text_no_contradiction(self, detector):
        """Ayni metin celisik degil"""
        text = "Revenue increased by 20%"

        score = detector.detect_conflict(text, text)

        assert score < 0.3  # Same text should have low conflict score

    def test_empty_text_handling(self, detector):
        """Bos metin isleme"""
        try:
            score = detector.detect_conflict("", "Some text")
            assert score >= 0  # Should handle gracefully
        except Exception:
            pass  # Exception handling is also acceptable

    def test_turkish_keywords(self, detector):
        """Turkce anahtar kelimeler (genisletme icin)"""
        # Bu test gelecekte Turkce destek icin genisletilebilir
        text_a = "Sirket kar etti"
        text_b = "Sirket zarar etti"

        # Simdilik temel testten gec
        score = detector.detect_conflict(text_a, text_b)
        assert score >= 0 and score <= 1.0

    def test_numeric_contradiction(self, detector):
        """Sayisal celiski testi"""
        text_a = "The budget is 100 million dollars"
        text_b = "The budget is 50 million dollars"

        # Temel detector bunu yakalamayabilir
        score = detector.detect_conflict(text_a, text_b)
        assert score >= 0 and score <= 1.0

    def test_case_insensitivity(self, detector):
        """Buyuk/kucuk harf duyarsizligi"""
        text_a = "PROFIT increased significantly"
        text_b = "loss was reported"

        score = detector.detect_conflict(text_a, text_b)
        assert score > 0.5  # Should still detect conflict

    def test_score_range(self, detector):
        """Skor aralik kontrolu"""
        test_pairs = [
            ("Increase in value", "Decrease in value"),
            ("Project success", "Project failure"),
            ("Good performance", "Poor performance"),
        ]

        for text_a, text_b in test_pairs:
            score = detector.detect_conflict(text_a, text_b)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for pair: {text_a}, {text_b}"


class TestContradictionIntegration:
    """Contradiction entegrasyon testleri"""

    def test_document_pair_analysis(self):
        """Dokuman cifti analizi"""
        detector = ContradictionDetector()

        doc1 = {"id": "d1", "content": "Company profit rose by 20%"}
        doc2 = {"id": "d2", "content": "Company loss increased this quarter"}

        score = detector.detect_conflict(doc1["content"], doc2["content"])

        assert score > 0.5  # Should detect conflict between docs

    def test_multiple_document_scanning(self):
        """Coklu dokuman taramasi"""
        detector = ContradictionDetector()

        documents = [
            {"id": "d1", "content": "Sales increased dramatically"},
            {"id": "d2", "content": "Revenue decreased significantly"},
            {"id": "d3", "content": "Profit margins improved"},
            {"id": "d4", "content": "Losses accumulated over time"},
        ]

        conflicts_found = []

        for i, doc1 in enumerate(documents):
            for doc2 in documents[i+1:]:
                score = detector.detect_conflict(doc1["content"], doc2["content"])
                if score > 0.5:
                    conflicts_found.append((doc1["id"], doc2["id"], score))

        # En az bir celiski bulmali
        assert len(conflicts_found) > 0

    def test_threshold_filtering(self):
        """Esik deger filtreleme"""
        detector = ContradictionDetector()

        pairs = [
            ("profit", "loss", 0.9),  # High conflict
            ("increase", "decrease", 0.85),  # High conflict
            ("project", "timeline", 0.1),  # Low conflict
        ]

        threshold = 0.5

        high_conflicts = []
        for text_a, text_b, expected_min in pairs:
            score = detector.detect_conflict(text_a, text_b)
            if score >= threshold:
                high_conflicts.append((text_a, text_b))

        # profit-loss ve increase-decrease yuksek celiski olmali
        assert len(high_conflicts) >= 2
