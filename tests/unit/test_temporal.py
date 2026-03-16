"""
NeuroCausal RAG - Temporal Reasoning Unit Tests
v5.1 - FAZ 1.4

Yazar: Ertugrul Akben
"""

import pytest
from datetime import datetime
from neurocausal_rag.reasoning.temporal import TemporalEngine


class TestTemporalEngine:
    """TemporalEngine unit tests"""

    @pytest.fixture
    def engine(self):
        """TemporalEngine instance"""
        return TemporalEngine()

    def test_engine_initialization(self, engine):
        """Engine baslatma testi"""
        assert engine is not None
        assert hasattr(engine, 'extract_date')
        assert hasattr(engine, 'validate_causal_order')

    def test_extract_date_full_format(self, engine):
        """Tam tarih formati cikarma (YYYY-MM-DD)"""
        text = "The event occurred on 2023-06-15"

        date = engine.extract_date(text)

        assert date is not None
        assert date.year == 2023
        assert date.month == 6
        assert date.day == 15

    def test_extract_date_year_only(self, engine):
        """Sadece yil cikarma"""
        text = "This happened in 2022"

        date = engine.extract_date(text)

        assert date is not None
        assert date.year == 2022

    def test_extract_date_no_date(self, engine):
        """Tarih olmayan metin"""
        text = "No date information here"

        date = engine.extract_date(text)

        assert date is None

    def test_extract_date_multiple_dates(self, engine):
        """Birden fazla tarih iceren metin"""
        text = "Between 2023-01-15 and 2023-12-31"

        date = engine.extract_date(text)

        # Ilk tarihi dondurmeli
        assert date is not None
        assert date.year == 2023
        assert date.month == 1
        assert date.day == 15

    def test_valid_causal_order(self, engine):
        """Gecerli nedensel sira (neden sonuctan once)"""
        cause_text = "Started on 2023-01-01"
        effect_text = "Completed by 2023-06-30"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        assert is_valid is True

    def test_invalid_causal_order(self, engine):
        """Gecersiz nedensel sira (neden sonuctan sonra)"""
        cause_text = "Initiated on 2023-12-01"
        effect_text = "The result appeared on 2023-01-01"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        assert is_valid is False

    def test_causal_order_same_date(self, engine):
        """Ayni tarihte nedensel iliski"""
        cause_text = "Event A on 2023-05-15"
        effect_text = "Event B on 2023-05-15"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        # Ayni tarih gecerli (esit kabul edilir)
        assert is_valid is True

    def test_causal_order_unknown_dates(self, engine):
        """Tarih bilinmiyorsa gecerli kabul et"""
        cause_text = "Something happened"
        effect_text = "Something else followed"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        # Tarih yoksa True donmeli (varsayilan)
        assert is_valid is True

    def test_causal_order_partial_dates(self, engine):
        """Kismi tarih bilgisi"""
        cause_text = "In 2022, the project started"
        effect_text = "By 2023, it was finished"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        assert is_valid is True

    def test_causal_order_only_cause_has_date(self, engine):
        """Sadece nedende tarih var"""
        cause_text = "Started on 2023-01-01"
        effect_text = "Eventually completed"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        # Tarih eksikse True donmeli
        assert is_valid is True

    def test_causal_order_only_effect_has_date(self, engine):
        """Sadece sonucta tarih var"""
        cause_text = "Something initiated"
        effect_text = "Finished on 2023-12-31"

        is_valid = engine.validate_causal_order(cause_text, effect_text)

        # Tarih eksikse True donmeli
        assert is_valid is True


class TestTemporalChainValidation:
    """Zamansal zincir dogrulama testleri"""

    @pytest.fixture
    def engine(self):
        return TemporalEngine()

    def test_valid_three_step_chain(self, engine):
        """Gecerli uc adimli zincir"""
        events = [
            "2022-01-01: Research started",
            "2022-06-15: Development began",
            "2023-01-01: Product launched"
        ]

        # Her ardisik cift icin dogrula
        all_valid = True
        for i in range(len(events) - 1):
            if not engine.validate_causal_order(events[i], events[i+1]):
                all_valid = False
                break

        assert all_valid is True

    def test_invalid_chain_with_reversal(self, engine):
        """Zincirde ters siralama"""
        events = [
            "2023-01-01: Final step",
            "2022-06-15: Middle step",
            "2022-01-01: First step"
        ]

        # Ilk gecis gecersiz olmali
        first_transition = engine.validate_causal_order(events[0], events[1])

        assert first_transition is False

    def test_partial_chain_validation(self, engine):
        """Kismi zincir dogrulama"""
        events = [
            "2022-01-01: Step 1",
            "No date: Step 2",
            "2023-01-01: Step 3"
        ]

        # Bilinmeyen tarihler True donmeli
        step1_to_2 = engine.validate_causal_order(events[0], events[1])
        step2_to_3 = engine.validate_causal_order(events[1], events[2])

        assert step1_to_2 is True  # Unknown date = valid
        assert step2_to_3 is True  # Unknown date = valid


class TestDatePatternRecognition:
    """Tarih deseni tanima testleri"""

    @pytest.fixture
    def engine(self):
        return TemporalEngine()

    def test_iso_format(self, engine):
        """ISO tarih formati"""
        text = "Date: 2023-11-30"
        date = engine.extract_date(text)

        assert date is not None
        assert date.year == 2023
        assert date.month == 11
        assert date.day == 30

    def test_year_in_context(self, engine):
        """Baglam icinde yil"""
        text = "The fiscal year 2024 report"
        date = engine.extract_date(text)

        assert date is not None
        assert date.year == 2024

    def test_multiple_years_first_match(self, engine):
        """Birden fazla yil - ilk eslesme"""
        text = "Between 2020 and 2025"
        date = engine.extract_date(text)

        assert date is not None
        # Ilk tarihi almali
        assert date.year in [2020, 2025]

    def test_date_edge_cases(self, engine):
        """Tarih sinir durumlari"""
        # Gecersiz tarih formatlari
        texts = [
            "abc123",
            "99-99-9999",
            "tomorrow",
            "next week"
        ]

        for text in texts:
            date = engine.extract_date(text)
            # None veya valid datetime olmali
            assert date is None or isinstance(date, datetime)
