"""
NeuroCausal RAG - Memory Store Tests
v5.2

Yazar: Ertugrul Akben
"""

import pytest
import os
import tempfile
import json
from neurocausal_rag.memory import (
    MemoryStore,
    MemoryNote,
    CausalFeedback,
    MemoryStats,
    create_memory_store
)
from neurocausal_rag.memory.store import FeedbackType


@pytest.fixture
def temp_db():
    """Geçici veritabanı"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def memory_store(temp_db):
    """Test memory store"""
    return MemoryStore(temp_db)


class TestMemoryNote:
    """MemoryNote dataclass testleri"""

    def test_note_creation(self):
        """Not oluşturma"""
        note = MemoryNote(
            id="test_note",
            content="Test content"
        )
        assert note.id == "test_note"
        assert note.content == "Test content"
        assert note.created_at != ""

    def test_note_with_related_docs(self):
        """İlişkili dokümanlarla not"""
        note = MemoryNote(
            id="test",
            content="Test",
            related_docs=["doc1", "doc2"]
        )
        assert len(note.related_docs) == 2


class TestCausalFeedback:
    """CausalFeedback dataclass testleri"""

    def test_feedback_creation(self):
        """Geri bildirim oluşturma"""
        fb = CausalFeedback(
            id="fb_test",
            source_id="doc1",
            target_id="doc2",
            relation_type="causes",
            feedback_type="positive"
        )
        assert fb.source_id == "doc1"
        assert fb.target_id == "doc2"


class TestMemoryStoreInit:
    """MemoryStore başlatma testleri"""

    def test_initialization(self, memory_store):
        """Store başlatma"""
        assert memory_store is not None
        assert os.path.exists(memory_store.db_path)

    def test_factory_function(self, temp_db):
        """Factory function"""
        store = create_memory_store(temp_db)
        assert isinstance(store, MemoryStore)


class TestNoteManagement:
    """Not yönetimi testleri"""

    def test_add_note(self, memory_store):
        """Not ekleme"""
        note = memory_store.add_note("Test note content")

        assert note.id.startswith("note_")
        assert note.content == "Test note content"

    def test_add_note_with_metadata(self, memory_store):
        """Metadata ile not ekleme"""
        note = memory_store.add_note(
            content="Test",
            related_docs=["doc1"],
            related_queries=["query1"],
            tags=["important"]
        )

        assert "doc1" in note.related_docs
        assert "important" in note.tags

    def test_get_notes(self, memory_store):
        """Notları getirme"""
        memory_store.add_note("Note 1")
        memory_store.add_note("Note 2")

        notes = memory_store.get_notes()

        assert len(notes) == 2

    def test_get_notes_by_tag(self, memory_store):
        """Etikete göre notları getirme"""
        memory_store.add_note("Tagged note", tags=["important"])
        memory_store.add_note("Other note", tags=["other"])

        notes = memory_store.get_notes(tag="important")

        assert len(notes) == 1
        assert "important" in notes[0].tags

    def test_delete_note(self, memory_store):
        """Not silme"""
        note = memory_store.add_note("To delete")

        deleted = memory_store.delete_note(note.id)

        assert deleted is True
        assert len(memory_store.get_notes()) == 0


class TestCausalManagement:
    """Nedensellik yönetimi testleri"""

    def test_add_causal_relation(self, memory_store):
        """Manuel nedensellik ekleme"""
        fb = memory_store.add_causal_relation(
            source_id="doc_a",
            target_id="doc_b",
            relation_type="causes",
            note="Kesinlikle neden olur"
        )

        assert fb.source_id == "doc_a"
        assert fb.target_id == "doc_b"
        assert fb.feedback_type == FeedbackType.MANUAL_ADD.value

    def test_remove_causal_relation(self, memory_store):
        """Nedensellik silme işareti"""
        fb = memory_store.remove_causal_relation(
            source_id="doc_x",
            target_id="doc_y",
            note="Bu ilişki yanlış"
        )

        assert fb.feedback_type == FeedbackType.MANUAL_DEL.value

    def test_add_positive_feedback(self, memory_store):
        """Olumlu geri bildirim"""
        fb = memory_store.add_feedback(
            source_id="a",
            target_id="b",
            relation_type="causes",
            is_positive=True,
            note="Bu işe yaradı"
        )

        assert fb.feedback_type == FeedbackType.POSITIVE.value
        assert fb.confidence == 1.0

    def test_add_negative_feedback(self, memory_store):
        """Olumsuz geri bildirim"""
        fb = memory_store.add_feedback(
            source_id="a",
            target_id="b",
            relation_type="causes",
            is_positive=False,
            note="Bu işe yaramadı"
        )

        assert fb.feedback_type == FeedbackType.NEGATIVE.value
        assert fb.confidence == 0.0


class TestFeedbackRetrieval:
    """Geri bildirim getirme testleri"""

    def test_get_all_feedbacks(self, memory_store):
        """Tüm geri bildirimleri getir"""
        memory_store.add_feedback("a", "b", "causes", True)
        memory_store.add_feedback("c", "d", "supports", False)

        feedbacks = memory_store.get_feedbacks()

        assert len(feedbacks) == 2

    def test_get_feedbacks_by_type(self, memory_store):
        """Türe göre geri bildirim getir"""
        memory_store.add_feedback("a", "b", "causes", True)
        memory_store.add_feedback("c", "d", "causes", False)

        positive = memory_store.get_feedbacks(
            feedback_type=FeedbackType.POSITIVE.value
        )

        assert len(positive) == 1

    def test_get_pending_additions(self, memory_store):
        """Bekleyen eklemeleri getir"""
        memory_store.add_causal_relation("a", "b", "causes")
        memory_store.add_causal_relation("c", "d", "supports")

        pending = memory_store.get_pending_additions()

        assert len(pending) == 2

    def test_mark_feedback_applied(self, memory_store):
        """Uygulandı olarak işaretle"""
        fb = memory_store.add_causal_relation("a", "b", "causes")

        memory_store.mark_feedback_applied(fb.id)

        pending = memory_store.get_pending_additions()
        assert len(pending) == 0


class TestStats:
    """İstatistik testleri"""

    def test_get_stats_empty(self, memory_store):
        """Boş store istatistikleri"""
        stats = memory_store.get_stats()

        assert stats.total_notes == 0
        assert stats.total_feedbacks == 0

    def test_get_stats_with_data(self, memory_store):
        """Verili store istatistikleri"""
        memory_store.add_note("Test")
        memory_store.add_feedback("a", "b", "causes", True)
        memory_store.add_feedback("c", "d", "causes", False)
        memory_store.add_causal_relation("x", "y", "causes")

        stats = memory_store.get_stats()

        assert stats.total_notes == 1
        assert stats.total_feedbacks == 3
        assert stats.positive_feedbacks == 1
        assert stats.negative_feedbacks == 1
        assert stats.manual_additions == 1


class TestReset:
    """Sıfırlama testleri"""

    def test_reset_without_confirm(self, memory_store):
        """Onaysız sıfırlama başarısız olmalı"""
        memory_store.add_note("Test")

        result = memory_store.reset(confirm=False)

        assert result is False
        assert len(memory_store.get_notes()) == 1

    def test_reset_with_confirm(self, memory_store):
        """Onaylı sıfırlama"""
        memory_store.add_note("Test")
        memory_store.add_feedback("a", "b", "causes", True)

        result = memory_store.reset(confirm=True)

        assert result is True
        assert len(memory_store.get_notes()) == 0
        assert len(memory_store.get_feedbacks()) == 0


class TestExportImport:
    """Dışa/içe aktarma testleri"""

    def test_export_to_json(self, memory_store, tmp_path):
        """JSON dışa aktarma"""
        memory_store.add_note("Test note")
        memory_store.add_feedback("a", "b", "causes", True)

        output_path = str(tmp_path / "export.json")
        memory_store.export_to_json(output_path)

        assert os.path.exists(output_path)

        with open(output_path, 'r') as f:
            data = json.load(f)

        assert len(data["notes"]) == 1
        assert len(data["feedbacks"]) == 1

    def test_import_from_json(self, memory_store, tmp_path):
        """JSON içe aktarma"""
        # Export data oluştur
        export_data = {
            "exported_at": "2025-01-01",
            "stats": {},
            "notes": [
                {
                    "id": "note_1",
                    "content": "Imported note",
                    "related_docs": ["doc1"],
                    "related_queries": [],
                    "tags": ["imported"]
                }
            ],
            "feedbacks": [
                {
                    "id": "fb_1",
                    "source_id": "x",
                    "target_id": "y",
                    "relation_type": "causes",
                    "feedback_type": "positive",
                    "user_note": "test",
                    "confidence": 1.0
                }
            ]
        }

        input_path = str(tmp_path / "import.json")
        with open(input_path, 'w') as f:
            json.dump(export_data, f)

        notes, feedbacks = memory_store.import_from_json(input_path)

        assert notes == 1
        assert feedbacks == 1

        # Doğrula
        all_notes = memory_store.get_notes()
        assert len(all_notes) == 1
        assert "Imported note" in all_notes[0].content
