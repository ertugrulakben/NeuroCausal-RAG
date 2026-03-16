"""
NeuroCausal RAG - Memory Store
v5.2 - Kalıcı Hafıza Sistemi

Özellikler:
1. Kullanıcı notları ve gözlemleri
2. Manuel nedensellik ekleme/silme
3. Model geri bildirimi entegrasyonu
4. Sıfırlama ve yedekleme

Yazar: Ertugrul Akben
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Geri bildirim türleri"""
    POSITIVE = "positive"      # Bu işe yaradı
    NEGATIVE = "negative"      # Bu işe yaramadı
    CORRECTION = "correction"  # Düzeltme
    MANUAL_ADD = "manual_add"  # Manuel ekleme
    MANUAL_DEL = "manual_del"  # Manuel silme


@dataclass
class MemoryNote:
    """Kullanıcı notu"""
    id: str
    content: str
    related_docs: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class CausalFeedback:
    """Nedensellik geri bildirimi"""
    id: str
    source_id: str           # Kaynak doküman
    target_id: str           # Hedef doküman
    relation_type: str       # causes, supports, etc.
    feedback_type: str       # positive, negative, correction
    user_note: str = ""      # Kullanıcı açıklaması
    confidence: float = 1.0  # Güven skoru
    created_at: str = ""
    is_applied: bool = False # Graf'a uygulandı mı

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class MemoryStats:
    """Hafıza istatistikleri"""
    total_notes: int = 0
    total_feedbacks: int = 0
    positive_feedbacks: int = 0
    negative_feedbacks: int = 0
    manual_additions: int = 0
    manual_deletions: int = 0
    last_updated: str = ""


class MemoryStore:
    """
    Kalıcı Hafıza Deposu

    SQLite tabanlı kalıcı depolama ile kullanıcı notlarını,
    manuel düzenlemeleri ve model geri bildirimlerini saklar.

    Kullanım:
        store = MemoryStore("memory.db")

        # Not ekle
        store.add_note("Bu sorgu iyi çalıştı", related_docs=["doc1"])

        # Nedensellik ekle (manuel)
        store.add_causal_relation("doc_a", "doc_b", "causes", note="Kesinlikle neden olur")

        # Geri bildirim ver
        store.add_feedback("doc_x", "doc_y", "causes", FeedbackType.POSITIVE)

        # Sıfırla
        store.reset()
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        auto_create: bool = True
    ):
        """
        Args:
            db_path: SQLite veritabanı dosya yolu
            auto_create: Veritabanı yoksa otomatik oluştur
        """
        self.db_path = db_path

        if auto_create:
            self._init_db()

    def _init_db(self) -> None:
        """Veritabanı tablolarını oluştur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Notlar tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                related_docs TEXT,
                related_queries TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')

        # Geri bildirimler tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedbacks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT,
                feedback_type TEXT,
                user_note TEXT,
                confidence REAL,
                created_at TEXT,
                is_applied INTEGER DEFAULT 0
            )
        ''')

        # İstatistikler tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        conn.commit()
        conn.close()

        logger.info(f"Memory database initialized: {self.db_path}")

    # ==================== NOT YÖNETİMİ ====================

    def add_note(
        self,
        content: str,
        related_docs: Optional[List[str]] = None,
        related_queries: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> MemoryNote:
        """
        Not ekle.

        Args:
            content: Not içeriği
            related_docs: İlişkili doküman ID'leri
            related_queries: İlişkili sorgular
            tags: Etiketler

        Returns:
            Oluşturulan not
        """
        note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        note = MemoryNote(
            id=note_id,
            content=content,
            related_docs=related_docs or [],
            related_queries=related_queries or [],
            tags=tags or []
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO notes (id, content, related_docs, related_queries, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            note.id,
            note.content,
            json.dumps(note.related_docs),
            json.dumps(note.related_queries),
            json.dumps(note.tags),
            note.created_at,
            note.updated_at
        ))

        conn.commit()
        conn.close()

        logger.info(f"Note added: {note_id}")
        return note

    def get_notes(
        self,
        tag: Optional[str] = None,
        doc_id: Optional[str] = None,
        limit: int = 100
    ) -> List[MemoryNote]:
        """Notları getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM notes"
        params = []

        if tag:
            query += " WHERE tags LIKE ?"
            params.append(f'%"{tag}"%')
        elif doc_id:
            query += " WHERE related_docs LIKE ?"
            params.append(f'%"{doc_id}"%')

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        notes = []
        for row in rows:
            notes.append(MemoryNote(
                id=row[0],
                content=row[1],
                related_docs=json.loads(row[2]) if row[2] else [],
                related_queries=json.loads(row[3]) if row[3] else [],
                tags=json.loads(row[4]) if row[4] else [],
                created_at=row[5],
                updated_at=row[6]
            ))

        return notes

    def delete_note(self, note_id: str) -> bool:
        """Notu sil"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    # ==================== NEDENSELLİK YÖNETİMİ ====================

    def add_causal_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "causes",
        note: str = "",
        confidence: float = 1.0
    ) -> CausalFeedback:
        """
        Manuel nedensellik ilişkisi ekle.

        Args:
            source_id: Kaynak doküman ID
            target_id: Hedef doküman ID
            relation_type: İlişki türü (causes, supports, requires, related)
            note: Açıklama
            confidence: Güven skoru (0-1)

        Returns:
            Oluşturulan geri bildirim
        """
        return self._add_feedback(
            source_id, target_id, relation_type,
            FeedbackType.MANUAL_ADD.value, note, confidence
        )

    def remove_causal_relation(
        self,
        source_id: str,
        target_id: str,
        note: str = ""
    ) -> CausalFeedback:
        """
        Nedensellik ilişkisini kaldır (silme işareti).

        Bu işlem graf'tan silmez, sadece silme işareti koyar.
        Uygulama sırasında bu işarete göre ilişki kaldırılır.
        """
        return self._add_feedback(
            source_id, target_id, "",
            FeedbackType.MANUAL_DEL.value, note, 1.0
        )

    def add_feedback(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        is_positive: bool,
        note: str = ""
    ) -> CausalFeedback:
        """
        Model/kullanıcı geri bildirimi ekle.

        Args:
            source_id: Kaynak doküman
            target_id: Hedef doküman
            relation_type: İlişki türü
            is_positive: Olumlu mu olumsuz mu
            note: Açıklama
        """
        feedback_type = FeedbackType.POSITIVE.value if is_positive else FeedbackType.NEGATIVE.value
        return self._add_feedback(
            source_id, target_id, relation_type,
            feedback_type, note, 1.0 if is_positive else 0.0
        )

    def _add_feedback(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        feedback_type: str,
        note: str,
        confidence: float
    ) -> CausalFeedback:
        """Geri bildirim ekle (internal)"""
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        feedback = CausalFeedback(
            id=feedback_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            feedback_type=feedback_type,
            user_note=note,
            confidence=confidence
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO feedbacks
            (id, source_id, target_id, relation_type, feedback_type, user_note, confidence, created_at, is_applied)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.id,
            feedback.source_id,
            feedback.target_id,
            feedback.relation_type,
            feedback.feedback_type,
            feedback.user_note,
            feedback.confidence,
            feedback.created_at,
            0
        ))

        conn.commit()
        conn.close()

        logger.info(f"Feedback added: {feedback_id} ({feedback_type})")
        return feedback

    def get_feedbacks(
        self,
        feedback_type: Optional[str] = None,
        applied_only: bool = False,
        pending_only: bool = False,
        limit: int = 100
    ) -> List[CausalFeedback]:
        """Geri bildirimleri getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM feedbacks WHERE 1=1"
        params = []

        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type)

        if applied_only:
            query += " AND is_applied = 1"
        elif pending_only:
            query += " AND is_applied = 0"

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        feedbacks = []
        for row in rows:
            feedbacks.append(CausalFeedback(
                id=row[0],
                source_id=row[1],
                target_id=row[2],
                relation_type=row[3],
                feedback_type=row[4],
                user_note=row[5],
                confidence=row[6],
                created_at=row[7],
                is_applied=bool(row[8])
            ))

        return feedbacks

    def mark_feedback_applied(self, feedback_id: str) -> bool:
        """Geri bildirimi uygulandı olarak işaretle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE feedbacks SET is_applied = 1 WHERE id = ?",
            (feedback_id,)
        )
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def get_pending_additions(self) -> List[CausalFeedback]:
        """Bekleyen manuel eklemeleri getir"""
        return self.get_feedbacks(
            feedback_type=FeedbackType.MANUAL_ADD.value,
            pending_only=True
        )

    def get_pending_deletions(self) -> List[CausalFeedback]:
        """Bekleyen manuel silmeleri getir"""
        return self.get_feedbacks(
            feedback_type=FeedbackType.MANUAL_DEL.value,
            pending_only=True
        )

    # ==================== İSTATİSTİKLER ====================

    def get_stats(self) -> MemoryStats:
        """Hafıza istatistiklerini getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Not sayısı
        cursor.execute("SELECT COUNT(*) FROM notes")
        total_notes = cursor.fetchone()[0]

        # Geri bildirim sayıları
        cursor.execute("SELECT COUNT(*) FROM feedbacks")
        total_feedbacks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE feedback_type = ?",
                      (FeedbackType.POSITIVE.value,))
        positive = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE feedback_type = ?",
                      (FeedbackType.NEGATIVE.value,))
        negative = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE feedback_type = ?",
                      (FeedbackType.MANUAL_ADD.value,))
        manual_add = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE feedback_type = ?",
                      (FeedbackType.MANUAL_DEL.value,))
        manual_del = cursor.fetchone()[0]

        conn.close()

        return MemoryStats(
            total_notes=total_notes,
            total_feedbacks=total_feedbacks,
            positive_feedbacks=positive,
            negative_feedbacks=negative,
            manual_additions=manual_add,
            manual_deletions=manual_del,
            last_updated=datetime.now().isoformat()
        )

    # ==================== SIFIRLAMA / YEDEK ====================

    def reset(self, confirm: bool = False) -> bool:
        """
        Tüm hafızayı sıfırla.

        UYARI: Bu işlem geri alınamaz!

        Args:
            confirm: Onay bayrağı (True olmalı)

        Returns:
            Başarılı ise True
        """
        if not confirm:
            logger.warning("Reset cancelled: confirm=False")
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM notes")
        cursor.execute("DELETE FROM feedbacks")
        cursor.execute("DELETE FROM stats")

        conn.commit()
        conn.close()

        logger.info("Memory store reset complete")
        return True

    def export_to_json(self, output_path: str) -> str:
        """
        Tüm hafızayı JSON olarak dışa aktar.

        Args:
            output_path: Çıktı dosya yolu

        Returns:
            Dosya yolu
        """
        notes = self.get_notes(limit=10000)
        feedbacks = self.get_feedbacks(limit=10000)
        stats = self.get_stats()

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "stats": asdict(stats),
            "notes": [asdict(n) for n in notes],
            "feedbacks": [asdict(f) for f in feedbacks]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Memory exported to: {output_path}")
        return output_path

    def import_from_json(self, input_path: str) -> Tuple[int, int]:
        """
        JSON'dan hafıza içe aktar.

        Args:
            input_path: Girdi dosya yolu

        Returns:
            (not_sayısı, geri_bildirim_sayısı)
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        notes_imported = 0
        feedbacks_imported = 0

        # Notları içe aktar
        for note_data in data.get("notes", []):
            self.add_note(
                content=note_data["content"],
                related_docs=note_data.get("related_docs", []),
                related_queries=note_data.get("related_queries", []),
                tags=note_data.get("tags", [])
            )
            notes_imported += 1

        # Geri bildirimleri içe aktar
        for fb_data in data.get("feedbacks", []):
            self._add_feedback(
                source_id=fb_data["source_id"],
                target_id=fb_data["target_id"],
                relation_type=fb_data.get("relation_type", ""),
                feedback_type=fb_data.get("feedback_type", ""),
                note=fb_data.get("user_note", ""),
                confidence=fb_data.get("confidence", 1.0)
            )
            feedbacks_imported += 1

        logger.info(f"Imported {notes_imported} notes, {feedbacks_imported} feedbacks")
        return notes_imported, feedbacks_imported


def create_memory_store(db_path: str = "memory.db") -> MemoryStore:
    """Factory function"""
    return MemoryStore(db_path)
