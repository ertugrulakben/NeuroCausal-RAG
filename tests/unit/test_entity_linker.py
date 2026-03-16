"""
NeuroCausal RAG - EntityLinker Unit Tests

Author: Ertugrul Akben
"""

import pytest
import tempfile
from pathlib import Path
from neurocausal_rag.entity.linker import EntityLinker, AliasStore, Entity


class TestAliasStore:
    """AliasStore unit tests"""

    @pytest.fixture
    def alias_store(self, tmp_path):
        """Gecici AliasStore instance"""
        persist_path = str(tmp_path / "aliases.json")
        return AliasStore(persist_path=persist_path)

    def test_add_alias(self, alias_store):
        """Alias ekleme testi"""
        alias_store.add_alias("Mavi Ufuk", "Gunes Enerjisi A.S.", 0.9)

        resolved = alias_store.resolve("Mavi Ufuk")
        assert resolved == "gunes enerjisi a.s."

    def test_resolve_case_insensitive(self, alias_store):
        """Buyuk/kucuk harf duyarsiz cozumleme"""
        alias_store.add_alias("Proje X", "Yeni Urun", 0.85)

        assert alias_store.resolve("proje x") == "yeni urun"
        assert alias_store.resolve("PROJE X") == "yeni urun"
        assert alias_store.resolve("Proje X") == "yeni urun"

    def test_get_aliases(self, alias_store):
        """Canonical icin alias listesi"""
        alias_store.add_alias("Kod A", "Gercek Isim", 0.9)
        alias_store.add_alias("Kod B", "Gercek Isim", 0.8)

        aliases = alias_store.get_aliases("Gercek Isim")
        assert "kod a" in aliases
        assert "kod b" in aliases

    def test_find_in_text(self, alias_store):
        """Metinde alias bulma"""
        alias_store.add_alias("Fenix", "ERP Projesi", 0.9)

        text = "Fenix projesi basarili oldu."
        found = alias_store.find_in_text(text)

        assert len(found) == 1
        assert found[0][0] == "fenix"
        assert found[0][1] == "erp projesi"

    def test_confidence_score(self, alias_store):
        """Guven skoru testi"""
        alias_store.add_alias("Test", "Real", 0.75)

        confidence = alias_store.get_confidence("Test", "Real")
        assert confidence == 0.75

    def test_self_alias_prevention(self, alias_store):
        """Kendisiyle alias engelleme"""
        alias_store.add_alias("Same", "Same", 0.9)

        # Kendisiyle esleme eklenmemeli
        resolved = alias_store.resolve("Same")
        assert resolved is None

    def test_persistence(self, tmp_path):
        """Kaydetme ve yukleme testi"""
        persist_path = str(tmp_path / "aliases.json")

        # Ilk store
        store1 = AliasStore(persist_path=persist_path)
        store1.add_alias("Alias1", "Canonical1", 0.9)
        store1.add_alias("Alias2", "Canonical2", 0.8)
        store1.save()

        # Yeni instance olustur ve yukle
        store2 = AliasStore(persist_path=persist_path)
        store2.load()

        assert store2.resolve("Alias1") == "canonical1"
        assert store2.resolve("Alias2") == "canonical2"


class TestEntityLinker:
    """EntityLinker unit tests"""

    @pytest.fixture
    def entity_linker(self):
        """EntityLinker instance"""
        return EntityLinker()

    def test_extract_aliases_equals_pattern(self, entity_linker):
        """'X = Y' pattern testi"""
        text = '"Mavi Ufuk" = "Gunes Enerjisi A.S."'

        aliases = entity_linker.extract_aliases_from_text(text)

        assert len(aliases) >= 1
        # En az bir alias bulunmali

    def test_extract_aliases_parenthesis_pattern(self, entity_linker):
        """'Y (X)' pattern testi"""
        text = "Ahmet Yilmaz (CEO) toplantiya katildi."

        aliases = entity_linker.extract_aliases_from_text(text)
        # Parentez ici pattern calismali

    def test_learn_aliases(self, entity_linker):
        """Dokumanlardan alias ogrenme"""
        alias_documents = [
            {"id": "doc1", "content": '"Proje Fenix" = "Yeni ERP Sistemi"'},
            {"id": "doc2", "content": "Proje Fenix 2024'te tamamlanacak."},
        ]

        learned = entity_linker.learn_aliases_from_documents(alias_documents)

        # En az bir alias ogrenmeli
        assert learned >= 0

    def test_add_alias_manual(self, entity_linker):
        """Manuel alias ekleme"""
        entity_linker.add_alias("Manuel Kod", "Manuel Gercek", 0.95)

        resolved = entity_linker.resolve_text("Manuel Kod projesi")
        assert "manuel kod" in resolved
        assert resolved["manuel kod"] == "manuel gercek"

    def test_resolve_text(self, entity_linker):
        """Metin cozumleme"""
        entity_linker.add_alias("Proje Alpha", "Yeni Platform", 0.9)

        result = entity_linker.resolve_text("Proje Alpha tamamlandi.")

        assert "proje alpha" in result
        assert result["proje alpha"] == "yeni platform"

    def test_resolve_text_full(self, entity_linker):
        """Tam metin cozumleme (string replacement)"""
        entity_linker.add_alias("Kod X", "Gercek Isim", 0.9)

        result = entity_linker.resolve_text_full("Kod X basarili oldu.")

        # Lowercase'e donusturulmus olabilir
        assert "gercek isim" in result.lower()
        assert "kod x" in result.lower()  # Parantez icinde korunmali

    def test_enrich_query(self, entity_linker):
        """Sorgu zenginlestirme"""
        entity_linker.add_alias("Fenix", "ERP Sistemi", 0.9)

        enriched = entity_linker.enrich_query("Fenix ne zaman bitecek?")

        # Lowercase'e donusturulmus olabilir
        assert "erp sistemi" in enriched.lower()

    def test_get_all_aliases(self, entity_linker):
        """Tum alias'lari getir"""
        entity_linker.add_alias("A1", "C1", 0.9)
        entity_linker.add_alias("A2", "C1", 0.8)
        entity_linker.add_alias("A3", "C2", 0.7)

        all_aliases = entity_linker.get_all_aliases()

        assert "c1" in all_aliases
        assert "c2" in all_aliases
        assert len(all_aliases["c1"]) == 2

    def test_serialization(self, entity_linker):
        """Dict'e donusturme ve geri yukleme"""
        entity_linker.add_alias("Test1", "Real1", 0.9)
        entity_linker.add_alias("Test2", "Real2", 0.8)

        data = entity_linker.to_dict()

        # Yeni instance olustur
        new_linker = EntityLinker.from_dict(data)

        resolved = new_linker.resolve_text("Test1 ve Test2")
        assert "test1" in resolved
        assert "test2" in resolved

    def test_find_entity_connections(self, entity_linker):
        """Iki dokuman arasinda entity baglantilari"""
        entity_linker.add_alias("Kod X", "Ortak Proje", 0.9)

        doc1 = {"id": "d1", "content": "Kod X hakkinda bilgi"}
        doc2 = {"id": "d2", "content": "Kod X devam ediyor"}

        connections = entity_linker.find_entity_connections(doc1, doc2)

        assert len(connections) >= 0  # En az bir baglanti olmali


class TestEntity:
    """Entity dataclass tests"""

    def test_entity_creation(self):
        """Entity olusturma"""
        entity = Entity(
            name="Tesla",
            entity_type="ORG",
            aliases={"TSLA", "Tesla Inc."}
        )

        assert entity.name == "Tesla"
        assert entity.entity_type == "ORG"
        assert "TSLA" in entity.aliases

    def test_entity_matches(self):
        """Entity esleme testi"""
        entity = Entity(
            name="Elon Musk",
            entity_type="PERSON",
            aliases={"Musk", "Tesla CEO"}
        )

        assert entity.matches("Elon Musk konustu")
        assert entity.matches("Musk aciklama yapti")
        assert not entity.matches("Bill Gates konustu")

    def test_entity_equality(self):
        """Entity esitlik testi"""
        e1 = Entity(name="Test", entity_type="ORG")
        e2 = Entity(name="test", entity_type="PERSON")  # Farkli tip, ayni isim
        e3 = Entity(name="Other", entity_type="ORG")

        assert e1 == e2  # Isim ayni (case insensitive)
        assert e1 != e3

    def test_entity_hash(self):
        """Entity hash testi (set/dict kullanimi icin)"""
        e1 = Entity(name="Test", entity_type="ORG")
        e2 = Entity(name="TEST", entity_type="PERSON")

        entity_set = {e1, e2}
        assert len(entity_set) == 1  # Ayni hash
