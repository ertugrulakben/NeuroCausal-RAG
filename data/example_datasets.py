"""
NeuroCausal RAG - Ornek Veri Setleri Yukleyici
Markdown dosyalarindan veri setlerini yukler

Yazar: Ertugrul Akben
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

# Proje dizini
BASE_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = BASE_DIR / "examples"


# =============================================================================
# VERI SETI TANIMLARI
# =============================================================================
DATASETS = {
    "climate": {
        "name": "Iklim Degisikligi",
        "icon": "🌍",
        "description": "Kuresel isinma, sera gazlari, buzul erimesi vb. 115 dokuman",
        "source": "climate_knowledge_base",
        "doc_count": 115
    },
    "supply_chain": {
        "name": "Tedarik Zinciri Krizi",
        "icon": "📦",
        "description": "Fabrika yangini → Urun gecikmesi. Gizli baglanti senaryosu.",
        "source": "supply_chain_crisis.md",
        "doc_count": 6
    },
    "secret_merger": {
        "name": "Gizli Satin Alma",
        "icon": "🔐",
        "description": "Kod adi ile gercek firma ismi arasindaki baglanti.",
        "source": "secret_merger.md",
        "doc_count": 6
    },
    "legal_impact": {
        "name": "Hukuki Domino Etkisi",
        "icon": "⚖️",
        "description": "Yasa degisikligi → Pazarlama cokusu. Sebep-sonuc zinciri.",
        "source": "legal_impact.md",
        "doc_count": 6
    },
    "stress_chain": {
        "name": "Stres Zinciri",
        "icon": "😰",
        "description": "Stres → Kortizol → Uyku → Dikkat → Kaza zinciri.",
        "source": "stress_chain.md",
        "doc_count": 4
    }
}


def parse_markdown_documents(content: str, source_name: str = "unknown") -> List[Dict]:
    """
    Markdown icerigini dokumanlara ayirir.

    Format:
    ### doc_id
    **Dokuman:** ...
    Icerik...

    ---
    """
    docs = []

    # ### ile baslayan bloklari bul
    pattern = r'###\s+(\w+)\s*\n(.*?)(?=###|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for doc_id, doc_content in matches:
        doc_content = doc_content.strip()
        if doc_content and not doc_content.startswith('---'):
            # --- ile biten kismi temizle
            doc_content = re.sub(r'\n---\s*$', '', doc_content).strip()

            if doc_content:
                docs.append({
                    'id': doc_id.strip(),
                    'content': doc_content,
                    'source': source_name,
                    'category': source_name
                })

    return docs


def load_markdown_dataset(filename: str) -> List[Dict]:
    """Markdown dosyasindan veri seti yukle."""
    filepath = EXAMPLES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    source_name = filename.replace('.md', '')
    return parse_markdown_documents(content, source_name)


def load_climate_dataset() -> List[Dict]:
    """Iklim veri setini yukle."""
    try:
        from data.climate_knowledge_base import get_documents
        docs = get_documents()
        for doc in docs:
            doc['source'] = 'climate'
        return docs
    except ImportError:
        return []


def load_dataset(dataset_key: str) -> List[Dict]:
    """
    Belirtilen veri setini yukle.

    Args:
        dataset_key: climate, supply_chain, secret_merger, legal_impact, stress_chain

    Returns:
        Dokuman listesi
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Gecersiz veri seti: {dataset_key}")

    dataset_info = DATASETS[dataset_key]
    source = dataset_info['source']

    if dataset_key == 'climate':
        return load_climate_dataset()
    elif source.endswith('.md'):
        return load_markdown_dataset(source)
    else:
        return []


def get_dataset_raw_content(dataset_key: str) -> str:
    """Veri setinin ham icerigini dondur (gosterim icin)."""
    if dataset_key not in DATASETS:
        return "Gecersiz veri seti"

    dataset_info = DATASETS[dataset_key]
    source = dataset_info['source']

    if dataset_key == 'climate':
        # Iklim verileri icin ozet
        docs = load_climate_dataset()
        lines = [f"# Iklim Degisikligi Veri Seti ({len(docs)} dokuman)\n"]
        for doc in docs[:10]:  # Ilk 10
            lines.append(f"### {doc['id']}")
            lines.append(doc['content'][:200] + "...\n")
        lines.append(f"\n... ve {len(docs) - 10} dokuman daha")
        return "\n".join(lines)

    elif source.endswith('.md'):
        filepath = EXAMPLES_DIR / source
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

    return "Icerik bulunamadi"


def get_available_datasets() -> Dict:
    """Mevcut veri setlerini dondur."""
    return DATASETS


def get_dataset_info(dataset_key: str) -> Dict:
    """Veri seti bilgilerini dondur."""
    return DATASETS.get(dataset_key, {})
