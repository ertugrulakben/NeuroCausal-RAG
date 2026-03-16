"""
NeuroCausal RAG - Ornek Veri Setleri Yukleyici
Markdown dosyalarindan veri setlerini yukler

Author: Ertugrul Akben
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

# Project directory
BASE_DIR = Path(__file__).parent.parent
EXAMPLES_DIR = BASE_DIR / "examples"


# =============================================================================
# DATASET DEFINITIONS
# =============================================================================
DATASETS = {
    "climate": {
        "name": "Climate Change",
        "icon": "🌍",
        "description": "Global warming, greenhouse gases, glacier melting etc. 115 documents",
        "source": "climate_knowledge_base",
        "doc_count": 115
    },
    "supply_chain": {
        "name": "Supply Chain Crisis",
        "icon": "📦",
        "description": "Factory fire -> Product delay. Hidden connection scenario.",
        "source": "supply_chain_crisis.md",
        "doc_count": 6
    },
    "secret_merger": {
        "name": "Secret Acquisition",
        "icon": "🔐",
        "description": "Connection between code name and real company name.",
        "source": "secret_merger.md",
        "doc_count": 6
    },
    "legal_impact": {
        "name": "Legal Domino Effect",
        "icon": "⚖️",
        "description": "Law change -> Marketing collapse. Cause-effect chain.",
        "source": "legal_impact.md",
        "doc_count": 6
    },
    "stress_chain": {
        "name": "Stress Chain",
        "icon": "😰",
        "description": "Stress -> Cortisol -> Sleep -> Attention -> Accident chain.",
        "source": "stress_chain.md",
        "doc_count": 4
    }
}


def parse_markdown_documents(content: str, source_name: str = "unknown") -> List[Dict]:
    """
    Split Markdown content into documents.

    Format:
    ### doc_id
    **Dokuman:** ...
    Icerik...

    ---
    """
    docs = []

    # Find blocks starting with ###
    pattern = r'###\s+(\w+)\s*\n(.*?)(?=###|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for doc_id, doc_content in matches:
        doc_content = doc_content.strip()
        if doc_content and not doc_content.startswith('---'):
            # Clean trailing ---
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
    """Load dataset from Markdown file."""
    filepath = EXAMPLES_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    source_name = filename.replace('.md', '')
    return parse_markdown_documents(content, source_name)


def load_climate_dataset() -> List[Dict]:
    """Load climate dataset."""
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
    Load specified dataset.

    Args:
        dataset_key: climate, supply_chain, secret_merger, legal_impact, stress_chain

    Returns:
        Dokuman listesi
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset_key}")

    dataset_info = DATASETS[dataset_key]
    source = dataset_info['source']

    if dataset_key == 'climate':
        return load_climate_dataset()
    elif source.endswith('.md'):
        return load_markdown_dataset(source)
    else:
        return []


def get_dataset_raw_content(dataset_key: str) -> str:
    """Return raw content of dataset (for display)."""
    if dataset_key not in DATASETS:
        return "Invalid dataset"

    dataset_info = DATASETS[dataset_key]
    source = dataset_info['source']

    if dataset_key == 'climate':
        # Summary for climate data
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

    return "Content not found"


def get_available_datasets() -> Dict:
    """Return available datasets."""
    return DATASETS


def get_dataset_info(dataset_key: str) -> Dict:
    """Return dataset info."""
    return DATASETS.get(dataset_key, {})
