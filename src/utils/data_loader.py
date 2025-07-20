"""Data loading utilities."""

import os
import json
from typing import List
from src.types.models import TestDocument


def load_test_documents(data_dir: str) -> List[TestDocument]:
    """Load test documents from JSON files in the data directory."""
    documents = []

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist!")
        return documents

    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return documents

    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                document = TestDocument(**data)
                documents.append(document)
                print(f"✅ Loaded {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")

    return documents


def save_results(results: List[dict], output_path: str) -> None:
    """Save benchmark results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📁 Results saved to {output_path}")