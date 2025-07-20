import os
import json
from datetime import datetime
from typing import Any

def create_results_folder(base_dir: str = "./results") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_json(data: Any, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)