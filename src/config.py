import os
from typing import List
from dotenv import load_dotenv
from model_types.models import ModelConfig

load_dotenv()


class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DATA_DIR = "../data"

MODEL_CONFIGURATIONS = [
    # Pure LLM models
    ModelConfig(ocr_model="llama3.2", extraction_model="llama3.2"),
    ModelConfig(ocr_model="llava", extraction_model="llava"),
    ModelConfig(ocr_model="gpt-oss:20b", extraction_model="gpt-oss:20b"),
    ModelConfig(ocr_model="deepseek-r1:8b", extraction_model="deepseek-r1:8b"),

    # PaddleOCR + LLM extraction
    ModelConfig(ocr_model="paddleocr", extraction_model="llama3.2"),
    ModelConfig(ocr_model="paddleocr", extraction_model="llava"),
    ModelConfig(ocr_model="paddleocr", extraction_model="gpt-oss:20b"),
    ModelConfig(ocr_model="paddleocr", extraction_model="deepseek-r1:8b"),

    # EasyOCR + LLM extraction
    ModelConfig(ocr_model="easyocr", extraction_model="llama3.2"),
    ModelConfig(ocr_model="easyocr", extraction_model="llava"),
    ModelConfig(ocr_model="easyocr", extraction_model="gpt-oss:20b"),
    ModelConfig(ocr_model="easyocr", extraction_model="deepseek-r1:8b"),

    # Tesseract + LLM extraction
    ModelConfig(ocr_model="tesseract", extraction_model="llama3.2"),
    ModelConfig(ocr_model="tesseract", extraction_model="llava"),
    ModelConfig(ocr_model="tesseract", extraction_model="gpt-oss:20b"),
    ModelConfig(ocr_model="tesseract", extraction_model="deepseek-r1:8b"),

    # Direct image extraction with LLMs
    ModelConfig(ocr_model="llama3.2", direct_image_extraction=True),
    ModelConfig(ocr_model="llava", direct_image_extraction=True),
    ModelConfig(ocr_model="gpt-oss:20b", direct_image_extraction=True),
    ModelConfig(ocr_model="deepseek-r1:8b", direct_image_extraction=True),
]

TOKEN_COSTS = {
    "llama3.2": {"input": 0.0, "output": 0.0},
    "llama3.2-vision": {"input": 0.0, "output": 0.0},
    "llava": {"input": 0.0, "output": 0.0},
    "gpt-oss:20b": {"input": 0.0, "output": 0.0},
    "deepseek-r1:8b": {"input": 0.0, "output": 0.0},

    "paddleocr": {"input": 0.0, "output": 0.0},
    "easyocr": {"input": 0.0, "output": 0.0},
    "tesseract": {"input": 0.0, "output": 0.0},
}


def get_filtered_model_configurations() -> List[ModelConfig]:
    from models.registry import get_available_models, is_ocr_only_model

    available_models = get_available_models()
    valid_configs = []

    for config in MODEL_CONFIGURATIONS:
        if config.ocr_model not in available_models:
            continue

        if config.extraction_model and config.extraction_model not in available_models:
            continue

        if is_ocr_only_model(config.ocr_model) and not config.extraction_model:
            continue

        if is_ocr_only_model(config.ocr_model) and config.direct_image_extraction:
            continue

        valid_configs.append(config)

    return valid_configs


def print_configuration_summary():
    print("\n🔧 Model Configuration Summary:")
    print("=" * 50)

    valid_configs = get_filtered_model_configurations()

    if not valid_configs:
        print("❌ No valid model configurations found!")
        print("   Please install the required OCR libraries and ensure models are available.")
        return

    ocr_groups = {}
    for config in valid_configs:
        ocr_model = config.ocr_model
        if ocr_model not in ocr_groups:
            ocr_groups[ocr_model] = []
        ocr_groups[ocr_model].append(config)

    for ocr_model, configs in ocr_groups.items():
        print(f"\n📖 OCR Model: {ocr_model}")
        for config in configs:
            if config.direct_image_extraction:
                print(f"   ➤ Direct image → JSON extraction")
            elif config.extraction_model:
                print(f"   ➤ OCR → {config.extraction_model} → JSON")
            else:
                print(f"   ➤ OCR only")

    print(f"\n📊 Total valid configurations: {len(valid_configs)}")