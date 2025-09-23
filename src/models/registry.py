from models.base import BaseModelProvider
from models.model_provider import ModelProvider
from models.ocr_providers import (
    PaddleOCRProvider,
    EasyOCRProvider,
    TesseractProvider,
    get_ocr_provider_availability,
    PADDLE_AVAILABLE,
    EASY_OCR_AVAILABLE,
    TESSERACT_AVAILABLE
)

LLM_MODELS = [
    "llama3.2",
    "llava",
    "gpt-oss:20b",
    "deepseek-r1:8b"
]

OCR_MODELS = []
if PADDLE_AVAILABLE:
    OCR_MODELS.append("paddleocr")
if EASY_OCR_AVAILABLE:
    OCR_MODELS.append("easyocr")
if TESSERACT_AVAILABLE:
    OCR_MODELS.append("tesseract")

SUPPORTED_MODELS = LLM_MODELS + OCR_MODELS


def get_model_provider(model_name: str) -> BaseModelProvider:
    if model_name not in SUPPORTED_MODELS:
        available_models = get_available_models()
        raise ValueError(
            f"Model '{model_name}' is not supported. Available models: {available_models}"
        )

    if model_name == "paddleocr":
        return PaddleOCRProvider()
    elif model_name == "easyocr":
        return EasyOCRProvider()
    elif model_name == "tesseract":
        return TesseractProvider()

    elif model_name in LLM_MODELS:
        return ModelProvider(model_name)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_available_models() -> list:
    return SUPPORTED_MODELS


def get_llm_models() -> list:
    return LLM_MODELS


def get_ocr_models() -> list:
    return OCR_MODELS


def is_ocr_only_model(model_name: str) -> bool:
    return model_name in OCR_MODELS


def print_model_availability():
    print("\n📋 Model Availability Report:")
    print("=" * 40)

    print("\n🤖 LLM Models (OCR + Extraction):")
    for model in LLM_MODELS:
        print(f"  ✅ {model}")

    print("\n👁️  OCR-Only Models:")
    ocr_availability = get_ocr_provider_availability()

    for ocr_model, available in ocr_availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {ocr_model}")
        if not available:
            if ocr_model == "paddleocr":
                print(f"     Install with: pip install paddleocr")
            elif ocr_model == "easyocr":
                print(f"     Install with: pip install easyocr")
            elif ocr_model == "tesseract":
                print(f"     Install with: pip install pytesseract pillow")
                print(f"     Also install Tesseract binary from: https://github.com/tesseract-ocr/tesseract")

    print(f"\n📊 Total Available: {len(SUPPORTED_MODELS)} models")
    print(f"   - LLM Models: {len(LLM_MODELS)}")
    print(f"   - OCR Models: {len(OCR_MODELS)}")