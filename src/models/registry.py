from models.base import BaseModelProvider
from models.model_provider import ModelProvider

SUPPORTED_MODELS = [
    "llama3.2",
    "llava",
    "gpt-oss:20b",
    "deepseek-r1:8b"
]

def get_model_provider(model_name: str) -> BaseModelProvider:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. Available models: {SUPPORTED_MODELS}"
        )
    return ModelProvider(model_name)

def get_available_models() -> list:
    return SUPPORTED_MODELS
