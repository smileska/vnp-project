"""Model registry for getting the appropriate provider for each model."""

from typing import Dict, Type
from src.models.base import BaseModelProvider
from src.models.gemini_provider import GeminiProvider
from src.models.llama_provider import LlamaProvider


# Model provider registry
MODEL_PROVIDERS: Dict[str, Type[BaseModelProvider]] = {
    # Google Gemini models
    "gemini-2.0-flash-001": GeminiProvider,
    "gemini-1.5-pro": GeminiProvider,
    "gemini-1.5-flash": GeminiProvider,

    # Local Llama models (via Ollama)
    "llama3.2-vision": LlamaProvider,
    "llava": LlamaProvider,

    # Hosted Llama models (via Together AI, Groq, etc.)
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": LlamaProvider,
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": LlamaProvider,
    "meta-llama/Llama-Vision-Free": LlamaProvider,
}


def get_model_provider(model_name: str) -> BaseModelProvider:
    """Get the appropriate provider for a model."""
    if model_name not in MODEL_PROVIDERS:
        available_models = list(MODEL_PROVIDERS.keys())
        raise ValueError(
            f"Model '{model_name}' is not supported. Available models: {available_models}"
        )

    provider_class = MODEL_PROVIDERS[model_name]
    return provider_class(model_name)


def get_available_models() -> list:
    """Get list of all available models."""
    return list(MODEL_PROVIDERS.keys())


def get_gemini_models() -> list:
    """Get list of available Gemini models."""
    return [model for model in MODEL_PROVIDERS.keys() if model.startswith("gemini")]


def get_llama_models() -> list:
    """Get list of available Llama models."""
    return [model for model in MODEL_PROVIDERS.keys() if "llama" in model.lower()]