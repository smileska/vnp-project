"""Configuration settings for the OCR benchmark."""

import os
from typing import List
from dotenv import load_dotenv
from src.types.models import ModelConfig

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision")

    # Benchmark settings
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
    RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Data directory
    DATA_DIR = "./data"


# Define which models to test
MODEL_CONFIGURATIONS: List[ModelConfig] = [
    # Google Gemini models
    ModelConfig(ocr_model="gemini-2.0-flash-001", extraction_model="gemini-2.0-flash-001"),
    ModelConfig(ocr_model="gemini-2.0-flash-001", extraction_model="gemini-2.0-flash-001", direct_image_extraction=True),
    ModelConfig(ocr_model="gemini-1.5-pro", extraction_model="gemini-1.5-pro"),
    ModelConfig(ocr_model="gemini-1.5-pro", extraction_model="gemini-1.5-pro", direct_image_extraction=True),

    # Llama models via Ollama (local)
    ModelConfig(ocr_model="llama3.2-vision", extraction_model="llama3.2-vision"),
    ModelConfig(ocr_model="llama3.2-vision", extraction_model="llama3.2-vision", direct_image_extraction=True),

    # Llama models via Together AI
    ModelConfig(ocr_model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", extraction_model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"),
    ModelConfig(ocr_model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", extraction_model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"),

    # Cross-model comparisons
    ModelConfig(ocr_model="gemini-2.0-flash-001", extraction_model="llama3.2-vision"),
    ModelConfig(ocr_model="llama3.2-vision", extraction_model="gemini-1.5-pro"),
]


# Token costs per million tokens (input, output) - Updated for available models
TOKEN_COSTS = {
    # Google Gemini models
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # Llama models (Together AI pricing)
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {"input": 0.18, "output": 0.18},
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": {"input": 1.20, "output": 1.20},

    # Local Ollama models (free)
    "llama3.2-vision": {"input": 0.0, "output": 0.0},
    "llava": {"input": 0.0, "output": 0.0},
}