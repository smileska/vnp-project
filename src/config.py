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
    ModelConfig(ocr_model="llama3.2", extraction_model="llama3.2"),
    ModelConfig(ocr_model="llava", extraction_model="llava"),
    ModelConfig(ocr_model="gpt-oss:20b", extraction_model="gpt-oss:20b"),
    ModelConfig(ocr_model="deepseek-r1:8b", extraction_model="deepseek-r1:8b"),
]


TOKEN_COSTS = {
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": {"input": 0.18, "output": 0.18},
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": {"input": 1.20, "output": 1.20},
    "llama3.2-vision": {"input": 0.0, "output": 0.0},
    "llava": {"input": 0.0, "output": 0.0},
}