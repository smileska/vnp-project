"""Type definitions for the OCR benchmark."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class ModelConfig:
    """Configuration for a model to test."""
    ocr_model: str
    extraction_model: Optional[str] = None
    direct_image_extraction: bool = False


class Usage(BaseModel):
    """Usage statistics for a request."""
    duration: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    total_cost: Optional[float] = None


class TestDocument(BaseModel):
    """A test document with ground truth data."""
    imageUrl: str
    metadata: Dict[str, Any]
    jsonSchema: Dict[str, Any]
    trueJsonOutput: Dict[str, Any]
    trueMarkdownOutput: str


class BenchmarkResult(BaseModel):
    """Result of running benchmark on one document with one model."""
    file_url: str
    metadata: Dict[str, Any]
    ocr_model: str
    extraction_model: str
    json_schema: Dict[str, Any]
    direct_image_extraction: bool = False
    true_markdown: str
    true_json: Dict[str, Any]
    predicted_markdown: Optional[str] = None
    predicted_json: Optional[Dict[str, Any]] = None
    text_similarity: Optional[float] = None
    json_accuracy: Optional[float] = None
    json_diff: Optional[Dict[str, Any]] = None
    usage: Optional[Usage] = None
    error: Optional[str] = None