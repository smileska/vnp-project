import json
import time
import base64
from typing import Dict, Any, Tuple
import requests
from .base import BaseModelProvider
from config import Config
from model_types.models import Usage
from utils.json_extractor import robust_json_extraction


class ModelProvider(BaseModelProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._setup_client()

    def _setup_client(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.headers = {"Content-Type": "application/json"}

    def _encode_image_url(self, image_url: str) -> str:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to download and encode image: {str(e)}")

    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        start_time = time.time()

        try:
            return await self._perform_ocr(image_url, start_time)

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Llama OCR error (ollama): {str(e)}")

    async def _perform_ocr(self, image_url: str, start_time: float) -> Tuple[str, Usage]:
        image_b64 = self._encode_image_url(image_url)

        payload = {
            "model": self.model_name,
            "prompt": """Convert this document to markdown format.

RULES:
- Extract ALL text content from the document
- Preserve the original structure and layout
- Use proper markdown formatting (headers, lists, tables)
- For tables, use markdown table format
- Include all numbers, dates, and details exactly as shown
- Use ☐ for empty checkboxes and ☑ for checked boxes
- Do not add any explanations or comments
- Return only the markdown content""",
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()

        duration = time.time() - start_time
        result = response.json()

        usage = Usage(
            duration=duration,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0
        )

        return result["response"], usage

    async def extract_json_from_text(
            self,
            text: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
            schema_str = json.dumps(json_schema, indent=2)

            prompt = f"""Extract JSON data from this text. Use the schema provided. Return ONLY JSON, no explanations.

Schema: {schema_str}

Text: {text}

JSON:"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            response_text = result["response"]

            usage = Usage(
                duration=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )

            # Use robust JSON extraction
            fallback_data = {"error": "extraction_failed", "raw_text": text[:200]}
            extracted_json, error = robust_json_extraction(response_text, fallback_data)

            if error:
                print(f"    ⚠️  JSON extraction issue: {error}")

            return extracted_json, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Llama JSON extraction error: {str(e)}")

    async def extract_json_from_image(
            self,
            image_url: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
            schema_str = json.dumps(json_schema, indent=2)

            prompt = f"""Extract JSON data from this image. Use the schema provided. Return ONLY JSON, no explanations.

Schema: {schema_str}

JSON:"""

            image_b64 = self._encode_image_url(image_url)
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            response_text = result["response"]

            usage = Usage(
                duration=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )

            # Use robust JSON extraction
            fallback_data = {"error": "extraction_failed"}
            extracted_json, error = robust_json_extraction(response_text, fallback_data)

            if error:
                print(f"    ⚠️  JSON extraction issue: {error}")

            return extracted_json, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Llama direct extraction error: {str(e)}")