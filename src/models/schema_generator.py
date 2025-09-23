import json
import time
import base64
import requests
from typing import Dict, Any, Tuple
from model_types.models import Usage
from config import Config
from utils.json_extractor import robust_json_extraction


class SchemaGenerator:

    def __init__(self, model_name: str = "llava", base_url: str = None):
        self.model_name = model_name
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.headers = {"Content-Type": "application/json"}

    def _encode_image_url(self, image_url: str) -> str:
        try:
            print(f"    📥 Downloading image from: {image_url}")
            response = requests.get(image_url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()

            image_data = response.content
            if len(image_data) == 0:
                raise Exception("Downloaded image is empty")

            print(f"    ✅ Downloaded {len(image_data)} bytes")
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to download and encode image: {str(e)}")

    async def generate_schema(self, image_url: str, document_type: str = "document") -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
            image_b64 = self._encode_image_url(image_url)

            prompt = f"""Look at this {document_type} image. Create a JSON schema for the data you see.

CRITICAL: Return ONLY the JSON schema. No explanations, no text, just JSON.

Format:
{{
  "type": "object",
  "properties": {{
    "field1": {{"type": "string"}},
    "field2": {{"type": "number"}}
  }}
}}"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json=payload,
                timeout=Config.REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            result = response.json()
            response_text = result["response"].strip()

            try:
                fallback_schema = self._create_fallback_schema(document_type)
                schema, error = robust_json_extraction(response_text, fallback_schema)

                if error:
                    print(f"    ⚠️  {error}")
                    print(f"    📝 Using fallback schema")

                if not isinstance(schema, dict) or "type" not in schema:
                    print(f"    ⚠️  Generated invalid schema, using fallback")
                    schema = fallback_schema

                usage = Usage(
                    duration=time.time() - start_time,
                    input_tokens=0,
                    output_tokens=0,
                    total_cost=0.0
                )

                return schema, usage

            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse generated schema as JSON: {str(e)}. Response: {response_text[:200]}")

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Schema generation error: {str(e)}")

    def _create_fallback_schema(self, document_type: str) -> Dict[str, Any]:
        if document_type == "receipt":
            return {
                "type": "object",
                "required": ["merchant", "total"],
                "properties": {
                    "merchant": {"type": "string"},
                    "total": {"type": "number"},
                    "date": {"type": "string"},
                    "items": {"type": "array"}
                }
            }
        elif document_type == "invoice":
            return {
                "type": "object",
                "required": ["invoice_number", "total"],
                "properties": {
                    "invoice_number": {"type": "string"},
                    "total": {"type": "number"},
                    "date": {"type": "string"},
                    "client": {"type": "string"}
                }
            }
        else:
            return {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "extracted_data": {"type": "object"}
                }
            }

    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        if not isinstance(schema, dict):
            return False

        required_keys = ["type", "properties"]
        return all(key in schema for key in required_keys)


class EvaluationModel:
    def __init__(self, model_name: str = "llama3.2", base_url: str = None):
        self.model_name = model_name
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.headers = {"Content-Type": "application/json"}

    async def extract_structured_data(
        self,
        ocr_text: str,
        json_schema: Dict[str, Any],
        document_context: str = ""
    ) -> Tuple[Dict[str, Any], Usage]:
        
        start_time = time.time()

        try:
            schema_str = json.dumps(json_schema, indent=2)

            prompt = f"""Extract JSON data from this text. Use the schema provided. Return ONLY JSON, no explanations.

Schema: {schema_str}

Text: {ocr_text}

JSON:"""

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json=payload,
                timeout=Config.REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            result = response.json()
            response_text = result["response"].strip()

            fallback_data = {"error": "extraction_failed", "raw_text": ocr_text[:200]}
            extracted_data, error = robust_json_extraction(response_text, fallback_data)

            if error:
                print(f"    ⚠️  {error}")

            usage = Usage(
                duration=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                total_cost=0.0
            )

            return extracted_data, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Data extraction error: {str(e)}")