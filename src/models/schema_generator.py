import json
import time
import base64
import requests
from typing import Dict, Any, Tuple
from model_types.models import Usage
from config import Config


class SchemaGenerator:

    def __init__(self, model_name: str = "llava", base_url: str = None):
        self.model_name = model_name
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.headers = {"Content-Type": "application/json"}

    def _encode_image_url(self, image_url: str) -> str:
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_data = response.content
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to download and encode image: {str(e)}")

    async def generate_schema(self, image_url: str, document_type: str = "document") -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
            image_b64 = self._encode_image_url(image_url)

            prompt = f"""Analyze this {document_type} image and create a comprehensive JSON schema that would capture all the structured data present.

REQUIREMENTS:
1. Identify all the different types of information in the document
2. Create a hierarchical JSON schema with appropriate nesting
3. Use proper JSON Schema format with "type", "properties", "required" fields
4. Include all visible fields, even if some values might be empty
5. Use appropriate data types (string, number, object, array)
6. Group related fields into logical objects
7. Mark essential fields as "required"

RULES:
- Return ONLY valid JSON Schema format
- Use descriptive property names
- Include "type": "object" at the root level
- Nest related information logically
- Consider arrays for repeated items (like line items, products, etc.)

Example structure for reference:
{{
  "type": "object",
  "required": ["field1", "field2"],
  "properties": {{
    "field1": {{"type": "string"}},
    "field2": {{"type": "number"}},
    "nested_info": {{
      "type": "object",
      "properties": {{
        "subfield": {{"type": "string"}}
      }}
    }}
  }}
}}

JSON Schema:"""

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

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            try:
                schema = json.loads(response_text.strip())

                if not isinstance(schema, dict) or "type" not in schema:
                    raise ValueError("Generated schema is not a valid JSON Schema")

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

            prompt = f"""You are a data extraction specialist. Extract structured data from the OCR text according to the provided JSON schema.

CONTEXT: {document_context if document_context else "General document"}

JSON SCHEMA:
{schema_str}

OCR TEXT:
{ocr_text}

INSTRUCTIONS:
1. Carefully read through the OCR text
2. Extract all relevant information that matches the schema
3. If a required field is not found, use null
4. If an optional field is not found, omit it or use null
5. Parse numbers as actual numbers, not strings
6. For dates, preserve the format found in the document
7. Be precise and accurate with data extraction
8. Return ONLY valid JSON that conforms to the schema

EXTRACTED JSON:"""

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

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            try:
                extracted_data = json.loads(response_text.strip())

                usage = Usage(
                    duration=time.time() - start_time,
                    input_tokens=0,
                    output_tokens=0,
                    total_cost=0.0
                )

                return extracted_data, usage

            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse extracted JSON: {str(e)}. Response: {response_text[:200]}")

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Data extraction error: {str(e)}")

if __name__ == "__main__":
    import asyncio


    async def test_schema_generation():
        schema_gen = SchemaGenerator()
        eval_model = EvaluationModel()

        test_image_url = "https://omni-demo-data.s3.us-east-1.amazonaws.com/templates/receipt.png"

        try:
            print("Generating schema from image...")
            schema, schema_usage = await schema_gen.generate_schema(test_image_url, "receipt")
            print("Generated Schema:")
            print(json.dumps(schema, indent=2))
            print(f"Schema generation took: {schema_usage.duration:.2f}s")

            sample_ocr_text = """
            NICK THE GREEK
            SOUVLAKI & GYRO HOUSE
            San Francisco
            121 spear street
            Suite B08
            san francisco, CA 94105
            (415) 757-0426

            November 8, 2024
            2:16 PM

            Ticket: 17
            Receipt: NKZ1

            TO GO

            Beef/Lamb Gyro Pita $12.50
            Gyro Bowl $13.25
            Pork Gyro Pita $16.50

            Subtotal $42.25
            SF Mandate (6%) $2.54
            8.625% (8.625%) $3.64
            Total $48.43
            """

            print("\nExtracting structured data...")
            extracted_data, eval_usage = await eval_model.extract_structured_data(
                sample_ocr_text,
                schema,
                "restaurant receipt"
            )
            print("Extracted Data:")
            print(json.dumps(extracted_data, indent=2))
            print(f"Data extraction took: {eval_usage.duration:.2f}s")

        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(test_schema_generation())