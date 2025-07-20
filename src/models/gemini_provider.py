import json
import time
import base64
from typing import Dict, Any, Tuple
import google.generativeai as genai
import requests
from src.models.base import BaseModelProvider
from src.types.models import Usage
from src.config import Config


class GeminiProvider(BaseModelProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        genai.configure(api_key=Config.GOOGLE_API_KEY)

        self.generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 20,
            "max_output_tokens": 4000,
        }

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def _download_image(self, image_url: str) -> bytes:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content

    def _create_image_part(self, image_url: str) -> Dict[str, Any]:
        try:
            image_data = self._download_image(image_url)

            if image_url.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_url.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_url.lower().endswith('.webp'):
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'

            return {
                'mime_type': mime_type,
                'data': image_data
            }
        except Exception as e:
            raise Exception(f"Failed to download image: {str(e)}")

    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        start_time = time.time()

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            image_part = self._create_image_part(image_url)

            prompt = """Convert this document to markdown format.

RULES:
- Extract ALL text content from the document
- Preserve the original structure and layout
- Use proper markdown formatting (headers, lists, tables)
- For tables, use markdown table format
- Include all numbers, dates, and details exactly as shown
- Use ☐ for empty checkboxes and ☑ for checked boxes
- Do not add any explanations or comments
- Return only the markdown content

Please convert this document:"""

            response = model.generate_content([prompt, image_part])

            duration = time.time() - start_time

            usage_metadata = response.usage_metadata
            input_cost, output_cost, total_cost = self.calculate_cost(
                usage_metadata.prompt_token_count if usage_metadata else 0,
                usage_metadata.candidates_token_count if usage_metadata else 0
            )

            usage = Usage(
                duration=duration,
                input_tokens=usage_metadata.prompt_token_count if usage_metadata else 0,
                output_tokens=usage_metadata.candidates_token_count if usage_metadata else 0,
                total_tokens=usage_metadata.total_token_count if usage_metadata else 0,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )

            text = response.text
            return text, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Gemini OCR error: {str(e)}")

    async def extract_json_from_text(
            self,
            text: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            schema_str = json.dumps(json_schema, indent=2)

            prompt = f"""Extract structured data from the following text according to this JSON schema:

{schema_str}

RULES:
- Return ONLY valid JSON that matches the schema
- If a field is not found in the text, use null
- Ensure all required fields are present
- Use the exact field names from the schema
- Parse numbers as numbers, not strings
- For dates, use the format found in the document

Text to extract from:
{text}

JSON:"""

            response = model.generate_content(prompt)

            duration = time.time() - start_time

            usage_metadata = response.usage_metadata
            input_cost, output_cost, total_cost = self.calculate_cost(
                usage_metadata.prompt_token_count if usage_metadata else 0,
                usage_metadata.candidates_token_count if usage_metadata else 0
            )

            usage = Usage(
                duration=duration,
                input_tokens=usage_metadata.prompt_token_count if usage_metadata else 0,
                output_tokens=usage_metadata.candidates_token_count if usage_metadata else 0,
                total_tokens=usage_metadata.total_token_count if usage_metadata else 0,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )

            response_text = response.text.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            extracted_json = json.loads(response_text.strip())

            return extracted_json, usage

        except json.JSONDecodeError as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Gemini JSON parsing error: {str(e)}. Response: {response.text[:200]}")
        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Gemini JSON extraction error: {str(e)}")

    async def extract_json_from_image(
            self,
            image_url: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            image_part = self._create_image_part(image_url)
            schema_str = json.dumps(json_schema, indent=2)

            prompt = f"""Extract structured data from this image according to the following JSON schema:

{schema_str}

RULES:
- Return ONLY valid JSON that matches the schema
- If a field is not found in the image, use null
- Ensure all required fields are present
- Use the exact field names from the schema
- Parse numbers as numbers, not strings
- For dates, use the format found in the document
- Read all text carefully and extract the requested information

JSON:"""

            response = model.generate_content([prompt, image_part])

            duration = time.time() - start_time

            usage_metadata = response.usage_metadata
            input_cost, output_cost, total_cost = self.calculate_cost(
                usage_metadata.prompt_token_count if usage_metadata else 0,
                usage_metadata.candidates_token_count if usage_metadata else 0
            )

            usage = Usage(
                duration=duration,
                input_tokens=usage_metadata.prompt_token_count if usage_metadata else 0,
                output_tokens=usage_metadata.candidates_token_count if usage_metadata else 0,
                total_tokens=usage_metadata.total_token_count if usage_metadata else 0,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )

            response_text = response.text.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            extracted_json = json.loads(response_text.strip())

            return extracted_json, usage

        except json.JSONDecodeError as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Gemini direct extraction JSON parsing error: {str(e)}. Response: {response.text[:200]}")
        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Gemini direct extraction error: {str(e)}")