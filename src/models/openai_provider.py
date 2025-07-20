"""OpenAI model provider."""

import json
import time
from typing import Dict, Any, Tuple
import openai
from src.models.base import BaseModelProvider
from src.types.models import Usage
from src.config import Config


class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider for GPT models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)

    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        """Perform OCR using OpenAI vision models."""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Convert the following document to markdown.
Return only the markdown with no explanation text. 

RULES:
- Include all information on the page
- Return tables in HTML format  
- Use ☐ and ☑ for checkboxes
- Preserve the original layout and structure"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0
            )

            duration = time.time() - start_time

            # Extract usage information
            usage_data = response.usage
            input_cost, output_cost, total_cost = self.calculate_cost(
                usage_data.prompt_tokens,
                usage_data.completion_tokens
            )

            usage = Usage(
                duration=duration,
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )

            text = response.choices[0].message.content
            return text, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"OpenAI OCR error: {str(e)}")

    async def extract_json_from_text(
            self,
            text: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        """Extract JSON from text using OpenAI."""
        start_time = time.time()

        try:
            schema_str = json.dumps(json_schema, indent=2)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"""Extract data from the document based on this JSON schema:

{schema_str}

Return only valid JSON that matches the schema. If some fields are not found, use null values."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=2000,
                temperature=0,
                response_format={"type": "json_object"}
            )

            duration = time.time() - start_time

            # Extract usage information
            usage_data = response.usage
            input_cost, output_cost, total_cost = self.calculate_cost(
                usage_data.prompt_tokens,
                usage_data.completion_tokens
            )

            usage = Usage(
                duration=duration,
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )

            json_text = response.choices[0].message.content
            extracted_json = json.loads(json_text)

            return extracted_json, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"OpenAI JSON extraction error: {str(e)}")

    async def extract_json_from_image(
            self,
            image_url: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        """Extract JSON directly from image using OpenAI vision."""
        start_time = time.time()

        try:
            schema_str = json.dumps(json_schema, indent=2)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Extract data from this image based on the following JSON schema:

{schema_str}

Return only valid JSON that matches the schema. If some fields are not found, use null values."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0,
                response_format={"type": "json_object"}
            )

            duration = time.time() - start_time

            # Extract usage information
            usage_data = response.usage
            input_cost, output_cost, total_cost = self.calculate_cost(
                usage_data.prompt_tokens,
                usage_data.completion_tokens
            )

            usage = Usage(
                duration=duration,
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )

            json_text = response.choices[0].message.content
            extracted_json = json.loads(json_text)

            return extracted_json, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"OpenAI direct extraction error: {str(e)}")