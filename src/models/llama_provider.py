import json
import time
import base64
from typing import Dict, Any, Tuple
import requests
from src.models.base import BaseModelProvider
from src.types.models import Usage
from src.config import Config


class LlamaProvider(BaseModelProvider):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.backend = self._determine_backend()
        self._setup_client()

    def _determine_backend(self) -> str:
        if self.model_name.startswith("meta-llama/"):
            if Config.TOGETHER_API_KEY:
                return "together"
            elif Config.GROQ_API_KEY:
                return "groq"

        return "ollama"

    def _setup_client(self):
        if self.backend == "together":
            self.base_url = "https://api.together.xyz/v1"
            self.headers = {
                "Authorization": f"Bearer {Config.TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
        elif self.backend == "groq":
            self.base_url = "https://api.groq.com/openai/v1"
            self.headers = {
                "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
        else:
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
            if self.backend == "ollama":
                return await self._perform_ocr_ollama(image_url, start_time)
            else:
                return await self._perform_ocr_api(image_url, start_time)

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Llama OCR error ({self.backend}): {str(e)}")

    async def _perform_ocr_ollama(self, image_url: str, start_time: float) -> Tuple[str, Usage]:
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

    async def _perform_ocr_api(self, image_url: str, start_time: float) -> Tuple[str, Usage]:
        image_b64 = self._encode_image_url(image_url)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Convert this document to markdown format.

RULES:
- Extract ALL text content from the document
- Preserve the original structure and layout
- Use proper markdown formatting (headers, lists, tables)
- For tables, use markdown table format
- Include all numbers, dates, and details exactly as shown
- Use ☐ for empty checkboxes and ☑ for checked boxes
- Do not add any explanations or comments
- Return only the markdown content"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4000,
            "temperature": 0
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()

        duration = time.time() - start_time
        result = response.json()

        usage_data = result.get("usage", {})
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)

        input_cost, output_cost, total_cost = self.calculate_cost(input_tokens, output_tokens)

        usage = Usage(
            duration=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=usage_data.get("total_tokens", input_tokens + output_tokens),
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )

        text = result["choices"][0]["message"]["content"]
        return text, usage

    async def extract_json_from_text(
            self,
            text: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        start_time = time.time()

        try:
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

            if self.backend == "ollama":
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
            else:
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 2000,
                    "temperature": 0
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                response_text = result["choices"][0]["message"]["content"]
                usage_data = result.get("usage", {})
                input_tokens = usage_data.get("prompt_tokens", 0)
                output_tokens = usage_data.get("completion_tokens", 0)

                input_cost, output_cost, total_cost = self.calculate_cost(input_tokens, output_tokens)

                usage = Usage(
                    duration=time.time() - start_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=usage_data.get("total_tokens", input_tokens + output_tokens),
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost
                )

            response_text = response_text.strip()

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
            raise Exception(f"Llama JSON parsing error: {str(e)}. Response: {response_text[:200]}")
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

            if self.backend == "ollama":
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
            else:
                image_b64 = self._encode_image_url(image_url)

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                response_text = result["choices"][0]["message"]["content"]

                usage_data = result.get("usage", {})
                input_tokens = usage_data.get("prompt_tokens", 0)
                output_tokens = usage_data.get("completion_tokens", 0)

                input_cost, output_cost, total_cost = self.calculate_cost(input_tokens, output_tokens)

                usage = Usage(
                    duration=time.time() - start_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=usage_data.get("total_tokens", input_tokens + output_tokens),
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost
                )

            response_text = response_text.strip()

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
            raise Exception(f"Llama direct extraction JSON parsing error: {str(e)}. Response: {response_text[:200]}")
        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Llama direct extraction error: {str(e)}")