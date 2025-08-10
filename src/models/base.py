from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from model_types.models import Usage


class BaseModelProvider(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        pass

    @abstractmethod
    async def extract_json_from_text(
            self,
            text: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        pass

    @abstractmethod
    async def extract_json_from_image(
            self,
            image_url: str,
            json_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Usage]:
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Tuple[float, float, float]:
        from src.config import TOKEN_COSTS

        if self.model_name not in TOKEN_COSTS:
            return 0.0, 0.0, 0.0

        costs = TOKEN_COSTS[self.model_name]
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost