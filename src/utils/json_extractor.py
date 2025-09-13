"""
Robust JSON extraction utility
"""

import json
import re
from typing import Dict, Any, Optional, Tuple


def robust_json_extraction(response: str, fallback_schema: Optional[Dict[str, Any]] = None) -> Tuple[
    Optional[dict], Optional[str]]:
    """
    Extract JSON from a model response that may contain markdown formatting or explanatory text.

    Args:
        response: Raw response from the model
        fallback_schema: Optional fallback schema to use if extraction fails

    Returns:
        Tuple of (parsed_json, error_message)
    """
    if not response or not response.strip():
        return fallback_schema or {}, "Empty response"

    # Handle responses that explicitly say no data found
    no_data_indicators = [
        "unable to find enough information",
        "no information available",
        "cannot extract",
        "insufficient data",
        "does not contain",
        "no data found",
        "i need the actual image data"
    ]

    response_lower = response.lower()
    for indicator in no_data_indicators:
        if indicator in response_lower:
            return fallback_schema or {}, None

    # Clean the response
    cleaned = response.strip()

    # Remove common prefixes and thinking tags
    prefixes_to_remove = [
        r"<think>.*?</think>",  # Remove thinking tags
        r"Based on.*?Here is the resulting JSON:\s*",
        r"Here is the JSON.*?:\s*",
        r"The extracted data in JSON format:\s*",
        r".*?Here's the structured data:\s*",
        r"Based on the provided.*?:\s*",
        r"To process the provided image.*?:",
        r"I can see.*?Here's the schema:",
        r"Looking at.*?schema:",
    ]

    for prefix_pattern in prefixes_to_remove:
        cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Try multiple extraction strategies
    strategies = [
        # Strategy 1: Look for ```json blocks
        lambda text: _extract_from_markdown_block(text, "json"),
        # Strategy 2: Look for ``` blocks (without json specifier)
        lambda text: _extract_from_markdown_block(text, ""),
        # Strategy 3: Look for any complete JSON object
        lambda text: _extract_complete_json_object(text),
        # Strategy 4: Look for JSON array
        lambda text: _extract_json_array(text),
        # Strategy 5: Try to fix incomplete JSON
        lambda text: _extract_and_fix_incomplete_json(text),
    ]

    for i, strategy in enumerate(strategies, 1):
        try:
            json_str = strategy(cleaned)
            if json_str:
                parsed = json.loads(json_str)
                return parsed, None
        except json.JSONDecodeError:
            continue
        except Exception:
            continue

    # If all strategies fail, return fallback
    return fallback_schema or {}, f"Could not extract valid JSON. Response preview: {response[:200]}..."


def _extract_from_markdown_block(text: str, block_type: str) -> Optional[str]:
    """Extract JSON from markdown code blocks"""
    if block_type:
        pattern = rf'```{block_type}\s*(\{{.*?\}})\s*```'
    else:
        pattern = r'```\s*(\{.*?\})\s*```'

    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_complete_json_object(text: str) -> Optional[str]:
    """Extract a complete JSON object by matching braces"""
    start = text.find('{')
    if start >= 0:
        brace_count = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    return text[start:end].strip()

    return None


def _extract_json_array(text: str) -> Optional[str]:
    """Extract JSON array"""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    return match.group(0).strip() if match else None


def _extract_and_fix_incomplete_json(text: str) -> Optional[str]:
    """Try to extract and fix incomplete JSON"""
    # Find the first {
    start = text.find('{')
    if start < 0:
        return None

    json_part = text[start:].strip()

    # Count braces and try to balance them
    open_braces = json_part.count('{')
    close_braces = json_part.count('}')

    if open_braces > close_braces:
        # Add missing closing braces
        json_part += '}' * (open_braces - close_braces)

    # Try to clean up common issues
    json_part = re.sub(r',\s*}', '}', json_part)  # Remove trailing commas
    json_part = re.sub(r',\s*]', ']', json_part)  # Remove trailing commas in arrays

    return json_part