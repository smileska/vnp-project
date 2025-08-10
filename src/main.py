import asyncio
import os
import json
import re
from datetime import datetime
from typing import Tuple, Optional, Any

from config import Config, MODEL_CONFIGURATIONS
from utils.data_loader import load_test_documents, save_results
from utils.file_utils import create_results_folder
from models.registry import get_model_provider, get_available_models
from evaluation.text_similarity import calculate_text_similarity
from evaluation.json_accuracy import calculate_json_accuracy
from model_types.models import BenchmarkResult, TestDocument, ModelConfig


def robust_json_extraction(response: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Extract JSON from a model response that may contain markdown formatting or explanatory text.

    Returns:
        Tuple of (parsed_json, error_message)
    """
    if not response or not response.strip():
        return None, "Empty response"

    # Handle responses that explicitly say no data found
    no_data_indicators = [
        "unable to find enough information",
        "no information available",
        "cannot extract",
        "insufficient data",
        "does not contain",
        "no data found"
    ]

    response_lower = response.lower()
    for indicator in no_data_indicators:
        if indicator in response_lower:
            # Return empty JSON structure based on common patterns
            return {}, None

    # Clean common prefixes that models add
    cleaned = response.strip()
    prefixes_to_remove = [
        r"Based on.*?Here is the resulting JSON:\s*",
        r"Here is the JSON.*?:\s*",
        r"The extracted data in JSON format:\s*",
        r".*?Here's the structured data:\s*",
        r"Based on the provided.*?:\s*",
    ]

    for prefix_pattern in prefixes_to_remove:
        cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Try multiple extraction strategies in order of preference
    strategies = [
        # Strategy 1: Look for ```json blocks
        lambda text: re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL),
        # Strategy 2: Look for ``` blocks (without json specifier)
        lambda text: re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL),
        # Strategy 3: Look for any complete JSON object (with proper braces)
        lambda text: re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text, re.DOTALL),
        # Strategy 4: Look for JSON array
        lambda text: re.search(r'(\[.*\])', text, re.DOTALL),
        # Strategy 5: Look for partial JSON and try to complete it
        lambda text: re.search(r'(\{.*)', text, re.DOTALL),
    ]

    for i, strategy in enumerate(strategies, 1):
        match = strategy(cleaned)
        if match:
            json_str = match.group(1).strip()

            # For strategy 5 (partial JSON), try to fix common issues
            if i == 5:
                # Count braces to see if we need to close them
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                missing_braces = open_braces - close_braces

                if missing_braces > 0:
                    json_str += '}' * missing_braces

            try:
                parsed = json.loads(json_str)
                return parsed, None
            except json.JSONDecodeError as e:
                # If it's strategy 5 and still fails, try with empty object
                if i == 5:
                    try:
                        # Try to extract just the first complete object
                        first_brace = json_str.find('{')
                        if first_brace >= 0:
                            brace_count = 0
                            end_pos = first_brace
                            for j, char in enumerate(json_str[first_brace:], first_brace):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_pos = j + 1
                                        break

                            if brace_count == 0:
                                partial_json = json_str[first_brace:end_pos]
                                return json.loads(partial_json), None
                    except:
                        continue
                continue  # Try next strategy

    # If all strategies fail, return empty object as fallback
    return {}, f"Could not extract valid JSON, returning empty object. Response preview: {response[:200]}..."


async def process_document_with_model(
        document: TestDocument,
        model_config: ModelConfig
) -> BenchmarkResult:
    result = BenchmarkResult(
        file_url=document.imageUrl,
        metadata=document.metadata,
        ocr_model=model_config.ocr_model,
        extraction_model=model_config.extraction_model or "",
        json_schema=document.jsonSchema,
        direct_image_extraction=model_config.direct_image_extraction,
        true_markdown=document.trueMarkdownOutput,
        true_json=document.trueJsonOutput
    )

    try:
        if model_config.direct_image_extraction:
            provider = get_model_provider(model_config.extraction_model or model_config.ocr_model)

            try:
                raw_response, usage = await provider.extract_json_from_image(
                    document.imageUrl,
                    document.jsonSchema
                )

                # Handle potential string responses with embedded JSON
                if isinstance(raw_response, str):
                    extracted_json, error = robust_json_extraction(raw_response)
                    if error:
                        raise ValueError(f"JSON extraction failed: {error}")
                    result.predicted_json = extracted_json
                else:
                    result.predicted_json = raw_response

                result.usage = usage

            except Exception as provider_error:
                # If the provider fails with JSON parsing, try to extract from the error message
                error_str = str(provider_error)
                if "Response:" in error_str:
                    # Extract the full response from the error message
                    response_start = error_str.find("Response:") + len("Response:")
                    response_part = error_str[response_start:].strip()

                    extracted_json, extraction_error = robust_json_extraction(response_part)
                    if extracted_json is not None:  # Could be empty dict {}
                        result.predicted_json = extracted_json
                        print(f"✅ Recovered JSON from error message for {document.imageUrl}")
                        # Set usage to None since we couldn't get it from the provider
                        result.usage = None
                    else:
                        # Still couldn't extract, but log what we tried
                        print(f"⚠️  Failed to recover JSON from error. Extraction error: {extraction_error}")
                        raise provider_error
                else:
                    raise provider_error
        else:
            ocr_provider = get_model_provider(model_config.ocr_model)
            extracted_text, ocr_usage = await ocr_provider.perform_ocr(document.imageUrl)
            result.predicted_markdown = extracted_text

            if model_config.extraction_model:
                extraction_provider = get_model_provider(model_config.extraction_model)

                try:
                    raw_response, extraction_usage = await extraction_provider.extract_json_from_text(
                        extracted_text,
                        document.jsonSchema
                    )

                    # Handle potential string responses with embedded JSON
                    if isinstance(raw_response, str):
                        extracted_json, error = robust_json_extraction(raw_response)
                        if error:
                            raise ValueError(f"JSON extraction failed: {error}")
                        result.predicted_json = extracted_json
                    else:
                        result.predicted_json = raw_response

                    total_usage = ocr_usage.model_copy() if ocr_usage else None
                    if total_usage and extraction_usage:
                        if total_usage.duration and extraction_usage.duration:
                            total_usage.duration += extraction_usage.duration
                        if total_usage.input_tokens and extraction_usage.input_tokens:
                            total_usage.input_tokens += extraction_usage.input_tokens
                        if total_usage.output_tokens and extraction_usage.output_tokens:
                            total_usage.output_tokens += extraction_usage.output_tokens
                        if total_usage.total_cost and extraction_usage.total_cost:
                            total_usage.total_cost += extraction_usage.total_cost

                    result.usage = total_usage or extraction_usage

                except Exception as provider_error:
                    # If the provider fails with JSON parsing, try to extract from the error message
                    error_str = str(provider_error)
                    if "Response:" in error_str:
                        # Extract the full response from the error message
                        response_start = error_str.find("Response:") + len("Response:")
                        response_part = error_str[response_start:].strip()

                        extracted_json, extraction_error = robust_json_extraction(response_part)
                        if extracted_json is not None:  # Could be empty dict {}
                            result.predicted_json = extracted_json
                            result.usage = ocr_usage
                            print(f"✅ Recovered JSON from error message for {document.imageUrl}")
                        else:
                            # Still couldn't extract, but log what we tried
                            print(f"⚠️  Failed to recover JSON from error. Extraction error: {extraction_error}")
                            raise provider_error
                    else:
                        raise provider_error
            else:
                result.usage = ocr_usage

        if result.predicted_markdown:
            result.text_similarity = calculate_text_similarity(
                document.trueMarkdownOutput,
                result.predicted_markdown
            )

        if result.predicted_json:
            json_accuracy, json_diff = calculate_json_accuracy(
                document.trueJsonOutput,
                result.predicted_json
            )
            result.json_accuracy = json_accuracy
            result.json_diff = json_diff

    except Exception as e:
        result.error = str(e)
        print(f"❌ Error processing {document.imageUrl} with {model_config.ocr_model}: {e}")

    return result


async def run_benchmark():
    print("🚀 Starting LLM OCR Comparison Benchmark")
    print("=" * 50)

    available_models = get_available_models()
    print(f"\n🤖 Available models: {', '.join(available_models)}")

    documents = load_test_documents(Config.DATA_DIR)
    if not documents:
        print("\n❌ No test documents found!")
        print("Please add JSON files to the data/ folder with the following structure:")
        print("""{
  "imageUrl": "https://example.com/image.png",
  "metadata": {"language": "EN", "documentType": "receipt"},
  "jsonSchema": {"type": "object", "properties": {...}},
  "trueJsonOutput": {...},
  "trueMarkdownOutput": "..."
}""")
        return

    print(f"\n📄 Loaded {len(documents)} test documents")
    print(f"🔧 Testing {len(MODEL_CONFIGURATIONS)} model configurations")

    valid_configs = []
    for config in MODEL_CONFIGURATIONS:
        if config.ocr_model in available_models:
            if not config.extraction_model or config.extraction_model in available_models:
                valid_configs.append(config)
            else:
                print(f"⚠️  Skipping config: extraction model '{config.extraction_model}' not available")
        else:
            print(f"⚠️  Skipping config: OCR model '{config.ocr_model}' not available")

    if not valid_configs:
        print("❌ No valid model configurations found!")
        return

    print(f"✅ Using {len(valid_configs)} valid model configurations")

    results_dir = create_results_folder(Config.RESULTS_DIR)
    print(f"\n📁 Results will be saved to: {results_dir}")

    all_results = []
    total_tasks = len(documents) * len(valid_configs)

    for model_config in valid_configs:
        model_name = f"{model_config.ocr_model}"
        if model_config.extraction_model and model_config.extraction_model != model_config.ocr_model:
            model_name += f" → {model_config.extraction_model}"
        if model_config.direct_image_extraction:
            model_name += " (Direct)"

        print(f"\n🔄 Processing with {model_name}...")

        for i, document in enumerate(documents, 1):
            print(f"  📄 Document {i}/{len(documents)}: {os.path.basename(document.imageUrl)}")
            result = await process_document_with_model(document, model_config)
            all_results.append(result.model_dump())

            # Print status for each document
            if result.error:
                print(f"    ❌ Error: {result.error}")
            else:
                if result.json_accuracy is not None:
                    print(f"    ✅ JSON accuracy: {result.json_accuracy:.3f}")
                if result.text_similarity is not None:
                    print(f"    📝 Text similarity: {result.text_similarity:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
    save_results(all_results, results_file)

    successful_results = [r for r in all_results if not r.get('error')]
    total_results = len(all_results)
    success_rate = len(successful_results) / total_results * 100 if total_results > 0 else 0

    json_accuracies = [r['json_accuracy'] for r in successful_results if r.get('json_accuracy') is not None]
    text_similarities = [r['text_similarity'] for r in successful_results if r.get('text_similarity') is not None]

    avg_json_accuracy = sum(json_accuracies) / len(json_accuracies) if json_accuracies else 0
    avg_text_similarity = sum(text_similarities) / len(text_similarities) if text_similarities else 0

    print("\n" + "=" * 50)
    print("📊 BENCHMARK COMPLETE!")
    print(f"✅ Processed {len(documents)} documents")
    print(f"✅ Tested {len(valid_configs)} model configurations")
    print(f"✅ Generated {len(all_results)} results")
    print(f"📈 Success rate: {success_rate:.1f}%")
    if json_accuracies:
        print(f"📊 Average JSON accuracy: {avg_json_accuracy:.3f}")
    if text_similarities:
        print(f"📊 Average text similarity: {avg_text_similarity:.3f}")
    print(f"📁 Results saved to: {results_file}")

    print("\n🎯 Next steps:")
    print("1. View results in dashboard:")
    print("   cd dashboard && streamlit run app.py")
    print("2. Or check the results JSON file directly")


if __name__ == "__main__":
    asyncio.run(run_benchmark())