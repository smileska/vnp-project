import asyncio
import os
from datetime import datetime
from typing import List
from tqdm import tqdm

from src.config import Config, MODEL_CONFIGURATIONS
from src.utils.data_loader import load_test_documents, save_results
from src.utils.file_utils import create_results_folder
from src.models.registry import get_model_provider, get_available_models
from src.evaluation.text_similarity import calculate_text_similarity
from src.evaluation.json_accuracy import calculate_json_accuracy
from src.types.models import BenchmarkResult, TestDocument, ModelConfig

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
            extracted_json, usage = await provider.extract_json_from_image(
                document.imageUrl,
                document.jsonSchema
            )
            result.predicted_json = extracted_json
            result.usage = usage
        else:
            ocr_provider = get_model_provider(model_config.ocr_model)
            extracted_text, ocr_usage = await ocr_provider.perform_ocr(document.imageUrl)
            result.predicted_markdown = extracted_text

            if model_config.extraction_model:
                extraction_provider = get_model_provider(model_config.extraction_model)
                extracted_json, extraction_usage = await extraction_provider.extract_json_from_text(
                    extracted_text,
                    document.jsonSchema
                )
                result.predicted_json = extracted_json

                total_usage = ocr_usage.copy() if ocr_usage else None
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

def check_api_keys():
    print("🔑 Checking API keys...")

    checks = []

    if Config.GOOGLE_API_KEY:
        checks.append("✅ Google API key found")
    else:
        checks.append("❌ Google API key missing")

    if Config.TOGETHER_API_KEY:
        checks.append("✅ Together AI API key found")
    else:
        checks.append("⚠️  Together AI API key missing (optional)")

    if Config.GROQ_API_KEY:
        checks.append("✅ Groq API key found")
    else:
        checks.append("⚠️  Groq API key missing (optional)")

    for check in checks:
        print(f"  {check}")

    if not any([Config.GOOGLE_API_KEY, Config.TOGETHER_API_KEY, Config.GROQ_API_KEY]):
        print("\n❌ ERROR: No API keys found!")
        print("Please add at least one API key to your .env file:")
        print("- GOOGLE_API_KEY for Gemini models")
        print("- TOGETHER_API_KEY for hosted Llama models")
        print("- GROQ_API_KEY for fast Llama inference")
        print("- Or install Ollama for local Llama models")
        return False

    return True

async def run_benchmark():
    print("🚀 Starting LLM OCR Comparison Benchmark")
    print("=" * 50)

    if not check_api_keys():
        return

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

    with tqdm(total=total_tasks, desc="Processing", ncols=100) as pbar:
        for model_config in valid_configs:
            model_name = f"{model_config.ocr_model}"
            if model_config.extraction_model and model_config.extraction_model != model_config.ocr_model:
                model_name += f" → {model_config.extraction_model}"
            if model_config.direct_image_extraction:
                model_name += " (Direct)"

            pbar.set_description(f"Testing {model_name}")

            for document in documents:
                result = await process_document_with_model(document, model_config)
                all_results.append(result.dict())
                pbar.update(1)

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