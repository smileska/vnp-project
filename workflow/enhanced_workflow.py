import asyncio
import os
import json
from datetime import datetime
from typing import Tuple, Optional, Any

from config import Config, get_filtered_model_configurations
from utils.data_loader import save_results
from utils.file_utils import create_results_folder
from models.registry import get_model_provider, get_available_models, is_ocr_only_model
from evaluation.text_similarity import calculate_text_similarity
from evaluation.json_accuracy import calculate_json_accuracy
from model_types.models import BenchmarkResult, ModelConfig, Usage
from models.schema_generator import SchemaGenerator, EvaluationModel


class DynamicTestDocument:
    def __init__(self, image_url: str, document_type: str = "document", ground_truth_json: Optional[dict] = None):
        self.image_url = image_url
        self.document_type = document_type
        self.ground_truth_json = ground_truth_json
        self.generated_schema = None
        self.metadata = {
            "document_type": document_type,
            "has_ground_truth": ground_truth_json is not None
        }


class EnhancedBenchmarkResult(BenchmarkResult):
    generated_schema: Optional[dict] = None
    schema_generation_time: Optional[float] = None
    workflow_type: str = "enhanced"


async def generate_schema_for_document(
        document: DynamicTestDocument,
        schema_generator: SchemaGenerator
) -> Tuple[dict, Usage]:
    print(f"Generating schema for {document.document_type}...")

    try:
        schema, usage = await schema_generator.generate_schema(
            document.image_url,
            document.document_type
        )
        document.generated_schema = schema
        print(f"Schema generated in {usage.duration:.2f}s")
        return schema, usage
    except Exception as e:
        print(f"Schema generation failed: {e}")
        basic_schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "extracted_info": {"type": "object"}
            }
        }
        usage = Usage(duration=0.0)
        return basic_schema, usage


async def process_document_enhanced_workflow(
        document: DynamicTestDocument,
        model_config: ModelConfig,
        schema_generator: SchemaGenerator,
        evaluation_model: EvaluationModel
) -> EnhancedBenchmarkResult:

    result = EnhancedBenchmarkResult(
        file_url=document.image_url,
        metadata=document.metadata,
        ocr_model=model_config.ocr_model,
        extraction_model=model_config.extraction_model or "",
        json_schema={},
        direct_image_extraction=model_config.direct_image_extraction,
        true_markdown="",
        true_json=document.ground_truth_json or {},
        workflow_type="enhanced"
    )

    try:
        schema, schema_usage = await generate_schema_for_document(document, schema_generator)
        result.generated_schema = schema
        result.json_schema = schema
        result.schema_generation_time = schema_usage.duration

        print(f"Performing OCR with {model_config.ocr_model}...")
        ocr_provider = get_model_provider(model_config.ocr_model)

        if is_ocr_only_model(model_config.ocr_model):
            extracted_text, ocr_usage = await ocr_provider.perform_ocr(document.image_url)
            result.predicted_markdown = extracted_text
            print(f"OCR completed in {ocr_usage.duration:.2f}s")

            print(f"Extracting structured data with evaluation model...")
            structured_data, eval_usage = await evaluation_model.extract_structured_data(
                extracted_text,
                schema,
                document.document_type
            )
            result.predicted_json = structured_data

            total_usage = Usage(
                duration=(schema_usage.duration or 0) + (ocr_usage.duration or 0) + (eval_usage.duration or 0),
                input_tokens=(ocr_usage.input_tokens or 0) + (eval_usage.input_tokens or 0),
                output_tokens=(ocr_usage.output_tokens or 0) + (eval_usage.output_tokens or 0),
                total_cost=(ocr_usage.total_cost or 0) + (eval_usage.total_cost or 0)
            )
            result.usage = total_usage

        else:
            print(f"Direct extraction with {model_config.ocr_model}...")
            structured_data, llm_usage = await ocr_provider.extract_json_from_image(
                document.image_url,
                schema
            )
            result.predicted_json = structured_data

            total_usage = Usage(
                duration=(schema_usage.duration or 0) + (llm_usage.duration or 0),
                input_tokens=llm_usage.input_tokens or 0,
                output_tokens=llm_usage.output_tokens or 0,
                total_cost=llm_usage.total_cost or 0
            )
            result.usage = total_usage

        if document.ground_truth_json and result.predicted_json:
            json_accuracy, json_diff = calculate_json_accuracy(
                document.ground_truth_json,
                result.predicted_json
            )
            result.json_accuracy = json_accuracy
            result.json_diff = json_diff
            print(f"JSON accuracy: {json_accuracy:.3f}")

    except Exception as e:
        result.error = str(e)
        print(f"Error: {e}")

    return result


async def run_enhanced_benchmark(test_images: list, document_types: list = None, ground_truths: list = None):

    print("Starting Enhanced LLM OCR Comparison Benchmark")
    print("=" * 60)

    schema_generator = SchemaGenerator()
    evaluation_model = EvaluationModel()

    documents = []
    for i, image_url in enumerate(test_images):
        doc_type = document_types[i] if document_types and i < len(document_types) else "document"
        ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None

        document = DynamicTestDocument(
            image_url=image_url,
            document_type=doc_type,
            ground_truth_json=ground_truth
        )
        documents.append(document)

    print(f"Created {len(documents)} test documents")

    valid_configs = get_filtered_model_configurations()
    if not valid_configs:
        print("No valid model configurations found!")
        return

    print(f"🔧 Testing {len(valid_configs)} model configurations")

    results_dir = create_results_folder(Config.RESULTS_DIR)
    print(f"Results will be saved to: {results_dir}")

    all_results = []
    total_tasks = len(documents) * len(valid_configs)
    completed_tasks = 0

    for model_config in valid_configs:
        model_name = f"{model_config.ocr_model}"
        if model_config.extraction_model and model_config.extraction_model != model_config.ocr_model:
            model_name += f" → {model_config.extraction_model}"

        print(f"\nProcessing with {model_name}...")

        for i, document in enumerate(documents, 1):
            print(f"  📄 Document {i}/{len(documents)}: {os.path.basename(document.image_url)}")

            try:
                result = await process_document_enhanced_workflow(
                    document,
                    model_config,
                    schema_generator,
                    evaluation_model
                )
                all_results.append(result.model_dump())

                if result.error:
                    print(f"Error: {result.error}")
                else:
                    status_msgs = []
                    if result.json_accuracy is not None:
                        status_msgs.append(f"JSON: {result.json_accuracy:.3f}")
                    if result.usage and result.usage.duration:
                        status_msgs.append(f"Time: {result.usage.duration:.1f}s")

                    if status_msgs:
                        print(f"{' | '.join(status_msgs)}")

            except Exception as e:
                print(f"Unexpected error: {e}")
                error_result = EnhancedBenchmarkResult(
                    file_url=document.image_url,
                    metadata=document.metadata,
                    ocr_model=model_config.ocr_model,
                    extraction_model=model_config.extraction_model or "",
                    json_schema={},
                    direct_image_extraction=model_config.direct_image_extraction,
                    true_markdown="",
                    true_json=document.ground_truth_json or {},
                    error=str(e),
                    workflow_type="enhanced"
                )
                all_results.append(error_result.model_dump())

            completed_tasks += 1
            progress = (completed_tasks / total_tasks) * 100
            print(f"Overall progress: {completed_tasks}/{total_tasks} ({progress:.1f}%)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"enhanced_benchmark_results_{timestamp}.json")
    save_results(all_results, results_file)

    successful_results = [r for r in all_results if not r.get('error')]
    total_results = len(all_results)
    success_rate = len(successful_results) / total_results * 100 if total_results > 0 else 0

    json_accuracies = [r['json_accuracy'] for r in successful_results if r.get('json_accuracy') is not None]
    avg_json_accuracy = sum(json_accuracies) / len(json_accuracies) if json_accuracies else 0

    schema_times = [r.get('schema_generation_time') for r in successful_results
                    if r.get('schema_generation_time') is not None]
    avg_schema_time = sum(schema_times) / len(schema_times) if schema_times else 0

    print("\n" + "=" * 60)
    print("ENHANCED BENCHMARK COMPLETE!")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Model configurations tested: {len(valid_configs)}")
    print(f"Total results generated: {len(all_results)}")
    print(f"Success rate: {success_rate:.1f}%")

    if json_accuracies:
        print(f"Average JSON accuracy: {avg_json_accuracy:.3f}")
    if avg_schema_time > 0:
        print(f"⚙Average schema generation time: {avg_schema_time:.1f}s")

    print(f"Results saved to: {results_file}")
    print("\nNext steps:")
    print("1. View detailed results in dashboard:")
    print("   cd dashboard && streamlit run enhanced_app.py")
    print("2. Compare enhanced vs traditional workflows")