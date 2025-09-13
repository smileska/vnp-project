import asyncio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from workflows.enhanced_workflow import run_enhanced_benchmark


def main():
    print("Testing Enhanced OCR Workflow")
    print("=" * 40)

    receipt_ground_truth = {
        "merchant": {
            "name": "Nick the Greek Souvlaki & Gyro House",
            "phone": "(415) 757-0426",
            "address": "121 Spear Street, Suite B08, San Francisco, CA 94105"
        },
        "receipt_details": {
            "date": "November 8, 2024",
            "time": "2:16 PM",
            "receipt_number": "NKZ1"
        },
        "totals": {
            "tax": 6.18,
            "total": 48.43,
            "subtotal": 42.25
        }
    }

    test_scenarios = [
        {
            "name": "Receipt Analysis",
            "images": [
                "https://omni-demo-data.s3.us-east-1.amazonaws.com/templates/receipt.png"
            ],
            "types": ["receipt"],
            "ground_truths": [receipt_ground_truth]
        },
        {
            "name": "Invoice Analysis (using receipt image as test)",
            "images": [
                "https://omni-demo-data.s3.us-east-1.amazonaws.com/templates/receipt.png"
            ],
            "types": ["invoice"],
            "ground_truths": [None]
        },
        {
            "name": "Form Analysis (using receipt image as test)",
            "images": [
                "https://omni-demo-data.s3.us-east-1.amazonaws.com/templates/receipt.png"
            ],
            "types": ["form"],
            "ground_truths": [None]
        }
    ]

    total_scenarios = len([s for s in test_scenarios if s["images"] and s["images"][0]])
    completed_scenarios = 0

    for scenario in test_scenarios:
        if not scenario["images"] or not scenario["images"][0]:
            print(f"Skipping {scenario['name']} - no test images configured")
            continue

        print(f"\nRunning scenario: {scenario['name']}")
        print("-" * 30)

        print(f"Image: {scenario['images'][0].split('/')[-1]}")
        print(f"Document type: {scenario['types'][0]}")
        print(
            f"Ground truth: {'Available' if scenario['ground_truths'] and scenario['ground_truths'][0] else 'None (schema generation test only)'}")

        try:
            asyncio.run(run_enhanced_benchmark(
                test_images=scenario["images"],
                document_types=scenario["types"],
                ground_truths=scenario["ground_truths"] if scenario["ground_truths"] and scenario["ground_truths"][
                    0] else None
            ))
            completed_scenarios += 1
            print(f"{scenario['name']} completed successfully")

        except Exception as e:
            print(f"{scenario['name']} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTest Results: {completed_scenarios}/{total_scenarios} scenarios completed!")

    print(f"\nWhat this test demonstrates:")
    print("1. Schema generation for different document types (receipt, invoice, form)")
    print("2. OCR extraction using your available models")
    print("3. AI-powered structured data extraction")
    print("4. Accuracy measurement (for receipt scenario with ground truth)")
    print("5. Performance timing for each component")

    print(f"\nNext steps:")
    print("1. Check the results in: cd dashboard && streamlit run enhanced_app.py")
    print("2. Compare how different models handle the same image as different document types")
    print("3. Add your own images by replacing URLs in the test scenarios")


def quick_test():
    print("Quick Test Mode - Receipt Only")
    print("=" * 40)

    receipt_ground_truth = {
        "merchant": {
            "name": "Nick the Greek Souvlaki & Gyro House",
            "phone": "(415) 757-0426",
            "address": "121 Spear Street, Suite B08, San Francisco, CA 94105"
        },
        "receipt_details": {
            "date": "November 8, 2024",
            "time": "2:16 PM",
            "receipt_number": "NKZ1"
        },
        "totals": {
            "tax": 6.18,
            "total": 48.43,
            "subtotal": 42.25
        }
    }

    test_images = ["https://omni-demo-data.s3.us-east-1.amazonaws.com/templates/receipt.png"]
    document_types = ["receipt"]
    ground_truths = [receipt_ground_truth]

    try:
        asyncio.run(run_enhanced_benchmark(test_images, document_types, ground_truths))
        print("Quick test completed successfully!")
        print("\nThis test:")
        print("- Generated a schema by analyzing the receipt image")
        print("- Performed OCR to extract text")
        print("- Used AI to convert text to structured JSON")
        print("- Calculated accuracy against the known ground truth")
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()