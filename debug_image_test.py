import requests
import base64
import json


def test_image_download():
    print("Testing image download...")

    image_url = "https://omni-demo-data.s3.us-east-1.amazonaws.com/templates/receipt.png"

    try:
        print(f"Downloading from: {image_url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(image_url, timeout=30, headers=headers)
        print(f"HTTP Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Content-Length: {len(response.content)} bytes")

        if response.status_code == 200 and len(response.content) > 0:
            print("Image download successful!")

            b64_data = base64.b64encode(response.content).decode('utf-8')
            print(f"📊 Base64 encoded length: {len(b64_data)} characters")
            print(f"📊 First 50 chars: {b64_data[:50]}...")

            return True, b64_data
        else:
            print("Image download failed!")
            return False, None

    except Exception as e:
        print(f"Error downloading image: {e}")
        return False, None


def test_ollama_connection():
    print("\nTesting Ollama connection...")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)

        if response.status_code == 200:
            models = response.json()
            print("Ollama is running!")
            print(f"Available models: {[model['name'] for model in models.get('models', [])]}")
            return True
        else:
            print(f"Ollama responded with status: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama at localhost:11434")
        print("   Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return False


def test_simple_text_generation():
    print("\nTesting simple text generation...")

    try:
        payload = {
            "model": "llama3.2",
            "prompt": "Say 'Hello, this is a test' and nothing else.",
            "stream": False
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("Text generation works!")
            print(f"Response: {result.get('response', 'No response field')}")
            return True
        else:
            print(f"Text generation failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"Error in text generation: {e}")
        return False


def test_image_with_model():
    print("\nTesting image analysis with vision model...")

    success, b64_image = test_image_download()
    if not success:
        print("Cannot test image model without image")
        return False

    try:
        payload = {
            "model": "llava",
            "prompt": "Describe what you see in this image in one sentence.",
            "images": [b64_image],
            "stream": False
        }

        print("Sending image to llava model...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            print("Image analysis works!")
            print(f"Model response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"Image analysis failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"Error in image analysis: {e}")
        return False


def main():
    print("OCR System Debug Test")
    print("=" * 40)

    tests = [
        test_image_download,
        test_ollama_connection,
        test_simple_text_generation,
        test_image_with_model
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 40)
    print("Debug Test Summary")
    print("=" * 40)

    test_names = [
        "Image Download",
        "Ollama Connection",
        "Text Generation",
        "Image Analysis"
    ]

    for name, result in zip(test_names, results):
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")

    if all(results):
        print("\nAll tests passed! Your system is ready.")
        print("   You can now run: python test_enhanced_workflow.py --quick")
    else:
        print("\nSome tests failed. Fix the issues above before running the main benchmark.")


if __name__ == "__main__":
    main()