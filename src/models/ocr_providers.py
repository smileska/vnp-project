import cv2
import numpy as np
import requests
import time
from typing import Tuple
from abc import ABC, abstractmethod

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    import easyocr

    EASY_OCR_AVAILABLE = True
except ImportError:
    EASY_OCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import io

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from models.base import BaseModelProvider
from model_types.models import Usage


class OCROnlyProvider(BaseModelProvider):
    """Base class for OCR-only providers that don't support JSON extraction"""

    def __init__(self, model_name: str):
        super().__init__(model_name)

    async def extract_json_from_text(self, text: str, json_schema: dict) -> Tuple[dict, Usage]:
        raise NotImplementedError("OCR-only models don't support JSON extraction from text")

    async def extract_json_from_image(self, image_url: str, json_schema: dict) -> Tuple[dict, Usage]:
        raise NotImplementedError("OCR-only models don't support JSON extraction from images")

    def _download_image(self, image_url: str) -> np.ndarray:
        """Download and convert image to OpenCV format"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            return image
        except Exception as e:
            raise Exception(f"Failed to download image from {image_url}: {str(e)}")


class PaddleOCRProvider(OCROnlyProvider):
    """PaddleOCR provider for OCR tasks"""

    def __init__(self):
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")

        super().__init__("paddleocr")
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        start_time = time.time()

        try:
            image = self._download_image(image_url)

            results = self.ocr_engine.ocr(image, cls=True)

            markdown_text = self._format_to_markdown(results)

            duration = time.time() - start_time
            usage = Usage(
                duration=duration,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )

            return markdown_text, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"PaddleOCR error: {str(e)}")

    def _format_to_markdown(self, results) -> str:
        """Convert PaddleOCR results to markdown format with better structure"""
        if not results or not results[0]:
            return ""

        lines = []
        last_y = None
        line_buffer = []

        for detection in results[0]:
            if detection and len(detection) >= 2:
                bbox = detection[0]
                text_info = detection[1]
                text = text_info[0] if isinstance(text_info, (list, tuple)) else text_info
                confidence = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0

                if confidence > 0.6:
                    y_coord = (bbox[0][1] + bbox[2][1]) / 2

                    if last_y is not None and abs(y_coord - last_y) > 20:
                        if line_buffer:
                            line_buffer.sort(key=lambda x: x[1])
                            line_text = ' '.join([item[0] for item in line_buffer])
                            if line_text.strip():
                                lines.append(line_text.strip())
                            line_buffer = []

                    x_coord = (bbox[0][0] + bbox[2][0]) / 2
                    line_buffer.append((text.strip(), x_coord))
                    last_y = y_coord

        if line_buffer:
            line_buffer.sort(key=lambda x: x[1])
            line_text = ' '.join([item[0] for item in line_buffer])
            if line_text.strip():
                lines.append(line_text.strip())

        return '\n'.join(lines) if lines else ""


class EasyOCRProvider(OCROnlyProvider):
    """EasyOCR provider for OCR tasks"""

    def __init__(self):
        if not EASY_OCR_AVAILABLE:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")

        super().__init__("easyocr")

        self.ocr_engine = easyocr.Reader(['en'], gpu=False)

    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        start_time = time.time()

        try:
            image = self._download_image(image_url)

            results = self.ocr_engine.readtext(image)

            markdown_text = self._format_to_markdown(results)

            duration = time.time() - start_time
            usage = Usage(
                duration=duration,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )

            return markdown_text, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"EasyOCR error: {str(e)}")

    def _format_to_markdown(self, results) -> str:
        """Convert EasyOCR results to markdown format with better structure"""
        if not results:
            return ""

        lines = []
        last_y = None
        line_buffer = []

        for detection in results:
            if len(detection) >= 2:
                bbox = detection[0]
                text = detection[1]
                confidence = detection[2] if len(detection) > 2 else 1.0

                if confidence > 0.6:
                    y_coords = [point[1] for point in bbox]
                    y_coord = sum(y_coords) / len(y_coords)

                    if last_y is not None and abs(y_coord - last_y) > 20:
                        if line_buffer:
                            line_buffer.sort(key=lambda x: x[1])
                            line_text = ' '.join([item[0] for item in line_buffer])
                            if line_text.strip():
                                lines.append(line_text.strip())
                            line_buffer = []

                    x_coords = [point[0] for point in bbox]
                    x_coord = sum(x_coords) / len(x_coords)
                    line_buffer.append((text.strip(), x_coord))
                    last_y = y_coord

        if line_buffer:
            line_buffer.sort(key=lambda x: x[1])
            line_text = ' '.join([item[0] for item in line_buffer])
            if line_text.strip():
                lines.append(line_text.strip())

        return '\n'.join(lines) if lines else ""


class TesseractProvider(OCROnlyProvider):
    """Tesseract OCR provider with enhanced configuration"""

    def __init__(self):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract dependencies not installed. Install with: pip install pytesseract pillow")

        super().__init__("tesseract")

        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract {version} detected")
        except Exception as e:
            raise Exception(f"Tesseract binary not found. Please install Tesseract OCR binary: {str(e)}")

    async def perform_ocr(self, image_url: str) -> Tuple[str, Usage]:
        start_time = time.time()

        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            image = self._preprocess_image(image)

            text = self._perform_multi_config_ocr(image)

            markdown_text = self._format_to_markdown(text)

            duration = time.time() - start_time
            usage = Usage(
                duration=duration,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )

            return markdown_text, usage

        except Exception as e:
            duration = time.time() - start_time
            usage = Usage(duration=duration)
            raise Exception(f"Tesseract OCR error: {str(e)}")

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        if image.mode != 'L':
            image = image.convert('L')

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        cv_image = cv2.medianBlur(cv_image, 3)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = clahe.apply(cv_image)

        _, cv_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return Image.fromarray(cv_image)

    def _perform_multi_config_ocr(self, image: Image.Image) -> str:
        """Try multiple Tesseract configurations and return best result"""
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 3',
            r'--oem 3 --psm 4',
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 13'
        ]

        results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if text and text.strip():
                    results.append(text)
            except Exception:
                continue

        if not results:
            try:
                return pytesseract.image_to_string(image)
            except Exception:
                return ""

        return max(results, key=len)

    def _format_to_markdown(self, text: str) -> str:
        """Enhanced formatting of Tesseract output to markdown"""
        if not text or not text.strip():
            return ""

        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if line:
                line = self._clean_ocr_text(line)
                if line:
                    formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _clean_ocr_text(self, text: str) -> str:
        """Clean common OCR errors"""
        if not text:
            return ""

        text = ' '.join(text.split())

        corrections = {
            '|': 'I',
            '0': 'O',
            '5': 'S',
            '1': 'l',
        }

        return text


def get_ocr_provider_availability():
    """Check which OCR providers are available"""
    availability = {
        'paddleocr': PADDLE_AVAILABLE,
        'easyocr': EASY_OCR_AVAILABLE,
        'tesseract': TESSERACT_AVAILABLE
    }

    if TESSERACT_AVAILABLE:
        try:
            pytesseract.get_tesseract_version()
            availability['tesseract'] = True
        except Exception:
            availability['tesseract'] = False

    return availability


def test_ocr_providers():
    """Test all available OCR providers with a simple test"""
    print("🧪 Testing OCR Providers")
    print("=" * 30)

    availability = get_ocr_provider_availability()

    for provider_name, available in availability.items():
        if available:
            try:
                if provider_name == 'paddleocr':
                    provider = PaddleOCRProvider()
                elif provider_name == 'easyocr':
                    provider = EasyOCRProvider()
                elif provider_name == 'tesseract':
                    provider = TesseractProvider()

                print(f"✅ {provider_name}: Ready")

            except Exception as e:
                print(f"❌ {provider_name}: Failed to initialize - {e}")
        else:
            print(f"⚠️  {provider_name}: Not available")


if __name__ == "__main__":
    test_ocr_providers()