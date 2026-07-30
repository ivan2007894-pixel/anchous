"""
OCR-based solver for text CAPTCHAs.

Uses EasyOCR for text recognition with image preprocessing.
"""

import logging
from pathlib import Path

from PIL import Image

from .captcha_types import TextCaptchaResult
from .image_utils import load_image, preprocess_for_ocr

logger = logging.getLogger(__name__)


class OCRSolver:
    """
    Text CAPTCHA solver using EasyOCR.

    Preprocessing pipeline:
    1. Grayscale conversion
    2. Upscaling for small images
    3. Contrast enhancement
    4. Noise reduction (median filter)
    5. Otsu binarization

    Then EasyOCR runs on the clean image.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = True,
    ):
        """
        Initialize OCR solver.

        Args:
            languages: List of language codes for EasyOCR (default: ["en"])
            gpu: Whether to use GPU for EasyOCR
        """
        import easyocr

        if languages is None:
            languages = ["en"]

        self.languages = languages
        logger.info(f"Loading EasyOCR with languages={languages}, gpu={gpu}...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        logger.info("EasyOCR loaded successfully")

    def solve(
        self,
        image_source: str | Path | bytes | Image.Image,
        preprocess: bool = True,
        allowlist: str | None = None,
    ) -> TextCaptchaResult:
        """
        Solve a text CAPTCHA.

        Args:
            image_source: CAPTCHA image — URL, file path, bytes, or PIL Image
            preprocess: Whether to apply preprocessing (recommended for most CAPTCHAs)
            allowlist: Optional string of allowed characters
                       (e.g., "abcdefghijklmnopqrstuvwxyz0123456789")

        Returns:
            TextCaptchaResult with recognized text and confidence
        """
        image = load_image(image_source)

        if preprocess:
            image = preprocess_for_ocr(image)

        # Convert PIL Image to numpy array for EasyOCR
        import numpy as np
        img_array = np.array(image)

        # Run OCR
        kwargs = {
            "detail": 1,  # Return bounding boxes + confidence
            "paragraph": False,
        }
        if allowlist:
            kwargs["allowlist"] = allowlist

        results = self.reader.readtext(img_array, **kwargs)

        if not results:
            logger.warning("No text detected in CAPTCHA image")
            return TextCaptchaResult(text="", confidence=0.0)

        # Combine all detected text segments
        # Sort by x-coordinate (left to right) for proper reading order
        results.sort(key=lambda r: r[0][0][0])  # Sort by top-left x

        raw_scores = []
        for bbox, text, conf in results:
            raw_scores.append((text, conf))

        # Join all text segments
        full_text = "".join(text for text, _ in raw_scores)
        avg_confidence = sum(conf for _, conf in raw_scores) / len(raw_scores)

        logger.info(
            f"OCR result: '{full_text}' (confidence={avg_confidence:.3f})"
        )

        return TextCaptchaResult(
            text=full_text,
            confidence=avg_confidence,
            raw_scores=raw_scores,
        )

    def solve_batch(
        self,
        image_sources: list[str | Path | bytes | Image.Image],
        preprocess: bool = True,
        allowlist: str | None = None,
    ) -> list[TextCaptchaResult]:
        """
        Solve multiple text CAPTCHAs.

        Args:
            image_sources: List of CAPTCHA images
            preprocess: Whether to apply preprocessing
            allowlist: Optional allowed characters

        Returns:
            List of TextCaptchaResult
        """
        return [
            self.solve(src, preprocess=preprocess, allowlist=allowlist)
            for src in image_sources
        ]
