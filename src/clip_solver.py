"""
CLIP-based CAPTCHA solver for image grid / image classification CAPTCHAs.

Uses OpenAI's CLIP model for zero-shot image classification.
Supports hCaptcha, reCAPTCHA v2, and similar grid-based CAPTCHAs.
"""

import logging
from pathlib import Path

import torch
import clip
from PIL import Image

from .captcha_types import (
    CaptchaType,
    GridSize,
    ImageCaptchaResult,
    TileResult,
)
from .image_utils import load_image, split_grid, preprocess_for_clip, draw_annotations, image_to_base64
from .prompts import CAPTCHA_PROMPTS, get_prompts, resolve_prompt

logger = logging.getLogger(__name__)


class CLIPSolver:
    """
    CAPTCHA solver using CLIP zero-shot classification.

    How it works:
    1. Load CAPTCHA grid image
    2. Split into individual tiles
    3. For each tile, compute CLIP similarity with positive and negative prompts
    4. Select tiles where positive similarity exceeds threshold

    Attributes:
        model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14")
        device: torch device (cuda/cpu)
        threshold: Minimum confidence score to select a tile
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str | None = None,
        threshold: float = 0.5,
    ):
        """
        Initialize CLIP solver.

        Args:
            model_name: CLIP model variant. Options:
                - "ViT-B/32" — fastest, good accuracy (default)
                - "ViT-B/16" — slower, better accuracy
                - "ViT-L/14" — slowest, best accuracy
                - "ViT-L/14@336px" — highest resolution, best for small details
            device: "cuda", "cpu", or None (auto-detect)
            threshold: Confidence threshold for tile selection (0.0 to 1.0)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.threshold = threshold

        logger.info(f"Loading CLIP model {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        logger.info(f"CLIP model loaded successfully")

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single image to CLIP feature space."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def _encode_images_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode multiple images in a single batch for efficiency."""
        image_inputs = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        features = self.model.encode_image(image_inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode text prompts to CLIP feature space."""
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between image and text features."""
        # image_features: (N_images, D)
        # text_features: (N_texts, D)
        # result: (N_images, N_texts)
        similarity = image_features @ text_features.T
        return similarity

    def solve(
        self,
        image_source: str | Path | bytes | Image.Image,
        prompt: str,
        grid: str = "3x3",
        threshold: float | None = None,
        return_annotated_image: bool = False,
    ) -> ImageCaptchaResult:
        """
        Solve an image grid CAPTCHA.

        Args:
            image_source: CAPTCHA image — URL, file path, bytes, or PIL Image
            prompt: CAPTCHA prompt text (e.g., "Select all images with a bus")
            grid: Grid size as string ("3x3" or "4x4")
            threshold: Override default threshold for this solve

        Returns:
            ImageCaptchaResult with selected tiles and confidence scores
        """
        threshold = threshold if threshold is not None else self.threshold
        grid_size = GridSize.from_string(grid)

        # 1. Resolve prompt to category
        category = resolve_prompt(prompt)
        if category is None:
            # Fallback: use the raw prompt directly
            logger.warning(
                f"Could not resolve prompt '{prompt}' to known category. "
                f"Using raw prompt as fallback."
            )
            category = prompt.lower().strip()

        prompts = get_prompts(category)
        logger.info(
            f"Solving CAPTCHA: category='{category}', grid={grid}, "
            f"threshold={threshold}"
        )

        # 2. Load and split image
        image = load_image(image_source)
        if grid_size == GridSize.GRID_1X1:
            tiles = [image]
        else:
            tiles = split_grid(image, grid_size.rows, grid_size.cols)

        # 3. Encode all tiles in batch
        tile_features = self._encode_images_batch(tiles)

        # 4. Encode positive and negative prompts
        pos_features = self._encode_texts(prompts["positive"])
        neg_features = self._encode_texts(prompts["negative"])

        # 5. Compute similarities
        pos_sim = self._compute_similarity(tile_features, pos_features)
        neg_sim = self._compute_similarity(tile_features, neg_features)

        # Average across all prompt variants
        pos_scores = pos_sim.mean(dim=1)  # (N_tiles,)
        neg_scores = neg_sim.mean(dim=1)  # (N_tiles,)

        # 6. Determine matches
        tile_results = []
        selected = []

        for i in range(len(tiles)):
            pos_score = pos_scores[i].item()
            neg_score = neg_scores[i].item()

            # Confidence = how much more similar to positive vs negative
            # Normalize to [0, 1] range using softmax-like scaling
            confidence = self._compute_confidence(pos_score, neg_score)
            is_match = confidence >= threshold

            tile_result = TileResult(
                index=i,
                positive_score=pos_score,
                negative_score=neg_score,
                is_match=is_match,
                confidence=confidence,
            )
            tile_results.append(tile_result)

            if is_match:
                selected.append(i)

            logger.debug(f"  {tile_result}")

        annotated_image_base64 = None
        if return_annotated_image and selected:
            # Draw dots on the full image
            annotated_img = draw_annotations(
                image,
                grid_size.rows,
                grid_size.cols,
                selected
            )
            annotated_image_base64 = image_to_base64(annotated_img)

        result = ImageCaptchaResult(
            captcha_type=CaptchaType.IMAGE_GRID,
            category=category,
            grid_size=grid_size,
            tiles=tile_results,
            selected_indices=selected,
            raw_prompt=prompt,
            annotated_image_base64=annotated_image_base64,
        )

        logger.info(
            f"Result: selected {len(selected)}/{len(tiles)} tiles, "
            f"avg_confidence={result.avg_confidence:.3f}"
        )

        return result

    def solve_single(
        self,
        image_source: str | Path | bytes | Image.Image,
        prompt: str,
        threshold: float | None = None,
    ) -> ImageCaptchaResult:
        """
        Solve a single image classification CAPTCHA (not a grid).

        Args:
            image_source: Image to classify
            prompt: What to look for
            threshold: Override default threshold

        Returns:
            ImageCaptchaResult with single tile result
        """
        return self.solve(
            image_source=image_source,
            prompt=prompt,
            grid="1x1",
            threshold=threshold,
        )

    def solve_tiles(
        self,
        tile_sources: list[str | Path | bytes | Image.Image],
        prompt: str,
        threshold: float | None = None,
        return_annotated_image: bool = False,
    ) -> ImageCaptchaResult:
        """
        Solve CAPTCHA when tiles are provided as separate images.
        Useful when tiles are already split (e.g., hCaptcha sends individual tile URLs).

        Args:
            tile_sources: List of tile images/URLs
            prompt: CAPTCHA prompt text
            threshold: Override default threshold

        Returns:
            ImageCaptchaResult with selected tiles
        """
        threshold = threshold if threshold is not None else self.threshold

        # Resolve category
        category = resolve_prompt(prompt)
        if category is None:
            category = prompt.lower().strip()

        prompts = get_prompts(category)

        # Load all tiles
        tiles = [load_image(src) for src in tile_sources]

        # Encode
        tile_features = self._encode_images_batch(tiles)
        pos_features = self._encode_texts(prompts["positive"])
        neg_features = self._encode_texts(prompts["negative"])

        pos_sim = self._compute_similarity(tile_features, pos_features)
        neg_sim = self._compute_similarity(tile_features, neg_features)
        pos_scores = pos_sim.mean(dim=1)
        neg_scores = neg_sim.mean(dim=1)

        tile_results = []
        selected = []

        for i in range(len(tiles)):
            pos_score = pos_scores[i].item()
            neg_score = neg_scores[i].item()
            confidence = self._compute_confidence(pos_score, neg_score)
            is_match = confidence >= threshold

            tile_results.append(TileResult(
                index=i,
                positive_score=pos_score,
                negative_score=neg_score,
                is_match=is_match,
                confidence=confidence,
            ))

            if is_match:
                selected.append(i)

        n_tiles = len(tiles)
        grid_size = (
            GridSize.GRID_3X3 if n_tiles == 9
            else GridSize.GRID_4X4 if n_tiles == 16
            else GridSize.GRID_1X1
        )

        annotated_image_base64 = None
        if return_annotated_image and selected:
            # Reconstruct grid image to annotate
            tile_w, tile_h = tiles[0].size
            full_w = tile_w * grid_size.cols
            full_h = tile_h * grid_size.rows
            full_image = Image.new("RGB", (full_w, full_h))
            
            for i, tile in enumerate(tiles):
                r = i // grid_size.cols
                c = i % grid_size.cols
                full_image.paste(tile, (c * tile_w, r * tile_h))
                
            annotated_img = draw_annotations(
                full_image,
                grid_size.rows,
                grid_size.cols,
                selected
            )
            annotated_image_base64 = image_to_base64(annotated_img)

        return ImageCaptchaResult(
            captcha_type=CaptchaType.IMAGE_GRID,
            category=category,
            grid_size=grid_size,
            tiles=tile_results,
            selected_indices=selected,
            raw_prompt=prompt,
            annotated_image_base64=annotated_image_base64,
        )

    def classify_image(
        self,
        image_source: str | Path | bytes | Image.Image,
        categories: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Classify a single image against multiple categories.
        Useful for determining what a CAPTCHA tile contains.

        Args:
            image_source: Image to classify
            categories: List of category names. If None, uses all known categories.

        Returns:
            Dict of {category: similarity_score}, sorted by score descending
        """
        if categories is None:
            categories = list(CAPTCHA_PROMPTS.keys())

        image = load_image(image_source)
        image_features = self._encode_image(image)

        results = {}
        for category in categories:
            prompts = get_prompts(category)
            text_features = self._encode_texts(prompts["positive"])
            sim = self._compute_similarity(image_features, text_features)
            results[category] = sim.mean().item()

        # Sort by score descending
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return results

    @staticmethod
    def _compute_confidence(pos_score: float, neg_score: float) -> float:
        """
        Compute confidence score from positive and negative similarity.
        Returns a value in [0, 1] range.
        """
        # Use softmax-style normalization
        import math
        temp = 10.0  # Temperature scaling — higher = more spread
        try:
            exp_pos = math.exp(pos_score * temp)
            exp_neg = math.exp(neg_score * temp)
            confidence = exp_pos / (exp_pos + exp_neg)
        except OverflowError:
            # If scores are very large, the higher one dominates
            confidence = 1.0 if pos_score > neg_score else 0.0
        return confidence

    def warmup(self) -> None:
        """
        Warmup the model with a dummy inference.
        Useful for accurate benchmarking after cold start.
        """
        dummy_image = Image.new("RGB", (224, 224), color="white")
        self._encode_image(dummy_image)
        self._encode_texts(["a test prompt"])
        logger.info("Model warmup complete")

    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "threshold": self.threshold,
            "known_categories": list(CAPTCHA_PROMPTS.keys()),
            "total_categories": len(CAPTCHA_PROMPTS),
        }
