"""
Tests for CAPTCHA solver.

These tests verify the solver logic without requiring GPU.
Run with: python -m pytest tests/ -v
"""

import math
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.captcha_types import (
    CaptchaType,
    GridSize,
    ImageCaptchaResult,
    TextCaptchaResult,
    TileResult,
)
from src.image_utils import split_grid, preprocess_for_ocr, _otsu_threshold
from src.prompts import resolve_prompt, get_prompts, CAPTCHA_PROMPTS


# ============================================================
# Tests for prompts.py
# ============================================================

class TestPrompts:
    """Tests for prompt resolution and lookup."""

    def test_resolve_known_prompt(self):
        """Known CAPTCHA prompts should resolve to categories."""
        assert resolve_prompt("Select all images with a bus") == "bus"
        assert resolve_prompt("select all images with a bus") == "bus"

    def test_resolve_hcaptcha_prompt(self):
        """hCaptcha-style prompts should resolve."""
        assert resolve_prompt("Please click each image containing a bus") == "bus"
        assert resolve_prompt("Please click each image containing a lion") == "lion"

    def test_resolve_unknown_prompt(self):
        """Unknown prompts should return None."""
        result = resolve_prompt("select all images with a unicorn")
        assert result is None

    def test_resolve_partial_match(self):
        """Prompts containing known keywords should fuzzy match."""
        result = resolve_prompt("click on the traffic light images")
        assert result == "traffic_light"

    def test_get_prompts_known_category(self):
        """Known categories should return positive and negative prompts."""
        prompts = get_prompts("bus")
        assert "positive" in prompts
        assert "negative" in prompts
        assert len(prompts["positive"]) > 0
        assert len(prompts["negative"]) > 0

    def test_get_prompts_unknown_category(self):
        """Unknown categories should return fallback prompts."""
        prompts = get_prompts("spaceship")
        assert "positive" in prompts
        assert "negative" in prompts
        assert any("spaceship" in p for p in prompts["positive"])

    def test_all_categories_have_both_prompts(self):
        """Every category should have both positive and negative prompts."""
        for category, prompts in CAPTCHA_PROMPTS.items():
            assert "positive" in prompts, f"{category} missing positive prompts"
            assert "negative" in prompts, f"{category} missing negative prompts"
            assert len(prompts["positive"]) >= 2, f"{category} needs ≥2 positive prompts"
            assert len(prompts["negative"]) >= 2, f"{category} needs ≥2 negative prompts"


# ============================================================
# Tests for captcha_types.py
# ============================================================

class TestCaptchaTypes:
    """Tests for type definitions and data models."""

    def test_grid_size_from_string(self):
        assert GridSize.from_string("3x3") == GridSize.GRID_3X3
        assert GridSize.from_string("4x4") == GridSize.GRID_4X4
        assert GridSize.from_string("1x1") == GridSize.GRID_1X1

    def test_grid_size_invalid(self):
        with pytest.raises(ValueError):
            GridSize.from_string("5x5")

    def test_grid_size_properties(self):
        grid = GridSize.GRID_3X3
        assert grid.rows == 3
        assert grid.cols == 3
        assert grid.total_tiles == 9

    def test_tile_result_repr(self):
        tile = TileResult(index=0, positive_score=0.8, negative_score=0.2,
                          is_match=True, confidence=0.9)
        assert "✓" in repr(tile)
        assert "0" in repr(tile)

    def test_image_captcha_result_to_dict(self):
        result = ImageCaptchaResult(
            category="bus",
            grid_size=GridSize.GRID_3X3,
            tiles=[
                TileResult(0, 0.8, 0.2, True, 0.9),
                TileResult(1, 0.3, 0.7, False, 0.3),
            ],
            selected_indices=[0],
        )
        d = result.to_dict()
        assert d["category"] == "bus"
        assert d["selected_tiles"] == [0]
        assert len(d["all_tiles"]) == 2

    def test_image_captcha_result_avg_confidence(self):
        result = ImageCaptchaResult(
            tiles=[
                TileResult(0, 0.8, 0.2, True, 0.9),
                TileResult(1, 0.7, 0.3, True, 0.8),
                TileResult(2, 0.3, 0.7, False, 0.3),
            ],
            selected_indices=[0, 1],
        )
        assert result.avg_confidence == pytest.approx(0.85, abs=0.01)

    def test_text_captcha_result_to_dict(self):
        result = TextCaptchaResult(
            text="abc123",
            confidence=0.95,
            raw_scores=[("abc", 0.96), ("123", 0.94)],
        )
        d = result.to_dict()
        assert d["text"] == "abc123"
        assert d["confidence"] == 0.95
        assert len(d["alternatives"]) == 2


# ============================================================
# Tests for image_utils.py
# ============================================================

class TestImageUtils:
    """Tests for image processing utilities."""

    def test_split_grid_3x3(self):
        """3x3 grid split should produce 9 tiles."""
        img = Image.new("RGB", (300, 300), "white")
        tiles = split_grid(img, rows=3, cols=3)
        assert len(tiles) == 9
        assert all(t.size == (100, 100) for t in tiles)

    def test_split_grid_4x4(self):
        """4x4 grid split should produce 16 tiles."""
        img = Image.new("RGB", (400, 400), "white")
        tiles = split_grid(img, rows=4, cols=4)
        assert len(tiles) == 16
        assert all(t.size == (100, 100) for t in tiles)

    def test_split_grid_preserves_content(self):
        """Tiles should contain the correct region of the source image."""
        img = Image.new("RGB", (200, 200), "white")
        # Draw a red pixel at (50, 50) — should be in tile index 0
        img.putpixel((50, 50), (255, 0, 0))
        tiles = split_grid(img, rows=2, cols=2)
        # Tile 0 is top-left (0:100, 0:100), pixel at (50,50) should be red
        assert tiles[0].getpixel((50, 50)) == (255, 0, 0)

    def test_preprocess_for_ocr(self):
        """OCR preprocessing should return an RGB image."""
        img = Image.new("RGB", (100, 30), "white")
        result = preprocess_for_ocr(img)
        assert result.mode == "RGB"
        # Image should be upscaled
        assert result.size[1] >= 30

    def test_otsu_threshold(self):
        """Otsu threshold should return a value between 0 and 255."""
        img = Image.new("L", (100, 100), 128)
        threshold = _otsu_threshold(img)
        assert 0 <= threshold <= 255

    def test_otsu_threshold_bimodal(self):
        """Otsu should find threshold between two peaks."""
        # Create bimodal image: half black, half white
        img = Image.new("L", (100, 100))
        for x in range(100):
            for y in range(100):
                img.putpixel((x, y), 0 if y < 50 else 255)
        threshold = _otsu_threshold(img)
        # Threshold should be somewhere in the middle
        assert 50 < threshold < 200


# ============================================================
# Tests for CLIPSolver (mocked — no GPU needed)
# ============================================================

class TestCLIPSolverLogic:
    """Test CLIPSolver helper methods without loading the actual model."""

    def test_compute_confidence_positive(self):
        """Higher positive score should give high confidence."""
        from src.clip_solver import CLIPSolver
        conf = CLIPSolver._compute_confidence(0.8, 0.2)
        assert conf > 0.9

    def test_compute_confidence_negative(self):
        """Higher negative score should give low confidence."""
        from src.clip_solver import CLIPSolver
        conf = CLIPSolver._compute_confidence(0.2, 0.8)
        assert conf < 0.1

    def test_compute_confidence_equal(self):
        """Equal scores should give ~0.5 confidence."""
        from src.clip_solver import CLIPSolver
        conf = CLIPSolver._compute_confidence(0.5, 0.5)
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_compute_confidence_range(self):
        """Confidence should always be in [0, 1]."""
        from src.clip_solver import CLIPSolver
        for pos in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for neg in [0.0, 0.1, 0.5, 0.9, 1.0]:
                conf = CLIPSolver._compute_confidence(pos, neg)
                assert 0.0 <= conf <= 1.0, f"Failed for pos={pos}, neg={neg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
