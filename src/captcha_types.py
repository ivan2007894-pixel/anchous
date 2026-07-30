"""
Type definitions and data models for CAPTCHA solver.
"""

from dataclasses import dataclass, field
from enum import Enum


class CaptchaType(Enum):
    """Supported CAPTCHA types."""
    IMAGE_GRID = "image_grid"        # hCaptcha / reCAPTCHA v2 grid selection
    IMAGE_SINGLE = "image_single"    # Single image classification
    TEXT = "text"                     # Text recognition CAPTCHA


class GridSize(Enum):
    """Common CAPTCHA grid sizes."""
    GRID_3X3 = (3, 3)
    GRID_4X4 = (4, 4)
    GRID_1X1 = (1, 1)  # Single image

    @property
    def rows(self) -> int:
        return self.value[0]

    @property
    def cols(self) -> int:
        return self.value[1]

    @property
    def total_tiles(self) -> int:
        return self.rows * self.cols

    @classmethod
    def from_string(cls, s: str) -> "GridSize":
        """Parse grid size from string like '3x3', '4x4'."""
        mapping = {
            "3x3": cls.GRID_3X3,
            "4x4": cls.GRID_4X4,
            "1x1": cls.GRID_1X1,
        }
        normalized = s.lower().strip()
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(f"Unknown grid size: {s}. Supported: {list(mapping.keys())}")


@dataclass
class TileResult:
    """Result for a single tile in the grid."""
    index: int
    positive_score: float
    negative_score: float
    is_match: bool
    confidence: float  # positive_score - negative_score, normalized

    def __repr__(self) -> str:
        status = "✓" if self.is_match else "✗"
        return f"Tile[{self.index}] {status} conf={self.confidence:.3f}"


@dataclass
class ImageCaptchaResult:
    """Result of solving an image grid CAPTCHA."""
    captcha_type: CaptchaType = CaptchaType.IMAGE_GRID
    category: str = ""
    grid_size: GridSize = GridSize.GRID_3X3
    tiles: list[TileResult] = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)
    raw_prompt: str = ""

    @property
    def confidence_scores(self) -> list[float]:
        """Confidence scores for selected tiles."""
        return [t.confidence for t in self.tiles if t.is_match]

    @property
    def avg_confidence(self) -> float:
        """Average confidence across selected tiles."""
        scores = self.confidence_scores
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "type": self.captcha_type.value,
            "category": self.category,
            "grid": f"{self.grid_size.rows}x{self.grid_size.cols}",
            "selected_tiles": self.selected_indices,
            "confidence": [round(s, 4) for s in self.confidence_scores],
            "avg_confidence": round(self.avg_confidence, 4),
            "all_tiles": [
                {
                    "index": t.index,
                    "positive_score": round(t.positive_score, 4),
                    "negative_score": round(t.negative_score, 4),
                    "is_match": t.is_match,
                    "confidence": round(t.confidence, 4),
                }
                for t in self.tiles
            ],
        }


@dataclass
class TextCaptchaResult:
    """Result of solving a text CAPTCHA."""
    captcha_type: CaptchaType = CaptchaType.TEXT
    text: str = ""
    confidence: float = 0.0
    raw_scores: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.captcha_type.value,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "alternatives": [
                {"text": t, "score": round(s, 4)}
                for t, s in self.raw_scores[:5]
            ],
        }
